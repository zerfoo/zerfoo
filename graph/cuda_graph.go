package graph

import (
	"context"
	"fmt"
	"log"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/tensor"
)

// nonCapturableOps lists instruction op names that must run outside CUDA graph
// capture. These ops perform CPU work or D2H copies that are incompatible with
// stream capture.
//
// EmbeddingLookup: reads token IDs from GPU via .Data() (D2H), does CPU
// float→int conversion.
//
// GroupedQueryAttention was previously non-capturable because it read
// cache.SeqLen() on the CPU for RoPE positions and used CPU-computed offsets
// for KV cache appends. Now that TensorCache uses a GPU-resident counter
// (offset_memcpy kernel) and GQA uses GPU RoPE selection (rope_select kernel),
// all position-dependent state is read from GPU memory at replay time, making
// GQA fully capturable.
var nonCapturableOps = map[string]bool{
	"EmbeddingLookup": true,
}

// CUDAGraphExecutor captures and replays a CUDA graph for an ExecutionPlan.
// It splits the plan into three regions:
//  1. Pre-capture: instructions that trigger D2H copies or have dynamic state
//  2. Capture region: GPU-only, position-independent instructions
//  3. Post-capture: any trailing non-capturable instructions
//
// During replay, regions 1 and 3 run normally while region 2 is replayed
// from the captured graph with near-zero launch overhead.
type CUDAGraphExecutor[T tensor.Numeric] struct {
	plan      *ExecutionPlan[T]
	stream    *cuda.Stream
	graphExec *cuda.GraphExec
	graph     *cuda.Graph
	warmups   int // number of warmup runs before capture
	calls     int // total calls so far
	failed    bool

	// Capture region boundaries: instructions [captureStart, captureEnd)
	// are captured into the CUDA graph. Instructions outside this range
	// run normally every call.
	captureStart int
	captureEnd   int

	// Fixed device buffer for the input token.
	inputDevPtr unsafe.Pointer
	inputBytes  int

	// Cache of GPU tensors for slots that arrive as CPU from pre-capture
	// (e.g. EmbeddingLookup with Q4K). Device addresses are reused across
	// replays so the captured graph stays valid.
	gpuSlotCache map[int]*tensor.TensorNumeric[T]

	// capturedSlots holds the tensors from scratchSlots that were written
	// during the capture run. These tensors' GPU buffers are the destinations
	// of the captured graph's operations. During replay, these must be
	// restored into scratchSlots after PrepareSlots (which resets them)
	// so that GraphLaunch writes to the same buffers and OutputTensor()
	// returns the correct result.
	capturedSlots map[int]*tensor.TensorNumeric[T]

	// onCaptured is called after successful capture, allowing the caller
	// to protect arena allocations from being reclaimed by Reset.
	onCaptured func()
}

// NewCUDAGraphExecutor creates a graph executor for the given plan.
// The optional onCaptured callback is invoked after a successful capture,
// allowing the caller to protect arena allocations from being reclaimed.
func NewCUDAGraphExecutor[T tensor.Numeric](plan *ExecutionPlan[T], streamPtr unsafe.Pointer, warmups int, onCaptured func()) *CUDAGraphExecutor[T] {
	if warmups < 1 {
		warmups = 1
	}

	// Determine capture region: find the first and last capturable instruction.
	n := plan.InstructionCount()
	captureStart := 0
	for captureStart < n && nonCapturableOps[plan.InstructionOpName(captureStart)] {
		captureStart++
	}
	captureEnd := n
	for captureEnd > captureStart && nonCapturableOps[plan.InstructionOpName(captureEnd-1)] {
		captureEnd--
	}

	// Check for non-capturable ops in the middle of the capture range.
	// If any exist, we can't capture a contiguous region.
	for i := captureStart; i < captureEnd; i++ {
		if nonCapturableOps[plan.InstructionOpName(i)] {
			log.Printf("cuda graph: non-capturable op %q at instruction %d inside capture range [%d, %d), disabling graph",
				plan.InstructionOpName(i), i, captureStart, captureEnd)
			return &CUDAGraphExecutor[T]{plan: plan, failed: true}
		}
	}

	if captureStart >= captureEnd {
		log.Printf("cuda graph: no capturable instructions found, graph disabled")
		return &CUDAGraphExecutor[T]{plan: plan, failed: true}
	}
	log.Printf("cuda graph: capture region is instructions [%d, %d) of %d total", captureStart, captureEnd, n)

	return &CUDAGraphExecutor[T]{
		plan:         plan,
		stream:       cuda.StreamFromPtr(streamPtr),
		warmups:      warmups,
		captureStart: captureStart,
		captureEnd:   captureEnd,
		gpuSlotCache: make(map[int]*tensor.TensorNumeric[T]),
		onCaptured:   onCaptured,
	}
}

// Run executes the plan, using graph capture/replay when available.
func (g *CUDAGraphExecutor[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	g.calls++

	if g.failed {
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Phase 1: Warmup runs.
	if g.calls <= g.warmups {
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Phase 2: Capture on first post-warmup call.
	if g.graphExec == nil {
		return g.captureAndRun(ctx, inputs...)
	}

	// Phase 3: Replay.
	return g.replay(ctx, inputs...)
}

// captureAndRun records the capturable region as a CUDA graph.
func (g *CUDAGraphExecutor[T]) captureAndRun(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Prepare slots with the real inputs.
	if err := g.plan.PrepareSlots(inputs...); err != nil {
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Run pre-capture instructions normally (e.g. EmbeddingLookup).
	if g.captureStart > 0 {
		if err := g.plan.RunInstructionRange(ctx, 0, g.captureStart); err != nil {
			g.failed = true
			log.Printf("cuda graph: pre-capture run failed: %v", err)
			return g.plan.RunInstructions(ctx, inputs...)
		}
	}

	// Ensure all slot data is GPU-resident before capture. Pre-capture
	// instructions (e.g. EmbeddingLookup with Q4K embedding tables) may
	// produce CPU tensors. Upload them now so the capture region sees only
	// GPU-resident data and avoids sync D2H copies that break capture.
	g.plan.EnsureSlotsGPU(g.gpuSlotCache)

	// Begin capture for the GPU-heavy region.
	if err := cuda.StreamBeginCapture(g.stream); err != nil {
		log.Printf("cuda graph: begin capture failed: %v", err)
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Run capturable instructions — GPU operations are recorded.
	captureErr := g.plan.RunInstructionRange(ctx, g.captureStart, g.captureEnd)

	// End capture.
	capturedGraph, endErr := cuda.StreamEndCapture(g.stream)
	if endErr != nil || captureErr != nil {
		if endErr != nil {
			log.Printf("cuda graph: end capture failed: %v", endErr)
		}
		if captureErr != nil {
			log.Printf("cuda graph: capture region failed: %v", captureErr)
		}
		g.failed = true
		if capturedGraph != nil {
			_ = cuda.GraphDestroy(capturedGraph)
		}
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.graph = capturedGraph

	// Instantiate executable graph.
	exec, err := cuda.GraphInstantiate(capturedGraph)
	if err != nil {
		log.Printf("cuda graph: instantiate failed: %v", err)
		_ = cuda.GraphDestroy(capturedGraph)
		g.graph = nil
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.graphExec = exec
	log.Printf("cuda graph: captured and instantiated successfully (instructions %d-%d)", g.captureStart, g.captureEnd-1)

	// Save all scratch slots written by captured instructions. These tensors
	// hold the GPU buffers that the captured graph writes to. During replay,
	// we must restore them after PrepareSlots (which resets scratchSlots).
	g.capturedSlots = make(map[int]*tensor.TensorNumeric[T])
	for i := g.captureStart; i < g.captureEnd; i++ {
		outSlot := g.plan.InstructionOutputIdx(i)
		if t := g.plan.ScratchSlot(outSlot); t != nil {
			g.capturedSlots[outSlot] = t
		}
	}

	// Notify the caller to protect arena allocations from reset.
	if g.onCaptured != nil {
		g.onCaptured()
	}

	// Launch the graph once to actually compute the results.
	if err := cuda.GraphLaunch(g.graphExec, g.stream); err != nil {
		return nil, fmt.Errorf("cuda graph: first launch failed: %w", err)
	}
	if err := g.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("cuda graph: sync after first launch: %w", err)
	}

	// Run post-capture instructions if any.
	if g.captureEnd < g.plan.InstructionCount() {
		if err := g.plan.RunInstructionRange(ctx, g.captureEnd, g.plan.InstructionCount()); err != nil {
			return nil, fmt.Errorf("cuda graph: post-capture run failed: %w", err)
		}
	}

	return g.plan.OutputTensor(), nil
}

// replay launches the pre-captured graph with updated input.
func (g *CUDAGraphExecutor[T]) replay(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Prepare slots with new inputs.
	if err := g.plan.PrepareSlots(inputs...); err != nil {
		return nil, fmt.Errorf("cuda graph replay: prepare slots: %w", err)
	}

	// Run pre-capture instructions normally.
	if g.captureStart > 0 {
		if err := g.plan.RunInstructionRange(ctx, 0, g.captureStart); err != nil {
			return nil, fmt.Errorf("cuda graph replay: pre-capture: %w", err)
		}
	}

	// Ensure pre-capture outputs are GPU-resident before replay.
	g.plan.EnsureSlotsGPU(g.gpuSlotCache)

	// Restore captured slots. PrepareSlots resets scratchSlots from p.slots,
	// which clears the tensors allocated during capture. The captured CUDA
	// graph writes to those GPU buffers, so we must restore the tensor
	// pointers so that OutputTensor() returns the correct result.
	for idx, t := range g.capturedSlots {
		g.plan.SetScratchSlot(idx, t)
	}

	// Replay the captured graph.
	if err := cuda.GraphLaunch(g.graphExec, g.stream); err != nil {
		return nil, fmt.Errorf("cuda graph: launch failed: %w", err)
	}
	if err := g.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("cuda graph: sync failed: %w", err)
	}

	// Run post-capture instructions if any.
	if g.captureEnd < g.plan.InstructionCount() {
		if err := g.plan.RunInstructionRange(ctx, g.captureEnd, g.plan.InstructionCount()); err != nil {
			return nil, fmt.Errorf("cuda graph replay: post-capture: %w", err)
		}
	}

	return g.plan.OutputTensor(), nil
}

// Destroy releases the CUDA graph resources.
func (g *CUDAGraphExecutor[T]) Destroy() {
	if g.graphExec != nil {
		_ = cuda.GraphExecDestroy(g.graphExec)
		g.graphExec = nil
	}
	if g.graph != nil {
		_ = cuda.GraphDestroy(g.graph)
		g.graph = nil
	}
	if g.inputDevPtr != nil {
		_ = cuda.Free(g.inputDevPtr)
		g.inputDevPtr = nil
	}
}
