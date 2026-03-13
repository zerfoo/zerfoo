package graph

import (
	"context"
	"fmt"
	"log"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/tensor"
)

// nonCapturableOps lists instruction op names that trigger D2H copies and
// must run outside CUDA graph capture. EmbeddingLookup reads token IDs
// from GPU via .Data() and does CPU float→int conversion.
var nonCapturableOps = map[string]bool{
	"EmbeddingLookup": true,
}

// CUDAGraphExecutor captures and replays a CUDA graph for an ExecutionPlan.
// It splits the plan into three regions:
//  1. Pre-capture: instructions that trigger D2H copies (e.g. EmbeddingLookup)
//  2. Capture region: GPU-only instructions (MatMul, RMSNorm, GQA, etc.)
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
}

// NewCUDAGraphExecutor creates a graph executor for the given plan.
func NewCUDAGraphExecutor[T tensor.Numeric](plan *ExecutionPlan[T], streamPtr unsafe.Pointer, warmups int) *CUDAGraphExecutor[T] {
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
