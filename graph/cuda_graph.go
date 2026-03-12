package graph

import (
	"context"
	"fmt"
	"log"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/tensor"
)

// CUDAGraphExecutor captures and replays a CUDA graph for an ExecutionPlan.
// On the first Run call, it executes the plan normally to establish stable
// memory addresses. On the second call, it captures all GPU operations into
// a CUDA graph and instantiates it. On subsequent calls, it replays the
// graph with near-zero launch overhead (~15us total vs ~7us per kernel).
//
// Requirements:
//   - The execution plan must use an arena allocator that resets between tokens
//     (so device pointer addresses are deterministic).
//   - All model weights must already be on GPU.
type CUDAGraphExecutor[T tensor.Numeric] struct {
	plan      *ExecutionPlan[T]
	stream    *cuda.Stream
	graphExec *cuda.GraphExec
	graph     *cuda.Graph
	warmups   int // number of warmup runs before capture
	calls     int // total calls so far
	failed    bool // true if capture failed permanently

	// Fixed device buffer for the input token. Updated via H2D memcpy
	// before each graph launch to avoid capturing H2D inside the graph.
	inputDevPtr unsafe.Pointer
	inputBytes  int
}

// NewCUDAGraphExecutor creates a graph executor for the given plan.
// streamPtr must be the cudaStream_t from the GPU engine.
// warmups controls how many normal runs happen before graph capture (minimum 1).
func NewCUDAGraphExecutor[T tensor.Numeric](plan *ExecutionPlan[T], streamPtr unsafe.Pointer, warmups int) *CUDAGraphExecutor[T] {
	if warmups < 1 {
		warmups = 1
	}
	return &CUDAGraphExecutor[T]{
		plan:    plan,
		stream:  cuda.StreamFromPtr(streamPtr),
		warmups: warmups,
	}
}

// Run executes the plan, using graph capture/replay when available.
// It handles three phases:
//  1. Warmup: run plan normally to establish arena addresses
//  2. Capture: record GPU operations into a CUDA graph
//  3. Replay: launch the captured graph (near-zero overhead)
func (g *CUDAGraphExecutor[T]) Run(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	g.calls++

	// If capture failed, fall back to normal execution permanently.
	if g.failed {
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Phase 1: Warmup runs - execute normally.
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

// captureAndRun records the plan execution as a CUDA graph.
func (g *CUDAGraphExecutor[T]) captureAndRun(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Pre-stage input on GPU at a fixed address.
	if err := g.stageInput(inputs); err != nil {
		log.Printf("cuda graph: failed to stage input, falling back: %v", err)
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Replace input with GPU-backed tensor so no H2D copies happen
	// inside the captured region.
	gpuInputs, err := g.gpuInputs(inputs)
	if err != nil {
		log.Printf("cuda graph: failed to create GPU inputs, falling back: %v", err)
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Begin capture. All GPU operations on this stream are now recorded.
	if err := cuda.StreamBeginCapture(g.stream); err != nil {
		log.Printf("cuda graph: begin capture failed, falling back: %v", err)
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}

	// Execute plan. Host-side code (slot manipulation) runs normally.
	// GPU operations (kernel launches, D2D memcpys) are recorded, not executed.
	_, runErr := g.plan.RunInstructions(ctx, gpuInputs...)

	// End capture and get the graph.
	capturedGraph, captureErr := cuda.StreamEndCapture(g.stream)
	if captureErr != nil || runErr != nil {
		// Capture failed. This typically happens when the forward pass
		// includes synchronous cudaMemcpy calls (e.g., GPUStorage.TrySlice)
		// that conflict with stream capture mode.
		if captureErr != nil {
			log.Printf("cuda graph: end capture failed: %v", captureErr)
		}
		if runErr != nil {
			log.Printf("cuda graph: plan run during capture failed: %v", runErr)
		}
		g.failed = true
		if capturedGraph != nil {
			_ = cuda.GraphDestroy(capturedGraph)
		}
		// Re-run the plan normally to get the actual result for this token.
		// The stream is restored to normal mode after EndCapture (even on error).
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.graph = capturedGraph

	// Instantiate executable graph.
	exec, err := cuda.GraphInstantiate(capturedGraph)
	if err != nil {
		log.Printf("cuda graph: instantiate failed, falling back: %v", err)
		_ = cuda.GraphDestroy(capturedGraph)
		g.graph = nil
		g.failed = true
		return g.plan.RunInstructions(ctx, inputs...)
	}
	g.graphExec = exec

	log.Printf("cuda graph: captured and instantiated successfully")

	// During capture, GPU operations were recorded but not executed.
	// Launch the graph once to actually compute the result.
	if err := cuda.GraphLaunch(g.graphExec, g.stream); err != nil {
		return nil, fmt.Errorf("cuda graph: first launch failed: %w", err)
	}
	if err := g.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("cuda graph: sync after first launch failed: %w", err)
	}

	return g.plan.OutputTensor(), nil
}

// replay launches the pre-captured graph with updated input.
func (g *CUDAGraphExecutor[T]) replay(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	_ = ctx // context checked by caller

	// Update the fixed input buffer with new token data.
	if err := g.updateInput(inputs); err != nil {
		return nil, fmt.Errorf("cuda graph replay: update input: %w", err)
	}

	// Launch the captured graph. This replays all ~338 kernel launches
	// with a single driver call (~15us vs ~2.37ms for individual launches).
	if err := cuda.GraphLaunch(g.graphExec, g.stream); err != nil {
		return nil, fmt.Errorf("cuda graph: launch failed: %w", err)
	}
	if err := g.stream.Synchronize(); err != nil {
		return nil, fmt.Errorf("cuda graph: sync failed: %w", err)
	}

	return g.plan.OutputTensor(), nil
}

// stageInput allocates a fixed device buffer and uploads the first input.
func (g *CUDAGraphExecutor[T]) stageInput(inputs []*tensor.TensorNumeric[T]) error {
	if len(inputs) == 0 {
		return fmt.Errorf("no inputs")
	}

	input := inputs[0]
	data := input.Data()
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	byteSize := len(data) * elemSize

	if g.inputDevPtr == nil {
		devPtr, err := cuda.Malloc(byteSize)
		if err != nil {
			return fmt.Errorf("malloc input buffer: %w", err)
		}
		g.inputDevPtr = devPtr
		g.inputBytes = byteSize
	}

	src := unsafe.Pointer(unsafe.SliceData(data))
	return cuda.Memcpy(g.inputDevPtr, src, byteSize, cuda.MemcpyHostToDevice)
}

// updateInput copies new token data to the fixed device buffer.
func (g *CUDAGraphExecutor[T]) updateInput(inputs []*tensor.TensorNumeric[T]) error {
	if len(inputs) == 0 {
		return fmt.Errorf("no inputs")
	}
	data := inputs[0].Data()
	src := unsafe.Pointer(unsafe.SliceData(data))
	var zero T
	elemSize := int(unsafe.Sizeof(zero))
	byteSize := len(data) * elemSize
	if byteSize > g.inputBytes {
		return fmt.Errorf("input size %d exceeds buffer %d", byteSize, g.inputBytes)
	}
	return cuda.Memcpy(g.inputDevPtr, src, byteSize, cuda.MemcpyHostToDevice)
}

// gpuInputs creates GPU-backed tensors pointing at the fixed device buffer.
func (g *CUDAGraphExecutor[T]) gpuInputs(inputs []*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs")
	}
	gpuIn := make([]*tensor.TensorNumeric[T], len(inputs))

	shape := inputs[0].Shape()
	gs, err := tensor.NewGPUStorageFromPtr[T](g.inputDevPtr, inputs[0].Size())
	if err != nil {
		return nil, err
	}
	gpuIn[0], err = tensor.NewWithStorage(shape, gs)
	if err != nil {
		return nil, err
	}

	for i := 1; i < len(inputs); i++ {
		gpuIn[i] = inputs[i]
	}
	return gpuIn, nil
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
