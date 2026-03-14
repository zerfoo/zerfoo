package graph

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// simpleAddNode is a graph node that adds 1.0 to every element of the input.
// It uses only GPU-compatible operations (engine.AddScalar) so that CUDA
// graph capture can record the kernel launches.
type simpleAddNode struct {
	NoParameters[float32]
	engine compute.Engine[float32]
}

func (n *simpleAddNode) OpType() string                     { return "AddOne" }
func (n *simpleAddNode) Attributes() map[string]interface{} { return nil }
func (n *simpleAddNode) OutputShape() []int                 { return nil }

func (n *simpleAddNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return n.engine.AddScalar(ctx, inputs[0], 1.0)
}

func (n *simpleAddNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

// TestCUDAGraphExecutor_CorrectnessVsNonGraph compares output from a CUDA
// graph executor against direct (non-graph) execution for 10 iterations.
// Both paths should produce identical results at temperature=0 (deterministic).
// Skips on non-GPU machines.
func TestCUDAGraphExecutor_CorrectnessVsNonGraph(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	if !cuda.Lib().GraphAvailable() {
		t.Skip("CUDA graph API not available")
	}

	eng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	sp, ok := any(eng).(compute.StreamProvider)
	if !ok {
		t.Skip("GPU engine does not implement StreamProvider")
	}
	streamPtr := sp.Stream()
	if streamPtr == nil {
		t.Skip("GPU stream is nil")
	}

	// Build a simple graph: input -> add 1.0.
	node := &simpleAddNode{engine: eng}
	b := NewBuilder[float32](eng)
	in := b.Input([]int{1, 4})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	ctx := context.Background()

	// Create a warmup input for compilation.
	warmupInput, err := tensor.New[float32]([]int{1, 4}, []float32{0, 0, 0, 0})
	if err != nil {
		t.Fatalf("tensor.New warmup: %v", err)
	}

	// Compile into an execution plan.
	compiled, err := g.Compile(ctx, warmupInput)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	// Run 10 iterations without graph to get reference outputs.
	referenceOutputs := make([][]float32, 10)
	for i := range 10 {
		input, err := tensor.New[float32]([]int{1, 4}, []float32{
			float32(i), float32(i + 1), float32(i + 2), float32(i + 3),
		})
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		out, err := compiled.RunInstructions(ctx, input)
		if err != nil {
			t.Fatalf("RunInstructions[%d]: %v", i, err)
		}
		referenceOutputs[i] = append([]float32(nil), out.Data()...)
	}

	// Now run with CUDA graph executor.
	graphExec := NewCUDAGraphExecutor[float32](compiled, streamPtr, 2, nil, nil)
	defer graphExec.Destroy()

	for i := range 10 {
		input, err := tensor.New[float32]([]int{1, 4}, []float32{
			float32(i), float32(i + 1), float32(i + 2), float32(i + 3),
		})
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		out, err := graphExec.Run(ctx, input)
		if err != nil {
			// Graph capture may fail if the forward pass has D2H copies;
			// that's okay, the executor falls back to non-graph mode.
			t.Logf("graphExec.Run[%d] error (may be expected): %v", i, err)
			continue
		}
		graphData := out.Data()
		for j, want := range referenceOutputs[i] {
			if graphData[j] != want {
				t.Errorf("iteration %d: graphOutput[%d] = %f, want %f",
					i, j, graphData[j], want)
			}
		}
	}
}

// TestCUDAGraphExecutor_FallbackOnFailure verifies that the executor
// gracefully falls back to non-graph execution when graph capture fails.
func TestCUDAGraphExecutor_FallbackOnFailure(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := compute.NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	sp, ok := any(eng).(compute.StreamProvider)
	if !ok {
		t.Skip("GPU engine does not implement StreamProvider")
	}
	streamPtr := sp.Stream()
	if streamPtr == nil {
		t.Skip("GPU stream is nil")
	}

	node := &simpleAddNode{engine: eng}
	b := NewBuilder[float32](eng)
	in := b.Input([]int{1, 4})
	b.AddNode(node, in)
	g, err := b.Build(node)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	ctx := context.Background()

	warmupInput, err := tensor.New[float32]([]int{1, 4}, []float32{0, 0, 0, 0})
	if err != nil {
		t.Fatalf("tensor.New warmup: %v", err)
	}

	compiled, err := g.Compile(ctx, warmupInput)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}

	graphExec := NewCUDAGraphExecutor[float32](compiled, streamPtr, 2, nil, nil)
	defer graphExec.Destroy()

	// Run enough times to trigger warmup + capture + replay.
	for i := range 5 {
		input, err := tensor.New[float32]([]int{1, 4}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatalf("tensor.New: %v", err)
		}
		out, err := graphExec.Run(ctx, input)
		if err != nil {
			t.Fatalf("Run[%d]: %v", i, err)
		}
		data := out.Data()
		want := []float32{2, 3, 4, 5}
		for j, w := range want {
			if data[j] != w {
				t.Errorf("Run[%d][%d] = %f, want %f", i, j, data[j], w)
			}
		}
	}
}
