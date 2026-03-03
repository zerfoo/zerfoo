package reducesum

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func newEngine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

func TestNew(t *testing.T) {
	engine := newEngine()
	r := New[float32](engine, []int{0}, true)

	if r == nil {
		t.Fatal("New returned nil")
	}
	if r.engine == nil {
		t.Error("engine is nil")
	}
}

func TestOpType(t *testing.T) {
	r := New[float32](newEngine(), []int{0}, true)
	if got := r.OpType(); got != "ReduceSum" {
		t.Errorf("OpType() = %q, want %q", got, "ReduceSum")
	}
}

func TestAttributes(t *testing.T) {
	r := New[float32](newEngine(), []int{1, 2}, false)

	attrs := r.Attributes()
	if attrs == nil {
		t.Fatal("Attributes() returned nil")
	}

	axes, ok := attrs["axes"].([]int)
	if !ok {
		t.Fatal("axes attribute missing or wrong type")
	}
	if len(axes) != 2 || axes[0] != 1 || axes[1] != 2 {
		t.Errorf("axes = %v, want [1, 2]", axes)
	}

	keepDims, ok := attrs["keepdims"].(bool)
	if !ok {
		t.Fatal("keepdims attribute missing or wrong type")
	}
	if keepDims {
		t.Error("keepdims = true, want false")
	}
}

func TestParameters(t *testing.T) {
	r := New[float32](newEngine(), []int{0}, true)
	if got := r.Parameters(); got != nil {
		t.Errorf("Parameters() = %v, want nil", got)
	}
}

func TestOutputShape_BeforeForward(t *testing.T) {
	r := New[float32](newEngine(), []int{0}, true)
	if got := r.OutputShape(); got != nil {
		t.Errorf("OutputShape() before forward = %v, want nil", got)
	}
}

func TestForward_SingleAxis_KeepDims(t *testing.T) {
	engine := newEngine()
	r := New[float32](engine, []int{1}, true)
	ctx := context.Background()

	// Input: [2, 3]
	input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	output, err := r.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// With keepDims=true, axis 1 reduced: output shape [2, 1]
	shape := output.Shape()
	if len(shape) != 2 || shape[0] != 2 || shape[1] != 1 {
		t.Errorf("output shape = %v, want [2, 1]", shape)
	}

	// Verify OutputShape is updated
	if os := r.OutputShape(); os == nil {
		t.Error("OutputShape() after forward is nil")
	}
}

func TestForward_SingleAxis_NoKeepDims(t *testing.T) {
	engine := newEngine()
	r := New[float32](engine, []int{1}, false)
	ctx := context.Background()

	input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	output, err := r.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Without keepDims, axis 1 removed: output shape [2]
	shape := output.Shape()
	if len(shape) != 1 || shape[0] != 2 {
		t.Errorf("output shape = %v, want [2]", shape)
	}
}

func TestForward_AllAxes(t *testing.T) {
	engine := newEngine()
	r := New[float32](engine, []int{}, true)
	ctx := context.Background()

	input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	output, err := r.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward with empty axes failed: %v", err)
	}

	if output == nil {
		t.Fatal("output is nil")
	}
}

func TestForward_Axis0_KeepDims(t *testing.T) {
	engine := newEngine()
	r := New[float32](engine, []int{0}, true)
	ctx := context.Background()

	input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	output, err := r.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] != 3 {
		t.Errorf("output shape = %v, want [1, 3]", shape)
	}
}

func TestBackward_KeepDims(t *testing.T) {
	engine := newEngine()
	r := New[float32](engine, []int{1}, true)
	ctx := context.Background()

	input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	_, err = r.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Output gradient shape matches forward output: [2, 1]
	dOut, err := tensor.New[float32]([]int{2, 1}, []float32{1, 1})
	if err != nil {
		t.Fatalf("failed to create dOut: %v", err)
	}

	grads, err := r.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("Backward returned %d gradients, want 1", len(grads))
	}

	// Gradient should be broadcast back to input shape [2, 3]
	gShape := grads[0].Shape()
	if len(gShape) != 2 || gShape[0] != 2 || gShape[1] != 3 {
		t.Errorf("gradient shape = %v, want [2, 3]", gShape)
	}
}

func TestBackward_NoKeepDims(t *testing.T) {
	engine := newEngine()
	r := New[float32](engine, []int{1}, false)
	ctx := context.Background()

	input, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	_, err = r.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Without keepDims, forward output shape is [2]
	dOut, err := tensor.New[float32]([]int{2}, []float32{1, 1})
	if err != nil {
		t.Fatalf("failed to create dOut: %v", err)
	}

	grads, err := r.Backward(ctx, types.FullBackprop, dOut, input)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if len(grads) != 1 {
		t.Fatalf("Backward returned %d gradients, want 1", len(grads))
	}

	gShape := grads[0].Shape()
	if len(gShape) != 2 || gShape[0] != 2 || gShape[1] != 3 {
		t.Errorf("gradient shape = %v, want [2, 3]", gShape)
	}
}

func TestBackward_WrongInputCount(t *testing.T) {
	r := New[float32](newEngine(), []int{0}, true)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Backward with no inputs should panic")
		}
	}()

	dOut, _ := tensor.New[float32]([]int{1}, nil)
	_, _ = r.Backward(context.Background(), types.FullBackprop, dOut)
}

func TestGraphNodeInterface(t *testing.T) {
	var _ graph.Node[float32] = (*ReduceSum[float32])(nil)
}
