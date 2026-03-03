package gather

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
	g := New[float32](engine)

	if g == nil {
		t.Fatal("New returned nil")
	}
	if g.engine == nil {
		t.Error("engine is nil")
	}
	if g.weights != nil {
		t.Error("weights should be nil for New()")
	}
}

func TestNewWithWeights(t *testing.T) {
	engine := newEngine()
	w, err := tensor.New[float32]([]int{10, 4}, nil)
	if err != nil {
		t.Fatalf("failed to create weights: %v", err)
	}

	g := NewWithWeights(engine, w)
	if g == nil {
		t.Fatal("NewWithWeights returned nil")
	}
	if !g.HasEmbeddedWeights() {
		t.Error("HasEmbeddedWeights() = false, want true")
	}
}

func TestHasEmbeddedWeights(t *testing.T) {
	engine := newEngine()

	tests := []struct {
		name string
		g    *Gather[float32]
		want bool
	}{
		{"without weights", New[float32](engine), false},
		{"with weights", func() *Gather[float32] {
			w, _ := tensor.New[float32]([]int{10, 4}, nil)
			return NewWithWeights(engine, w)
		}(), true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.g.HasEmbeddedWeights(); got != tc.want {
				t.Errorf("HasEmbeddedWeights() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestOpType(t *testing.T) {
	g := New[float32](newEngine())
	if got := g.OpType(); got != "Gather" {
		t.Errorf("OpType() = %q, want %q", got, "Gather")
	}
}

func TestAttributes(t *testing.T) {
	g := New[float32](newEngine())
	if got := g.Attributes(); got != nil {
		t.Errorf("Attributes() = %v, want nil", got)
	}
}

func TestParameters(t *testing.T) {
	g := New[float32](newEngine())
	if got := g.Parameters(); got != nil {
		t.Errorf("Parameters() = %v, want nil", got)
	}
}

func TestOutputShape(t *testing.T) {
	g := New[float32](newEngine())
	// Before forward, output shape should be nil
	if got := g.OutputShape(); got != nil {
		t.Errorf("OutputShape() before forward = %v, want nil", got)
	}
}

func TestForward_WithEmbeddedWeights_1DIndices(t *testing.T) {
	engine := newEngine()

	// Create a simple weight matrix [4, 3]
	wData := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	w, err := tensor.New[float32]([]int{4, 3}, wData)
	if err != nil {
		t.Fatalf("failed to create weights: %v", err)
	}

	g := NewWithWeights(engine, w)
	ctx := context.Background()

	// 1D indices [2] -> should be reshaped to [1, 2]
	idxData := []float32{0, 2}
	indices, err := tensor.New[float32]([]int{2}, idxData)
	if err != nil {
		t.Fatalf("failed to create indices: %v", err)
	}

	output, err := g.Forward(ctx, indices)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Output shape: [1, 2, 3]
	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 3 {
		t.Errorf("output shape = %v, want [1, 2, 3]", shape)
	}

	// Verify OutputShape is updated
	if os := g.OutputShape(); os == nil {
		t.Error("OutputShape() after forward is nil")
	}
}

func TestForward_WithEmbeddedWeights_2DIndices(t *testing.T) {
	engine := newEngine()

	wData := []float32{1, 2, 3, 4, 5, 6}
	w, err := tensor.New[float32]([]int{3, 2}, wData)
	if err != nil {
		t.Fatalf("failed to create weights: %v", err)
	}

	g := NewWithWeights(engine, w)
	ctx := context.Background()

	// 2D indices [1, 2]
	idxData := []float32{0, 2}
	indices, err := tensor.New[float32]([]int{1, 2}, idxData)
	if err != nil {
		t.Fatalf("failed to create indices: %v", err)
	}

	output, err := g.Forward(ctx, indices)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 2 {
		t.Errorf("output shape = %v, want [1, 2, 2]", shape)
	}
}

func TestForward_WithEmbeddedWeights_WrongInputCount(t *testing.T) {
	engine := newEngine()
	w, _ := tensor.New[float32]([]int{4, 3}, nil)
	g := NewWithWeights(engine, w)
	ctx := context.Background()

	i1, _ := tensor.New[float32]([]int{2}, nil)
	i2, _ := tensor.New[float32]([]int{2}, nil)

	_, err := g.Forward(ctx, i1, i2)
	if err == nil {
		t.Error("Forward with 2 inputs on embedded-weights gather should fail")
	}
}

func TestForward_WithoutWeights_WrongInputCount(t *testing.T) {
	engine := newEngine()

	tests := []struct {
		name   string
		inputs []*tensor.TensorNumeric[float32]
	}{
		{"zero inputs", nil},
		{"one input", func() []*tensor.TensorNumeric[float32] {
			i, _ := tensor.New[float32]([]int{2}, nil)
			return []*tensor.TensorNumeric[float32]{i}
		}()},
		{"three inputs", func() []*tensor.TensorNumeric[float32] {
			i1, _ := tensor.New[float32]([]int{2}, nil)
			i2, _ := tensor.New[float32]([]int{2}, nil)
			i3, _ := tensor.New[float32]([]int{2}, nil)
			return []*tensor.TensorNumeric[float32]{i1, i2, i3}
		}()},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g := New[float32](engine)
			_, err := g.Forward(context.Background(), tc.inputs...)
			if err == nil {
				t.Error("Forward should fail with wrong input count")
			}
		})
	}
}

func TestForward_WithoutWeights_WrongIndicesType(t *testing.T) {
	engine := newEngine()
	g := New[float32](engine)
	ctx := context.Background()

	// Both inputs are float32, but indices needs to be *tensor.TensorNumeric[int]
	params, _ := tensor.New[float32]([]int{4, 3}, nil)
	indices, _ := tensor.New[float32]([]int{1, 2}, nil)

	_, err := g.Forward(ctx, params, indices)
	if err == nil {
		t.Error("Forward with float32 indices should fail (expects int)")
	}
}

func TestBackward_WrongIndicesType(t *testing.T) {
	engine := newEngine()
	g := New[float32](engine)
	ctx := context.Background()

	params, _ := tensor.New[float32]([]int{4, 3}, nil)
	indices, _ := tensor.New[float32]([]int{1, 2}, nil)
	dOut, _ := tensor.New[float32]([]int{1, 2, 3}, nil)

	_, err := g.Backward(ctx, types.FullBackprop, dOut, params, indices)
	if err == nil {
		t.Error("Backward with float32 indices should fail (expects int)")
	}
}

func newIntEngine() compute.Engine[int] {
	return compute.NewCPUEngine[int](numeric.IntOps{})
}

func TestForward_WithoutWeights_IntIndices(t *testing.T) {
	engine := newIntEngine()
	g := New[int](engine)
	ctx := context.Background()

	// params: [4, 3] embedding table
	params, err := tensor.New[int]([]int{4, 3}, []int{
		10, 20, 30,
		40, 50, 60,
		70, 80, 90,
		100, 110, 120,
	})
	if err != nil {
		t.Fatalf("failed to create params: %v", err)
	}

	// indices: [1, 2] (fetch rows 0 and 2)
	indices, err := tensor.New[int]([]int{1, 2}, []int{0, 2})
	if err != nil {
		t.Fatalf("failed to create indices: %v", err)
	}

	output, err := g.Forward(ctx, params, indices)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Output shape: [1, 2, 3]
	shape := output.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 2 || shape[2] != 3 {
		t.Errorf("output shape = %v, want [1, 2, 3]", shape)
	}
}

func TestBackward_WithIntIndices(t *testing.T) {
	engine := newIntEngine()
	g := New[int](engine)
	ctx := context.Background()

	params, _ := tensor.New[int]([]int{4, 3}, nil)
	indices, _ := tensor.New[int]([]int{1, 2}, []int{0, 2})
	dOut, _ := tensor.New[int]([]int{2, 3}, []int{1, 1, 1, 1, 1, 1})

	grads, err := g.Backward(ctx, types.FullBackprop, dOut, params, indices)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if len(grads) != 2 {
		t.Fatalf("Backward returned %d gradients, want 2", len(grads))
	}

	// First gradient should have same shape as params
	if grads[0] == nil {
		t.Fatal("gradient[0] is nil")
	}
	gShape := grads[0].Shape()
	if len(gShape) != 2 || gShape[0] != 4 || gShape[1] != 3 {
		t.Errorf("gradient[0] shape = %v, want [4, 3]", gShape)
	}

	// Second gradient should be nil (indices don't have gradients)
	if grads[1] != nil {
		t.Error("gradient[1] should be nil")
	}
}

func TestGraphNodeInterface(t *testing.T) {
	var _ graph.Node[float32] = (*Gather[float32])(nil)
}
