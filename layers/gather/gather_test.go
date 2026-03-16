package gather

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
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

func TestForward_WithoutWeights_Float32Indices(t *testing.T) {
	engine := newEngine()
	g := New[float32](engine)
	ctx := context.Background()

	// General Gather now accepts float32 indices (converts to int internally).
	params, _ := tensor.New[float32]([]int{4, 3}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	indices, _ := tensor.New[float32]([]int{}, []float32{1}) // scalar index

	out, err := g.Forward(ctx, params, indices)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if out == nil {
		t.Fatal("output is nil")
	}
}

func TestForward_ScalarGatherFrom1DData(t *testing.T) {
	engine := newEngine()
	g := New[float32](engine)
	ctx := context.Background()

	// 1D data (e.g. output of Shape op) with scalar index → 0D scalar result.
	params, _ := tensor.New[float32]([]int{3}, []float32{10, 20, 30})
	indices, _ := tensor.New[float32]([]int{}, []float32{1}) // scalar index

	out, err := g.Forward(ctx, params, indices)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Output should be 0D (scalar).
	if len(out.Shape()) != 0 {
		t.Errorf("output shape = %v, want [] (0D scalar)", out.Shape())
	}

	// Value should be 20 (element at index 1).
	if out.Data()[0] != 20 {
		t.Errorf("output value = %v, want 20", out.Data()[0])
	}
}

func TestForward_ScalarGatherNegativeIndex(t *testing.T) {
	engine := newEngine()
	g := New[float32](engine)
	ctx := context.Background()

	// Negative index: -1 means last element.
	params, _ := tensor.New[float32]([]int{4}, []float32{100, 200, 300, 400})
	indices, _ := tensor.New[float32]([]int{}, []float32{-1})

	out, err := g.Forward(ctx, params, indices)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if len(out.Shape()) != 0 {
		t.Errorf("output shape = %v, want [] (0D scalar)", out.Shape())
	}
	if out.Data()[0] != 400 {
		t.Errorf("output value = %v, want 400", out.Data()[0])
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

func TestNewWithIndices(t *testing.T) {
	engine := newEngine()
	idx, err := tensor.New[int]([]int{1}, []int{1})
	if err != nil {
		t.Fatalf("failed to create indices: %v", err)
	}
	g := NewWithIndices[float32](engine, idx)
	if g == nil {
		t.Fatal("NewWithIndices returned nil")
	}
	if g.HasEmbeddedWeights() {
		t.Error("HasEmbeddedWeights() should be false for NewWithIndices")
	}
}

func TestForward_WithEmbeddedIndices_ScalarGather(t *testing.T) {
	engine := newEngine()

	// Embedded index: scalar 1 → pick element at index 1.
	idx, _ := tensor.New[int]([]int{1}, []int{1})
	g := NewWithIndices[float32](engine, idx)

	// Data input: 1D shape tensor [10, 20, 30].
	data, _ := tensor.New[float32]([]int{3}, []float32{10, 20, 30})

	out, err := g.Forward(context.Background(), data)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if len(out.Shape()) != 0 {
		t.Errorf("output shape = %v, want [] (0D scalar)", out.Shape())
	}
	if out.Data()[0] != 20 {
		t.Errorf("output value = %v, want 20", out.Data()[0])
	}
}

func TestForward_WithEmbeddedIndices_WrongInputCount(t *testing.T) {
	engine := newEngine()
	idx, _ := tensor.New[int]([]int{1}, []int{0})
	g := NewWithIndices[float32](engine, idx)

	d1, _ := tensor.New[float32]([]int{3}, nil)
	d2, _ := tensor.New[float32]([]int{3}, nil)

	_, err := g.Forward(context.Background(), d1, d2)
	if err == nil {
		t.Error("Forward with 2 inputs on embedded-indices gather should fail")
	}
}

func TestBuildGather_WithEmbeddedIndices(t *testing.T) {
	engine := newEngine()
	attrs := map[string]interface{}{
		"axis":                  0,
		"/Constant_output_0": []int64{2},
	}
	node, err := BuildGather[float32](engine, numeric.Float32Ops{}, "/Gather", nil, attrs)
	if err != nil {
		t.Fatalf("BuildGather: %v", err)
	}
	if node.OpType() != "Gather" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Gather")
	}

	// Verify it works with 1 input (data from Shape).
	data, _ := tensor.New[float32]([]int{4}, []float32{10, 20, 30, 40})
	out, err := node.Forward(context.Background(), data)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if out.Data()[0] != 30 {
		t.Errorf("output = %v, want 30", out.Data()[0])
	}
}

func TestGraphNodeInterface(t *testing.T) {
	var _ graph.Node[float32] = (*Gather[float32])(nil)
}

func TestForward_ScalarGather_OutOfBoundsClamping(t *testing.T) {
	engine := newEngine()
	ctx := context.Background()

	tests := []struct {
		name      string
		pShape    []int
		pData     []float32
		idx       float32
		wantFirst float32
	}{
		{
			// start >= len(data): idx=3 on [3,2] -> start=6, len=6 -> clamp
			name:      "start_at_end",
			pShape:    []int{3, 2},
			pData:     []float32{10, 20, 30, 40, 50, 60},
			idx:       3,
			wantFirst: 50, // clamped to last row [50, 60]
		},
		{
			// end > len(data): idx=2 on [3] with 1D data -> stride=1, end=3
			// but start=2, end=3 -> end <= len(3), no clamp needed; try larger
			// idx=5 on [3,2] -> start=10, end=12 > len=6 -> clamp
			name:      "end_past_data",
			pShape:    []int{3, 2},
			pData:     []float32{10, 20, 30, 40, 50, 60},
			idx:       5,
			wantFirst: 50, // clamped
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			g := New[float32](engine)
			params, _ := tensor.New[float32](tc.pShape, tc.pData)
			indices, _ := tensor.New[float32]([]int{}, []float32{tc.idx})

			out, err := g.Forward(ctx, params, indices)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
			if out.Data()[0] != tc.wantFirst {
				t.Errorf("output[0] = %v, want %v", out.Data()[0], tc.wantFirst)
			}
		})
	}
}

func TestForward_GeneralGather_1DIndicesReshape(t *testing.T) {
	engine := newEngine()
	g := New[float32](engine)
	ctx := context.Background()

	// 2D params, 1D indices (not scalar) -> triggers reshape to [1, N]
	params, _ := tensor.New[float32]([]int{4, 3}, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	// 1D indices with >1 element -> hits the reshape path at line 172-181
	indices, _ := tensor.New[float32]([]int{2}, []float32{0, 2})

	out, err := g.Forward(ctx, params, indices)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	// After reshape: indices [1,2], params [4,3] -> output [1,2,3]
	if len(out.Shape()) != 3 || out.Shape()[0] != 1 || out.Shape()[1] != 2 || out.Shape()[2] != 3 {
		t.Errorf("output shape = %v, want [1, 2, 3]", out.Shape())
	}
}
