package core

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

func makeEngine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

func makeOps() numeric.Float32Ops {
	return numeric.Float32Ops{}
}

func makeTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	out, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatalf("makeTensor: %v", err)
	}
	return out
}

// ---------- Mul ----------

func TestMul(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	m := NewMul(engine)

	a := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
	b := makeTensor(t, []int{2, 2}, []float32{5, 6, 7, 8})

	// Forward
	out, err := m.Forward(ctx, a, b)
	if err != nil {
		t.Fatalf("Mul Forward: %v", err)
	}
	want := []float32{5, 12, 21, 32}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("Mul Forward[%d] = %v, want %v", i, v, want[i])
		}
	}

	// OutputShape
	if s := m.OutputShape(); len(s) != 2 || s[0] != 2 || s[1] != 2 {
		t.Errorf("Mul OutputShape = %v, want [2 2]", s)
	}

	// Parameters
	if p := m.Parameters(); p != nil {
		t.Errorf("Mul Parameters = %v, want nil", p)
	}

	// OpType
	if op := m.OpType(); op != "Mul" {
		t.Errorf("Mul OpType = %q, want %q", op, "Mul")
	}

	// Attributes
	if a := m.Attributes(); a != nil {
		t.Errorf("Mul Attributes = %v, want nil", a)
	}

	// Backward
	grad := makeTensor(t, []int{2, 2}, []float32{1, 1, 1, 1})
	grads, err := m.Backward(ctx, types.FullBackprop, grad, a, b)
	if err != nil {
		t.Fatalf("Mul Backward: %v", err)
	}
	if len(grads) != 2 {
		t.Fatalf("Mul Backward len = %d, want 2", len(grads))
	}
	// gradA = grad * b, gradB = grad * a
	wantA := []float32{5, 6, 7, 8}
	wantB := []float32{1, 2, 3, 4}
	for i := range grads[0].Data() {
		if grads[0].Data()[i] != wantA[i] {
			t.Errorf("Mul gradA[%d] = %v, want %v", i, grads[0].Data()[i], wantA[i])
		}
		if grads[1].Data()[i] != wantB[i] {
			t.Errorf("Mul gradB[%d] = %v, want %v", i, grads[1].Data()[i], wantB[i])
		}
	}

	// graph.Node interface
	var _ graph.Node[float32] = m
}

// ---------- Sub ----------

func TestSub(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	s := NewSub(engine)

	a := makeTensor(t, []int{1, 3}, []float32{10, 20, 30})
	b := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})

	// Forward (2 inputs)
	out, err := s.Forward(ctx, a, b)
	if err != nil {
		t.Fatalf("Sub Forward: %v", err)
	}
	want := []float32{9, 18, 27}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("Sub Forward[%d] = %v, want %v", i, v, want[i])
		}
	}

	// Forward (1 input = negate)
	out, err = s.Forward(ctx, a)
	if err != nil {
		t.Fatalf("Sub Forward negate: %v", err)
	}
	wantNeg := []float32{-10, -20, -30}
	for i, v := range out.Data() {
		if v != wantNeg[i] {
			t.Errorf("Sub negate[%d] = %v, want %v", i, v, wantNeg[i])
		}
	}

	// Forward (3 inputs = error)
	_, err = s.Forward(ctx, a, b, a)
	if err == nil {
		t.Error("Sub Forward with 3 inputs should return error")
	}

	// OutputShape
	if os := s.OutputShape(); len(os) == 0 {
		t.Error("Sub OutputShape should not be empty after Forward")
	}

	// Parameters
	if p := s.Parameters(); p != nil {
		t.Errorf("Sub Parameters = %v, want nil", p)
	}

	// OpType
	if op := s.OpType(); op != "Sub" {
		t.Errorf("Sub OpType = %q, want %q", op, "Sub")
	}

	// Attributes
	if attr := s.Attributes(); attr != nil {
		t.Errorf("Sub Attributes = %v, want nil", attr)
	}

	// Backward
	grad := makeTensor(t, []int{1, 3}, []float32{1, 1, 1})
	grads, err := s.Backward(ctx, types.FullBackprop, grad, a, b)
	if err != nil {
		t.Fatalf("Sub Backward: %v", err)
	}
	if len(grads) != 2 {
		t.Fatalf("Sub Backward len = %d, want 2", len(grads))
	}
	// gradA = grad, gradB = -grad
	for i := range grads[0].Data() {
		if grads[0].Data()[i] != 1 {
			t.Errorf("Sub gradA[%d] = %v, want 1", i, grads[0].Data()[i])
		}
		if grads[1].Data()[i] != -1 {
			t.Errorf("Sub gradB[%d] = %v, want -1", i, grads[1].Data()[i])
		}
	}
}

// ---------- Reshape ----------

func TestReshape(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	r := NewReshape(engine, []int{3, 2})

	input := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	// Forward
	out, err := r.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Reshape Forward: %v", err)
	}
	if s := out.Shape(); s[0] != 3 || s[1] != 2 {
		t.Errorf("Reshape Forward shape = %v, want [3 2]", s)
	}

	// OutputShape
	if os := r.OutputShape(); os[0] != 3 || os[1] != 2 {
		t.Errorf("Reshape OutputShape = %v, want [3 2]", os)
	}

	// Parameters
	if p := r.Parameters(); p != nil {
		t.Errorf("Reshape Parameters = %v, want nil", p)
	}

	// OpType
	if op := r.OpType(); op != "Reshape" {
		t.Errorf("Reshape OpType = %q, want %q", op, "Reshape")
	}

	// Attributes
	attr := r.Attributes()
	if attr == nil {
		t.Fatal("Reshape Attributes should not be nil")
	}
	if _, ok := attr["shape"]; !ok {
		t.Error("Reshape Attributes missing 'shape'")
	}

	// Backward
	grad := makeTensor(t, []int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	grads, err := r.Backward(ctx, types.FullBackprop, grad, input)
	if err != nil {
		t.Fatalf("Reshape Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("Reshape Backward len = %d, want 1", len(grads))
	}
	if s := grads[0].Shape(); s[0] != 2 || s[1] != 3 {
		t.Errorf("Reshape Backward shape = %v, want [2 3]", s)
	}
}

// ---------- Shape ----------

func TestShapeLayer(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	s := New(engine)

	input := makeTensor(t, []int{2, 3, 4}, make([]float32, 24))

	// Forward
	out, err := s.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Shape Forward: %v", err)
	}
	// Output should be [3] tensor with values [2, 3, 4]
	if s := out.Shape(); len(s) != 1 || s[0] != 3 {
		t.Errorf("Shape Forward shape = %v, want [3]", s)
	}
	want := []float32{2, 3, 4}
	for i, v := range out.Data() {
		if v != want[i] {
			t.Errorf("Shape Forward[%d] = %v, want %v", i, v, want[i])
		}
	}

	// OutputShape
	if os := s.OutputShape(); len(os) != 1 || os[0] != 3 {
		t.Errorf("Shape OutputShape = %v, want [3]", os)
	}

	// Parameters
	if p := s.Parameters(); p != nil {
		t.Errorf("Shape Parameters = %v, want nil", p)
	}

	// OpType
	if op := s.OpType(); op != "Shape" {
		t.Errorf("Shape OpType = %q, want %q", op, "Shape")
	}

	// Attributes
	if attr := s.Attributes(); attr != nil {
		t.Errorf("Shape Attributes = %v, want nil", attr)
	}

	// Backward (returns nil, nil)
	grads, err := s.Backward(ctx, types.FullBackprop, nil)
	if err != nil {
		t.Fatalf("Shape Backward: %v", err)
	}
	if grads != nil {
		t.Errorf("Shape Backward = %v, want nil", grads)
	}
}

// ---------- Cast ----------

func TestCast(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	c := NewCast(engine)

	input := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	// Forward (identity for same type)
	out, err := c.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Cast Forward: %v", err)
	}
	if out != input {
		t.Error("Cast Forward should return same tensor for same type")
	}

	// OutputShape
	if os := c.OutputShape(); os[0] != 2 || os[1] != 3 {
		t.Errorf("Cast OutputShape = %v, want [2 3]", os)
	}

	// Parameters
	if p := c.Parameters(); p != nil {
		t.Errorf("Cast Parameters = %v, want nil", p)
	}

	// OpType
	if op := c.OpType(); op != "Cast" {
		t.Errorf("Cast OpType = %q, want %q", op, "Cast")
	}

	// Attributes
	if attr := c.Attributes(); attr != nil {
		t.Errorf("Cast Attributes = %v, want nil", attr)
	}

	// Backward
	grad := makeTensor(t, []int{2, 3}, []float32{1, 1, 1, 1, 1, 1})
	grads, err := c.Backward(ctx, types.FullBackprop, grad, input)
	if err != nil {
		t.Fatalf("Cast Backward: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("Cast Backward len = %d, want 1", len(grads))
	}
	if grads[0] != grad {
		t.Error("Cast Backward should pass gradient through unchanged")
	}
}

// ---------- Unsqueeze ----------

func TestUnsqueeze(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	tests := []struct {
		name        string
		axes        []int
		inputShape  []int
		wantShape   []int
		wantBackShape []int
	}{
		{
			name:        "axis_0",
			axes:        []int{0},
			inputShape:  []int{3, 4},
			wantShape:   []int{1, 3, 4},
			wantBackShape: []int{3, 4},
		},
		{
			name:        "axis_2",
			axes:        []int{2},
			inputShape:  []int{3, 4},
			wantShape:   []int{3, 4, 1},
			wantBackShape: []int{3, 4},
		},
		{
			name:        "negative_axis",
			axes:        []int{-1},
			inputShape:  []int{3, 4},
			wantShape:   []int{3, 4, 1},
			wantBackShape: []int{3, 4},
		},
		{
			name:        "multiple_axes",
			axes:        []int{0, 2},
			inputShape:  []int{3, 4},
			wantShape:   []int{1, 3, 1, 4},
			wantBackShape: []int{3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u := NewUnsqueeze(engine, tt.axes)

			size := 1
			for _, d := range tt.inputShape {
				size *= d
			}
			input := makeTensor(t, tt.inputShape, make([]float32, size))

			// Forward
			out, err := u.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Unsqueeze Forward: %v", err)
			}
			gotShape := out.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("shape len = %d, want %d", len(gotShape), len(tt.wantShape))
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tt.wantShape[i])
				}
			}

			// OutputShape
			os := u.OutputShape()
			for i := range os {
				if os[i] != tt.wantShape[i] {
					t.Errorf("OutputShape[%d] = %d, want %d", i, os[i], tt.wantShape[i])
				}
			}

			// Backward
			gradSize := 1
			for _, d := range tt.wantShape {
				gradSize *= d
			}
			grad := makeTensor(t, tt.wantShape, make([]float32, gradSize))
			grads, err := u.Backward(ctx, types.FullBackprop, grad, input)
			if err != nil {
				t.Fatalf("Unsqueeze Backward: %v", err)
			}
			backShape := grads[0].Shape()
			for i := range backShape {
				if backShape[i] != tt.wantBackShape[i] {
					t.Errorf("backShape[%d] = %d, want %d", i, backShape[i], tt.wantBackShape[i])
				}
			}
		})
	}

	// Test Parameters, OpType, Attributes
	u := NewUnsqueeze(engine, []int{0})
	if p := u.Parameters(); p != nil {
		t.Errorf("Unsqueeze Parameters = %v, want nil", p)
	}
	if op := u.OpType(); op != "Unsqueeze" {
		t.Errorf("Unsqueeze OpType = %q, want %q", op, "Unsqueeze")
	}
	attr := u.Attributes()
	if attr == nil {
		t.Fatal("Unsqueeze Attributes should not be nil")
	}
	if _, ok := attr["axes"]; !ok {
		t.Error("Unsqueeze Attributes missing 'axes'")
	}
}

// ---------- MatMul ----------

func TestMatMul(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	m := NewMatMul(engine)

	a := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := makeTensor(t, []int{3, 2}, []float32{1, 2, 3, 4, 5, 6})

	// Forward (compatible dims)
	out, err := m.Forward(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul Forward: %v", err)
	}
	if s := out.Shape(); s[0] != 2 || s[1] != 2 {
		t.Errorf("MatMul Forward shape = %v, want [2 2]", s)
	}

	// OutputShape
	if os := m.OutputShape(); os[0] != 2 || os[1] != 2 {
		t.Errorf("MatMul OutputShape = %v, want [2 2]", os)
	}

	// Parameters
	if p := m.Parameters(); p != nil {
		t.Errorf("MatMul Parameters = %v, want nil", p)
	}

	// OpType
	if op := m.OpType(); op != "MatMul" {
		t.Errorf("MatMul OpType = %q, want %q", op, "MatMul")
	}

	// Attributes
	if attr := m.Attributes(); attr != nil {
		t.Errorf("MatMul Attributes = %v, want nil", attr)
	}

	// Forward error: wrong number of inputs
	_, err = m.Forward(ctx, a)
	if err == nil {
		t.Error("MatMul Forward with 1 input should error")
	}

	// Forward: dimension mismatch requiring transpose
	// a is [2,3], c is [2,3]. a's inner dim (3) != c's outer dim (2), but a's inner (3) == c's inner (3)
	c := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	out, err = m.Forward(ctx, a, c)
	if err != nil {
		t.Fatalf("MatMul Forward with transpose: %v", err)
	}
	if s := out.Shape(); s[0] != 2 || s[1] != 2 {
		t.Errorf("MatMul Forward transposed shape = %v, want [2 2]", s)
	}

	// Forward: incompatible dimensions
	d := makeTensor(t, []int{4, 5}, make([]float32, 20))
	_, err = m.Forward(ctx, a, d)
	if err == nil {
		t.Error("MatMul Forward with incompatible dims should error")
	}

	// Backward uses naive matmul (no transpose): gradA=grad@b, gradB=a@grad
	// Use square matrices so shapes are compatible
	aSq := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
	bSq := makeTensor(t, []int{2, 2}, []float32{5, 6, 7, 8})
	_, _ = m.Forward(ctx, aSq, bSq)
	grad := makeTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
	grads, err := m.Backward(ctx, types.FullBackprop, grad, aSq, bSq)
	if err != nil {
		t.Fatalf("MatMul Backward: %v", err)
	}
	if len(grads) != 2 {
		t.Fatalf("MatMul Backward len = %d, want 2", len(grads))
	}
}

// ---------- Add (extended) ----------

func TestAdd_Extended(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	add := NewAdd[float32](engine)

	// Parameters
	if p := add.Parameters(); p != nil {
		t.Errorf("Add Parameters = %v, want nil", p)
	}

	// OutputShape
	if os := add.OutputShape(); os != nil {
		t.Errorf("Add OutputShape = %v, want nil", os)
	}

	// OpType
	if op := add.OpType(); op != "Add" {
		t.Errorf("Add OpType = %q, want %q", op, "Add")
	}

	// Attributes
	if attr := add.Attributes(); attr != nil {
		t.Errorf("Add Attributes = %v, want nil", attr)
	}

	// Forward error: wrong number of inputs
	_, err := add.Forward(context.Background(), makeTensor(t, []int{1}, []float32{1}))
	if err == nil {
		t.Error("Add Forward with 1 input should error")
	}

	// BuildAdd
	node, err := BuildAdd(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildAdd: %v", err)
	}
	if node.OpType() != "Add" {
		t.Errorf("BuildAdd OpType = %q, want %q", node.OpType(), "Add")
	}
}
