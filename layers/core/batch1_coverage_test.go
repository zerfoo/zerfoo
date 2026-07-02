package core

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/types"
)

// ---------- ConstantOfShape ----------

func TestConstantOfShapeForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name      string
		value     float32
		shapeIn   []float32
		wantShape []int
		wantData  []float32
	}{
		{
			name:      "scalar_fill_2x3",
			value:     7,
			shapeIn:   []float32{2, 3},
			wantShape: []int{2, 3},
			wantData:  []float32{7, 7, 7, 7, 7, 7},
		},
		{
			name:      "single_element",
			value:     42,
			shapeIn:   []float32{1},
			wantShape: []int{1},
			wantData:  []float32{42},
		},
		{
			name:      "3d_shape",
			value:     0,
			shapeIn:   []float32{2, 1, 3},
			wantShape: []int{2, 1, 3},
			wantData:  []float32{0, 0, 0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &ConstantOfShape[float32]{engine: engine, value: tt.value}

			shape := makeTensor(t, []int{len(tt.shapeIn)}, tt.shapeIn)
			out, err := node.Forward(ctx, shape)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := out.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("shape = %v, want %v", gotShape, tt.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tt.wantShape[i])
				}
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	// Error: wrong number of inputs
	node := &ConstantOfShape[float32]{engine: engine, value: 1}
	_, err := node.Forward(ctx)
	if err == nil {
		t.Error("Forward with 0 inputs should error")
	}

	// Backward returns error
	_, err = node.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := node.OpType(); op != "ConstantOfShape" {
		t.Errorf("OpType = %q, want %q", op, "ConstantOfShape")
	}
	attrs := node.Attributes()
	if attrs == nil {
		t.Fatal("Attributes should not be nil")
	}
	if node.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if node.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildConstantOfShape with float64 attribute
	built, err := BuildConstantOfShape(engine, ops, "test", nil, map[string]any{"value": float64(3.5)})
	if err != nil {
		t.Fatalf("BuildConstantOfShape float64: %v", err)
	}
	if built.OpType() != "ConstantOfShape" {
		t.Errorf("built OpType = %q", built.OpType())
	}

	// BuildConstantOfShape with float32 attribute
	_, err = BuildConstantOfShape(engine, ops, "test", nil, map[string]any{"value": float32(2.0)})
	if err != nil {
		t.Fatalf("BuildConstantOfShape float32: %v", err)
	}

	// BuildConstantOfShape with int64 attribute
	_, err = BuildConstantOfShape(engine, ops, "test", nil, map[string]any{"value": int64(5)})
	if err != nil {
		t.Fatalf("BuildConstantOfShape int64: %v", err)
	}

	// BuildConstantOfShape with no value attribute
	_, err = BuildConstantOfShape(engine, ops, "test", nil, map[string]any{})
	if err != nil {
		t.Fatalf("BuildConstantOfShape no value: %v", err)
	}
}

// ---------- Div ----------

func TestDivForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name     string
		a, b     []float32
		shape    []int
		wantData []float32
	}{
		{
			name:     "element_wise",
			a:        []float32{10, 20, 30, 40},
			b:        []float32{2, 4, 5, 8},
			shape:    []int{2, 2},
			wantData: []float32{5, 5, 6, 5},
		},
		{
			name:     "divide_by_one",
			a:        []float32{3, 7},
			b:        []float32{1, 1},
			shape:    []int{2},
			wantData: []float32{3, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDiv[float32](engine)
			a := makeTensor(t, tt.shape, tt.a)
			b := makeTensor(t, tt.shape, tt.b)

			out, err := d.Forward(ctx, a, b)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	d := NewDiv[float32](engine)

	// Error: wrong number of inputs
	a := makeTensor(t, []int{2}, []float32{1, 2})
	_, err := d.Forward(ctx, a)
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Backward returns error
	_, err = d.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := d.OpType(); op != "Div" {
		t.Errorf("OpType = %q, want %q", op, "Div")
	}
	if d.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if d.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if d.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildDiv
	built, err := BuildDiv(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildDiv: %v", err)
	}
	if built.OpType() != "Div" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Equal ----------

func TestEqualForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name     string
		a, b     []float32
		shapeA   []int
		shapeB   []int
		wantData []float32
	}{
		{
			name:     "matching_elements",
			a:        []float32{1, 2, 3, 4},
			b:        []float32{1, 0, 3, 0},
			shapeA:   []int{4},
			shapeB:   []int{4},
			wantData: []float32{1, 0, 1, 0},
		},
		{
			name:     "all_equal",
			a:        []float32{5, 5},
			b:        []float32{5, 5},
			shapeA:   []int{2},
			shapeB:   []int{2},
			wantData: []float32{1, 1},
		},
		{
			name:     "scalar_broadcast_b",
			a:        []float32{1, 2, 3},
			b:        []float32{2},
			shapeA:   []int{3},
			shapeB:   []int{1},
			wantData: []float32{0, 1, 0},
		},
		{
			name:     "scalar_broadcast_a",
			a:        []float32{3},
			b:        []float32{1, 3, 5},
			shapeA:   []int{1},
			shapeB:   []int{3},
			wantData: []float32{0, 1, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eq := &Equal[float32]{engine: engine, ops: ops}
			a := makeTensor(t, tt.shapeA, tt.a)
			b := makeTensor(t, tt.shapeB, tt.b)

			out, err := eq.Forward(ctx, a, b)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	eq := &Equal[float32]{engine: engine, ops: ops}

	// Error: wrong number of inputs
	_, err := eq.Forward(ctx, makeTensor(t, []int{1}, []float32{1}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Error: mismatched sizes (non-scalar)
	a := makeTensor(t, []int{3}, []float32{1, 2, 3})
	b := makeTensor(t, []int{2}, []float32{1, 2})
	_, err = eq.Forward(ctx, a, b)
	if err == nil {
		t.Error("Forward with mismatched sizes should error")
	}

	// Backward returns error
	_, err = eq.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := eq.OpType(); op != "Equal" {
		t.Errorf("OpType = %q, want %q", op, "Equal")
	}
	if eq.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if eq.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if eq.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildEqual
	built, err := BuildEqual(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildEqual: %v", err)
	}
	if built.OpType() != "Equal" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Expand ----------

func TestExpandForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name        string
		inputShape  []int
		inputData   []float32
		targetShape []float32
		wantShape   []int
		wantData    []float32
	}{
		{
			name:        "broadcast_1x3_to_2x3",
			inputShape:  []int{1, 3},
			inputData:   []float32{1, 2, 3},
			targetShape: []float32{2, 3},
			wantShape:   []int{2, 3},
			wantData:    []float32{1, 2, 3, 1, 2, 3},
		},
		{
			name:        "no_change",
			inputShape:  []int{2, 2},
			inputData:   []float32{1, 2, 3, 4},
			targetShape: []float32{2, 2},
			wantShape:   []int{2, 2},
			wantData:    []float32{1, 2, 3, 4},
		},
		{
			name:        "add_batch_dim",
			inputShape:  []int{3},
			inputData:   []float32{10, 20, 30},
			targetShape: []float32{2, 3},
			wantShape:   []int{2, 3},
			wantData:    []float32{10, 20, 30, 10, 20, 30},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expand := &Expand[float32]{engine: engine}
			input := makeTensor(t, tt.inputShape, tt.inputData)
			shape := makeTensor(t, []int{len(tt.targetShape)}, tt.targetShape)

			out, err := expand.Forward(ctx, input, shape)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			gotShape := out.Shape()
			if len(gotShape) != len(tt.wantShape) {
				t.Fatalf("shape = %v, want %v", gotShape, tt.wantShape)
			}
			for i := range gotShape {
				if gotShape[i] != tt.wantShape[i] {
					t.Errorf("shape[%d] = %d, want %d", i, gotShape[i], tt.wantShape[i])
				}
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	expand := &Expand[float32]{engine: engine}

	// Error: wrong number of inputs
	_, err := expand.Forward(ctx, makeTensor(t, []int{2}, []float32{1, 2}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Backward returns error
	_, err = expand.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := expand.OpType(); op != "Expand" {
		t.Errorf("OpType = %q, want %q", op, "Expand")
	}
	if expand.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if expand.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if expand.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildExpand
	built, err := BuildExpand(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildExpand: %v", err)
	}
	if built.OpType() != "Expand" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Greater ----------

func TestGreaterForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name     string
		a, b     []float32
		shapeA   []int
		shapeB   []int
		wantData []float32
	}{
		{
			name:     "element_wise",
			a:        []float32{5, 2, 8, 1},
			b:        []float32{3, 4, 6, 1},
			shapeA:   []int{4},
			shapeB:   []int{4},
			wantData: []float32{1, 0, 1, 0},
		},
		{
			name:     "scalar_broadcast_b",
			a:        []float32{1, 5, 3},
			b:        []float32{3},
			shapeA:   []int{3},
			shapeB:   []int{1},
			wantData: []float32{0, 1, 0},
		},
		{
			name:     "scalar_broadcast_a",
			a:        []float32{5},
			b:        []float32{3, 7, 5},
			shapeA:   []int{1},
			shapeB:   []int{3},
			wantData: []float32{1, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := &Greater[float32]{engine: engine, ops: ops}
			a := makeTensor(t, tt.shapeA, tt.a)
			b := makeTensor(t, tt.shapeB, tt.b)

			out, err := g.Forward(ctx, a, b)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	g := &Greater[float32]{engine: engine, ops: ops}

	// Error: wrong number of inputs
	_, err := g.Forward(ctx, makeTensor(t, []int{1}, []float32{1}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Error: mismatched sizes
	a := makeTensor(t, []int{3}, []float32{1, 2, 3})
	b := makeTensor(t, []int{2}, []float32{1, 2})
	_, err = g.Forward(ctx, a, b)
	if err == nil {
		t.Error("Forward with mismatched sizes should error")
	}

	// Backward returns error
	_, err = g.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := g.OpType(); op != "Greater" {
		t.Errorf("OpType = %q, want %q", op, "Greater")
	}
	if g.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if g.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if g.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildGreater
	built, err := BuildGreater(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildGreater: %v", err)
	}
	if built.OpType() != "Greater" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Neg ----------

func TestNegForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name     string
		input    []float32
		shape    []int
		wantData []float32
	}{
		{
			name:     "positive_values",
			input:    []float32{1, 2, 3},
			shape:    []int{3},
			wantData: []float32{-1, -2, -3},
		},
		{
			name:     "negative_values",
			input:    []float32{-5, -10},
			shape:    []int{2},
			wantData: []float32{5, 10},
		},
		{
			name:     "zero",
			input:    []float32{0},
			shape:    []int{1},
			wantData: []float32{0},
		},
		{
			name:     "mixed",
			input:    []float32{-3, 0, 7, -1},
			shape:    []int{2, 2},
			wantData: []float32{3, 0, -7, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &Neg[float32]{engine: engine, ops: ops}
			input := makeTensor(t, tt.shape, tt.input)

			out, err := n.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if v != tt.wantData[i] {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	n := &Neg[float32]{engine: engine, ops: ops}

	// Error: wrong number of inputs
	_, err := n.Forward(ctx)
	if err == nil {
		t.Error("Forward with 0 inputs should error")
	}

	// Backward returns error
	_, err = n.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := n.OpType(); op != "Neg" {
		t.Errorf("OpType = %q, want %q", op, "Neg")
	}
	if n.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if n.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if n.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildNeg
	built, err := BuildNeg(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildNeg: %v", err)
	}
	if built.OpType() != "Neg" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Pow ----------

func TestPowForward(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()
	ctx := context.Background()

	tests := []struct {
		name     string
		base     []float32
		exp      []float32
		shape    []int
		wantData []float32
	}{
		{
			name:     "integer_exponent",
			base:     []float32{2, 3, 4},
			exp:      []float32{2, 2, 2},
			shape:    []int{3},
			wantData: []float32{4, 9, 16},
		},
		{
			name:     "fractional_exponent",
			base:     []float32{4, 9, 16},
			exp:      []float32{0.5, 0.5, 0.5},
			shape:    []int{3},
			wantData: []float32{2, 3, 4},
		},
		{
			name:     "zero_exponent",
			base:     []float32{5, 10},
			exp:      []float32{0, 0},
			shape:    []int{2},
			wantData: []float32{1, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewPow[float32](engine)
			base := makeTensor(t, tt.shape, tt.base)
			exp := makeTensor(t, tt.shape, tt.exp)

			out, err := p.Forward(ctx, base, exp)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if math.Abs(float64(v-tt.wantData[i])) > 1e-5 {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	p := NewPow[float32](engine)

	// Error: wrong number of inputs
	_, err := p.Forward(ctx, makeTensor(t, []int{1}, []float32{1}))
	if err == nil {
		t.Error("Forward with 1 input should error")
	}

	// Backward returns error
	_, err = p.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := p.OpType(); op != "Pow" {
		t.Errorf("OpType = %q, want %q", op, "Pow")
	}
	if p.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if p.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if p.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildPow
	built, err := BuildPow(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildPow: %v", err)
	}
	if built.OpType() != "Pow" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Sqrt ----------

func TestSqrtForward(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name     string
		input    []float32
		shape    []int
		wantData []float32
	}{
		{
			name:     "positive_values",
			input:    []float32{4, 9, 16, 25},
			shape:    []int{4},
			wantData: []float32{2, 3, 4, 5},
		},
		{
			name:     "zero",
			input:    []float32{0},
			shape:    []int{1},
			wantData: []float32{0},
		},
		{
			name:     "one",
			input:    []float32{1},
			shape:    []int{1},
			wantData: []float32{1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewSqrt[float32](engine)
			input := makeTensor(t, tt.shape, tt.input)

			out, err := s.Forward(ctx, input)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			for i, v := range out.Data() {
				if math.Abs(float64(v-tt.wantData[i])) > 1e-5 {
					t.Errorf("data[%d] = %v, want %v", i, v, tt.wantData[i])
				}
			}
		})
	}

	s := NewSqrt[float32](engine)

	// Error: wrong number of inputs
	_, err := s.Forward(ctx)
	if err == nil {
		t.Error("Forward with 0 inputs should error")
	}

	// Backward returns error
	_, err = s.Backward(ctx, types.FullBackprop, nil)
	if err == nil {
		t.Error("Backward should return error")
	}

	// OpType, Attributes, OutputShape, Parameters
	if op := s.OpType(); op != "Sqrt" {
		t.Errorf("OpType = %q, want %q", op, "Sqrt")
	}
	if s.Attributes() != nil {
		t.Error("Attributes should be nil")
	}
	if s.OutputShape() != nil {
		t.Error("OutputShape should be nil")
	}
	if s.Parameters() != nil {
		t.Error("Parameters should be nil")
	}

	// BuildSqrt
	built, err := BuildSqrt(engine, ops, "test", nil, nil)
	if err != nil {
		t.Fatalf("BuildSqrt: %v", err)
	}
	if built.OpType() != "Sqrt" {
		t.Errorf("built OpType = %q", built.OpType())
	}
}

// ---------- Helper: NewAdd, NewSub, NewMul constructors ----------

func TestBuildConstructors(t *testing.T) {
	engine := makeEngine()
	ops := makeOps()

	// Verify constructors exist and return correct OpType
	tests := []struct {
		name string
		fn   func(compute.Engine[float32], numeric.Float32Ops, string, map[string]any) (string, error)
	}{
		{
			name: "BuildDiv",
			fn: func(e compute.Engine[float32], o numeric.Float32Ops, n string, a map[string]any) (string, error) {
				node, err := BuildDiv(e, o, n, nil, a)
				if err != nil {
					return "", err
				}
				return node.OpType(), nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opType, err := tt.fn(engine, ops, "test", nil)
			if err != nil {
				t.Fatalf("%s: %v", tt.name, err)
			}
			if opType == "" {
				t.Errorf("%s returned empty OpType", tt.name)
			}
		})
	}
}
