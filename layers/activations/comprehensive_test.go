package activations

import (
	"context"
	"fmt"
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
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatalf("makeTensor: %v", err)
	}
	return tn
}

// errEngine wraps a real engine and injects errors.
type errEngine struct {
	compute.Engine[float32]
	calls   map[string]int
	failOn  map[string]int
	failErr error
}

func newErrEngine(failOn map[string]int) *errEngine {
	return &errEngine{
		Engine:  compute.NewCPUEngine[float32](numeric.Float32Ops{}),
		calls:   make(map[string]int),
		failOn:  failOn,
		failErr: fmt.Errorf("injected error"),
	}
}

func (e *errEngine) check(op string) error {
	e.calls[op]++
	if n, ok := e.failOn[op]; ok && e.calls[op] >= n {
		return e.failErr
	}
	return nil
}

func (e *errEngine) UnaryOp(ctx context.Context, a *tensor.TensorNumeric[float32], fn func(float32) float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("UnaryOp"); err != nil {
		return nil, err
	}
	return e.Engine.UnaryOp(ctx, a, fn, dst...)
}

func (e *errEngine) Mul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *errEngine) MulScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("MulScalar"); err != nil {
		return nil, err
	}
	return e.Engine.MulScalar(ctx, a, scalar, dst...)
}

func (e *errEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *errEngine) AddScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("AddScalar"); err != nil {
		return nil, err
	}
	return e.Engine.AddScalar(ctx, a, scalar, dst...)
}

func (e *errEngine) Exp(ctx context.Context, a *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Exp"); err != nil {
		return nil, err
	}
	return e.Engine.Exp(ctx, a, dst...)
}

func (e *errEngine) Div(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Div"); err != nil {
		return nil, err
	}
	return e.Engine.Div(ctx, a, b, dst...)
}

func (e *errEngine) Split(ctx context.Context, a *tensor.TensorNumeric[float32], numSplits int, axis int) ([]*tensor.TensorNumeric[float32], error) {
	if err := e.check("Split"); err != nil {
		return nil, err
	}
	return e.Engine.Split(ctx, a, numSplits, axis)
}

func (e *errEngine) Concat(ctx context.Context, inputs []*tensor.TensorNumeric[float32], axis int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Concat"); err != nil {
		return nil, err
	}
	return e.Engine.Concat(ctx, inputs, axis, dst...)
}

func (e *errEngine) Tanh(ctx context.Context, a *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Tanh"); err != nil {
		return nil, err
	}
	return e.Engine.Tanh(ctx, a, dst...)
}

func (e *errEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

// ---------- BaseActivation OpType / Attributes ----------

func TestBaseActivation_OpTypeAttributes(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	base := NewBaseActivation(eng, ops, "TestOp",
		WithForwardOp(ops.ReLU),
		WithBackwardOp(ops.ReLUGrad))

	if base.OpType() != "TestOp" {
		t.Errorf("OpType = %q, want %q", base.OpType(), "TestOp")
	}
	attrs := base.Attributes()
	if attrs == nil {
		t.Error("Attributes should not be nil")
	}
}

// ---------- BaseActivation Forward input count error ----------

func TestBaseActivation_ForwardInputCountError(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	base := NewBaseActivation(eng, ops, "ReLU",
		WithForwardOp(ops.ReLU),
		WithBackwardOp(ops.ReLUGrad))

	// 0 inputs
	_, err := base.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}

	// 2 inputs
	a := makeTensor(t, []int{2}, []float32{1, 2})
	b := makeTensor(t, []int{2}, []float32{3, 4})
	_, err = base.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected error for 2 inputs")
	}
}

// ---------- BaseActivation Forward engine error ----------

func TestBaseActivation_ForwardEngineError(t *testing.T) {
	eng := newErrEngine(map[string]int{"UnaryOp": 1})
	ops := makeOps()
	base := NewBaseActivation[float32](eng, ops, "ReLU",
		WithForwardOp(ops.ReLU),
		WithBackwardOp(ops.ReLUGrad))

	input := makeTensor(t, []int{2}, []float32{1, 2})
	_, err := base.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected engine error")
	}
}

// ---------- BaseActivation Backward errors ----------

func TestBaseActivation_BackwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := makeOps()

	t.Run("derivative_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"UnaryOp": 2}) // #1=Forward
		base := NewBaseActivation[float32](eng, ops, "ReLU",
			WithForwardOp(ops.ReLU),
			WithBackwardOp(ops.ReLUGrad))
		input := makeTensor(t, []int{2}, []float32{1, 2})
		_, _ = base.Forward(ctx, input)
		grad := makeTensor(t, []int{2}, []float32{1, 1})
		_, err := base.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error from UnaryOp")
		}
	})

	t.Run("mul_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Mul": 1})
		base := NewBaseActivation[float32](eng, ops, "ReLU",
			WithForwardOp(ops.ReLU),
			WithBackwardOp(ops.ReLUGrad))
		input := makeTensor(t, []int{2}, []float32{1, 2})
		_, _ = base.Forward(ctx, input)
		grad := makeTensor(t, []int{2}, []float32{1, 1})
		_, err := base.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error from Mul")
		}
	})
}

// ---------- Gelu ----------

func TestGelu(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	gelu := NewGelu(eng, ops)

	t.Run("forward", func(t *testing.T) {
		input := makeTensor(t, []int{4}, []float32{-1, 0, 0.5, 1})
		out, err := gelu.Forward(context.Background(), input)
		if err != nil {
			t.Fatal(err)
		}
		if len(out.Data()) != 4 {
			t.Errorf("output len = %d, want 4", len(out.Data()))
		}
		// GELU(0) should be 0
		if out.Data()[1] != 0 {
			t.Errorf("GELU(0) = %v, want 0", out.Data()[1])
		}
	})

	t.Run("backward", func(t *testing.T) {
		ctx := context.Background()
		input := makeTensor(t, []int{4}, []float32{-1, 0, 0.5, 1})
		_, _ = gelu.Forward(ctx, input)
		grad := makeTensor(t, []int{4}, []float32{1, 1, 1, 1})
		grads, err := gelu.Backward(ctx, types.FullBackprop, grad)
		if err != nil {
			t.Fatal(err)
		}
		if len(grads) != 1 || len(grads[0].Data()) != 4 {
			t.Error("unexpected gradient shape")
		}
	})
}

// ---------- BuildGelu ----------

func TestBuildGelu(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	node, err := BuildGelu(eng, ops, "gelu", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if node.OpType() != "Gelu" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Gelu")
	}
}

// ---------- FastGelu ----------

func TestFastGelu_Full(t *testing.T) {
	eng := makeEngine()
	fg := NewFastGelu[float32](eng)

	t.Run("optype", func(t *testing.T) {
		if fg.OpType() != "FastGelu" {
			t.Errorf("OpType = %q", fg.OpType())
		}
	})

	t.Run("attributes", func(t *testing.T) {
		if fg.Attributes() != nil {
			t.Error("expected nil attributes")
		}
	})

	t.Run("parameters", func(t *testing.T) {
		if fg.Parameters() != nil {
			t.Error("expected nil parameters")
		}
	})

	t.Run("output_shape", func(t *testing.T) {
		if fg.OutputShape() != nil {
			t.Error("expected nil output shape")
		}
	})

	t.Run("forward_input_error", func(t *testing.T) {
		_, err := fg.Forward(context.Background())
		if err == nil {
			t.Error("expected error for 0 inputs")
		}
	})

	t.Run("backward", func(t *testing.T) {
		grads, err := fg.Backward(context.Background(), types.FullBackprop, nil)
		if err != nil {
			t.Fatal(err)
		}
		if grads != nil {
			t.Error("expected nil grads")
		}
	})
}

// ---------- FastGelu Forward error paths ----------

func TestFastGelu_ForwardErrors(t *testing.T) {
	ctx := context.Background()
	input := makeTensor(t, []int{2}, []float32{0.5, 1.0})

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"Mul1", map[string]int{"Mul": 1}},
		{"Mul2", map[string]int{"Mul": 2}},
		{"MulScalar1", map[string]int{"MulScalar": 1}},
		{"Add1", map[string]int{"Add": 1}},
		{"MulScalar2", map[string]int{"MulScalar": 2}},
		{"Tanh1", map[string]int{"Tanh": 1}},
		{"AddScalar1", map[string]int{"AddScalar": 1}},
		{"Mul3", map[string]int{"Mul": 3}},
		{"MulScalar3", map[string]int{"MulScalar": 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			fg := NewFastGelu[float32](eng)
			_, err := fg.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- BuildFastGelu ----------

func TestBuildFastGelu(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	node, err := BuildFastGelu(eng, ops, "fg", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if node.OpType() != "FastGelu" {
		t.Errorf("OpType = %q", node.OpType())
	}
}

// ---------- LeakyReLU OpType / Attributes ----------

func TestLeakyReLU_OpTypeAttributes(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	lr := NewLeakyReLU(eng, ops, WithAlpha[float32](0.1))

	if lr.OpType() != "LeakyReLU" {
		t.Errorf("OpType = %q", lr.OpType())
	}
	attrs := lr.Attributes()
	if attrs["alpha"] != 0.1 {
		t.Errorf("alpha = %v, want 0.1", attrs["alpha"])
	}
}

// ---------- LeakyReLU Forward input error ----------

func TestLeakyReLU_ForwardInputError(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	lr := NewLeakyReLU(eng, ops)

	_, err := lr.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

// ---------- LeakyReLU Forward/Backward engine errors ----------

func TestLeakyReLU_ForwardEngineError(t *testing.T) {
	eng := newErrEngine(map[string]int{"UnaryOp": 1})
	ops := makeOps()
	lr := NewLeakyReLU[float32](eng, ops)

	input := makeTensor(t, []int{2}, []float32{1, -1})
	_, err := lr.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected engine error")
	}
}

func TestLeakyReLU_BackwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := makeOps()

	t.Run("unary_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"UnaryOp": 2})
		lr := NewLeakyReLU[float32](eng, ops)
		input := makeTensor(t, []int{2}, []float32{1, -1})
		_, _ = lr.Forward(ctx, input)
		grad := makeTensor(t, []int{2}, []float32{1, 1})
		_, err := lr.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error")
		}
	})

	t.Run("mul_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Mul": 1})
		lr := NewLeakyReLU[float32](eng, ops)
		input := makeTensor(t, []int{2}, []float32{1, -1})
		_, _ = lr.Forward(ctx, input)
		grad := makeTensor(t, []int{2}, []float32{1, 1})
		_, err := lr.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error")
		}
	})
}

// ---------- SwiGLU ----------

func TestSwiGLU_Full(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	sg := NewSwiGLU(eng, ops)

	t.Run("optype", func(t *testing.T) {
		if sg.OpType() != "SwiGLU" {
			t.Errorf("OpType = %q", sg.OpType())
		}
	})

	t.Run("attributes", func(t *testing.T) {
		attrs := sg.Attributes()
		if attrs == nil || len(attrs) != 0 {
			t.Error("expected empty map")
		}
	})

	t.Run("parameters", func(t *testing.T) {
		if sg.Parameters() != nil {
			t.Error("expected nil parameters")
		}
	})

	t.Run("output_shape_before_forward", func(t *testing.T) {
		if sg.OutputShape() != nil {
			t.Error("expected nil before forward")
		}
	})

	t.Run("forward_input_error", func(t *testing.T) {
		_, err := sg.Forward(context.Background())
		if err == nil {
			t.Error("expected error for 0 inputs")
		}
	})

	t.Run("forward_odd_dim", func(t *testing.T) {
		input := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
		_, err := sg.Forward(context.Background(), input)
		if err == nil {
			t.Error("expected error for odd last dim")
		}
	})

	t.Run("forward_0dim", func(t *testing.T) {
		input, _ := tensor.New[float32]([]int{}, []float32{1.0})
		_, err := sg.Forward(context.Background(), input)
		if err == nil {
			t.Error("expected error for 0-dimensional input")
		}
	})

	t.Run("forward_backward", func(t *testing.T) {
		ctx := context.Background()
		input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
		out, err := sg.Forward(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		// Output shape should be [1, 2] (last dim halved)
		if out.Shape()[1] != 2 {
			t.Errorf("output shape = %v, want [1, 2]", out.Shape())
		}

		// OutputShape after forward
		outShape := sg.OutputShape()
		if outShape[1] != 2 {
			t.Errorf("OutputShape = %v", outShape)
		}

		grad := makeTensor(t, out.Shape(), []float32{1, 1})
		grads, err := sg.Backward(ctx, types.FullBackprop, grad, input)
		if err != nil {
			t.Fatal(err)
		}
		if len(grads) != 1 || len(grads[0].Data()) != 4 {
			t.Error("unexpected gradient shape")
		}
	})

	t.Run("backward_input_error", func(t *testing.T) {
		ctx := context.Background()
		input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
		_, _ = sg.Forward(ctx, input)
		grad := makeTensor(t, []int{1, 2}, []float32{1, 1})
		// 0 inputs to backward
		_, err := sg.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error for 0 backward inputs")
		}
	})
}

// ---------- SwiGLU Forward engine errors ----------

func TestSwiGLU_ForwardEngineErrors(t *testing.T) {
	ctx := context.Background()
	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"Split", map[string]int{"Split": 1}},
		{"Sigmoid_Exp", map[string]int{"Exp": 1}},
		{"Mul_output", map[string]int{"Mul": 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			sg := NewSwiGLU[float32](eng, makeOps())
			_, err := sg.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- SwiGLU Backward engine errors ----------

func TestSwiGLU_BackwardEngineErrors(t *testing.T) {
	ctx := context.Background()
	ops := makeOps()

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// Backward calls: Split#2, Mul#2(dLdx1), Mul#3(dLdgate), MulScalar#1(negGate), AddScalar#2(oneMinusGate), Mul#4(sigmoidGrad), Mul#5(dLdx2), Concat#1
		{"Split", map[string]int{"Split": 2}},
		{"dLdx1_Mul", map[string]int{"Mul": 2}},
		{"dLdgate_Mul", map[string]int{"Mul": 3}},
		{"negGate_MulScalar", map[string]int{"MulScalar": 1}},
		{"oneMinusGate_AddScalar", map[string]int{"AddScalar": 2}},
		{"sigmoidGrad_Mul", map[string]int{"Mul": 4}},
		{"dLdx2_Mul", map[string]int{"Mul": 5}},
		{"Concat", map[string]int{"Concat": 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			sg := NewSwiGLU[float32](eng, ops)
			input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
			out, err := sg.Forward(ctx, input)
			if err != nil {
				t.Skipf("Forward failed: %v", err)
			}
			grad := makeTensor(t, out.Shape(), []float32{1, 1})
			_, err = sg.Backward(ctx, types.FullBackprop, grad, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- BuildTanh ----------

func TestBuildTanh(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	node, err := BuildTanh(eng, ops, "tanh", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if node.OpType() != "Tanh" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Tanh")
	}
}

// ---------- testActivationCoverage helper ----------

func TestActivationCoverageHelper(t *testing.T) {
	eng := makeEngine()
	ops := makeOps()
	testActivationCoverage(t, func() ActivationLayer[float32] {
		return NewBaseActivation(eng, ops, "ReLU",
			WithForwardOp(ops.ReLU),
			WithBackwardOp(ops.ReLUGrad))
	})
}

// ---------- Interface conformance ----------

var (
	_ graph.Node[float32] = (*BaseActivation[float32])(nil)
	_ graph.Node[float32] = (*FastGelu[float32])(nil)
	_ graph.Node[float32] = (*SwiGLU[float32])(nil)
	_ graph.Node[float32] = (*LeakyReLU[float32])(nil)
	_ graph.Node[float32] = (*Gelu[float32])(nil)
	_ graph.Node[float32] = (*Tanh[float32])(nil)
)
