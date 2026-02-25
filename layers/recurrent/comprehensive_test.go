package recurrent

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

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

func (e *errEngine) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("MatMul"); err != nil {
		return nil, err
	}
	return e.Engine.MatMul(ctx, a, b, dst...)
}

func (e *errEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
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

func (e *errEngine) Reshape(ctx context.Context, a *tensor.TensorNumeric[float32], shape []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Reshape"); err != nil {
		return nil, err
	}
	return e.Engine.Reshape(ctx, a, shape, dst...)
}

func (e *errEngine) Transpose(ctx context.Context, a *tensor.TensorNumeric[float32], axes []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Transpose"); err != nil {
		return nil, err
	}
	return e.Engine.Transpose(ctx, a, axes, dst...)
}

func (e *errEngine) Sum(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Sum"); err != nil {
		return nil, err
	}
	return e.Engine.Sum(ctx, a, axis, keepDims, dst...)
}

func (e *errEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

func makeEngine() compute.Engine[float32] {
	return compute.NewCPUEngine[float32](numeric.Float32Ops{})
}

func makeTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatalf("makeTensor: %v", err)
	}
	return tn
}

func TestSimpleRNN_OpType(t *testing.T) {
	rnn, err := NewSimpleRNN[float32]("rnn", makeEngine(), numeric.Float32Ops{}, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	if rnn.OpType() != "SimpleRNN" {
		t.Errorf("OpType = %q, want %q", rnn.OpType(), "SimpleRNN")
	}
}

func TestSimpleRNN_Attributes(t *testing.T) {
	rnn, err := NewSimpleRNN[float32]("rnn", makeEngine(), numeric.Float32Ops{}, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	attrs := rnn.Attributes()
	if attrs["input_dim"] != 4 {
		t.Errorf("input_dim = %v, want 4", attrs["input_dim"])
	}
	if attrs["hidden_dim"] != 3 {
		t.Errorf("hidden_dim = %v, want 3", attrs["hidden_dim"])
	}
}

func TestSimpleRNN_OutputShape(t *testing.T) {
	rnn, err := NewSimpleRNN[float32]("rnn", makeEngine(), numeric.Float32Ops{}, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	os := rnn.OutputShape()
	if len(os) != 2 || os[1] != 3 {
		t.Errorf("OutputShape = %v, want [-1, 3]", os)
	}
}

func TestSimpleRNN_NewErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := makeEngine()

	t.Run("zero_input_dim", func(t *testing.T) {
		_, err := NewSimpleRNN[float32]("rnn", eng, ops, 0, 3)
		if err == nil {
			t.Error("expected error for zero input dim")
		}
	})

	t.Run("zero_hidden_dim", func(t *testing.T) {
		_, err := NewSimpleRNN[float32]("rnn", eng, ops, 4, 0)
		if err == nil {
			t.Error("expected error for zero hidden dim")
		}
	})
}

func TestSimpleRNN_Forward_InputError(t *testing.T) {
	rnn, _ := NewSimpleRNN[float32]("rnn", makeEngine(), numeric.Float32Ops{}, 4, 3)
	_, err := rnn.Forward(context.Background())
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestSimpleRNN_Backward_UnsupportedMode(t *testing.T) {
	ctx := context.Background()
	rnn, _ := NewSimpleRNN[float32]("rnn", makeEngine(), numeric.Float32Ops{}, 4, 3)

	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	out, err := rnn.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	grad := makeTensor(t, out.Shape(), make([]float32, out.Shape()[0]*out.Shape()[1]))
	_, err = rnn.Backward(ctx, types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error for unsupported backward mode")
	}
}

func TestSimpleRNN_Backward_OneStep(t *testing.T) {
	ctx := context.Background()
	rnn, _ := NewSimpleRNN[float32]("rnn", makeEngine(), numeric.Float32Ops{}, 4, 3)

	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	out, err := rnn.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	grad := makeTensor(t, out.Shape(), []float32{0.1, 0.1, 0.1})
	grads, err := rnn.Backward(ctx, types.OneStepApproximation, grad)
	if err != nil {
		t.Fatal(err)
	}
	if len(grads) != 2 {
		t.Errorf("grads len = %d, want 2", len(grads))
	}
}

func TestSimpleRNN_ParameterCount(t *testing.T) {
	rnn, _ := NewSimpleRNN[float32]("rnn", makeEngine(), numeric.Float32Ops{}, 4, 3)
	params := rnn.Parameters()
	// inputWeights(1) + hiddenWeights(1) + bias(1)
	if len(params) != 3 {
		t.Errorf("params len = %d, want 3", len(params))
	}
}

// ---------- Registry init tests ----------

func TestSimpleRNN_Registry(t *testing.T) {
	eng := makeEngine()
	ops := numeric.Float32Ops{}

	builder, err := model.GetLayerBuilder[float32]("SimpleRNN")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}

	t.Run("valid", func(t *testing.T) {
		node, err := builder(eng, ops, "rnn", nil, map[string]any{
			"input_dim":  4,
			"hidden_dim": 3,
		})
		if err != nil {
			t.Fatal(err)
		}
		if node.OpType() != "SimpleRNN" {
			t.Errorf("OpType = %q", node.OpType())
		}
	})

	t.Run("missing_input_dim", func(t *testing.T) {
		_, err := builder(eng, ops, "rnn", nil, map[string]any{
			"hidden_dim": 3,
		})
		if err == nil {
			t.Error("expected error for missing input_dim")
		}
	})

	t.Run("missing_hidden_dim", func(t *testing.T) {
		_, err := builder(eng, ops, "rnn", nil, map[string]any{
			"input_dim": 4,
		})
		if err == nil {
			t.Error("expected error for missing hidden_dim")
		}
	})
}

// ---------- Forward error paths via errEngine ----------

func TestSimpleRNN_ForwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// Forward: inputWeights.Forward = MatMul#1
		{"inputWeights_error", map[string]int{"MatMul": 1}},
		// Forward: hiddenWeights.Forward = MatMul#2
		{"hiddenWeights_error", map[string]int{"MatMul": 2}},
		// Forward: engine.Add = Add#1 (after inputWeights + hiddenWeights Add#1)
		// Note: bias internally uses Add too. The engine.Add in Forward is the first Add call.
		{"add_error", map[string]int{"Add": 1}},
		// Forward: bias.Forward = Add#2
		{"bias_error", map[string]int{"Add": 2}},
		// Forward: activation.Forward = UnaryOp#1
		{"activation_error", map[string]int{"UnaryOp": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			rnn, err := NewSimpleRNN[float32]("rnn", eng, ops, 4, 3)
			if err != nil {
				t.Fatalf("NewSimpleRNN: %v", err)
			}
			input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
			_, err = rnn.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- Backward error paths via errEngine ----------

func TestSimpleRNN_BackwardErrors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}

	// Forward succeeds, then backward fails at various points.
	// In OneStepApproximation backward:
	// 1. activation.Backward = UnaryOp#2 (UnaryOp#1 was forward activation)
	// 2. inputWeights.Backward = multiple engine calls (Reshape, Transpose, MatMul, Add)
	// 3. hiddenWeights.Backward = more engine calls

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// activation.Backward uses UnaryOp (derivative) then Mul
		// Forward uses: UnaryOp#1 (activation)
		// Backward: UnaryOp#2 (derivative of tanh)
		{"activation_backward_error", map[string]int{"UnaryOp": 2}},
		// inputWeights.Backward: Reshape calls in Linear.Backward
		// Forward: MatMul#1(iw), MatMul#2(hw), Add#1, Add#2(bias), UnaryOp#1
		// Backward: UnaryOp#2(tanh deriv), Mul#1(grad*deriv), then inputWeights.Backward starts
		// Linear.Backward calls: Reshape#1, Reshape#2, Transpose#1, MatMul#3, Add#3, Transpose#2, MatMul#4, Reshape#3
		// Fail at Reshape#1 to trigger inputWeights.Backward error
		{"inputWeights_backward_error", map[string]int{"Reshape": 1}},
		// hiddenWeights.Backward: runs after inputWeights.Backward
		// inputWeights.Backward calls: Reshape#1,#2, Transpose#1, MatMul#3, Add#3, Transpose#2, MatMul#4, Reshape#3
		// hiddenWeights.Backward calls: Reshape#4
		// Fail at Reshape#4 to trigger hiddenWeights.Backward error
		{"hiddenWeights_backward_error", map[string]int{"Reshape": 4}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			rnn, err := NewSimpleRNN[float32]("rnn", eng, ops, 4, 3)
			if err != nil {
				t.Fatalf("NewSimpleRNN: %v", err)
			}
			input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
			out, err := rnn.Forward(ctx, input)
			if err != nil {
				t.Skipf("Forward failed: %v", err)
			}
			grad := makeTensor(t, out.Shape(), make([]float32, out.Shape()[0]*out.Shape()[1]))
			_, err = rnn.Backward(ctx, types.OneStepApproximation, grad)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}
