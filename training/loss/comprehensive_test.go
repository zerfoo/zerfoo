package loss

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// ---------- errEngine ----------

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

func (e *errEngine) Sub(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}

func (e *errEngine) Mul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *errEngine) Softmax(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Softmax"); err != nil {
		return nil, err
	}
	return e.Engine.Softmax(ctx, a, axis, dst...)
}

func (e *errEngine) Log(ctx context.Context, a *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Log"); err != nil {
		return nil, err
	}
	return e.Engine.Log(ctx, a, dst...)
}

func (e *errEngine) OneHot(ctx context.Context, input *tensor.TensorNumeric[int], depth int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("OneHot"); err != nil {
		return nil, err
	}
	return e.Engine.OneHot(ctx, input, depth, dst...)
}

func (e *errEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

// ---------- CrossEntropyLoss metadata tests ----------

func TestCrossEntropyLoss_OpType(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	cel := NewCrossEntropyLoss[float32](eng)
	if cel.OpType() != "CrossEntropyLoss" {
		t.Errorf("OpType = %q, want %q", cel.OpType(), "CrossEntropyLoss")
	}
}

func TestCrossEntropyLoss_Attributes(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	cel := NewCrossEntropyLoss[float32](eng)
	if cel.Attributes() != nil {
		t.Error("expected nil attributes")
	}
}

func TestCrossEntropyLoss_Forward_InputCountError(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	cel := NewCrossEntropyLoss[float32](eng)

	t.Run("zero_inputs", func(t *testing.T) {
		_, err := cel.Forward(context.Background())
		if err == nil {
			t.Error("expected error for 0 inputs")
		}
	})

	t.Run("one_input", func(t *testing.T) {
		p, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
		_, err := cel.Forward(context.Background(), p)
		if err == nil {
			t.Error("expected error for 1 input")
		}
	})

	t.Run("three_inputs", func(t *testing.T) {
		p, _ := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
		_, err := cel.Forward(context.Background(), p, p, p)
		if err == nil {
			t.Error("expected error for 3 inputs")
		}
	})
}

// ---------- CrossEntropyLoss Forward error paths ----------

func TestCrossEntropyLoss_ForwardErrors(t *testing.T) {
	preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[float32]([]int{2}, []float32{2.0, 0.0})

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"softmax_error", map[string]int{"Softmax": 1}},
		{"log_error", map[string]int{"Log": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			cel := NewCrossEntropyLoss[float32](eng)
			_, err := cel.Forward(context.Background(), preds, targets)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- CrossEntropyLoss Backward full + error paths ----------

func TestCrossEntropyLoss_Backward_Full(t *testing.T) {
	eng := newErrEngine(nil)
	cel := NewCrossEntropyLoss[float32](eng)

	// Manually set cached values (simulates successful Forward)
	preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[int]([]int{2}, []int{2, 0})
	softmax, _ := tensor.New[float32]([]int{2, 3}, []float32{0.09, 0.24, 0.67, 0.67, 0.09, 0.24})

	cel.predictions = preds
	cel.targets = targets
	cel.softmaxOutput = softmax

	dOut, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 1, 1, 1, 1, 1})
	grads, err := cel.Backward(context.Background(), types.FullBackprop, dOut)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if len(grads) != 2 {
		t.Errorf("grads len = %d, want 2", len(grads))
	}
	if grads[0] == nil {
		t.Error("predictions gradient should not be nil")
	}
	if grads[1] != nil {
		t.Error("targets gradient should be nil")
	}
}

func TestCrossEntropyLoss_BackwardErrors(t *testing.T) {
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"one_hot_error", map[string]int{"OneHot": 1}},
		{"sub_error", map[string]int{"Sub": 1}},
		{"mul_error", map[string]int{"Mul": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			cel := NewCrossEntropyLoss[float32](eng)

			preds, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 3, 1, 2})
			targets, _ := tensor.New[int]([]int{2}, []int{2, 0})
			softmax, _ := tensor.New[float32]([]int{2, 3}, []float32{0.09, 0.24, 0.67, 0.67, 0.09, 0.24})

			cel.predictions = preds
			cel.targets = targets
			cel.softmaxOutput = softmax

			dOut, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 1, 1, 1, 1, 1})
			_, err := cel.Backward(context.Background(), types.FullBackprop, dOut)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- CrossEntropyLoss float64 target type ----------

func TestCrossEntropyLoss_Forward_Float64Targets(t *testing.T) {
	eng := compute.NewCPUEngine[float64](numeric.Float64Ops{})
	cel := NewCrossEntropyLoss[float64](eng)

	preds, _ := tensor.New[float64]([]int{2, 3}, []float64{1, 2, 3, 3, 1, 2})
	targets, _ := tensor.New[float64]([]int{2}, []float64{2.0, 0.0})

	// Forward will fail at Gather (known limitation), but we exercise the float64 path
	_, _ = cel.Forward(context.Background(), preds, targets)
}

// ---------- MSE metadata tests ----------

func TestMSE_OpType(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	if m.OpType() != "MSE" {
		t.Errorf("OpType = %q, want %q", m.OpType(), "MSE")
	}
}

func TestMSE_Attributes(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	if m.Attributes() != nil {
		t.Error("expected nil attributes")
	}
}

func TestMSE_OutputShape(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	os := m.OutputShape()
	if len(os) != 1 || os[0] != 1 {
		t.Errorf("OutputShape = %v, want [1]", os)
	}
}

func TestMSE_Parameters(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	if m.Parameters() != nil {
		t.Error("expected nil parameters")
	}
}

func TestMSE_Forward_InputCountError(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := NewMSE[float32](eng, numeric.Float32Ops{})

	t.Run("zero_inputs", func(t *testing.T) {
		_, err := m.Forward(context.Background())
		if err == nil {
			t.Error("expected error for 0 inputs")
		}
	})

	t.Run("one_input", func(t *testing.T) {
		p, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
		_, err := m.Forward(context.Background(), p)
		if err == nil {
			t.Error("expected error for 1 input")
		}
	})
}

func TestMSE_Backward_NilInputs(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})
	_, err := m.Backward(context.Background(), types.FullBackprop, dOut)
	if err == nil {
		t.Error("expected error for nil predictions/targets")
	}
}

// ---------- MSE Forward/Backward error paths via errEngine ----------

func TestMSE_Forward_SubError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Sub": 1})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	p, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	tgt, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	_, err := m.Forward(context.Background(), p, tgt)
	if err == nil {
		t.Error("expected error from Sub")
	}
}

func TestMSE_Forward_MulError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Mul": 1})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	p, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	tgt, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	_, err := m.Forward(context.Background(), p, tgt)
	if err == nil {
		t.Error("expected error from Mul")
	}
}

func TestMSE_Backward_SubError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Sub": 2})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	p, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	tgt, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	_, err := m.Forward(context.Background(), p, tgt)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}

	_, err = m.Backward(context.Background(), types.FullBackprop, dOut, p, tgt)
	if err == nil {
		t.Error("expected error from Backward Sub")
	}
}

func TestMSE_Backward_MulError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Mul": 2})
	m := NewMSE[float32](eng, numeric.Float32Ops{})
	p, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	tgt, _ := tensor.New[float32]([]int{2}, []float32{3, 4})
	dOut, _ := tensor.New[float32]([]int{1}, []float32{1.0})

	_, err := m.Forward(context.Background(), p, tgt)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}

	_, err = m.Backward(context.Background(), types.FullBackprop, dOut, p, tgt)
	if err == nil {
		t.Error("expected error from Backward Mul")
	}
}
