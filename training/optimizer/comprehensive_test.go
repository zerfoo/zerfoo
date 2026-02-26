package optimizer

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// ---------- countingEngine for AdamW error injection at specific call counts ----------

type countingEngine[T tensor.Numeric] struct {
	compute.Engine[T]
	calls   map[string]int
	failOn  map[string]int // fail at Nth call
	failErr error
}

func newCountingEngine[T tensor.Numeric](real compute.Engine[T], failOn map[string]int) *countingEngine[T] {
	return &countingEngine[T]{
		Engine:  real,
		calls:   make(map[string]int),
		failOn:  failOn,
		failErr: fmt.Errorf("injected error"),
	}
}

func (e *countingEngine[T]) check(op string) error {
	e.calls[op]++
	if n, ok := e.failOn[op]; ok && e.calls[op] >= n {
		return e.failErr
	}
	return nil
}

func (e *countingEngine[T]) Zeros(ctx context.Context, dst *tensor.TensorNumeric[T], shape []int) error {
	if err := e.check("Zeros"); err != nil {
		return err
	}
	return e.Engine.Zeros(ctx, dst, shape)
}

func (e *countingEngine[T]) MulScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := e.check("MulScalar"); err != nil {
		return nil, err
	}
	return e.Engine.MulScalar(ctx, a, scalar, dst...)
}

func (e *countingEngine[T]) Add(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := e.check("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *countingEngine[T]) Mul(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := e.check("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *countingEngine[T]) Sqrt(ctx context.Context, a *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := e.check("Sqrt"); err != nil {
		return nil, err
	}
	return e.Engine.Sqrt(ctx, a, dst...)
}

func (e *countingEngine[T]) AddScalar(ctx context.Context, a *tensor.TensorNumeric[T], scalar T, dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := e.check("AddScalar"); err != nil {
		return nil, err
	}
	return e.Engine.AddScalar(ctx, a, scalar, dst...)
}

func (e *countingEngine[T]) Div(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := e.check("Div"); err != nil {
		return nil, err
	}
	return e.Engine.Div(ctx, a, b, dst...)
}

func (e *countingEngine[T]) Sub(ctx context.Context, a, b *tensor.TensorNumeric[T], dst ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if err := e.check("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}

// ---------- AdamW Step error paths with counting engine ----------
// AdamW.Step call sequence per parameter (first call, state not yet initialized):
// Zeros#1(m), Zeros#2(v), MulScalar#1(m*beta1), MulScalar#2(grad*(1-beta1)),
// Add#1(mNew+gradScaled), MulScalar#3(v*beta2), Mul#1(grad*grad),
// MulScalar#4(gradSquared*(1-beta2)), Add#2(vNew+gradSquaredScaled),
// Sqrt#1(v), AddScalar#1(sqrtV+eps), Div#1(m/sqrtVPlusEps),
// MulScalar#5(updateTerm*alpha), MulScalar#6(paramValue*lrWd),
// Sub#1(param-updateTermScaled), Sub#2(paramNew-weightDecayTerm)

func TestAdamW_Step_CountingErrors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	real := compute.NewCPUEngine[float32](ops)

	createParam := func(t *testing.T) *graph.Parameter[float32] {
		t.Helper()
		value, err := tensor.New[float32]([]int{2}, []float32{1.0, 2.0})
		if err != nil {
			t.Fatal(err)
		}
		gradient, err := tensor.New[float32]([]int{2}, []float32{0.1, 0.1})
		if err != nil {
			t.Fatal(err)
		}
		param, err := graph.NewParameter("p", value, tensor.New[float32])
		if err != nil {
			t.Fatal(err)
		}
		param.Gradient = gradient
		return param
	}

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// State initialization errors
		{"zeros_m_error", map[string]int{"Zeros": 1}},
		{"zeros_v_error", map[string]int{"Zeros": 2}},
		// Moment update errors
		{"mulscalar_m_beta1", map[string]int{"MulScalar": 1}},
		{"mulscalar_grad_1_minus_beta1", map[string]int{"MulScalar": 2}},
		{"add_m_update", map[string]int{"Add": 1}},
		{"mulscalar_v_beta2", map[string]int{"MulScalar": 3}},
		{"mul_grad_squared", map[string]int{"Mul": 1}},
		{"mulscalar_gradsq_1_minus_beta2", map[string]int{"MulScalar": 4}},
		{"add_v_update", map[string]int{"Add": 2}},
		// Update computation errors
		{"sqrt_v", map[string]int{"Sqrt": 1}},
		{"addscalar_eps", map[string]int{"AddScalar": 1}},
		{"div_m_sqrtv", map[string]int{"Div": 1}},
		{"mulscalar_update_alpha", map[string]int{"MulScalar": 5}},
		{"mulscalar_weight_decay", map[string]int{"MulScalar": 6}},
		{"sub_param_update", map[string]int{"Sub": 1}},
		{"sub_weight_decay", map[string]int{"Sub": 2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newCountingEngine[float32](real, tt.failOn)
			adamw := NewAdamW[float32](eng, 0.001, 0.9, 0.999, 1e-8, 0.01)
			param := createParam(t)
			err := adamw.Step(ctx, []*graph.Parameter[float32]{param})
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- SGD Clip: float16 and float8 type coverage ----------

func runClipTest[T tensor.Numeric](
	t *testing.T,
	ops numeric.Arithmetic[T],
	values, gradients, expected []T,
	threshold float32,
) {
	t.Helper()
	engine := compute.NewCPUEngine[T](ops)
	sgd := NewSGD[T](engine, ops, 0.1)

	value, err := tensor.New[T]([]int{len(values)}, values)
	if err != nil {
		t.Fatal(err)
	}
	gradient, err := tensor.New[T]([]int{len(gradients)}, gradients)
	if err != nil {
		t.Fatal(err)
	}
	param, err := graph.NewParameter("p", value, tensor.New[T])
	if err != nil {
		t.Fatal(err)
	}
	param.Gradient = gradient

	sgd.Clip(context.Background(), []*graph.Parameter[T]{param}, threshold)

	if !reflect.DeepEqual(param.Gradient.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, param.Gradient.Data())
	}
}

func TestSGD_Clip_Float16(t *testing.T) {
	f := float16.FromFloat32
	runClipTest(t, numeric.Float16Ops{},
		[]float16.Float16{f(1), f(2), f(3), f(4)},
		[]float16.Float16{f(-10), f(0.5), f(10), f(-0.5)},
		[]float16.Float16{f(-1.0), f(0.5), f(1.0), f(-0.5)},
		1.0,
	)
}

func TestSGD_Clip_Float8(t *testing.T) {
	f := float8.FromFloat64
	runClipTest(t, numeric.Float8Ops{},
		[]float8.Float8{f(1), f(2), f(3), f(4)},
		[]float8.Float8{f(-10), f(0.5), f(10), f(-0.5)},
		[]float8.Float8{f(-1.0), f(0.5), f(1.0), f(-0.5)},
		1.0,
	)
}

// ---------- SGD Clip: values within threshold (no clipping) ----------

func TestSGD_Clip_NoClipping(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	sgd := NewSGD[float32](engine, ops, 0.1)

	value, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	gradient, _ := tensor.New[float32]([]int{2}, []float32{0.3, -0.2})
	param, _ := graph.NewParameter("p", value, tensor.New[float32])
	param.Gradient = gradient

	sgd.Clip(context.Background(), []*graph.Parameter[float32]{param}, 1.0)

	expected := []float32{0.3, -0.2}
	if !reflect.DeepEqual(param.Gradient.Data(), expected) {
		t.Errorf("expected %v, got %v", expected, param.Gradient.Data())
	}
}
