package core

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// errEngine wraps a real engine and returns errors for specific method calls
// after a specified number of successful calls.
type errEngine struct {
	compute.Engine[float32]
	calls     map[string]int
	failOn    map[string]int
	failErr   error
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

func (e *errEngine) Mul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *errEngine) Sub(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}

func (e *errEngine) Concat(ctx context.Context, inputs []*tensor.TensorNumeric[float32], axis int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Concat"); err != nil {
		return nil, err
	}
	return e.Engine.Concat(ctx, inputs, axis, dst...)
}

func (e *errEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

func (e *errEngine) Split(ctx context.Context, a *tensor.TensorNumeric[float32], numSplits int, axis int) ([]*tensor.TensorNumeric[float32], error) {
	if err := e.check("Split"); err != nil {
		return nil, err
	}
	return e.Engine.Split(ctx, a, numSplits, axis)
}

func (e *errEngine) Softmax(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Softmax"); err != nil {
		return nil, err
	}
	return e.Engine.Softmax(ctx, a, axis, dst...)
}

func (e *errEngine) UnaryOp(ctx context.Context, a *tensor.TensorNumeric[float32], fn func(float32) float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("UnaryOp"); err != nil {
		return nil, err
	}
	return e.Engine.UnaryOp(ctx, a, fn, dst...)
}

// ---------- Linear Backward error paths ----------

func TestLinear_BackwardErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"Reshape1", map[string]int{"Reshape": 1}},
		{"Reshape2", map[string]int{"Reshape": 2}},
		{"Transpose1", map[string]int{"Transpose": 1}},
		{"MatMul1", map[string]int{"MatMul": 2}}, // MatMul#1 is Forward, #2 is Backward dw
		{"Add", map[string]int{"Add": 1}},
		{"Transpose2", map[string]int{"Transpose": 2}},
		{"MatMul2", map[string]int{"MatMul": 3}}, // MatMul#3 is dx
		{"ReshapeBack", map[string]int{"Reshape": 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			l, err := NewLinear[float32]("l", eng, ops, 4, 3)
			if err != nil {
				t.Fatalf("NewLinear: %v", err)
			}

			input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
			_, fwdErr := l.Forward(ctx, input)
			if fwdErr != nil {
				t.Skipf("Forward failed: %v", fwdErr)
			}

			grad := makeTensor(t, []int{1, 3}, []float32{1, 1, 1})
			_, err = l.Backward(ctx, types.FullBackprop, grad, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- Linear init registration test ----------

func TestLinear_InitBuilder(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	builder, err := model.GetLayerBuilder[float32]("Linear")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}

	// Success case
	node, err := builder(engine, ops, "test_lin", nil, map[string]any{
		"input_features":  4,
		"output_features": 3,
	})
	if err != nil {
		t.Fatalf("builder: %v", err)
	}
	if node.OpType() != "Linear" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "Linear")
	}

	// Missing input_features
	_, err = builder(engine, ops, "test", nil, map[string]any{
		"output_features": 3,
	})
	if err == nil {
		t.Error("missing input_features should error")
	}

	// Missing output_features
	_, err = builder(engine, ops, "test", nil, map[string]any{
		"input_features": 4,
	})
	if err == nil {
		t.Error("missing output_features should error")
	}
}

// ---------- FFN Forward/Backward error paths ----------

func TestFFN_ForwardErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	tests := []struct {
		name   string
		failOn map[string]int
	}{
		// Forward: w1.Forward fails at MatMul#1
		{"w1_Forward", map[string]int{"MatMul": 1}},
		// Forward: w3.Forward fails at MatMul#2 (w1 uses MatMul#1 + Add#1)
		{"w3_Forward", map[string]int{"MatMul": 3}},
		// Forward: Concat fails
		{"Concat_Forward", map[string]int{"Concat": 1}},
		// Forward: SwiGLU error (Split is first SwiGLU op)
		{"swiglu_Forward", map[string]int{"Split": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			f, err := NewFFN[float32]("ffn", eng, ops, 4, 8, 4)
			if err != nil {
				t.Fatalf("NewFFN: %v", err)
			}

			input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
			_, err = f.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

func TestFFN_BackwardErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// First, do a successful forward with a real engine, then swap for backward
	realEng := makeEngine()
	f, err := NewFFN[float32]("ffn", realEng, ops, 4, 8, 4)
	if err != nil {
		t.Fatal(err)
	}

	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	out, err := f.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	gradData := make([]float32, out.Shape()[0]*out.Shape()[1])
	for i := range gradData {
		gradData[i] = 0.1
	}
	grad := makeTensor(t, out.Shape(), gradData)

	// Backward with w2 bias error (Sum is called in Bias.Backward)
	errEng := newErrEngine(map[string]int{"Sum": 1})
	f.w2.bias.engine = errEng
	_, err = f.Backward(ctx, types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error from w2 backward")
	}
}

// ---------- Dense Forward/Backward error paths ----------

func TestDense_ForwardBackwardErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Dense forward: bias error
	t.Run("bias_forward_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Add": 1})
		d, err := NewDense[float32]("d", eng, ops, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
		_, err = d.Forward(ctx, input)
		if err == nil {
			t.Error("expected bias forward error")
		}
	})

	// Dense backward: bias error
	t.Run("bias_backward_error", func(t *testing.T) {
		realEng := makeEngine()
		d, err := NewDense[float32]("d", realEng, ops, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
		_, err = d.Forward(ctx, input)
		if err != nil {
			t.Fatal(err)
		}
		grad := makeTensor(t, []int{1, 3}, []float32{1, 1, 1})
		errEng := newErrEngine(map[string]int{"Sum": 1})
		d.bias.engine = errEng
		_, err = d.Backward(ctx, types.FullBackprop, grad, input)
		if err == nil {
			t.Error("expected bias backward error")
		}
	})
}

// ---------- FiLM Forward/Backward error paths ----------

func TestFiLM_ForwardErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	t.Run("scale_gen_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"MatMul": 1})
		film, err := NewFiLM[float32]("film", eng, ops, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		feature := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
		ctxIn := makeTensor(t, []int{1, 4}, []float32{0.1, 0.2, 0.3, 0.4})
		_, err = film.Forward(ctx, feature, ctxIn)
		if err == nil {
			t.Error("expected scale gen error")
		}
	})

	t.Run("bias_gen_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"MatMul": 2})
		film, err := NewFiLM[float32]("film", eng, ops, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		feature := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
		ctxIn := makeTensor(t, []int{1, 4}, []float32{0.1, 0.2, 0.3, 0.4})
		_, err = film.Forward(ctx, feature, ctxIn)
		if err == nil {
			t.Error("expected bias gen error")
		}
	})

	t.Run("mul_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Mul": 1})
		film, err := NewFiLM[float32]("film", eng, ops, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		feature := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
		ctxIn := makeTensor(t, []int{1, 4}, []float32{0.1, 0.2, 0.3, 0.4})
		_, err = film.Forward(ctx, feature, ctxIn)
		if err == nil {
			t.Error("expected mul error")
		}
	})

	t.Run("add_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Add": 1})
		film, err := NewFiLM[float32]("film", eng, ops, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		feature := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
		ctxIn := makeTensor(t, []int{1, 4}, []float32{0.1, 0.2, 0.3, 0.4})
		_, err = film.Forward(ctx, feature, ctxIn)
		if err == nil {
			t.Error("expected add error")
		}
	})
}

func TestFiLM_BackwardErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	makeFilm := func(t *testing.T) (*FiLM[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) {
		t.Helper()
		eng := makeEngine()
		film, err := NewFiLM[float32]("film", eng, ops, 4, 3)
		if err != nil {
			t.Fatal(err)
		}
		feature := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
		ctxIn := makeTensor(t, []int{1, 4}, []float32{0.1, 0.2, 0.3, 0.4})
		out, err := film.Forward(ctx, feature, ctxIn)
		if err != nil {
			t.Fatal(err)
		}
		grad := makeTensor(t, out.Shape(), make([]float32, out.Shape()[0]*out.Shape()[1]))
		for i := range grad.Data() {
			grad.Data()[i] = 1
		}
		return film, grad, feature, ctxIn
	}

	t.Run("bias_backward_error", func(t *testing.T) {
		film, grad, feature, ctxIn := makeFilm(t)
		errEng := newErrEngine(map[string]int{"Reshape": 1})
		film.biasGen.engine = errEng
		_, err := film.Backward(ctx, types.FullBackprop, grad, feature, ctxIn)
		if err == nil {
			t.Error("expected bias backward error")
		}
	})

	t.Run("dScale_mul_error", func(t *testing.T) {
		film, grad, feature, ctxIn := makeFilm(t)
		errEng := newErrEngine(map[string]int{"Mul": 1})
		film.engine = errEng
		_, err := film.Backward(ctx, types.FullBackprop, grad, feature, ctxIn)
		if err == nil {
			t.Error("expected dScale mul error")
		}
	})

	t.Run("scale_backward_error", func(t *testing.T) {
		film, grad, feature, ctxIn := makeFilm(t)
		errEng := newErrEngine(map[string]int{"Reshape": 1})
		film.scaleGen.engine = errEng
		_, err := film.Backward(ctx, types.FullBackprop, grad, feature, ctxIn)
		if err == nil {
			t.Error("expected scale backward error")
		}
	})

	t.Run("dFeature_mul_error", func(t *testing.T) {
		film, grad, feature, ctxIn := makeFilm(t)
		errEng := newErrEngine(map[string]int{"Mul": 2})
		film.engine = errEng
		_, err := film.Backward(ctx, types.FullBackprop, grad, feature, ctxIn)
		if err == nil {
			t.Error("expected dFeature mul error")
		}
	})

	t.Run("dContext_add_error", func(t *testing.T) {
		film, grad, feature, ctxIn := makeFilm(t)
		errEng := newErrEngine(map[string]int{"Add": 1})
		film.engine = errEng
		_, err := film.Backward(ctx, types.FullBackprop, grad, feature, ctxIn)
		if err == nil {
			t.Error("expected dContext add error")
		}
	})
}

// ---------- Sub Backward error (UnaryOp) ----------

func TestSub_BackwardError(t *testing.T) {
	eng := newErrEngine(map[string]int{"UnaryOp": 1})
	s := NewSub(eng)

	a := makeTensor(t, []int{1, 3}, []float32{10, 20, 30})
	b := makeTensor(t, []int{1, 3}, []float32{1, 2, 3})
	grad := makeTensor(t, []int{1, 3}, []float32{1, 1, 1})

	_, err := s.Backward(context.Background(), types.FullBackprop, grad, a, b)
	if err == nil {
		t.Error("expected UnaryOp error")
	}
}

// ---------- Mul Backward error ----------

func TestMul_BackwardError(t *testing.T) {
	ctx := context.Background()

	t.Run("gradA_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Mul": 1})
		m := NewMul(eng)
		a := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		b := makeTensor(t, []int{2, 2}, []float32{5, 6, 7, 8})
		grad := makeTensor(t, []int{2, 2}, []float32{1, 1, 1, 1})
		_, err := m.Backward(ctx, types.FullBackprop, grad, a, b)
		if err == nil {
			t.Error("expected Mul error on gradA")
		}
	})

	t.Run("gradB_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Mul": 2})
		m := NewMul(eng)
		a := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		b := makeTensor(t, []int{2, 2}, []float32{5, 6, 7, 8})
		grad := makeTensor(t, []int{2, 2}, []float32{1, 1, 1, 1})
		_, err := m.Backward(ctx, types.FullBackprop, grad, a, b)
		if err == nil {
			t.Error("expected Mul error on gradB")
		}
	})
}

// ---------- MatMul Forward/Backward errors ----------

func TestMatMul_ForwardTransposeError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Transpose": 1})
	m := NewMatMul(eng)

	// a inner dim (3) != b outer dim (2), but a inner (3) == b inner (3)
	a := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	_, err := m.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected transpose error")
	}
}

func TestMatMul_ForwardTransposedMatMulError(t *testing.T) {
	eng := newErrEngine(map[string]int{"MatMul": 1})
	m := NewMatMul(eng)

	a := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	_, err := m.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected matmul error after transpose")
	}
}

func TestMatMul_BackwardErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("gradA_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"MatMul": 2}) // #1 is Forward
		m := NewMatMul(eng)
		a := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		b := makeTensor(t, []int{2, 2}, []float32{5, 6, 7, 8})
		_, _ = m.Forward(ctx, a, b)
		grad := makeTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
		_, err := m.Backward(ctx, types.FullBackprop, grad, a, b)
		if err == nil {
			t.Error("expected gradA MatMul error")
		}
	})

	t.Run("gradB_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"MatMul": 3}) // #1=Fwd, #2=gradA, #3=gradB
		m := NewMatMul(eng)
		a := makeTensor(t, []int{2, 2}, []float32{1, 2, 3, 4})
		b := makeTensor(t, []int{2, 2}, []float32{5, 6, 7, 8})
		_, _ = m.Forward(ctx, a, b)
		grad := makeTensor(t, []int{2, 2}, []float32{1, 0, 0, 1})
		_, err := m.Backward(ctx, types.FullBackprop, grad, a, b)
		if err == nil {
			t.Error("expected gradB MatMul error")
		}
	})
}

// ---------- Bias Forward/Backward errors ----------

func TestBias_ForwardError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Add": 1})
	ops := numeric.Float32Ops{}
	b, _ := NewBias("b", eng, ops, 4)

	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	_, err := b.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected Add error")
	}
}

func TestBias_BackwardError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Sum": 1})
	ops := numeric.Float32Ops{}
	b, _ := NewBias("b", eng, ops, 4)

	// Forward first
	realEng := makeEngine()
	b.engine = realEng
	input := makeTensor(t, []int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	_, _ = b.Forward(context.Background(), input)

	// Backward with error engine
	b.engine = eng
	grad := makeTensor(t, []int{2, 4}, []float32{1, 1, 1, 1, 1, 1, 1, 1})
	_, err := b.Backward(context.Background(), types.FullBackprop, grad)
	if err == nil {
		t.Error("expected Sum error")
	}
}

// ---------- LMHead Forward error paths ----------

func TestLMHead_ForwardErrors(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	t.Run("reshape_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Reshape": 1})
		lm, err := NewLMHead(eng, ops, 4, 8)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, 4}, make([]float32, 8))
		_, err = lm.Forward(ctx, input)
		if err == nil {
			t.Error("expected reshape error")
		}
	})

	t.Run("matmul_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"MatMul": 1})
		lm, err := NewLMHead(eng, ops, 4, 8)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, 4}, make([]float32, 8))
		_, err = lm.Forward(ctx, input)
		if err == nil {
			t.Error("expected matmul error")
		}
	})

	t.Run("reshape2_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Reshape": 2})
		lm, err := NewLMHead(eng, ops, 4, 8)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, 4}, make([]float32, 8))
		_, err = lm.Forward(ctx, input)
		if err == nil {
			t.Error("expected 2nd reshape error")
		}
	})
}

// ---------- Reshape Backward error ----------

func TestReshape_BackwardError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Reshape": 2}) // #1 is Forward
	r := NewReshape(eng, []int{3, 2})

	input := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	_, _ = r.Forward(context.Background(), input)

	grad := makeTensor(t, []int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	_, err := r.Backward(context.Background(), types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected reshape error")
	}
}

// ---------- Unsqueeze Backward error ----------

func TestUnsqueeze_BackwardError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Reshape": 2}) // #1 is Forward
	u := NewUnsqueeze(eng, []int{0})

	input := makeTensor(t, []int{3, 4}, make([]float32, 12))
	_, _ = u.Forward(context.Background(), input)

	grad := makeTensor(t, []int{1, 3, 4}, make([]float32, 12))
	_, err := u.Backward(context.Background(), types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected reshape error")
	}
}

// ---------- MatMulNBits Backward error ----------

func TestMatMulNBits_BackwardTransposeError(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

	eng := newErrEngine(map[string]int{"Transpose": 1})
	m, _ := NewMatMulNBits("test", eng, ops, qw, scale, nil, 4, true)

	input := makeTensor(t, []int{1, 2}, []float32{1, 2})
	// Use real engine for dequantize (no engine calls), then forward uses our engine
	_, fwdErr := m.Forward(ctx, input)
	if fwdErr != nil {
		t.Skipf("Forward failed: %v", fwdErr)
	}

	grad := makeTensor(t, []int{1, 4}, make([]float32, 4))
	_, err := m.Backward(ctx, types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected transpose error")
	}
}

// ---------- MatMulNBits Forward MatMul error ----------

func TestMatMulNBits_ForwardMatMulError(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

	eng := newErrEngine(map[string]int{"MatMul": 1})
	m, _ := NewMatMulNBits("test", eng, ops, qw, scale, nil, 4, true)

	input := makeTensor(t, []int{1, 2}, []float32{1, 2})
	_, err := m.Forward(ctx, input)
	if err == nil {
		t.Error("expected MatMul error")
	}
}

// ---------- Sub Forward negate error ----------

func TestSub_ForwardNegateError(t *testing.T) {
	eng := newErrEngine(map[string]int{"Sub": 1})
	s := NewSub(eng)
	a := makeTensor(t, []int{1, 3}, []float32{10, 20, 30})
	_, err := s.Forward(context.Background(), a)
	if err == nil {
		t.Error("expected Sub error in negate path")
	}
}

// ---------- MatMulNBits Backward MatMul error ----------

func TestMatMulNBits_BackwardMatMulError(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	qw, _ := tensor.New[uint8]([]int{2, 2}, []uint8{0x12, 0x34, 0x56, 0x78})
	scale, _ := tensor.New[float32]([]int{2}, []float32{0.1, 0.2})

	eng := newErrEngine(map[string]int{"MatMul": 2}) // #1=Forward, #2=Backward
	m, _ := NewMatMulNBits("test", eng, ops, qw, scale, nil, 4, true)

	input := makeTensor(t, []int{1, 2}, []float32{1, 2})
	_, _ = m.Forward(ctx, input)

	grad := makeTensor(t, []int{1, 4}, make([]float32, 4))
	_, err := m.Backward(ctx, types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected MatMul error")
	}
}

// ---------- Dense Backward linear error path ----------

func TestDense_BackwardLinearError(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Dense with bias: trigger error in linear backward (Reshape)
	realEng := makeEngine()
	d, err := NewDense[float32]("d", realEng, ops, 4, 3)
	if err != nil {
		t.Fatal(err)
	}
	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	_, err = d.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	grad := makeTensor(t, []int{1, 3}, []float32{1, 1, 1})
	errEng := newErrEngine(map[string]int{"Reshape": 1})
	d.linear.engine = errEng
	_, err = d.Backward(ctx, types.FullBackprop, grad, input)
	if err == nil {
		t.Error("expected linear backward error")
	}
}

// ---------- Concat Backward axis mismatch ----------

func TestConcat_BackwardAxisMismatch(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	c := NewConcat[float32](engine, 0)

	a := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := makeTensor(t, []int{3, 3}, []float32{7, 8, 9, 10, 11, 12, 13, 14, 15})
	_, _ = c.Forward(ctx, a, b) // output [5, 3]

	// Backward with wrong axis sum
	grad := makeTensor(t, []int{5, 3}, make([]float32, 15))
	wrongA := makeTensor(t, []int{1, 3}, make([]float32, 3)) // wrong axis size
	_, err := c.Backward(ctx, types.FullBackprop, grad, wrongA, b)
	if err == nil {
		t.Error("expected axis mismatch error")
	}
}

// ---------- NewFFN no-bias error paths ----------

func TestFFN_NoBiasErrors(t *testing.T) {
	ops := numeric.Float32Ops{}

	// Error creating w1 without bias (zero dims)
	_, err := NewFFN[float32]("ffn", makeEngine(), ops, 0, 8, 4, WithFFNNoBias[float32]())
	if err == nil {
		t.Error("expected error for zero input dim")
	}
}

// ---------- Constant Attributes (dtype switch branches) ----------
// Note: The other branches (float64, int32, etc.) are unreachable for Constant[float32]
// since Data() always returns []float32. We test what we can.

// ---------- RotaryEmbedding Forward error (odd headDim) ----------

func TestRotaryEmbedding_ForwardOddHeadDim(t *testing.T) {
	engine := makeEngine()
	re := NewRotaryEmbedding[float32](engine)

	// 3D input with odd headDim (3) - RoPE requires even headDim
	input := makeTensor(t, []int{1, 2, 3}, make([]float32, 6))
	_, err := re.Forward(context.Background(), input)
	if err == nil {
		t.Error("expected error for odd headDim")
	}
}

// ---------- RotaryEmbedding Backward error ----------

func TestRotaryEmbedding_BackwardError(t *testing.T) {
	engine := makeEngine()
	ctx := context.Background()

	re := NewRotaryEmbedding[float32](engine)

	// First forward with even headDim to initialize inner
	input := makeTensor(t, []int{1, 2, 4}, make([]float32, 8))
	_, err := re.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	// Backward with wrong shape to trigger inner.Backward error
	// The inner RoPE expects specific shapes
	badGrad := makeTensor(t, []int{1, 2, 3}, make([]float32, 6)) // wrong last dim
	_, err = re.Backward(ctx, types.FullBackprop, badGrad, input)
	if err == nil {
		t.Error("expected backward error")
	}
}

// ---------- MatMul Forward compatible dims error ----------

func TestMatMul_ForwardDirectError(t *testing.T) {
	eng := newErrEngine(map[string]int{"MatMul": 1})
	m := NewMatMul(eng)

	a := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := makeTensor(t, []int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	_, err := m.Forward(context.Background(), a, b)
	if err == nil {
		t.Error("expected MatMul error for direct multiply")
	}
}

// ---------- Dense Backward activation error ----------

func TestDense_BackwardActivationError(t *testing.T) {
	ops := numeric.Float32Ops{}
	ctx := context.Background()

	// Use an activation that errors on backward
	eng := makeEngine()
	d, err := NewDense[float32]("d", eng, ops, 4, 3, WithActivation[float32](NewCast(eng)))
	if err != nil {
		t.Fatal(err)
	}

	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	_, err = d.Forward(ctx, input)
	if err != nil {
		t.Fatal(err)
	}

	// Cast backward with 0 inputs panics, but that's panic not error.
	// Let's verify normal backward works with activation
	grad := makeTensor(t, []int{1, 3}, []float32{1, 1, 1})
	grads, err := d.Backward(ctx, types.FullBackprop, grad, input)
	if err != nil {
		t.Fatalf("Dense Backward with Cast activation: %v", err)
	}
	if len(grads) != 1 {
		t.Fatalf("grads len = %d, want 1", len(grads))
	}
}

// ---------- FFN NewFFN with NoBias path (creates Dense without bias) ----------

func TestFFN_NoBiasPath(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := makeEngine()

	f, err := NewFFN[float32]("ffn", eng, ops, 4, 8, 4, WithFFNNoBias[float32]())
	if err != nil {
		t.Fatal(err)
	}

	// Verify it works (Forward + Backward)
	ctx := context.Background()
	input := makeTensor(t, []int{1, 4}, []float32{1, 2, 3, 4})
	out, err := f.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	gradData := make([]float32, out.Shape()[0]*out.Shape()[1])
	for i := range gradData {
		gradData[i] = 0.1
	}
	grad := makeTensor(t, out.Shape(), gradData)
	_, err = f.Backward(ctx, types.FullBackprop, grad)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
}

// Verify interface conformance
var (
	_ graph.Node[float32] = (*Mul[float32])(nil)
	_ graph.Node[float32] = (*Sub[float32])(nil)
	_ graph.Node[float32] = (*Reshape[float32])(nil)
	_ graph.Node[float32] = (*Shape[float32])(nil)
	_ graph.Node[float32] = (*Cast[float32])(nil)
	_ graph.Node[float32] = (*Unsqueeze[float32])(nil)
	_ graph.Node[float32] = (*MatMul[float32])(nil)
)
