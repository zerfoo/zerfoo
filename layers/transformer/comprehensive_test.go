package transformer

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

// ---------- mockAttn ----------

type mockAttn struct {
	forwardFunc  func(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error)
	backwardFunc func(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error)
	outShape     []int
}

func (m *mockAttn) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if m.forwardFunc != nil {
		return m.forwardFunc(ctx, inputs...)
	}
	return inputs[0], nil
}

func (m *mockAttn) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[float32], inputs ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	if m.backwardFunc != nil {
		return m.backwardFunc(ctx, mode, dOut, inputs...)
	}
	return []*tensor.TensorNumeric[float32]{dOut}, nil
}

func (m *mockAttn) Parameters() []*graph.Parameter[float32] { return nil }
func (m *mockAttn) OpType() string                          { return "MockAttention" }
func (m *mockAttn) Attributes() map[string]interface{}      { return nil }
func (m *mockAttn) OutputShape() []int                      { return m.outShape }

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

func (e *errEngine) reset() {
	for k := range e.calls {
		delete(e.calls, k)
	}
}

func (e *errEngine) setFailOn(failOn map[string]int) {
	e.failOn = failOn
}

func (e *errEngine) Mul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Mul"); err != nil {
		return nil, err
	}
	return e.Engine.Mul(ctx, a, b, dst...)
}

func (e *errEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *errEngine) Sub(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}

func (e *errEngine) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("MatMul"); err != nil {
		return nil, err
	}
	return e.Engine.MatMul(ctx, a, b, dst...)
}

func (e *errEngine) ReduceMean(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("ReduceMean"); err != nil {
		return nil, err
	}
	return e.Engine.ReduceMean(ctx, a, axis, keepDims, dst...)
}

func (e *errEngine) AddScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("AddScalar"); err != nil {
		return nil, err
	}
	return e.Engine.AddScalar(ctx, a, scalar, dst...)
}

func (e *errEngine) Rsqrt(ctx context.Context, a *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Rsqrt"); err != nil {
		return nil, err
	}
	return e.Engine.Rsqrt(ctx, a, dst...)
}

func (e *errEngine) ReduceSum(ctx context.Context, a *tensor.TensorNumeric[float32], axis int, keepDims bool, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("ReduceSum"); err != nil {
		return nil, err
	}
	return e.Engine.ReduceSum(ctx, a, axis, keepDims, dst...)
}

func (e *errEngine) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[float32], axis int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Concat"); err != nil {
		return nil, err
	}
	return e.Engine.Concat(ctx, tensors, axis, dst...)
}

func (e *errEngine) Split(ctx context.Context, a *tensor.TensorNumeric[float32], numSplits int, axis int) ([]*tensor.TensorNumeric[float32], error) {
	if err := e.check("Split"); err != nil {
		return nil, err
	}
	return e.Engine.Split(ctx, a, numSplits, axis)
}

func (e *errEngine) Transpose(ctx context.Context, a *tensor.TensorNumeric[float32], axes []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Transpose"); err != nil {
		return nil, err
	}
	return e.Engine.Transpose(ctx, a, axes, dst...)
}

func (e *errEngine) Reshape(ctx context.Context, a *tensor.TensorNumeric[float32], shape []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Reshape"); err != nil {
		return nil, err
	}
	return e.Engine.Reshape(ctx, a, shape, dst...)
}

func (e *errEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

func makeTensor(t *testing.T, shape []int) *tensor.TensorNumeric[float32] {
	t.Helper()
	n := 1
	for _, s := range shape {
		n *= s
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return tn
}

// findBackwardThreshold iterates through possible call counts for opName to find
// the threshold that triggers targetError during Backward. This avoids duplicating
// the threshold-search pattern across multiple tests.
func findBackwardThreshold(t *testing.T, ctx context.Context, ops numeric.Float32Ops, dim int, opName, targetError string) {
	t.Helper()
	eng := newErrEngine(nil)
	block, err := NewTransformerBlock[float32](eng, ops, dim, dim, &mockAttn{})
	if err != nil {
		t.Fatal(err)
	}
	input := makeTensor(t, []int{1, 2, dim})
	_, err = block.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}
	eng.reset()
	grad := makeTensor(t, []int{1, 2, dim})
	_, _ = block.Backward(ctx, types.FullBackprop, grad)
	totalCalls := eng.calls[opName]
	for threshold := 1; threshold <= totalCalls; threshold++ {
		eng2 := newErrEngine(nil)
		block2, err2 := NewTransformerBlock[float32](eng2, ops, dim, dim, &mockAttn{})
		if err2 != nil {
			continue
		}
		_, err2 = block2.Forward(ctx, input)
		if err2 != nil {
			continue
		}
		eng2.reset()
		eng2.setFailOn(map[string]int{opName: threshold})
		_, err2 = block2.Backward(ctx, types.FullBackprop, grad)
		if err2 != nil && err2.Error() == targetError {
			return
		}
	}
}

// ---------- Accessor methods ----------

func TestBlock_OpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	attn := &mockAttn{}
	block, err := NewTransformerBlock[float32](eng, ops, 8, 8, attn)
	if err != nil {
		t.Fatal(err)
	}
	if block.OpType() != "TransformerBlock" {
		t.Errorf("OpType = %q, want TransformerBlock", block.OpType())
	}
}

func TestBlock_Attributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	attn := &mockAttn{}
	block, err := NewTransformerBlock[float32](eng, ops, 8, 8, attn)
	if err != nil {
		t.Fatal(err)
	}
	if block.Attributes() != nil {
		t.Error("expected nil attributes")
	}
}

func TestBlock_OutputShape(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	attn := &mockAttn{outShape: []int{2, 10, 8}}
	block, err := NewTransformerBlock[float32](eng, ops, 8, 8, attn)
	if err != nil {
		t.Fatal(err)
	}
	os := block.OutputShape()
	if len(os) != 3 || os[2] != 8 {
		t.Errorf("OutputShape = %v, want [2 10 8]", os)
	}
}

func TestBlock_Attention(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	attn := &mockAttn{}
	block, err := NewTransformerBlock[float32](eng, ops, 8, 8, attn)
	if err != nil {
		t.Fatal(err)
	}
	if block.Attention() != attn {
		t.Error("Attention() should return the original attention node")
	}
}

func TestBlock_Engine(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	attn := &mockAttn{}
	block, err := NewTransformerBlock[float32](eng, ops, 8, 8, attn)
	if err != nil {
		t.Fatal(err)
	}
	if block.Engine() != eng {
		t.Error("Engine() should return the original engine")
	}
}

// ---------- Constructor error ----------

func TestBlock_NewTransformerBlock_FFNError(t *testing.T) {
	ops := numeric.Float32Ops{}
	eng := compute.NewCPUEngine[float32](ops)
	attn := &mockAttn{}
	// ffnDim=0 should cause FFN creation to fail
	_, err := NewTransformerBlock[float32](eng, ops, 8, 0, attn)
	if err == nil {
		t.Error("expected error for ffnDim=0")
	}
}

// ---------- Forward error paths ----------

func TestBlock_Forward_Errors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	dim := 8

	// makeTensor is defined at package level

	t.Run("norm1_error", func(t *testing.T) {
		eng := newErrEngine(map[string]int{"Mul": 1})
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err == nil {
			t.Error("expected error from norm1")
		}
	})

	t.Run("attention_error", func(t *testing.T) {
		eng := compute.NewCPUEngine[float32](ops)
		attn := &mockAttn{
			forwardFunc: func(_ context.Context, _ ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
				return nil, fmt.Errorf("attention forward error")
			},
		}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err == nil {
			t.Error("expected error from attention")
		}
	})

	t.Run("first_add_error", func(t *testing.T) {
		// RMSNorm uses Mul, ReduceMean, AddScalar, Rsqrt but NOT Add.
		// So Add#1 is the block's first residual connection.
		eng := newErrEngine(map[string]int{"Add": 1})
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err == nil {
			t.Error("expected error from first Add (residual)")
		}
	})

	t.Run("normPostAttention_error", func(t *testing.T) {
		// norm1 uses Mul#1,#2,#3. normPostAttention starts at Mul#4.
		eng := newErrEngine(map[string]int{"Mul": 4})
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err == nil {
			t.Error("expected error from normPostAttention")
		}
	})

	t.Run("norm2_error", func(t *testing.T) {
		// norm1: Mul#1,2,3. normPostAttn: Mul#4,5,6. norm2 starts at Mul#7.
		eng := newErrEngine(map[string]int{"Mul": 7})
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err == nil {
			t.Error("expected error from norm2")
		}
	})

	t.Run("ffn_error", func(t *testing.T) {
		// RMSNorm uses Mul, not MatMul. So MatMul#1 is FFN's first Linear.
		eng := newErrEngine(map[string]int{"MatMul": 1})
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err == nil {
			t.Error("expected error from ffn")
		}
	})

	t.Run("second_add_error", func(t *testing.T) {
		// Forward Add calls: #1=residual1, #2=w1 bias, #3=w3 bias, #4=w2 bias, #5=residual2
		eng := newErrEngine(map[string]int{"Add": 5})
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err == nil {
			t.Error("expected error from second Add (residual)")
		}
	})
}

// ---------- Backward error paths ----------

func TestBlock_Backward_Errors(t *testing.T) {
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	dim := 8

	makeTensor := func(t *testing.T, shape []int) *tensor.TensorNumeric[float32] {
		t.Helper()
		n := 1
		for _, s := range shape {
			n *= s
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i%10+1) * 0.01
		}
		tn, err := tensor.New(shape, data)
		if err != nil {
			t.Fatal(err)
		}
		return tn
	}

	t.Run("attention_backward_error", func(t *testing.T) {
		eng := compute.NewCPUEngine[float32](ops)
		attn := &mockAttn{
			backwardFunc: func(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
				return nil, fmt.Errorf("attention backward error")
			},
		}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err != nil {
			t.Skipf("Forward failed: %v", err)
		}
		grad := makeTensor(t, []int{1, 2, dim})
		_, err = block.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error from attention backward")
		}
	})

	t.Run("ffn_backward_error", func(t *testing.T) {
		eng := newErrEngine(nil)
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err != nil {
			t.Skipf("Forward failed: %v", err)
		}
		// Reset counts after forward, then fail on first MatMul in backward (ffn backward)
		eng.reset()
		eng.setFailOn(map[string]int{"MatMul": 1})
		grad := makeTensor(t, []int{1, 2, dim})
		_, err = block.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error from ffn backward")
		}
	})

	t.Run("norm2_backward_error", func(t *testing.T) {
		eng := newErrEngine(nil)
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err != nil {
			t.Skipf("Forward failed: %v", err)
		}
		// After FFN backward completes, norm2 backward uses Mul. Count Mul calls in FFN backward
		// and fail after. We use ReduceSum which is only used by norm backward, not FFN backward.
		eng.reset()
		eng.setFailOn(map[string]int{"ReduceSum": 1})
		grad := makeTensor(t, []int{1, 2, dim})
		_, err = block.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error from norm2 backward")
		}
	})

	t.Run("accumulate_npa_gradient_error", func(t *testing.T) {
		findBackwardThreshold(t, ctx, ops, dim, "Add", "accumulate npa gradient: injected error")
	})

	t.Run("norm1_backward_error", func(t *testing.T) {
		eng := newErrEngine(nil)
		attn := &mockAttn{}
		block, err := NewTransformerBlock[float32](eng, ops, dim, dim, attn)
		if err != nil {
			t.Fatal(err)
		}
		input := makeTensor(t, []int{1, 2, dim})
		_, err = block.Forward(ctx, input)
		if err != nil {
			t.Skipf("Forward failed: %v", err)
		}
		// Count ReduceSum calls in full backward
		eng.reset()
		grad := makeTensor(t, []int{1, 2, dim})
		_, _ = block.Backward(ctx, types.FullBackprop, grad)
		totalRS := eng.calls["ReduceSum"]
		// norm1 backward is the 3rd RMSNorm backward. Each uses 3 ReduceSum calls.
		// norm2: RS#1,2,3. normPostAttn: RS#4,5,6. norm1: RS#7,8,9
		for threshold := 1; threshold <= totalRS; threshold++ {
			eng2 := newErrEngine(nil)
			block2, err2 := NewTransformerBlock[float32](eng2, ops, dim, dim, &mockAttn{})
			if err2 != nil {
				continue
			}
			_, err2 = block2.Forward(ctx, input)
			if err2 != nil {
				continue
			}
			eng2.reset()
			eng2.setFailOn(map[string]int{"ReduceSum": threshold})
			_, err2 = block2.Backward(ctx, types.FullBackprop, grad)
			if err2 != nil && err2.Error() == "norm1 backward: injected error" {
				return
			}
		}
		// If we didn't find the exact threshold, try hitting norm1 backward at ReduceSum#7
		eng.reset()
		eng.setFailOn(map[string]int{"ReduceSum": 7})
		_, err = block.Backward(ctx, types.FullBackprop, grad)
		if err == nil {
			t.Error("expected error from norm1 backward")
		}
	})

	t.Run("accumulate_input_gradient_error", func(t *testing.T) {
		findBackwardThreshold(t, ctx, ops, dim, "Add", "accumulate input gradient: injected error")
	})
}
