package embeddings

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

// ---------- TokenEmbedding metadata ----------

func TestTokenEmbedding_OpType(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	if te.OpType() != "TokenEmbedding" {
		t.Errorf("OpType = %q, want TokenEmbedding", te.OpType())
	}
}

func TestTokenEmbedding_Attributes(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	attrs := te.Attributes()
	if attrs == nil {
		t.Fatal("expected non-nil attributes")
	}
	if attrs["vocab_size"] != 10 {
		t.Errorf("vocab_size = %v, want 10", attrs["vocab_size"])
	}
	if attrs["embedding_dim"] != 4 {
		t.Errorf("embedding_dim = %v, want 4", attrs["embedding_dim"])
	}
}

// ---------- NewTokenEmbeddingFromParam ----------

func TestNewTokenEmbeddingFromParam(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	t.Run("valid", func(t *testing.T) {
		tableTensor, _ := tensor.New[float32]([]int{5, 3}, make([]float32, 15))
		param, _ := graph.NewParameter("emb_table", tableTensor, tensor.New[float32])
		te, err := NewTokenEmbeddingFromParam(eng, param)
		if err != nil {
			t.Fatal(err)
		}
		if te.vocabSize != 5 || te.embeddingDim != 3 {
			t.Errorf("got vocabSize=%d embeddingDim=%d, want 5,3", te.vocabSize, te.embeddingDim)
		}
	})

	t.Run("invalid_shape", func(t *testing.T) {
		tableTensor, _ := tensor.New[float32]([]int{15}, make([]float32, 15))
		param, _ := graph.NewParameter("emb_table", tableTensor, tensor.New[float32])
		_, err := NewTokenEmbeddingFromParam(eng, param)
		if err == nil {
			t.Error("expected error for 1D embedding table")
		}
	})
}

// ---------- RotaryPositionalEmbedding metadata ----------

func TestRotaryPositionalEmbedding_OpType(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, err := NewRotaryPositionalEmbedding[float32](context.Background(), eng, 4, 10)
	if err != nil {
		t.Fatal(err)
	}
	if rpe.OpType() != "RotaryPositionalEmbedding" {
		t.Errorf("OpType = %q, want RotaryPositionalEmbedding", rpe.OpType())
	}
}

func TestRotaryPositionalEmbedding_Attributes(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, err := NewRotaryPositionalEmbedding[float32](context.Background(), eng, 4, 10)
	if err != nil {
		t.Fatal(err)
	}
	if rpe.Attributes() != nil {
		t.Error("expected nil attributes")
	}
}

// ---------- RotaryPositionalEmbedding.Scale ----------

func TestRotaryPositionalEmbedding_Scale(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, eng, 4, 10)
	if err != nil {
		t.Fatal(err)
	}

	// Get original cos values for comparison
	origCos := make([]float32, len(rpe.cosAngles.Data()))
	copy(origCos, rpe.cosAngles.Data())

	err = rpe.Scale(ctx, 2.0)
	if err != nil {
		t.Fatalf("Scale: %v", err)
	}

	// Verify cos angles were scaled
	scaledCos := rpe.cosAngles.Data()
	for i, v := range scaledCos {
		expected := origCos[i] * 2.0
		diff := v - expected
		if diff < -1e-5 || diff > 1e-5 {
			t.Errorf("cos[%d] = %v, want %v", i, v, expected)
			break
		}
	}
}

// ---------- RotaryPositionalEmbedding Forward errors ----------

func TestRotaryPositionalEmbedding_Forward_InputCount(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, _ := NewRotaryPositionalEmbedding[float32](ctx, eng, 4, 10)

	_, err := rpe.Forward(ctx)
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}

func TestRotaryPositionalEmbedding_Forward_1DInput(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, _ := NewRotaryPositionalEmbedding[float32](ctx, eng, 4, 10)

	input, _ := tensor.New[float32]([]int{4}, []float32{1, 2, 3, 4})
	_, err := rpe.Forward(ctx, input)
	if err == nil {
		t.Error("expected error for 1D input")
	}
}

// ---------- RotaryPositionalEmbedding Forward/Backward error paths via errEngine ----------

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

func (e *errEngine) Sub(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Sub"); err != nil {
		return nil, err
	}
	return e.Engine.Sub(ctx, a, b, dst...)
}

func (e *errEngine) Add(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Add"); err != nil {
		return nil, err
	}
	return e.Engine.Add(ctx, a, b, dst...)
}

func (e *errEngine) Concat(ctx context.Context, tensors []*tensor.TensorNumeric[float32], axis int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Concat"); err != nil {
		return nil, err
	}
	return e.Engine.Concat(ctx, tensors, axis, dst...)
}

func (e *errEngine) MulScalar(ctx context.Context, a *tensor.TensorNumeric[float32], scalar float32, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("MulScalar"); err != nil {
		return nil, err
	}
	return e.Engine.MulScalar(ctx, a, scalar, dst...)
}

func (e *errEngine) Reshape(ctx context.Context, a *tensor.TensorNumeric[float32], shape []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if err := e.check("Reshape"); err != nil {
		return nil, err
	}
	return e.Engine.Reshape(ctx, a, shape, dst...)
}

func (e *errEngine) Gather(ctx context.Context, params *tensor.TensorNumeric[float32], indices *tensor.TensorNumeric[int], output *tensor.TensorNumeric[float32]) error {
	if err := e.check("Gather"); err != nil {
		return err
	}
	return e.Engine.Gather(ctx, params, indices, output)
}

func (e *errEngine) ScatterAdd(ctx context.Context, dEmbeddingTable *tensor.TensorNumeric[float32], indices *tensor.TensorNumeric[int], dOut *tensor.TensorNumeric[float32]) error {
	if err := e.check("ScatterAdd"); err != nil {
		return err
	}
	return e.Engine.ScatterAdd(ctx, dEmbeddingTable, indices, dOut)
}

func (e *errEngine) Zeros(ctx context.Context, a *tensor.TensorNumeric[float32], shape []int) error {
	if err := e.check("Zeros"); err != nil {
		return err
	}
	return e.Engine.Zeros(ctx, a, shape)
}

func (e *errEngine) Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

// ---------- RotaryPositionalEmbedding Forward engine errors ----------

func TestRotaryPositionalEmbedding_ForwardErrors(t *testing.T) {
	ctx := context.Background()

	// Forward calls: Mul#1(x0*cos), Mul#2(x1*sin), Sub#1(rotX0), Mul#3(x1*cos), Mul#4(x0*sin), Add#1(rotX1), Concat#1
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"mul1_error", map[string]int{"Mul": 1}},
		{"mul2_error", map[string]int{"Mul": 2}},
		{"sub_error", map[string]int{"Sub": 1}},
		{"mul3_error", map[string]int{"Mul": 3}},
		{"mul4_error", map[string]int{"Mul": 4}},
		{"add_error", map[string]int{"Add": 1}},
		{"concat_error", map[string]int{"Concat": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eng := newErrEngine(tt.failOn)
			rpe, err := NewRotaryPositionalEmbedding[float32](ctx, eng, 4, 10)
			if err != nil {
				t.Fatal(err)
			}
			input, _ := tensor.New[float32]([]int{1, 2, 4}, make([]float32, 8))
			_, err = rpe.Forward(ctx, input)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- RotaryPositionalEmbedding Backward engine errors ----------

func TestRotaryPositionalEmbedding_BackwardErrors(t *testing.T) {
	ctx := context.Background()

	// Backward calls: Mul#1(dR0*cos), Mul#2(dR1*sin), Add#1(dLdxRot0), Mul#3(dR1*cos), Mul#4(dR0*sin), Sub#1(dLdxRot1), Concat#1
	tests := []struct {
		name   string
		failOn map[string]int
	}{
		{"mul1_error", map[string]int{"Mul": 1}},
		{"mul2_error", map[string]int{"Mul": 2}},
		{"add_error", map[string]int{"Add": 1}},
		{"mul3_error", map[string]int{"Mul": 3}},
		{"mul4_error", map[string]int{"Mul": 4}},
		{"sub_error", map[string]int{"Sub": 1}},
		{"concat_error", map[string]int{"Concat": 1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use real engine for Forward, then errEngine for Backward
			realEng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
			rpe, err := NewRotaryPositionalEmbedding[float32](ctx, realEng, 4, 10)
			if err != nil {
				t.Fatal(err)
			}
			input, _ := tensor.New[float32]([]int{1, 2, 4}, make([]float32, 8))
			_, err = rpe.Forward(ctx, input)
			if err != nil {
				t.Skipf("Forward failed: %v", err)
			}

			// Replace engine with errEngine for backward
			errEng := newErrEngine(tt.failOn)
			rpe.engine = errEng

			grad, _ := tensor.New[float32]([]int{1, 2, 4}, make([]float32, 8))
			_, err = rpe.Backward(ctx, types.FullBackprop, grad)
			if err == nil {
				t.Error("expected error")
			}
		})
	}
}

// ---------- RotaryPositionalEmbedding Scale errors ----------

func TestRotaryPositionalEmbedding_Scale_MulScalarError(t *testing.T) {
	ctx := context.Background()
	realEng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, realEng, 4, 10)
	if err != nil {
		t.Fatal(err)
	}

	// Replace engine with errEngine that fails on MulScalar
	errEng := newErrEngine(map[string]int{"MulScalar": 1})
	rpe.engine = errEng

	err = rpe.Scale(ctx, 2.0)
	if err == nil {
		t.Error("expected error from Scale MulScalar")
	}
}

func TestRotaryPositionalEmbedding_Scale_SecondMulScalarError(t *testing.T) {
	ctx := context.Background()
	realEng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, realEng, 4, 10)
	if err != nil {
		t.Fatal(err)
	}

	errEng := newErrEngine(map[string]int{"MulScalar": 2})
	rpe.engine = errEng

	err = rpe.Scale(ctx, 2.0)
	if err == nil {
		t.Error("expected error from Scale second MulScalar")
	}
}

// ---------- TokenEmbedding Forward/Backward error paths ----------

func TestTokenEmbedding_Forward_GatherError(t *testing.T) {
	ctx := context.Background()
	eng := newErrEngine(map[string]int{"Gather": 1})
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	// Use float32 indices (0.0 = token 0)
	input, _ := tensor.New[float32]([]int{2}, []float32{0, 1})
	_, err = te.Forward(ctx, input)
	if err == nil {
		t.Error("expected error from Gather")
	}
}

func TestTokenEmbedding_Forward_ReshapeError(t *testing.T) {
	ctx := context.Background()
	eng := newErrEngine(map[string]int{"Reshape": 1})
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.New[float32]([]int{2}, []float32{0, 1})
	_, err = te.Forward(ctx, input)
	if err == nil {
		t.Error("expected error from Reshape")
	}
}

func TestTokenEmbedding_Backward_ZerosError(t *testing.T) {
	ctx := context.Background()
	eng := newErrEngine(nil)
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.New[float32]([]int{2}, []float32{0, 1})
	_, err = te.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}

	eng.reset()
	eng.setFailOn(map[string]int{"Zeros": 1})
	grad, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))
	_, err = te.Backward(ctx, types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error from Zeros")
	}
}

func TestTokenEmbedding_Backward_ScatterAddError(t *testing.T) {
	ctx := context.Background()
	eng := newErrEngine(nil)
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.New[float32]([]int{2}, []float32{0, 1})
	_, err = te.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}

	eng.reset()
	eng.setFailOn(map[string]int{"ScatterAdd": 1})
	grad, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))
	_, err = te.Backward(ctx, types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error from ScatterAdd")
	}
}

func TestTokenEmbedding_Backward_ReshapeError(t *testing.T) {
	ctx := context.Background()
	eng := newErrEngine(nil)
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.New[float32]([]int{2}, []float32{0, 1})
	_, err = te.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}

	eng.reset()
	eng.setFailOn(map[string]int{"Reshape": 1})
	grad, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))
	_, err = te.Backward(ctx, types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error from Reshape")
	}
}

func TestTokenEmbedding_Backward_1DGradError(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	input, _ := tensor.New[float32]([]int{2}, []float32{0, 1})
	_, err = te.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}

	// 1D gradient should fail (must have at least 2 dims)
	grad, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})
	_, err = te.Backward(ctx, types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error for 1D gradient")
	}
}

// ---------- RoPE Forward Slice error (seqLen exceeds precomputed) ----------

func TestRotaryPositionalEmbedding_Forward_SliceError(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	// Create RoPE with maxSeqLen=2
	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, eng, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Input with seqLen=5 exceeds precomputed seqLen=2
	input, _ := tensor.New[float32]([]int{1, 5, 4}, make([]float32, 20))
	_, err = rpe.Forward(ctx, input)
	if err == nil {
		t.Error("expected error from Forward Slice (seqLen exceeds precomputed)")
	}
}

// ---------- RoPE Backward Slice errors ----------

func TestRotaryPositionalEmbedding_Backward_SliceError(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	rpe, err := NewRotaryPositionalEmbedding[float32](ctx, eng, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Forward with seqLen=2 (within range)
	input, _ := tensor.New[float32]([]int{1, 2, 4}, make([]float32, 8))
	_, err = rpe.Forward(ctx, input)
	if err != nil {
		t.Skipf("Forward failed: %v", err)
	}
	// Backward with larger seqLen gradient triggers Slice error
	grad, _ := tensor.New[float32]([]int{1, 5, 4}, make([]float32, 20))
	_, err = rpe.Backward(ctx, types.FullBackprop, grad)
	if err == nil {
		t.Error("expected error from Backward Slice")
	}
}

// ---------- TokenEmbedding Forward unsupported type ----------

func TestTokenEmbedding_Forward_UnsupportedType(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[int](numeric.IntOps{})
	te, err := NewTokenEmbedding[int](eng, 10, 4)
	if err != nil {
		// Constructor may fail for int type due to RandomUniform, which is acceptable
		t.Skipf("NewTokenEmbedding failed for int type: %v", err)
	}
	input, _ := tensor.New[int]([]int{2}, []int{0, 1})
	_, err = te.Forward(ctx, input)
	if err == nil {
		t.Error("expected error for unsupported element type int")
	}
}

func TestTokenEmbedding_Forward_InputCount(t *testing.T) {
	ctx := context.Background()
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	te, err := NewTokenEmbedding[float32](eng, 10, 4)
	if err != nil {
		t.Fatal(err)
	}
	_, err = te.Forward(ctx)
	if err == nil {
		t.Error("expected error for 0 inputs")
	}
}
