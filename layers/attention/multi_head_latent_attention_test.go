package attention

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/testing/testutils"
)

// newTestMLA creates a MultiHeadLatentAttention layer with small dimensions
// suitable for unit testing.
func newTestMLA(t *testing.T) *MultiHeadLatentAttention[float32] {
	t.Helper()
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	numHeads := 2
	headDim := 4
	kvLoraDim := 3
	hiddenDim := 8

	qkDim := numHeads * headDim // 8

	wQ, err := core.NewDense[float32]("wQ", engine, ops, hiddenDim, qkDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wQ: %v", err)
	}
	wDKV, err := core.NewDense[float32]("wDKV", engine, ops, hiddenDim, kvLoraDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wDKV: %v", err)
	}
	wUK, err := core.NewDense[float32]("wUK", engine, ops, kvLoraDim, qkDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wUK: %v", err)
	}
	wUV, err := core.NewDense[float32]("wUV", engine, ops, kvLoraDim, qkDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wUV: %v", err)
	}
	wO, err := core.NewDense[float32]("wO", engine, ops, qkDim, hiddenDim, core.WithoutBias[float32]())
	if err != nil {
		t.Fatalf("wO: %v", err)
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](ctx, engine, headDim, 16)
	if err != nil {
		t.Fatalf("rope: %v", err)
	}

	return NewMultiHeadLatentAttention(engine, ops, numHeads, headDim, kvLoraDim, wQ, wDKV, wUK, wUV, wO, rope)
}

func TestMultiHeadLatentAttention_Forward_Shape(t *testing.T) {
	mla := newTestMLA(t)

	batch := 1
	seqLen := 3
	hiddenDim := 8

	input, err := tensor.New[float32]([]int{batch, seqLen, hiddenDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = float32(i%7+1) * 0.01
	}

	out, err := mla.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expected := []int{batch, seqLen, hiddenDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Errorf("output shape = %v, want %v", out.Shape(), expected)
	}
}

func TestMultiHeadLatentAttention_Forward_Batch(t *testing.T) {
	mla := newTestMLA(t)

	batch := 2
	seqLen := 4
	hiddenDim := 8

	input, err := tensor.New[float32]([]int{batch, seqLen, hiddenDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = float32(i%11+1) * 0.02
	}

	out, err := mla.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expected := []int{batch, seqLen, hiddenDim}
	if !testutils.IntSliceEqual(expected, out.Shape()) {
		t.Errorf("output shape = %v, want %v", out.Shape(), expected)
	}

	// Verify no NaN in output.
	for i, v := range out.Data() {
		if v != v {
			t.Fatalf("NaN at index %d", i)
		}
	}
}

func TestMultiHeadLatentAttention_Forward_InputValidation(t *testing.T) {
	mla := newTestMLA(t)
	ctx := context.Background()

	input1, _ := tensor.New[float32]([]int{1, 2, 8}, nil)
	input2, _ := tensor.New[float32]([]int{1, 2, 8}, nil)

	_, err := mla.Forward(ctx, input1, input2)
	if err == nil {
		t.Error("expected error for 2 inputs, got nil")
	}

	_, err = mla.Forward(ctx)
	if err == nil {
		t.Error("expected error for 0 inputs, got nil")
	}
}

func TestMultiHeadLatentAttention_Parameters(t *testing.T) {
	mla := newTestMLA(t)
	params := mla.Parameters()

	// 5 Dense layers without bias = 5 weight parameters.
	if len(params) != 5 {
		t.Errorf("len(Parameters()) = %d, want 5", len(params))
	}
}

func TestMultiHeadLatentAttention_OpType(t *testing.T) {
	mla := newTestMLA(t)
	if got := mla.OpType(); got != "MultiHeadLatentAttention" {
		t.Errorf("OpType() = %q, want %q", got, "MultiHeadLatentAttention")
	}
}

func TestMultiHeadLatentAttention_Attributes(t *testing.T) {
	mla := newTestMLA(t)
	attrs := mla.Attributes()

	tests := []struct {
		key  string
		want int
	}{
		{"num_heads", 2},
		{"head_dim", 4},
		{"kv_lora_dim", 3},
	}
	for _, tt := range tests {
		v, ok := attrs[tt.key].(int)
		if !ok {
			t.Errorf("Attributes()[%q] missing or not int", tt.key)
			continue
		}
		if v != tt.want {
			t.Errorf("Attributes()[%q] = %d, want %d", tt.key, v, tt.want)
		}
	}
}

func TestMultiHeadLatentAttention_OutputShape(t *testing.T) {
	mla := newTestMLA(t)

	// Before Forward, OutputShape is nil.
	if s := mla.OutputShape(); s != nil {
		t.Errorf("OutputShape() before Forward = %v, want nil", s)
	}

	input, _ := tensor.New[float32]([]int{1, 3, 8}, nil)
	for i := range input.Data() {
		input.Data()[i] = 0.1
	}

	_, err := mla.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expected := []int{1, 3, 8}
	if !testutils.IntSliceEqual(expected, mla.OutputShape()) {
		t.Errorf("OutputShape() = %v, want %v", mla.OutputShape(), expected)
	}
}
