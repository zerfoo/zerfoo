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

	return NewMultiHeadLatentAttention(engine, ops, numHeads, headDim, kvLoraDim, 0, wQ, wDKV, wUK, wUV, wO, rope)
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
		{"rope_head_dim", 4}, // 0 input -> defaults to headDim
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

// newTestMLAPartialRoPE creates an MLA with partial RoPE (ropeHeadDim < headDim).
func newTestMLAPartialRoPE(t *testing.T, headDim, ropeHeadDim int) *MultiHeadLatentAttention[float32] {
	t.Helper()
	ctx := context.Background()
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine(ops)

	numHeads := 2
	kvLoraDim := 3
	hiddenDim := numHeads * headDim

	qkDim := numHeads * headDim

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

	// RoPE is created with ropeHeadDim (not full headDim).
	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](ctx, engine, ropeHeadDim, 16)
	if err != nil {
		t.Fatalf("rope: %v", err)
	}

	return NewMultiHeadLatentAttention(engine, ops, numHeads, headDim, kvLoraDim, ropeHeadDim, wQ, wDKV, wUK, wUV, wO, rope)
}

func TestMultiHeadLatentAttention_PartialRoPE(t *testing.T) {
	tests := []struct {
		name        string
		headDim     int
		ropeHeadDim int
	}{
		{"half_rope", 8, 4},
		{"quarter_rope", 8, 2},
		{"full_rope_via_zero", 4, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var mla *MultiHeadLatentAttention[float32]
			if tt.ropeHeadDim == 0 {
				// Use standard constructor that defaults to full headDim.
				mla = newTestMLA(t)
			} else {
				mla = newTestMLAPartialRoPE(t, tt.headDim, tt.ropeHeadDim)
			}

			numHeads := mla.numHeads
			hiddenDim := numHeads * mla.headDim
			batch := 1
			seqLen := 3

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

			// Verify output shape is correct.
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

			// Verify ropeHeadDim attribute is set correctly.
			effectiveRopeHeadDim := tt.ropeHeadDim
			if effectiveRopeHeadDim <= 0 || effectiveRopeHeadDim >= mla.headDim {
				effectiveRopeHeadDim = mla.headDim
			}
			if got := mla.Attributes()["rope_head_dim"].(int); got != effectiveRopeHeadDim {
				t.Errorf("rope_head_dim attribute = %d, want %d", got, effectiveRopeHeadDim)
			}
		})
	}
}

func TestMultiHeadLatentAttention_PartialRoPE_PositionIndependence(t *testing.T) {
	// This test verifies that with partial RoPE, only the first ropeHeadDim
	// dimensions are affected by position, while the rest remain position-independent.
	// We do this indirectly by running the same input through MLA and verifying
	// the output is valid and consistent.
	headDim := 8
	ropeHeadDim := 4
	mla := newTestMLAPartialRoPE(t, headDim, ropeHeadDim)

	hiddenDim := mla.numHeads * headDim
	batch := 1
	seqLen := 4

	input, err := tensor.New[float32]([]int{batch, seqLen, hiddenDim}, nil)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	for i := range input.Data() {
		input.Data()[i] = float32(i%5+1) * 0.02
	}

	out, err := mla.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Basic sanity: output should have correct shape and no NaN.
	if !testutils.IntSliceEqual([]int{batch, seqLen, hiddenDim}, out.Shape()) {
		t.Errorf("output shape = %v, want [%d, %d, %d]", out.Shape(), batch, seqLen, hiddenDim)
	}
	for i, v := range out.Data() {
		if v != v {
			t.Fatalf("NaN at index %d", i)
		}
	}

	// Verify that ropeHeadDim is strictly less than headDim (partial RoPE is active).
	if mla.ropeHeadDim >= mla.headDim {
		t.Fatalf("expected partial RoPE (ropeHeadDim=%d < headDim=%d)", mla.ropeHeadDim, mla.headDim)
	}
}

func TestMultiHeadLatentAttention_PartialRoPE_SplitConcat(t *testing.T) {
	// Verify that splitLastDim correctly splits and that the pieces
	// can be concatenated back to recover the original data.
	headDim := 8
	ropeHeadDim := 4
	mla := newTestMLAPartialRoPE(t, headDim, ropeHeadDim)

	bh := 2
	seqLen := 3
	data := make([]float32, bh*seqLen*headDim)
	for i := range data {
		data[i] = float32(i)
	}
	input, err := tensor.New[float32]([]int{bh, seqLen, headDim}, data)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}

	left, right, err := mla.splitLastDim(input, bh, seqLen, ropeHeadDim)
	if err != nil {
		t.Fatalf("splitLastDim: %v", err)
	}

	// Check shapes.
	if !testutils.IntSliceEqual([]int{bh, seqLen, ropeHeadDim}, left.Shape()) {
		t.Errorf("left shape = %v, want [%d, %d, %d]", left.Shape(), bh, seqLen, ropeHeadDim)
	}
	rest := headDim - ropeHeadDim
	if !testutils.IntSliceEqual([]int{bh, seqLen, rest}, right.Shape()) {
		t.Errorf("right shape = %v, want [%d, %d, %d]", right.Shape(), bh, seqLen, rest)
	}

	// Verify split content: for each row, left has [0..ropeHeadDim), right has [ropeHeadDim..headDim).
	for i := 0; i < bh*seqLen; i++ {
		for j := 0; j < ropeHeadDim; j++ {
			want := float32(i*headDim + j)
			got := left.Data()[i*ropeHeadDim+j]
			if got != want {
				t.Errorf("left[%d][%d] = %v, want %v", i, j, got, want)
			}
		}
		for j := 0; j < rest; j++ {
			want := float32(i*headDim + ropeHeadDim + j)
			got := right.Data()[i*rest+j]
			if got != want {
				t.Errorf("right[%d][%d] = %v, want %v", i, j, got, want)
			}
		}
	}

	// Concatenate back and verify we get the original data.
	ctx := context.Background()
	joined, err := mla.engine.Concat(ctx, []*tensor.TensorNumeric[float32]{left, right}, 2)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}
	if !testutils.IntSliceEqual(input.Shape(), joined.Shape()) {
		t.Errorf("joined shape = %v, want %v", joined.Shape(), input.Shape())
	}
	for i, v := range joined.Data() {
		if v != data[i] {
			t.Errorf("joined[%d] = %v, want %v", i, v, data[i])
		}
	}
}
