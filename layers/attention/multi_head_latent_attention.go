package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// MultiHeadLatentAttention implements Multi-head Latent Attention (MLA)
// as used in DeepSeek V3/R1. MLA compresses KV into a low-rank latent
// vector, dramatically reducing KV cache size.
//
// Partial RoPE: RoPE is applied only to the first ropeHeadDim dimensions
// of Q and K. The remaining dimensions are position-independent, matching
// the DeepSeek V3 paper specification.
type MultiHeadLatentAttention[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Dimensions
	numHeads    int
	headDim     int
	ropeHeadDim int // RoPE applied to first ropeHeadDim dims only
	kvLoraDim   int

	// Projections
	wQ   *core.Dense[T] // query projection [hidden, numHeads*headDim]
	wDKV *core.Dense[T] // down-project KV [hidden, kvLoraDim]
	wUK  *core.Dense[T] // up-project keys [kvLoraDim, numHeads*headDim]
	wUV  *core.Dense[T] // up-project values [kvLoraDim, numHeads*headDim]
	wO   *core.Dense[T] // output projection [numHeads*headDim, hidden]

	rope *embeddings.RotaryPositionalEmbedding[T]
	sdpa *ScaledDotProductAttention[T]

	outputShape []int
}

// NewMultiHeadLatentAttention creates a new MLA layer.
// ropeHeadDim specifies how many of the headDim dimensions receive RoPE.
// If ropeHeadDim <= 0 or >= headDim, RoPE is applied to all dimensions
// (backwards-compatible behavior).
func NewMultiHeadLatentAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	numHeads, headDim, kvLoraDim, ropeHeadDim int,
	wQ, wDKV, wUK, wUV, wO *core.Dense[T],
	rope *embeddings.RotaryPositionalEmbedding[T],
) *MultiHeadLatentAttention[T] {
	if ropeHeadDim <= 0 || ropeHeadDim >= headDim {
		ropeHeadDim = headDim
	}
	return &MultiHeadLatentAttention[T]{
		engine:      engine,
		ops:         ops,
		numHeads:    numHeads,
		headDim:     headDim,
		ropeHeadDim: ropeHeadDim,
		kvLoraDim:   kvLoraDim,
		wQ:          wQ,
		wDKV:        wDKV,
		wUK:         wUK,
		wUV:         wUV,
		wO:          wO,
		rope:        rope,
		sdpa:        NewScaledDotProductAttention(engine, headDim),
	}
}

// Forward computes the MLA forward pass.
// Input: [batch, seqLen, hidden]
// Output: [batch, seqLen, hidden]
func (m *MultiHeadLatentAttention[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MLA expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	batch := shape[0]
	seqLen := shape[1]

	// 1. Query projection: [batch, seqLen, hidden] -> [batch, seqLen, numHeads*headDim]
	qProj, err := m.wQ.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("query projection: %w", err)
	}

	// 2. KV down-projection: [batch, seqLen, hidden] -> [batch, seqLen, kvLoraDim]
	cKV, err := m.wDKV.Forward(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("kv down-projection: %w", err)
	}

	// 3. Up-project keys: [batch, seqLen, kvLoraDim] -> [batch, seqLen, numHeads*headDim]
	kProj, err := m.wUK.Forward(ctx, cKV)
	if err != nil {
		return nil, fmt.Errorf("key up-projection: %w", err)
	}

	// 4. Up-project values: [batch, seqLen, kvLoraDim] -> [batch, seqLen, numHeads*headDim]
	vProj, err := m.wUV.Forward(ctx, cKV)
	if err != nil {
		return nil, fmt.Errorf("value up-projection: %w", err)
	}

	// 5. Reshape Q/K/V to multi-head format: [batch, seqLen, numHeads, headDim]
	qHeads, err := m.engine.Reshape(ctx, qProj, []int{batch, seqLen, m.numHeads, m.headDim})
	if err != nil {
		return nil, err
	}
	kHeads, err := m.engine.Reshape(ctx, kProj, []int{batch, seqLen, m.numHeads, m.headDim})
	if err != nil {
		return nil, err
	}
	vHeads, err := m.engine.Reshape(ctx, vProj, []int{batch, seqLen, m.numHeads, m.headDim})
	if err != nil {
		return nil, err
	}

	// 6. Transpose to [batch, numHeads, seqLen, headDim]
	qHeads, err = m.engine.Transpose(ctx, qHeads, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	kHeads, err = m.engine.Transpose(ctx, kHeads, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	vHeads, err = m.engine.Transpose(ctx, vHeads, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// 7. Apply partial RoPE to Q and K.
	// Only the first ropeHeadDim dimensions get RoPE; the rest are
	// position-independent (per the DeepSeek V3 paper).
	qHeads, err = m.applyPartialRoPE(ctx, qHeads, batch, seqLen)
	if err != nil {
		return nil, fmt.Errorf("rope Q: %w", err)
	}
	kHeads, err = m.applyPartialRoPE(ctx, kHeads, batch, seqLen)
	if err != nil {
		return nil, fmt.Errorf("rope K: %w", err)
	}

	// 8. Flatten heads into batch for SDPA: [batch, numHeads, seqLen, headDim] -> [batch*numHeads, seqLen, headDim]
	qFlat, err := m.engine.Reshape(ctx, qHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}
	kFlat, err := m.engine.Reshape(ctx, kHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}
	vFlat, err := m.engine.Reshape(ctx, vHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}

	// 9. Scaled dot-product attention: [batch*numHeads, seqLen, headDim]
	attnOut, err := m.sdpa.Forward(ctx, qFlat, kFlat, vFlat, nil)
	if err != nil {
		return nil, fmt.Errorf("sdpa: %w", err)
	}

	// 10. Reshape back: [batch*numHeads, seqLen, headDim] -> [batch, seqLen, numHeads*headDim]
	attnOut, err = m.engine.Reshape(ctx, attnOut, []int{batch, seqLen, m.numHeads * m.headDim})
	if err != nil {
		return nil, err
	}

	// 11. Output projection: [batch, seqLen, numHeads*headDim] -> [batch, seqLen, hidden]
	output, err := m.wO.Forward(ctx, attnOut)
	if err != nil {
		return nil, fmt.Errorf("output projection: %w", err)
	}

	m.outputShape = output.Shape()
	return output, nil
}

// applyPartialRoPE applies RoPE to only the first ropeHeadDim dimensions
// of the input tensor (shape [batch, numHeads, seqLen, headDim]).
// When ropeHeadDim == headDim, this applies RoPE to all dimensions.
func (m *MultiHeadLatentAttention[T]) applyPartialRoPE(
	ctx context.Context,
	heads *tensor.TensorNumeric[T],
	batch, seqLen int,
) (*tensor.TensorNumeric[T], error) {
	bh := batch * m.numHeads

	if m.ropeHeadDim == m.headDim {
		// Full RoPE: apply to all dimensions (no splitting needed).
		flat, err := m.engine.Reshape(ctx, heads, []int{bh, seqLen, m.headDim})
		if err != nil {
			return nil, err
		}
		roped, err := m.rope.Forward(ctx, flat)
		if err != nil {
			return nil, err
		}
		return m.engine.Reshape(ctx, roped, []int{batch, m.numHeads, seqLen, m.headDim})
	}

	// Partial RoPE: split along last dim into [ropeHeadDim] and [headDim-ropeHeadDim].
	flat, err := m.engine.Reshape(ctx, heads, []int{bh, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}

	ropePart, passthru, err := m.splitLastDim(flat, bh, seqLen, m.ropeHeadDim)
	if err != nil {
		return nil, fmt.Errorf("split for partial rope: %w", err)
	}

	// Apply RoPE only to the first ropeHeadDim dimensions.
	roped, err := m.rope.Forward(ctx, ropePart)
	if err != nil {
		return nil, err
	}

	// Concatenate roped and passthrough parts back along the last axis.
	joined, err := m.engine.Concat(ctx, []*tensor.TensorNumeric[T]{roped, passthru}, 2)
	if err != nil {
		return nil, fmt.Errorf("concat after partial rope: %w", err)
	}

	return m.engine.Reshape(ctx, joined, []int{batch, m.numHeads, seqLen, m.headDim})
}

// splitLastDim splits a 3D tensor [bh, seqLen, dim] along the last axis
// into two tensors: [bh, seqLen, splitAt] and [bh, seqLen, dim-splitAt].
func (m *MultiHeadLatentAttention[T]) splitLastDim(
	t *tensor.TensorNumeric[T],
	bh, seqLen, splitAt int,
) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], error) {
	dim := m.headDim
	rest := dim - splitAt
	data := t.Data()

	leftData := make([]T, bh*seqLen*splitAt)
	rightData := make([]T, bh*seqLen*rest)

	for i := 0; i < bh*seqLen; i++ {
		srcOff := i * dim
		copy(leftData[i*splitAt:(i+1)*splitAt], data[srcOff:srcOff+splitAt])
		copy(rightData[i*rest:(i+1)*rest], data[srcOff+splitAt:srcOff+dim])
	}

	left, err := tensor.New[T]([]int{bh, seqLen, splitAt}, leftData)
	if err != nil {
		return nil, nil, err
	}
	right, err := tensor.New[T]([]int{bh, seqLen, rest}, rightData)
	if err != nil {
		return nil, nil, err
	}
	return left, right, nil
}

// Backward computes gradients for MLA (not yet implemented).
func (m *MultiHeadLatentAttention[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("MLA backward not implemented")
}

// Parameters returns all trainable parameters.
func (m *MultiHeadLatentAttention[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, m.wQ.Parameters()...)
	params = append(params, m.wDKV.Parameters()...)
	params = append(params, m.wUK.Parameters()...)
	params = append(params, m.wUV.Parameters()...)
	params = append(params, m.wO.Parameters()...)
	return params
}

// OutputShape returns the output shape.
func (m *MultiHeadLatentAttention[T]) OutputShape() []int {
	return m.outputShape
}

// OpType returns the layer operation type.
func (m *MultiHeadLatentAttention[T]) OpType() string {
	return "MultiHeadLatentAttention"
}

// Attributes returns the layer attributes.
func (m *MultiHeadLatentAttention[T]) Attributes() map[string]any {
	return map[string]any{
		"num_heads":     m.numHeads,
		"head_dim":      m.headDim,
		"rope_head_dim": m.ropeHeadDim,
		"kv_lora_dim":   m.kvLoraDim,
	}
}

// Compile-time interface check.
var _ graph.Node[float32] = (*MultiHeadLatentAttention[float32])(nil)
