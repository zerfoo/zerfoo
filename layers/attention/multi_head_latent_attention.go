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
type MultiHeadLatentAttention[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Dimensions
	numHeads  int
	headDim   int
	kvLoraDim int

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
func NewMultiHeadLatentAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	numHeads, headDim, kvLoraDim int,
	wQ, wDKV, wUK, wUV, wO *core.Dense[T],
	rope *embeddings.RotaryPositionalEmbedding[T],
) *MultiHeadLatentAttention[T] {
	return &MultiHeadLatentAttention[T]{
		engine:    engine,
		ops:       ops,
		numHeads:  numHeads,
		headDim:   headDim,
		kvLoraDim: kvLoraDim,
		wQ:        wQ,
		wDKV:      wDKV,
		wUK:       wUK,
		wUV:       wUV,
		wO:        wO,
		rope:      rope,
		sdpa:      NewScaledDotProductAttention(engine, headDim),
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

	// 7. Apply RoPE to Q and K per-head.
	// Reshape to [batch*numHeads, seqLen, headDim] for RoPE.
	qFlat, err := m.engine.Reshape(ctx, qHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}
	kFlat, err := m.engine.Reshape(ctx, kHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}

	qRoPE, err := m.rope.Forward(ctx, qFlat)
	if err != nil {
		return nil, fmt.Errorf("rope Q: %w", err)
	}
	kRoPE, err := m.rope.Forward(ctx, kFlat)
	if err != nil {
		return nil, fmt.Errorf("rope K: %w", err)
	}

	// Reshape back to [batch, numHeads, seqLen, headDim]
	qHeads, err = m.engine.Reshape(ctx, qRoPE, []int{batch, m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}
	kHeads, err = m.engine.Reshape(ctx, kRoPE, []int{batch, m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}

	// 8. Flatten heads into batch for SDPA: [batch, numHeads, seqLen, headDim] -> [batch*numHeads, seqLen, headDim]
	qFlat2, err := m.engine.Reshape(ctx, qHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}
	kFlat2, err := m.engine.Reshape(ctx, kHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}
	vFlat2, err := m.engine.Reshape(ctx, vHeads, []int{batch * m.numHeads, seqLen, m.headDim})
	if err != nil {
		return nil, err
	}

	// 9. Scaled dot-product attention: [batch*numHeads, seqLen, headDim]
	attnOut, err := m.sdpa.Forward(ctx, qFlat2, kFlat2, vFlat2, nil)
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
		"num_heads":    m.numHeads,
		"head_dim":     m.headDim,
		"kv_lora_dim":  m.kvLoraDim,
	}
}

// Compile-time interface check.
var _ graph.Node[float32] = (*MultiHeadLatentAttention[float32])(nil)
