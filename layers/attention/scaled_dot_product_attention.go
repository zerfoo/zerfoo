package attention

import (
	"context"
	"errors"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// ScaledDotProductAttention implements the scaled dot-product attention mechanism.
type ScaledDotProductAttention[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	headDim float64 // Dimension of each head, used for scaling
}

// NewScaledDotProductAttention creates a new ScaledDotProductAttention layer.
func NewScaledDotProductAttention[T tensor.Numeric](engine compute.Engine[T], headDim int) *ScaledDotProductAttention[T] {
	return &ScaledDotProductAttention[T]{
		engine:  engine,
		headDim: float64(headDim),
	}
}

// Forward computes the scaled dot-product attention.
// Q, K, V are expected to be 3D tensors (batch_size, seq_len, head_dim).
func (sdpa *ScaledDotProductAttention[T]) Forward(ctx context.Context, q, k, v *tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	// 1. MatMul Q and K^T
	// (batch, seq_len_q, head_dim) x (batch, head_dim, seq_len_k) -> (batch, seq_len_q, seq_len_k)
	kTransposed, err := sdpa.engine.Transpose(ctx, k, nil) // Transpose K
	if err != nil {
		return nil, err
	}
	attentionScores, err := sdpa.engine.MatMul(ctx, q, kTransposed, nil)
	if err != nil {
		return nil, err
	}

	// 2. Scale attention scores
	scaleFactor := sdpa.engine.Ops().FromFloat64(1.0 / math.Sqrt(sdpa.headDim))
	scaledAttentionScores, err := sdpa.engine.MulScalar(ctx, attentionScores, scaleFactor, nil)
	if err != nil {
		return nil, err
	}

	// 3. Apply Softmax
	attentionWeights, err := sdpa.engine.Softmax(ctx, scaledAttentionScores, -1, nil) // Softmax along the last dimension
	if err != nil {
		return nil, err
	}

	// 4. MatMul attention weights and V
	// (batch, seq_len_q, seq_len_k) x (batch, seq_len_k, head_dim) -> (batch, seq_len_q, head_dim)
	output, err := sdpa.engine.MatMul(ctx, attentionWeights, v, nil)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for ScaledDotProductAttention.
// dOut is the gradient from the subsequent layer.
func (sdpa *ScaledDotProductAttention[T]) Backward(_ context.Context, _ *tensor.Tensor[T], _, _, _ *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	// Placeholder for backward pass. This would involve complex chain rule applications.
	// For now, return nil gradients for Q, K, V.
	return []*tensor.Tensor[T]{nil, nil, nil}, errors.New("ScaledDotProductAttention backward pass not yet implemented")
}
