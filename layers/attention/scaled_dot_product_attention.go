package attention

import (
	"context"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// ScaledDotProductAttention implements the scaled dot-product attention mechanism.
type ScaledDotProductAttention[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	headDim float64 // Dimension of each head, used for scaling

	// Cached tensors for backward pass
	q                *tensor.TensorNumeric[T]
	k                *tensor.TensorNumeric[T]
	v                *tensor.TensorNumeric[T]
	attentionWeights *tensor.TensorNumeric[T]
}

// ScaledDotProductAttentionOptions holds configuration options for ScaledDotProductAttention.
type ScaledDotProductAttentionOptions[T tensor.Numeric] struct {
	// No specific options for now, but kept for consistency.
}

// ScaledDotProductAttentionOption applies an option to ScaledDotProductAttentionOptions.
type ScaledDotProductAttentionOption[T tensor.Numeric] func(*ScaledDotProductAttentionOptions[T])

// NewScaledDotProductAttention creates a new ScaledDotProductAttention layer.
func NewScaledDotProductAttention[T tensor.Numeric](engine compute.Engine[T], headDim int, opts ...ScaledDotProductAttentionOption[T]) *ScaledDotProductAttention[T] {
	options := &ScaledDotProductAttentionOptions[T]{}
	for _, opt := range opts {
		opt(options)
	}

	return &ScaledDotProductAttention[T]{
		engine:  engine,
		headDim: float64(headDim),
	}
}

// Forward computes the scaled dot-product attention.
// Q, K, V are expected to be 3D tensors (batch_size, seq_len, head_dim).
// mask is an optional 4D tensor (batch_size, num_heads, seq_len_q, seq_len_k).
func (sdpa *ScaledDotProductAttention[T]) Forward(ctx context.Context, q, k, v *tensor.TensorNumeric[T], mask *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Cache inputs for backward pass
	sdpa.q = q
	sdpa.k = k
	sdpa.v = v

	// 1. MatMul Q and K^T
	// (batch, seq_len_q, head_dim) x (batch, head_dim, seq_len_k) -> (batch, seq_len_q, seq_len_k)
	kTransposed, err := sdpa.engine.Transpose(ctx, k, []int{0, 2, 1}) // Transpose K for 3D tensor
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

	// 3. Apply mask
	if mask != nil {
		batchSize := q.Shape()[0]
		numHeads := mask.Shape()[1]
		seqLenQ := q.Shape()[1]
		seqLenK := k.Shape()[1]
		reshapedScores, err := sdpa.engine.Reshape(ctx, scaledAttentionScores, []int{batchSize / numHeads, numHeads, seqLenQ, seqLenK})
		if err != nil {
			return nil, err
		}
		maskedScores, err := sdpa.engine.Add(ctx, reshapedScores, mask, nil)
		if err != nil {
			return nil, err
		}
		scaledAttentionScores, err = sdpa.engine.Reshape(ctx, maskedScores, []int{batchSize, seqLenQ, seqLenK})
		if err != nil {
			return nil, err
		}
	}

	// 4. Apply Softmax
	attentionWeights, err := sdpa.engine.Softmax(ctx, scaledAttentionScores, -1, nil) // Softmax along the last dimension
	if err != nil {
		return nil, err
	}
	sdpa.attentionWeights = attentionWeights // Cache for backward pass

	// 5. MatMul attention weights and V
	// (batch, seq_len_q, seq_len_k) x (batch, seq_len_k, head_dim) -> (batch, seq_len_q, head_dim)
	output, err := sdpa.engine.MatMul(ctx, attentionWeights, v, nil)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for ScaledDotProductAttention.
// dOut is the gradient from the subsequent layer.
func (sdpa *ScaledDotProductAttention[T]) Backward(ctx context.Context, dOut *tensor.TensorNumeric[T], _, _, _ *tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// 1. Gradient w.r.t. V
	attentionWeightsTransposed, err := sdpa.engine.Transpose(ctx, sdpa.attentionWeights, []int{0, 2, 1})
	if err != nil {
		return nil, err
	}
	dV, err := sdpa.engine.MatMul(ctx, attentionWeightsTransposed, dOut, nil)
	if err != nil {
		return nil, err
	}

	// 2. Gradient w.r.t. attention_weights
	vTransposed, err := sdpa.engine.Transpose(ctx, sdpa.v, []int{0, 2, 1})
	if err != nil {
		return nil, err
	}
	dAttentionWeights, err := sdpa.engine.MatMul(ctx, dOut, vTransposed, nil)
	if err != nil {
		return nil, err
	}

	// 3. Gradient w.r.t. scaled_attention_scores (through softmax)
	// dL/dx = (dL/dy - sum(dL/dy * y)) * y
	mul, err := sdpa.engine.Mul(ctx, dAttentionWeights, sdpa.attentionWeights)
	if err != nil {
		return nil, err
	}
	sum, err := sdpa.engine.ReduceSum(ctx, mul, -1, true)
	if err != nil {
		return nil, err
	}
	sub, err := sdpa.engine.Sub(ctx, dAttentionWeights, sum)
	if err != nil {
		return nil, err
	}
	dScaledAttentionScores, err := sdpa.engine.Mul(ctx, sub, sdpa.attentionWeights)
	if err != nil {
		return nil, err
	}

	// 4. Gradient w.r.t. attention_scores (through scaling)
	scaleFactor := sdpa.engine.Ops().FromFloat64(1.0 / math.Sqrt(sdpa.headDim))
	dAttentionScores, err := sdpa.engine.MulScalar(ctx, dScaledAttentionScores, scaleFactor, nil)
	if err != nil {
		return nil, err
	}

	// 5. Gradient w.r.t. Q and K
	dQ, err := sdpa.engine.MatMul(ctx, dAttentionScores, sdpa.k, nil)
	if err != nil {
		return nil, err
	}
	dAttentionScoresTransposed, err := sdpa.engine.Transpose(ctx, dAttentionScores, []int{0, 2, 1})
	if err != nil {
		return nil, err
	}
	dK, err := sdpa.engine.MatMul(ctx, dAttentionScoresTransposed, sdpa.q, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dQ, dK, dV}, nil
}
