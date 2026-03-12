package attention

import (
	"context"
	"fmt"
	"log"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// negInfValue returns a large negative value (-1e9) for floating point types.
func negInfValue[T tensor.Numeric]() T {
	var zero T
	if p, ok := any(&zero).(*float32); ok {
		*p = -1e9
		return zero
	}
	if p, ok := any(&zero).(*float64); ok {
		*p = -1e9
		return zero
	}
	return zero
}

// ScaledDotProductAttention implements the scaled dot-product attention mechanism.
type ScaledDotProductAttention[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	headDim float64 // Dimension of each head, used for scaling
	causal  bool    // if true, apply causal masking to attention scores

	// Cached tensors for backward pass
	q                *tensor.TensorNumeric[T]
	k                *tensor.TensorNumeric[T]
	v                *tensor.TensorNumeric[T]
	attentionWeights *tensor.TensorNumeric[T]
}

// SetCausal enables or disables causal (lower-triangular) masking.
func (sdpa *ScaledDotProductAttention[T]) SetCausal(causal bool) {
	sdpa.causal = causal
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
func (sdpa *ScaledDotProductAttention[T]) Forward(ctx context.Context, q, k, v, mask *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Cache inputs for backward pass
	sdpa.q = q
	sdpa.k = k
	sdpa.v = v

	// Try fused flash attention when no arbitrary mask is provided.
	// Flash attention handles causal masking internally via the causal flag.
	if mask == nil {
		if result, err := tryFlashForward(q, k, v, int(sdpa.headDim), sdpa.causal); result != nil || err != nil {
			return result, err
		}
	}

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
	// Compute head dimension robustly to avoid division by zero
	d := sdpa.headDim
	if d <= 0 {
		// Fallback to deriving from Q's last dimension
		if q == nil || len(q.Shape()) < 3 {
			return nil, fmt.Errorf("ScaledDotProductAttention: invalid Q shape %v to infer head dimension", q.Shape())
		}
		d = float64(q.Shape()[2])
	}
	if d <= 0 {
		return nil, fmt.Errorf("ScaledDotProductAttention: headDim must be > 0, got %v", d)
	}
	scaleFactor := sdpa.engine.Ops().FromFloat64(1.0 / math.Sqrt(d))

	scaledAttentionScores, err := sdpa.engine.MulScalar(ctx, attentionScores, scaleFactor, nil)
	if err != nil {
		return nil, err
	}

	// 3. Apply mask (explicit 4D mask or causal)
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
	} else if sdpa.causal {
		// Apply causal masking directly to 3D scores (batch, seqQ, seqK).
		// Set positions where q_pos < k_pos to -inf.
		data := scaledAttentionScores.Data()
		shape := scaledAttentionScores.Shape()
		batch, seqQ, seqK := shape[0], shape[1], shape[2]
		offset := seqK - seqQ // cached tokens are always visible
		negInf := negInfValue[T]()
		for b := range batch {
			for qi := range seqQ {
				for ki := range seqK {
					if ki > qi+offset {
						data[(b*seqQ+qi)*seqK+ki] = negInf
					}
				}
			}
		}
	}

	// Debug: check attention scores before softmax and print problematic rows
	if scoreData, ok := any(scaledAttentionScores.Data()).([]float32); ok {
		nanCount := 0
		infCount := 0
		for _, v := range scoreData {
			if math.IsNaN(float64(v)) {
				nanCount++
			}
			if math.IsInf(float64(v), 0) {
				infCount++
			}
		}
		shape := scaledAttentionScores.Shape()
		seqK := shape[len(shape)-1]
		seqQ := shape[len(shape)-2]
		// Always print first few rows of first SDPA call for debugging
		if shape[0] == 4 && seqQ == 8 {
			for b := 0; b < 1; b++ {
				for qi := 0; qi < 4; qi++ {
					row := scoreData[b*seqQ*seqK+qi*seqK : b*seqQ*seqK+(qi+1)*seqK]
					log.Printf("[SDPA-DBG] pre-softmax batch=%d qi=%d: %v", b, qi, row)
				}
			}
		}
		if nanCount > 0 || infCount > 0 {
			log.Printf("[SDPA-NaN] pre-softmax scores: NaN=%d Inf=%d total=%d shape=%v", nanCount, infCount, len(scoreData), scaledAttentionScores.Shape())
		}
	}

	// 4. Apply Softmax
	attentionWeights, err := sdpa.engine.Softmax(ctx, scaledAttentionScores, -1, nil) // Softmax along the last dimension
	if err != nil {
		return nil, err
	}

	// Debug: check attention weights after softmax
	if wData, ok := any(attentionWeights.Data()).([]float32); ok {
		nanCount := 0
		for _, v := range wData {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				nanCount++
			}
		}
		if nanCount > 0 {
			log.Printf("[SDPA-NaN] post-softmax weights: NaN=%d/%d shape=%v", nanCount, len(wData), attentionWeights.Shape())
			// Print first few rows to understand the pattern
			shape := attentionWeights.Shape()
			seqK := shape[len(shape)-1]
			for b := 0; b < shape[0] && b < 2; b++ {
				for qi := 0; qi < shape[1] && qi < 8; qi++ {
					row := wData[b*shape[1]*seqK+qi*seqK : b*shape[1]*seqK+(qi+1)*seqK]
					hasNaN := false
					for _, v := range row {
						if math.IsNaN(float64(v)) {
							hasNaN = true
							break
						}
					}
					if hasNaN {
						log.Printf("[SDPA-NaN]   batch=%d qi=%d: %v", b, qi, row)
					}
				}
			}
		}
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
func (sdpa *ScaledDotProductAttention[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut, _, _, _ *tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
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
	// Use the same robust head dimension computation as in Forward
	d := sdpa.headDim
	if d <= 0 {
		if sdpa.q == nil || len(sdpa.q.Shape()) < 3 {
			return nil, fmt.Errorf("ScaledDotProductAttention: cannot infer headDim in Backward; cached Q is invalid")
		}
		d = float64(sdpa.q.Shape()[2])
	}
	scaleFactor := sdpa.engine.Ops().FromFloat64(1.0 / math.Sqrt(d))

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
