package functional

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// MultiHeadAttentionBackward computes gradients for multi-head scaled dot-product attention.
// dOutput: gradient from upstream [seq_len, d_model]
// q, k, v: original inputs [seq_len, d_model]
// nHeads: number of attention heads
// Returns: dQ, dK, dV [seq_len, d_model]
func MultiHeadAttentionBackward[T tensor.Float](ctx context.Context, engine compute.Engine[T], ops numeric.Arithmetic[T],
	dOutput, q, k, v *tensor.TensorNumeric[T], nHeads int) (dQ, dK, dV *tensor.TensorNumeric[T], err error) {

	if dOutput == nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: dOutput is nil")
	}
	if q == nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: q is nil")
	}
	if k == nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: k is nil")
	}
	if v == nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: v is nil")
	}

	qShape := q.Shape()
	if len(qShape) != 2 {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: q must be 2D [seq_len, d_model], got %v", qShape)
	}
	seqLen, dModel := qShape[0], qShape[1]
	if dModel%nHeads != 0 {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: d_model (%d) must be divisible by nHeads (%d)", dModel, nHeads)
	}
	dHead := dModel / nHeads

	// Split inputs into per-head tensors: [seq_len, d_model] -> nHeads * [seq_len, d_head]
	qHeads, err := splitHeads(ctx, engine, q, seqLen, nHeads, dHead)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: split q: %w", err)
	}
	kHeads, err := splitHeads(ctx, engine, k, seqLen, nHeads, dHead)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: split k: %w", err)
	}
	vHeads, err := splitHeads(ctx, engine, v, seqLen, nHeads, dHead)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: split v: %w", err)
	}

	// Split dOutput into per-head gradients
	dOutHeads, err := splitHeads(ctx, engine, dOutput, seqLen, nHeads, dHead)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: split dOutput: %w", err)
	}

	scale := T(1.0 / math.Sqrt(float64(dHead)))

	dQHeads := make([]*tensor.TensorNumeric[T], nHeads)
	dKHeads := make([]*tensor.TensorNumeric[T], nHeads)
	dVHeads := make([]*tensor.TensorNumeric[T], nHeads)

	for h := range nHeads {
		// Recompute forward attention weights for this head:
		// scores_h = Q_h @ K_h^T * scale -> softmax
		khT, err := engine.Transpose(ctx, kHeads[h], []int{1, 0})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: transpose k head %d: %w", h, err)
		}
		scores, err := engine.MatMul(ctx, qHeads[h], khT)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: scores head %d: %w", h, err)
		}
		scores, err = engine.MulScalar(ctx, scores, scale)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: scale scores head %d: %w", h, err)
		}
		attnWeights, err := engine.Softmax(ctx, scores, -1)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: softmax head %d: %w", h, err)
		}

		// Backward: output_h = attnWeights_h @ V_h
		// dV_h = attnWeights_h^T @ dOutput_h  [seq_len, d_head]
		attnWeightsT, err := engine.Transpose(ctx, attnWeights, []int{1, 0})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: transpose attnWeights head %d: %w", h, err)
		}
		dVHeads[h], err = engine.MatMul(ctx, attnWeightsT, dOutHeads[h])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: dV head %d: %w", h, err)
		}

		// dAttnWeights_h = dOutput_h @ V_h^T  [seq_len, seq_len]
		vhT, err := engine.Transpose(ctx, vHeads[h], []int{1, 0})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: transpose v head %d: %w", h, err)
		}
		dAttnWeights, err := engine.MatMul(ctx, dOutHeads[h], vhT)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: dAttnWeights head %d: %w", h, err)
		}

		// dScores_h = SoftmaxBackward(dAttnWeights_h, attnWeights_h)  [seq_len, seq_len]
		dScores, err := SoftmaxBackward(ctx, engine, ops, dAttnWeights, attnWeights)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: softmax backward head %d: %w", h, err)
		}

		// dQ_h = scale * dScores_h @ K_h  [seq_len, d_head]
		dQh, err := engine.MatMul(ctx, dScores, kHeads[h])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: dQ matmul head %d: %w", h, err)
		}
		dQHeads[h], err = engine.MulScalar(ctx, dQh, scale)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: dQ scale head %d: %w", h, err)
		}

		// dK_h = scale * dScores_h^T @ Q_h  [seq_len, d_head]
		dScoresT, err := engine.Transpose(ctx, dScores, []int{1, 0})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: transpose dScores head %d: %w", h, err)
		}
		dKh, err := engine.MatMul(ctx, dScoresT, qHeads[h])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: dK matmul head %d: %w", h, err)
		}
		dKHeads[h], err = engine.MulScalar(ctx, dKh, scale)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: dK scale head %d: %w", h, err)
		}
	}

	// Concatenate per-head gradients back to [seq_len, d_model]
	dQ, err = engine.Concat(ctx, dQHeads, 1)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: concat dQ: %w", err)
	}
	dK, err = engine.Concat(ctx, dKHeads, 1)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: concat dK: %w", err)
	}
	dV, err = engine.Concat(ctx, dVHeads, 1)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("functional.MultiHeadAttentionBackward: concat dV: %w", err)
	}

	return dQ, dK, dV, nil
}
