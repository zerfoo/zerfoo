package functional

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// Linear computes y = x @ weight^T + bias. If bias is nil, computes y = x @ weight^T.
// x: [*, in_features], weight: [out_features, in_features], bias: [out_features]
func Linear[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	x, weight *tensor.TensorNumeric[T], bias *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {

	weightT, err := engine.Transpose(ctx, weight, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("functional.Linear: transpose weight: %w", err)
	}

	result, err := engine.MatMul(ctx, x, weightT)
	if err != nil {
		return nil, fmt.Errorf("functional.Linear: matmul: %w", err)
	}

	if bias != nil {
		result, err = engine.Add(ctx, result, bias)
		if err != nil {
			return nil, fmt.Errorf("functional.Linear: add bias: %w", err)
		}
	}

	return result, nil
}

// MultiHeadAttention computes multi-head scaled dot-product attention.
// q, k, v: [seq_len, d_model]. nHeads: number of attention heads.
// Returns: [seq_len, d_model]
func MultiHeadAttention[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	q, k, v *tensor.TensorNumeric[T], nHeads int) (*tensor.TensorNumeric[T], error) {

	qShape := q.Shape()
	if len(qShape) != 2 {
		return nil, fmt.Errorf("functional.MultiHeadAttention: q must be 2D [seq_len, d_model], got %v", qShape)
	}
	seqLen, dModel := qShape[0], qShape[1]
	if dModel%nHeads != 0 {
		return nil, fmt.Errorf("functional.MultiHeadAttention: d_model (%d) must be divisible by nHeads (%d)", dModel, nHeads)
	}
	dHead := dModel / nHeads

	// Split into heads: [seq_len, d_model] -> [nHeads, seq_len, d_head]
	qHeads, err := splitHeads(ctx, engine, q, seqLen, nHeads, dHead)
	if err != nil {
		return nil, fmt.Errorf("functional.MultiHeadAttention: split q: %w", err)
	}
	kHeads, err := splitHeads(ctx, engine, k, seqLen, nHeads, dHead)
	if err != nil {
		return nil, fmt.Errorf("functional.MultiHeadAttention: split k: %w", err)
	}
	vHeads, err := splitHeads(ctx, engine, v, seqLen, nHeads, dHead)
	if err != nil {
		return nil, fmt.Errorf("functional.MultiHeadAttention: split v: %w", err)
	}

	scale := T(1.0 / math.Sqrt(float64(dHead)))
	headOutputs := make([]*tensor.TensorNumeric[T], nHeads)

	for h := range nHeads {
		// scores = qh @ kh^T, shape [seq_len, seq_len]
		khT, err := engine.Transpose(ctx, kHeads[h], []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("functional.MultiHeadAttention: transpose k head %d: %w", h, err)
		}
		scores, err := engine.MatMul(ctx, qHeads[h], khT)
		if err != nil {
			return nil, fmt.Errorf("functional.MultiHeadAttention: matmul scores head %d: %w", h, err)
		}

		// Scale by 1/sqrt(d_head)
		scores, err = engine.MulScalar(ctx, scores, scale)
		if err != nil {
			return nil, fmt.Errorf("functional.MultiHeadAttention: scale head %d: %w", h, err)
		}

		// Softmax along last axis
		attnWeights, err := engine.Softmax(ctx, scores, -1)
		if err != nil {
			return nil, fmt.Errorf("functional.MultiHeadAttention: softmax head %d: %w", h, err)
		}

		// output = attn_weights @ v, shape [seq_len, d_head]
		headOut, err := engine.MatMul(ctx, attnWeights, vHeads[h])
		if err != nil {
			return nil, fmt.Errorf("functional.MultiHeadAttention: matmul output head %d: %w", h, err)
		}
		headOutputs[h] = headOut
	}

	// Concatenate heads along last axis: nHeads * [seq_len, d_head] -> [seq_len, d_model]
	result, err := engine.Concat(ctx, headOutputs, 1)
	if err != nil {
		return nil, fmt.Errorf("functional.MultiHeadAttention: concat heads: %w", err)
	}

	return result, nil
}

// splitHeads reshapes [seq_len, d_model] into nHeads separate [seq_len, d_head] tensors.
func splitHeads[T tensor.Numeric](ctx context.Context, engine compute.Engine[T],
	x *tensor.TensorNumeric[T], seqLen, nHeads, dHead int) ([]*tensor.TensorNumeric[T], error) {

	// Reshape to [seq_len, nHeads, d_head]
	reshaped, err := engine.Reshape(ctx, x, []int{seqLen, nHeads, dHead})
	if err != nil {
		return nil, err
	}

	// Transpose to [nHeads, seq_len, d_head]
	transposed, err := engine.Transpose(ctx, reshaped, []int{1, 0, 2})
	if err != nil {
		return nil, err
	}

	// Split along axis 0 into nHeads tensors of [seq_len, d_head]
	// After split each piece is [1, seq_len, d_head], reshape to [seq_len, d_head]
	pieces, err := engine.Split(ctx, transposed, nHeads, 0)
	if err != nil {
		return nil, err
	}

	heads := make([]*tensor.TensorNumeric[T], nHeads)
	for i, p := range pieces {
		heads[i], err = engine.Reshape(ctx, p, []int{seqLen, dHead})
		if err != nil {
			return nil, err
		}
	}
	return heads, nil
}
