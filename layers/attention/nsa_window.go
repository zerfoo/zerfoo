package attention

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// NSAWindowAttention implements the sliding window attention path for
// Native Sparse Attention (NSA). It takes pre-projected Q, K, V tensors
// with shape [batch, heads, seq, dim] and applies causal sliding window
// attention, restricting each query to attend only to keys within a
// fixed-size window.
type NSAWindowAttention[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	windowSize int
	numHeads   int
	numKVHeads int
	headDim    int
}

// NewNSAWindowAttention creates a new NSAWindowAttention layer.
//
// Parameters:
//   - engine: compute engine for tensor operations
//   - ops: arithmetic operations for the numeric type
//   - windowSize: number of past tokens each query can attend to (the window
//     spans from max(0, q-windowSize) to q inclusive)
//   - numHeads: number of query attention heads
//   - numKVHeads: number of key/value attention heads
//   - headDim: dimension of each attention head
func NewNSAWindowAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	windowSize, numHeads, numKVHeads, headDim int,
) (*NSAWindowAttention[T], error) {
	if windowSize <= 0 {
		return nil, fmt.Errorf("NSAWindowAttention: windowSize must be > 0, got %d", windowSize)
	}
	if numHeads <= 0 {
		return nil, fmt.Errorf("NSAWindowAttention: numHeads must be > 0, got %d", numHeads)
	}
	if numKVHeads <= 0 {
		return nil, fmt.Errorf("NSAWindowAttention: numKVHeads must be > 0, got %d", numKVHeads)
	}
	if numHeads%numKVHeads != 0 {
		return nil, fmt.Errorf("NSAWindowAttention: numHeads (%d) must be divisible by numKVHeads (%d)", numHeads, numKVHeads)
	}
	if headDim <= 0 {
		return nil, fmt.Errorf("NSAWindowAttention: headDim must be > 0, got %d", headDim)
	}
	return &NSAWindowAttention[T]{
		engine:     engine,
		ops:        ops,
		windowSize: windowSize,
		numHeads:   numHeads,
		numKVHeads: numKVHeads,
		headDim:    headDim,
	}, nil
}

// Forward computes sliding window attention over pre-projected Q, K, V.
//
// Input shapes:
//   - Q: [batch, numHeads, seqQ, headDim]
//   - K: [batch, numKVHeads, seqKV, headDim]
//   - V: [batch, numKVHeads, seqKV, headDim]
//
// Output shape: [batch, numHeads, seqQ, headDim]
//
// For each query position q, attention is restricted to keys in the range
// [max(0, q - windowSize + 1), q] (causal sliding window). Positions outside
// this window receive -1e9 additive masking before softmax.
func (nw *NSAWindowAttention[T]) Forward(ctx context.Context, Q, K, V *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	qShape := Q.Shape()
	kShape := K.Shape()
	if len(qShape) != 4 || len(kShape) != 4 {
		return nil, fmt.Errorf("NSAWindowAttention: expected 4D tensors, got Q %v, K %v", qShape, kShape)
	}

	batch := qShape[0]
	seqQ := qShape[2]
	seqKV := kShape[2]

	// Expand KV heads to match query heads via repetition if needed.
	groupSize := nw.numHeads / nw.numKVHeads
	if groupSize > 1 {
		var err error
		K, err = nw.expandKVHeads(ctx, K, groupSize)
		if err != nil {
			return nil, fmt.Errorf("NSAWindowAttention: expanding K heads: %w", err)
		}
		V, err = nw.expandKVHeads(ctx, V, groupSize)
		if err != nil {
			return nil, fmt.Errorf("NSAWindowAttention: expanding V heads: %w", err)
		}
	}

	// Reshape from [batch, heads, seq, dim] to [batch*heads, seq, dim] for SDPA.
	q3d, err := nw.engine.Reshape(ctx, Q, []int{batch * nw.numHeads, seqQ, nw.headDim})
	if err != nil {
		return nil, err
	}
	k3d, err := nw.engine.Reshape(ctx, K, []int{batch * nw.numHeads, seqKV, nw.headDim})
	if err != nil {
		return nil, err
	}
	v3d, err := nw.engine.Reshape(ctx, V, []int{batch * nw.numHeads, seqKV, nw.headDim})
	if err != nil {
		return nil, err
	}

	// Build the causal sliding window mask [1, 1, seqQ, seqKV].
	mask, err := nw.buildWindowMask(seqQ, seqKV)
	if err != nil {
		return nil, err
	}

	// Use ScaledDotProductAttention for the core computation.
	sdpa := NewScaledDotProductAttention[T](nw.engine, nw.headDim)
	out3d, err := sdpa.Forward(ctx, q3d, k3d, v3d, mask)
	if err != nil {
		return nil, err
	}

	// Reshape back to [batch, heads, seqQ, headDim].
	return nw.engine.Reshape(ctx, out3d, []int{batch, nw.numHeads, seqQ, nw.headDim})
}

// buildWindowMask creates a causal sliding window mask of shape [1, 1, seqQ, seqKV].
// Positions within the window get 0; positions outside get -1e9.
func (nw *NSAWindowAttention[T]) buildWindowMask(seqQ, seqKV int) (*tensor.TensorNumeric[T], error) {
	largeNeg := nw.ops.FromFloat64(-1e9)
	zero := nw.ops.FromFloat64(0)
	data := make([]T, seqQ*seqKV)

	// offset handles the case where seqKV > seqQ (e.g. KV cache with
	// prefill tokens). Query position i corresponds to absolute position
	// i + offset in the KV sequence.
	offset := seqKV - seqQ

	for i := range seqQ {
		absPos := i + offset
		// Causal: can only attend to positions <= absPos.
		// Window: can only attend to positions >= absPos - windowSize + 1.
		windowStart := absPos - nw.windowSize + 1
		if windowStart < 0 {
			windowStart = 0
		}
		for j := range seqKV {
			if j >= windowStart && j <= absPos {
				data[i*seqKV+j] = zero
			} else {
				data[i*seqKV+j] = largeNeg
			}
		}
	}

	return tensor.New[T]([]int{1, 1, seqQ, seqKV}, data)
}

// expandKVHeads repeats KV heads to match query head count.
// Input: [batch, numKVHeads, seq, dim] -> Output: [batch, numKVHeads*groupSize, seq, dim]
func (nw *NSAWindowAttention[T]) expandKVHeads(ctx context.Context, t *tensor.TensorNumeric[T], groupSize int) (*tensor.TensorNumeric[T], error) {
	shape := t.Shape()
	batch, kvHeads, seq, dim := shape[0], shape[1], shape[2], shape[3]

	srcData := t.Data()
	expanded := make([]T, batch*kvHeads*groupSize*seq*dim)

	for b := range batch {
		for h := range kvHeads {
			srcOffset := ((b*kvHeads + h) * seq) * dim
			for g := range groupSize {
				dstHead := h*groupSize + g
				dstOffset := ((b*kvHeads*groupSize + dstHead) * seq) * dim
				copy(expanded[dstOffset:dstOffset+seq*dim], srcData[srcOffset:srcOffset+seq*dim])
			}
		}
	}

	return tensor.New[T]([]int{batch, kvHeads * groupSize, seq, dim}, expanded)
}

// WindowSize returns the configured window size.
func (nw *NSAWindowAttention[T]) WindowSize() int {
	return nw.windowSize
}

// Scale returns the attention scaling factor (1/sqrt(headDim)).
func (nw *NSAWindowAttention[T]) Scale() float64 {
	return 1.0 / math.Sqrt(float64(nw.headDim))
}
