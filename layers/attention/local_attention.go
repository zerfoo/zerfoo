package attention

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// LocalAttention implements a local, sliding-window self-attention mechanism.
type LocalAttention[T tensor.Numeric] struct {
	gqa        *GroupedQueryAttention[T]
	windowSize int
	ops        numeric.Arithmetic[T]
}

// LocalAttentionOptions holds configuration options for the LocalAttention layer.
type LocalAttentionOptions[T tensor.Numeric] struct {
	Base      float64
	MaxSeqLen int
}

// LocalAttentionOption is a function that applies an option to LocalAttentionOptions.
type LocalAttentionOption[T tensor.Numeric] func(*LocalAttentionOptions[T])

// WithLocalRopeBase sets the base for Rotary Positional Embeddings.
//
// base: The base for Rotary Positional Embeddings.
func WithLocalRopeBase[T tensor.Numeric](base float64) LocalAttentionOption[T] {
	return func(o *LocalAttentionOptions[T]) {
		o.Base = base
	}
}

// WithLocalMaxSeqLen sets the maximum sequence length for Rotary Positional Embeddings.
//
// maxSeqLen: The maximum sequence length for Rotary Positional Embeddings.
func WithLocalMaxSeqLen[T tensor.Numeric](maxSeqLen int) LocalAttentionOption[T] {
	return func(o *LocalAttentionOptions[T]) {
		o.MaxSeqLen = maxSeqLen
	}
}

// NewLocalAttention creates a new LocalAttention layer.
func NewLocalAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads, windowSize int,
	opts ...LocalAttentionOption[T],
) (*LocalAttention[T], error) {
	// Default options
	options := &LocalAttentionOptions[T]{
		Base:      10000.0,
		MaxSeqLen: 2048,
	}
	for _, opt := range opts {
		opt(options)
	}

	gqa, err := NewGroupedQueryAttention[T](
		engine, ops, modelDim, numQueryHeads, numKeyValueHeads,
		WithRopeBase[T](options.Base),
		WithMaxSeqLen[T](options.MaxSeqLen),
	)
	if err != nil {
		return nil, err
	}

	return &LocalAttention[T]{
		gqa:        gqa,
		windowSize: windowSize,
		ops:        ops,
	}, nil
}

// Parameters returns the parameters of the LocalAttention layer.
func (la *LocalAttention[T]) Parameters() []*graph.Parameter[T] {
	return la.gqa.Parameters()
}

// Forward computes the forward pass of the LocalAttention layer.
func (la *LocalAttention[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	seqLen := input.Shape()[1]

	mask, err := la.createLocalAttentionMask(seqLen)
	if err != nil {
		return nil, err
	}

	return la.gqa.Forward(ctx, input, mask)
}

func (la *LocalAttention[T]) createLocalAttentionMask(seqLen int) (*tensor.TensorNumeric[T], error) {
	mask, err := tensor.New[T]([]int{1, 1, seqLen, seqLen}, nil)
	if err != nil {
		return nil, err
	}
	// Fill with a large negative number
	largeNegative := la.ops.FromFloat64(-1e9)
	for i := 0; i < seqLen*seqLen; i++ { //nolint:intrange // Keep classic index loop for clarity and performance
		mask.Data()[i] = largeNegative
	}

	for i := 0; i < seqLen; i++ { //nolint:intrange // Keep classic index loop for clarity and performance
		start := i - la.windowSize
		if start < 0 {
			start = 0
		}

		end := i + la.windowSize + 1
		if end > seqLen {
			end = seqLen
		}

		for j := start; j < end; j++ {
			mask.Data()[i*seqLen+j] = la.ops.FromFloat64(0)
		}
	}

	return mask, nil
}

// Backward delegates the backward pass to the wrapped GroupedQueryAttention.
func (la *LocalAttention[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return la.gqa.Backward(ctx, mode, dOut, inputs...)
}

// OutputShape returns the output shape of the LocalAttention layer.
func (la *LocalAttention[T]) OutputShape() []int {
	return la.gqa.OutputShape()
}

// Attributes returns the attributes of the LocalAttention layer.
func (la *LocalAttention[T]) Attributes() map[string]interface{} {
	attrs := la.gqa.Attributes()
	attrs["window_size"] = la.windowSize
	return attrs
}

// OpType returns the operation type of the LocalAttention layer.
func (la *LocalAttention[T]) OpType() string {
	return "LocalAttention"
}
