package attention

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
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

// WithRopeBase sets the base for Rotary Positional Embeddings.
func WithLocalRopeBase[T tensor.Numeric](base float64) LocalAttentionOption[T] {
	return func(o *LocalAttentionOptions[T]) {
		o.Base = base
	}
}

// WithMaxSeqLen sets the maximum sequence length for Rotary Positional Embeddings.
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
	for i := 0; i < seqLen*seqLen; i++ {
		mask.Data()[i] = largeNegative
	}

	for i := 0; i < seqLen; i++ {
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

// Backward is not implemented
func (la *LocalAttention[T]) Backward(ctx context.Context, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return la.gqa.Backward(ctx, dOut, inputs...)
}

func (la *LocalAttention[T]) OutputShape() []int {
	return la.gqa.OutputShape()
}