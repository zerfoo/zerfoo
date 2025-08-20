package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// LocalSlidingWindowAttention implements a local sliding window attention mechanism.
type LocalSlidingWindowAttention[T tensor.Numeric] struct {
	gqa         *GroupedQueryAttention[T]
	windowSize  int
	outputShape []int
}

// NewLocalSlidingWindowAttention creates a new LocalSlidingWindowAttention layer.
func NewLocalSlidingWindowAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads, windowSize int,
	epsilon T,
	base float64,
	maxSeqLen int,
) (*LocalSlidingWindowAttention[T], error) {
	gqa, err := NewGroupedQueryAttention[T](engine, ops, modelDim, numQueryHeads, numKeyValueHeads, base, maxSeqLen)
	if err != nil {
		return nil, fmt.Errorf("failed to create GroupedQueryAttention: %w", err)
	}
	return &LocalSlidingWindowAttention[T]{
		gqa:        gqa,
		windowSize: windowSize,
	}, nil
}

// OutputShape returns the output shape of the LocalSlidingWindowAttention.
func (lswa *LocalSlidingWindowAttention[T]) OutputShape() []int {
	return lswa.outputShape
}

// Parameters returns the parameters of the LocalSlidingWindowAttention layer.
func (lswa *LocalSlidingWindowAttention[T]) Parameters() []*graph.Parameter[T] {
	return lswa.gqa.Parameters()
}

// Forward computes the local sliding window attention.
func (lswa *LocalSlidingWindowAttention[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	input := inputs[0]
	lswa.outputShape = input.Shape()
	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	numHeads := lswa.gqa.numQueryHeads

	// Create sliding window mask
	mask, err := tensor.New[T]([]int{batchSize, numHeads, seqLen, seqLen}, nil)
	if err != nil {
		return nil, err
	}
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					if j > i || j < i-lswa.windowSize {
						if err := mask.Set(lswa.gqa.ops.FromFloat64(-1e9), b, h, i, j); err != nil {
							return nil, err
						}
					}
				}
			}
		}
	}

	return lswa.gqa.Forward(ctx, input, mask)
}

// Backward computes the gradients for LocalSlidingWindowAttention.
func (lswa *LocalSlidingWindowAttention[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	// The backward pass is the same as GQA for now.
	return lswa.gqa.Backward(ctx, dOut, inputs...)
}
