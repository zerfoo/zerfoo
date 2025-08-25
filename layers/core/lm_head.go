package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// LMHead is a linear layer that maps hidden states to vocabulary logits.
type LMHead[T tensor.Numeric] struct {
	linear *Linear[T]
	engine compute.Engine[T]
}

// NewLMHead creates a new LMHead.
func NewLMHead[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], hiddenDim, vocabSize int) (*LMHead[T], error) {
	linear, err := NewLinear[T]("lm_head", engine, ops, hiddenDim, vocabSize)
	if err != nil {
		return nil, err
	}

	return &LMHead[T]{linear: linear, engine: engine}, nil
}

// SetWeights sets the weights of the LMHead. This is useful for sharing weights
// with a token embedding layer.
func (h *LMHead[T]) SetWeights(weights *tensor.TensorNumeric[T]) {
	h.linear.weights.Value = weights
}

// Forward computes the forward pass of the LMHead.
func (h *LMHead[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	originalShape := input.Shape()
	batchSize := originalShape[0]
	seqLen := originalShape[1]
	hiddenDim := originalShape[2]

	reshapedInput, err := h.engine.Reshape(ctx, input, []int{batchSize * seqLen, hiddenDim})
	if err != nil {
		return nil, err
	}

	output, err := h.linear.Forward(ctx, reshapedInput)
	if err != nil {
		return nil, err
	}

	vocabSize := h.linear.OutputShape()[1]

	output, err = h.engine.Reshape(ctx, output, []int{batchSize, seqLen, vocabSize})
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the backward pass of the LMHead.
func (h *LMHead[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return h.linear.Backward(ctx, mode, dOut, inputs...)
}

// Parameters returns the parameters of the LMHead.
func (h *LMHead[T]) Parameters() []*graph.Parameter[T] {
	return h.linear.Parameters()
}

// OutputShape returns the output shape of the LMHead.
func (h *LMHead[T]) OutputShape() []int {
	return h.linear.OutputShape()
}
