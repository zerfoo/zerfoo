// Package activations provides neural network activation functions.
package activations

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// SwiGLU implements the SwiGLU activation function.
type SwiGLU[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	sigmoid *Sigmoid[T] // SwiGLU uses Sigmoid internally

	// Cached tensors for backward pass
	lastInput *tensor.Tensor[T]
	gate      *tensor.Tensor[T] // The sigmoid(x2) part
}

// NewSwiGLU creates a new SwiGLU activation layer.
func NewSwiGLU[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *SwiGLU[T] {
	return &SwiGLU[T]{
		engine:  engine,
		ops:     ops,
		sigmoid: NewSigmoid[T](engine, ops),
	}
}

// OutputShape returns the output shape of SwiGLU.
// Input shape is (..., 2 * feature_dim). Output shape is (..., feature_dim).
func (s *SwiGLU[T]) OutputShape(inputShapes ...[]int) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("SwiGLU: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputShapes))
	}
	inputShape := inputShapes[0]
	if len(inputShape) < 1 {
		return nil, errors.New("SwiGLU input must have at least one dimension")
	}
	lastDim := inputShape[len(inputShape)-1]
	if lastDim%2 != 0 {
		return nil, fmt.Errorf("last dimension of input (%d) must be even for SwiGLU", lastDim)
	}
	outputShape := make([]int, len(inputShape))
	copy(outputShape, inputShape)
	outputShape[len(outputShape)-1] = lastDim / 2

	return outputShape, nil
}

// Parameters returns an empty slice as SwiGLU has no trainable parameters.
func (s *SwiGLU[T]) Parameters() []graph.Parameter[T] {
	return nil
}

// Forward computes the SwiGLU activation.
// Input: A tensor with its last dimension being 2 * feature_dim.
func (s *SwiGLU[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SwiGLU: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := inputs[0]
	s.lastInput = input // Cache input for backward

	inputShape := s.lastInput.Shape()
	// featureDim := lastDim / 2

	// Split input into x1 and x2 along the last dimension
	splitTensors, err := s.engine.Split(ctx, s.lastInput, 2, len(inputShape)-1)
	if err != nil {
		return nil, err
	}
	x1 := splitTensors[0]
	x2 := splitTensors[1]

	// Compute gate = sigmoid(x2)
	gate, err := s.sigmoid.Forward(ctx, x2)
	if err != nil {
		return nil, err
	}
	s.gate = gate // Cache gate for backward

	// Compute output = x1 * gate
	output, err := s.engine.Mul(ctx, x1, gate, nil)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for SwiGLU.
func (s *SwiGLU[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("invalid input count: %w", graph.ErrInvalidInputCount)
	}
	// dOut shape: (..., feature_dim)
	// s.lastInput shape: (..., 2 * feature_dim)

	inputShape := s.lastInput.Shape()

	// Re-split original input into x1 and x2
	splitTensors, err := s.engine.Split(ctx, s.lastInput, 2, len(inputShape)-1)
	if err != nil {
		return nil, err
	}
	x1 := splitTensors[0]
	_ = splitTensors[1]

	// dL/dx1 = dOut * gate
	dLdx1, err := s.engine.Mul(ctx, dOut, s.gate, nil)
	if err != nil {
		return nil, err
	}

	// dL/dgate = dOut * x1
	dLdgate, err := s.engine.Mul(ctx, dOut, x1, nil)
	if err != nil {
		return nil, err
	}

	// dL/dx2 = dL/dgate * dgate/dx2
	// is the derivative of sigmoid(x2), which is sigmoid(x2) * (1 - sigmoid(x2))
	// We already have gate = sigmoid(x2)
	oneMinusGate, err := s.engine.UnaryOp(ctx, s.gate, func(val T) T { return s.ops.Sub(s.ops.One(), val) })
	if err != nil {
		return nil, err
	}
	sigmoidGrad, err := s.engine.Mul(ctx, s.gate, oneMinusGate, nil)
	if err != nil {
		return nil, err
	}
	dLdx2, err := s.engine.Mul(ctx, dLdgate, sigmoidGrad, nil)
	if err != nil {
		return nil, err
	}

	// Concatenate dL/dx1 and dL/dx2
	dInput, err := s.engine.Concat(ctx, []*tensor.Tensor[T]{dLdx1, dLdx2}, len(inputShape)-1, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor[T]{dInput}, nil
}
