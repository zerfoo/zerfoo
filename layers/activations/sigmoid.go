package activations

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// Sigmoid implements the sigmoid activation function using composed engine
// primitives: sigmoid(x) = exp(x) / (1 + exp(x)). This avoids UnaryOp
// which is opaque to the tracing compiler.
type Sigmoid[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	lastInput   *tensor.TensorNumeric[T]
	lastOutput  *tensor.TensorNumeric[T]
	outputShape []int
}

// NewSigmoid creates a new Sigmoid activation function.
func NewSigmoid[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Sigmoid[T] {
	return &Sigmoid[T]{engine: engine, ops: ops}
}

// OpType returns the operation type.
func (s *Sigmoid[T]) OpType() string { return "Sigmoid" }

// Attributes returns empty attributes.
func (s *Sigmoid[T]) Attributes() map[string]interface{} {
	return make(map[string]interface{})
}

// OutputShape returns the output shape.
func (s *Sigmoid[T]) OutputShape() []int { return s.outputShape }

// Parameters returns nil (no trainable parameters).
func (s *Sigmoid[T]) Parameters() []*graph.Parameter[T] { return nil }

// Forward computes sigmoid(x) = exp(x) / (1 + exp(x)).
func (s *Sigmoid[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Sigmoid: %w, expected 1, got %d", graph.ErrInvalidInputCount, len(inputs))
	}

	s.lastInput = inputs[0]
	s.outputShape = inputs[0].Shape()

	// sigmoid(x) = exp(x) / (1 + exp(x))
	expX, err := s.engine.Exp(ctx, inputs[0])
	if err != nil {
		return nil, fmt.Errorf("Sigmoid: exp: %w", err)
	}
	onePlusExpX, err := s.engine.AddScalar(ctx, expX, s.ops.One())
	if err != nil {
		return nil, fmt.Errorf("Sigmoid: add scalar: %w", err)
	}
	result, err := s.engine.Div(ctx, expX, onePlusExpX)
	if err != nil {
		return nil, fmt.Errorf("Sigmoid: div: %w", err)
	}

	s.lastOutput = result
	return result, nil
}

// Backward computes sigmoid gradient: dsigmoid/dx = sigmoid(x) * (1 - sigmoid(x)).
func (s *Sigmoid[T]) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
	oneMinusSig, err := s.engine.MulScalar(ctx, s.lastOutput, s.ops.FromFloat64(-1))
	if err != nil {
		return nil, err
	}
	oneMinusSig, err = s.engine.AddScalar(ctx, oneMinusSig, s.ops.One())
	if err != nil {
		return nil, err
	}
	derivative, err := s.engine.Mul(ctx, s.lastOutput, oneMinusSig)
	if err != nil {
		return nil, err
	}
	inputGrad, err := s.engine.Mul(ctx, outputGradient, derivative)
	if err != nil {
		return nil, err
	}
	return []*tensor.TensorNumeric[T]{inputGrad}, nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*Sigmoid[float32])(nil)
