package recurrent

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// SimpleRNN is a simple recurrent neural network layer.
// It maintains a hidden state that is updated at each forward pass.
// newState = tanh(Wx*x + Wh*h + b)
type SimpleRNN[T tensor.Numeric] struct {
	engine        compute.Engine[T]
	inputWeights  *core.Linear[T]
	hiddenWeights *core.Linear[T]
	bias          *graph.Parameter[T]
	hiddenState   *tensor.TensorNumeric[T]
	lastInput     *tensor.TensorNumeric[T]
	lastSum       *tensor.TensorNumeric[T]
	outputShape   []int
}

// NewSimpleRNN creates a new SimpleRNN layer.
func NewSimpleRNN[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, hiddenDim int,
) (*SimpleRNN[T], error) {
	if name == "" {
		return nil, errors.New("layer name cannot be empty")
	}

	inputWeights, err := core.NewLinear[T](name+"_input_weights", engine, ops, inputDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create input weights: %w", err)
	}

	hiddenWeights, err := core.NewLinear[T](name+"_hidden_weights", engine, ops, hiddenDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create hidden weights: %w", err)
	}

	biasData := make([]T, hiddenDim)
	biasTensor, err := tensor.New[T]([]int{1, hiddenDim}, biasData)
	if err != nil {
		return nil, fmt.Errorf("failed to create bias tensor: %w", err)
	}
	bias, err := graph.NewParameter[T](name+"_bias", biasTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("failed to create bias parameter: %w", err)
	}

	hiddenState, _ := tensor.New[T]([]int{1, hiddenDim}, make([]T, hiddenDim))

	return &SimpleRNN[T]{
		engine:        engine,
		inputWeights:  inputWeights,
		hiddenWeights: hiddenWeights,
		bias:          bias,
		hiddenState:   hiddenState,
		outputShape:   []int{1, hiddenDim},
	}, nil
}

// OutputShape returns the output shape of the layer.
func (r *SimpleRNN[T]) OutputShape() []int {
	return r.outputShape
}

// Forward performs the forward pass.
// It takes one input: the current input to the sequence.
// The hidden state is managed internally.
func (r *SimpleRNN[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SimpleRNN: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	r.lastInput = inputs[0]

	inputTerm, err := r.inputWeights.Forward(ctx, r.lastInput)
	if err != nil {
		return nil, err
	}

	hiddenTerm, err := r.hiddenWeights.Forward(ctx, r.hiddenState)
	if err != nil {
		return nil, err
	}

	sum, err := r.engine.Add(ctx, inputTerm, hiddenTerm)
	if err != nil {
		return nil, err
	}
	sum, err = r.engine.Add(ctx, sum, r.bias.Value)
	if err != nil {
		return nil, err
	}
	r.lastSum = sum

	// Apply tanh activation
	newState, err := r.engine.Tanh(ctx, sum)
	if err != nil {
		return nil, err
	}

	r.hiddenState = newState
	r.outputShape = newState.Shape()
	return newState, nil
}

// Backward computes the gradients.
func (r *SimpleRNN[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Gradient of tanh is (1 - tanh^2(x))
	tanhGrad, err := r.engine.TanhPrime(ctx, r.lastSum, outputGradient)
	if err != nil {
		return nil, err
	}

	// Gradient for the bias is just the tanh gradient
	// Sum over the batch axis and keep dimensions to match bias shape [1, hiddenDim]
	r.bias.Gradient, err = r.engine.Sum(ctx, tanhGrad, 0, true)
	if err != nil {
		return nil, err
	}

	// Backprop through the linear layers
	dInput, err := r.inputWeights.Backward(ctx, mode, tanhGrad, r.lastInput)
	if err != nil {
		return nil, err
	}

	dHidden, err := r.hiddenWeights.Backward(ctx, mode, tanhGrad, r.hiddenState)
	if err != nil {
		return nil, err
	}

	switch mode {
	case types.OneStepApproximation:
		// For one-step approximation, we do not propagate the gradient back to the previous hidden state.
		// We can return a zero gradient for the hidden state connection.
		zeroGrad, _ := tensor.New[T](r.hiddenState.Shape(), make([]T, r.hiddenState.Size()))
		return []*tensor.TensorNumeric[T]{dInput[0], zeroGrad}, nil
	case types.FullBackprop:
		// Full BPTT is not implemented yet. The gradient for the hidden state would be passed
		// to the previous time step in a full BPTT implementation.
		return []*tensor.TensorNumeric[T]{dInput[0], dHidden[0]}, nil
	default:
		return nil, fmt.Errorf("unsupported backward mode: %v", mode)
	}
}

// Parameters returns the trainable parameters of the layer.
func (r *SimpleRNN[T]) Parameters() []*graph.Parameter[T] {
	params := r.inputWeights.Parameters()
	params = append(params, r.hiddenWeights.Parameters()...)
	params = append(params, r.bias)
	return params
}

// OpType returns the operation type.
func (r *SimpleRNN[T]) OpType() string {
	return "SimpleRNN"
}

// Attributes returns the attributes of the layer.
func (r *SimpleRNN[T]) Attributes() map[string]interface{} {
	return nil
}

// Ensure SimpleRNN implements the graph.Node interface.
var _ graph.Node[float32] = (*SimpleRNN[float32])(nil)

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*SimpleRNN[float32])(nil)
