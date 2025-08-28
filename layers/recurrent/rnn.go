package recurrent

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// SimpleRNN is a simple recurrent neural network layer.
type SimpleRNN[T tensor.Numeric] struct {
	name          string
	engine        compute.Engine[T]
	ops           numeric.Arithmetic[T]
	inputWeights  *core.Linear[T]
	hiddenWeights *core.Linear[T]
	bias          *core.Bias[T]
	activation    graph.Node[T]
	hiddenState   *tensor.TensorNumeric[T]
	lastInput     *tensor.TensorNumeric[T]
	inputDim      int
	hiddenDim     int
}

// NewSimpleRNN creates a new SimpleRNN layer.
func NewSimpleRNN[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, hiddenDim int,
) (*SimpleRNN[T], error) {
	inputWeights, err := core.NewLinear[T](name+"_input_weights", engine, ops, inputDim, hiddenDim)
	if err != nil {
		return nil, err
	}

	hiddenWeights, err := core.NewLinear[T](name+"_hidden_weights", engine, ops, hiddenDim, hiddenDim)
	if err != nil {
		return nil, err
	}

	bias, err := core.NewBias[T](name+"_bias", engine, ops, hiddenDim)
	if err != nil {
		return nil, err
	}

	return &SimpleRNN[T]{
		name:          name,
		engine:        engine,
		ops:           ops,
		inputWeights:  inputWeights,
		hiddenWeights: hiddenWeights,
		bias:          bias,
		activation:    activations.NewTanh[T](engine, ops),
		inputDim:      inputDim,
		hiddenDim:     hiddenDim,
	}, nil
}

// OpType returns the operation type of the layer.
func (r *SimpleRNN[T]) OpType() string {
	return "SimpleRNN"
}

// Attributes returns the attributes of the layer.
func (r *SimpleRNN[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_dim":  r.inputDim,
		"hidden_dim": r.hiddenDim,
	}
}

// OutputShape returns the output shape of the layer.
func (r *SimpleRNN[T]) OutputShape() []int {
	return r.hiddenWeights.OutputShape()
}

// Forward computes the forward pass of the layer.
func (r *SimpleRNN[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SimpleRNN requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]
	r.lastInput = input

	if r.hiddenState == nil {
		// Initialize hidden state with proper batch dimension
		batchSize := input.Shape()[0]
		r.hiddenState, _ = tensor.New[T]([]int{batchSize, r.hiddenDim}, make([]T, batchSize*r.hiddenDim))
	}

	inputContribution, err := r.inputWeights.Forward(ctx, input)
	if err != nil {
		return nil, err
	}

	hiddenContribution, err := r.hiddenWeights.Forward(ctx, r.hiddenState)
	if err != nil {
		return nil, err
	}

	sum, err := r.engine.Add(ctx, inputContribution, hiddenContribution)
	if err != nil {
		return nil, err
	}

	// Add bias
	biasedSum, err := r.bias.Forward(ctx, sum)
	if err != nil {
		return nil, err
	}

	newState, err := r.activation.Forward(ctx, biasedSum)
	if err != nil {
		return nil, err
	}

	r.hiddenState = newState
	return newState, nil
}

// Backward computes the gradients.
func (r *SimpleRNN[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if mode == types.OneStepApproximation {
		tanhGrad, err := r.activation.Backward(ctx, mode, outputGradient, r.hiddenState)
		if err != nil {
			return nil, err
		}

		dInput, err := r.inputWeights.Backward(ctx, mode, tanhGrad[0], r.lastInput)
		if err != nil {
			return nil, err
		}

		dHidden, err := r.hiddenWeights.Backward(ctx, mode, tanhGrad[0], r.hiddenState)
		if err != nil {
			return nil, err
		}

		return []*tensor.TensorNumeric[T]{dInput[0], dHidden[0]}, nil
	}

	return nil, fmt.Errorf("unsupported backward mode: %v", mode)
}

// Parameters returns the parameters of the layer.
func (r *SimpleRNN[T]) Parameters() []*graph.Parameter[T] {
	params := r.inputWeights.Parameters()
	params = append(params, r.hiddenWeights.Parameters()...)
	params = append(params, r.bias.Parameters()...)
	return params
}

func init() {
	model.RegisterLayer("SimpleRNN", func(engine compute.Engine[float32], ops numeric.Arithmetic[float32], name string, params map[string]*graph.Parameter[float32], attributes map[string]interface{}) (graph.Node[float32], error) {
		inputDim, ok := attributes["input_dim"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'input_dim' for SimpleRNN")
		}
		hiddenDim, ok := attributes["hidden_dim"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'hidden_dim' for SimpleRNN")
		}
		return NewSimpleRNN[float32](name, engine, ops, inputDim, hiddenDim)
	})
}