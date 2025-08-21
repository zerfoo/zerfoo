package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/components"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// FFN (Feed-Forward Network) implements a two-layer MLP with SwiGLU activation.
type FFN[T tensor.Numeric] struct {
	name    string
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]

	w1      *Dense[T] // First linear layer
	w3      *Dense[T] // Gate linear layer for SwiGLU
	w2      *Dense[T] // Second linear layer
	swiglu  *activations.SwiGLU[T]

	// Cached tensors for backward pass
	inputTensor *tensor.Tensor[T]
	w1Output    *tensor.Tensor[T]
	w3Output    *tensor.Tensor[T]
	swiGLUOutput *tensor.Tensor[T]
	w2Output    *tensor.Tensor[T]
	outputShape []int
}

// FFNOptions holds configuration options for the FFN layer.
type FFNOptions[T tensor.Numeric] struct {
	Initializer components.WeightInitializer[T]
	WithBias    bool
}

// FFNOption is a function that applies an option to FFNOptions.
type FFNOption[T tensor.Numeric] func(*FFNOptions[T])

// WithFFNInitializer sets a custom weight initializer for the FFN layer.
func WithFFNInitializer[T tensor.Numeric](initializer components.WeightInitializer[T]) FFNOption[T] {
	return func(o *FFNOptions[T]) {
		o.Initializer = initializer
	}
}

// WithFFNBias sets whether the FFN layer should include bias terms.
func WithFFNBias[T tensor.Numeric](withBias bool) FFNOption[T] {
	return func(o *FFNOptions[T]) {
		o.WithBias = withBias
	}
}

// NewFFN creates a new Feed-Forward Network layer.
// name: The name of the FFN layer.
// engine: The compute engine to use for tensor operations.
// ops: Numeric operations for the given type T.
// inputDim: The input dimension of the FFN.
// hiddenDim: The dimension of the hidden layer.
// outputDim: The output dimension of the FFN.
func NewFFN[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputDim, hiddenDim, outputDim int, opts ...FFNOption[T]) (*FFN[T], error) {
	// Default options
	options := &FFNOptions[T]{
		Initializer: components.NewXavierInitializer(ops),
		WithBias:    true,
	}
	for _, opt := range opts {
		opt(options)
	}

	// Create Dense layer options based on FFN options
	var denseOpts []DenseOption[T]
	if !options.WithBias {
		denseOpts = append(denseOpts, WithBias[T](false))
	}

	// Create Linear layer options based on FFN options
	var linearOpts []LinearOption[T]
	if options.Initializer != nil {
		linearOpts = append(linearOpts, WithInitializer[T](options.Initializer))
	}

	// W1: inputDim -> hiddenDim
	w1, err := NewDense[T](name+"_w1", engine, ops, inputDim, hiddenDim, denseOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create W1 dense layer: %w", err)
	}

	// W3: inputDim -> hiddenDim (for the gate in SwiGLU)
	w3, err := NewDense[T](name+"_w3", engine, ops, inputDim, hiddenDim, denseOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create W3 dense layer: %w", err)
	}

	// W2: hiddenDim -> outputDim
	w2, err := NewDense[T](name+"_w2", engine, ops, hiddenDim, outputDim, denseOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create W2 dense layer: %w", err)
	}

	// Apply custom initializer to existing layers if specified
	if options.Initializer != nil {
		// Initialize W1 weights
		w1WeightsData, err := options.Initializer.Initialize(inputDim, hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize W1 weights: %w", err)
		}
		w1WeightsTensor, err := tensor.New[T]([]int{inputDim, hiddenDim}, w1WeightsData)
		if err != nil {
			return nil, fmt.Errorf("failed to create W1 weights tensor: %w", err)
		}
		w1.linear.weights.Value = w1WeightsTensor

		// Initialize W2 weights
		w2WeightsData, err := options.Initializer.Initialize(hiddenDim, outputDim)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize W2 weights: %w", err)
		}
		w2WeightsTensor, err := tensor.New[T]([]int{hiddenDim, outputDim}, w2WeightsData)
		if err != nil {
			return nil, fmt.Errorf("failed to create W2 weights tensor: %w", err)
		}
		w2.linear.weights.Value = w2WeightsTensor

		// Initialize W3 weights
		w3WeightsData, err := options.Initializer.Initialize(inputDim, hiddenDim)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize W3 weights: %w", err)
		}
		w3WeightsTensor, err := tensor.New[T]([]int{inputDim, hiddenDim}, w3WeightsData)
		if err != nil {
			return nil, fmt.Errorf("failed to create W3 weights tensor: %w", err)
		}
		w3.linear.weights.Value = w3WeightsTensor
	}

	swiglu := activations.NewSwiGLU[T](engine, ops)

	return &FFN[T]{
		name:   name,
		engine: engine,
		ops:    ops,
		w1:     w1,
		w3:     w3,
		w2:     w2,
		swiglu: swiglu,
	}, nil
}

// OutputShape returns the output shape of the FFN layer.
func (ffn *FFN[T]) OutputShape() []int {
	return ffn.outputShape
}

// Parameters returns the trainable parameters of the FFN layer.
func (ffn *FFN[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]

	// Append parameters from w1
	params = append(params, ffn.w1.Parameters()...)
	// Append parameters from w3
	params = append(params, ffn.w3.Parameters()...)
	// Append parameters from w2
	params = append(params, ffn.w2.Parameters()...)
	return params
}

// Forward computes the forward pass of the FFN.
func (ffn *FFN[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FFN: expected 1 input tensor, got %d", len(inputs))
	}
	input := inputs[0]
	ffn.inputTensor = input // Cache for backward pass

	// Linear Layer 1
	w1Output, err := ffn.w1.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	ffn.w1Output = w1Output // Cache for backward pass

	// Gate Linear Layer (W3)
	w3Output, err := ffn.w3.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	ffn.w3Output = w3Output // Cache for backward pass

	// Concatenate w1Output and w3Output for SwiGLU
	swigluInput, err := ffn.engine.Concat(ctx, []*tensor.Tensor[T]{w1Output, w3Output}, len(w1Output.Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU: %w", err)
	}

	// SwiGLU Activation
	swiGLUOutput, err := ffn.swiglu.Forward(ctx, swigluInput)
	if err != nil {
		return nil, err
	}
	ffn.swiGLUOutput = swiGLUOutput // Cache for backward pass

	// Linear Layer 2
	w2Output, err := ffn.w2.Forward(ctx, swiGLUOutput)
	if err != nil {
		return nil, err
	}
	ffn.w2Output = w2Output // Cache for backward pass
	ffn.outputShape = w2Output.Shape()

	return w2Output, nil
}

// Backward computes the backward pass of the FFN.
func (ffn *FFN[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	// Backward through W2
	dSwiGLUOutput, err := ffn.w2.Backward(ctx, dOut, ffn.swiGLUOutput)
	if err != nil {
		return nil, err
	}
	dSwiGLUOutputTensor := dSwiGLUOutput[0]

	// Concatenate w1Output and w3Output to reconstruct swigluInput for backward
	swigluInput, err := ffn.engine.Concat(ctx, []*tensor.Tensor[T]{ffn.w1Output, ffn.w3Output}, len(ffn.w1Output.Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU backward: %w", err)
	}

	// Backward through SwiGLU
	dSwiGLUInputs, err := ffn.swiglu.Backward(ctx, dSwiGLUOutputTensor, swigluInput)
	if err != nil {
		return nil, err
	}
	dSwiGLUInputTensor := dSwiGLUInputs[0]

	// Split the gradient back for w1 and w3
	splitGrads, err := ffn.engine.Split(ctx, dSwiGLUInputTensor, 2, len(dSwiGLUInputTensor.Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to split gradients for w1 and w3: %w", err)
	}
	dW1Output, dW3Output := splitGrads[0], splitGrads[1]

	// Backward through W1
	dInputW1, err := ffn.w1.Backward(ctx, dW1Output, ffn.inputTensor)
	if err != nil {
		return nil, err
	}
	dInputW1Tensor := dInputW1[0]

	// Backward through W3
	dInputW3, err := ffn.w3.Backward(ctx, dW3Output, ffn.inputTensor)
	if err != nil {
		return nil, err
	}
	dInputW3Tensor := dInputW3[0]

	// Sum gradients for the input tensor
	dInputTotal, err := ffn.engine.Add(ctx, dInputW1Tensor, dInputW3Tensor)
	if err != nil {
		return nil, err
	}

	return []*tensor.Tensor[T]{dInputTotal}, nil
}

// Engine returns the compute engine of the FFN layer.
func (ffn *FFN[T]) Engine() compute.Engine[T] {
	return ffn.engine
}