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
	"github.com/zerfoo/zerfoo/types"
)

// FFN (Feed-Forward Network) implements a two-layer MLP with SwiGLU activation.
type FFN[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	w1     *Dense[T] // First linear layer
	w3     *Dense[T] // Gate linear layer for SwiGLU
	w2     *Dense[T] // Second linear layer
	swiglu *activations.SwiGLU[T]

	// Cached tensors for backward pass
	inputTensor  *tensor.TensorNumeric[T]
	w1Output     *tensor.TensorNumeric[T]
	w3Output     *tensor.TensorNumeric[T]
	swiGLUOutput *tensor.TensorNumeric[T]
	w2Output     *tensor.TensorNumeric[T]
	outputShape  []int
}

// OpType returns the operation type.
func (f *FFN[T]) OpType() string {
	return "FFN"
}

// Attributes returns the attributes.
func (f *FFN[T]) Attributes() map[string]interface{} {
	return make(map[string]interface{})
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

	// Note: Linear layer options are not used directly here because Dense handles initialization internally.

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
func (f *FFN[T]) OutputShape() []int {
	return f.outputShape
}

// Parameters returns the trainable parameters of the FFN layer.
func (f *FFN[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]

	// Append parameters from w1
	params = append(params, f.w1.Parameters()...)
	// Append parameters from w3
	params = append(params, f.w3.Parameters()...)
	// Append parameters from w2
	params = append(params, f.w2.Parameters()...)

	return params
}

// Forward computes the forward pass of the FFN.
func (f *FFN[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FFN: expected 1 input tensor, got %d", len(inputs))
	}

	input := inputs[0]
	f.inputTensor = input // Cache for backward pass

	// Linear Layer 1
	w1Output, err := f.w1.Forward(ctx, input)
	if err != nil {
		return nil, err
	}

	f.w1Output = w1Output // Cache for backward pass

	// Gate Linear Layer (W3)
	w3Output, err := f.w3.Forward(ctx, input)
	if err != nil {
		return nil, err
	}

	f.w3Output = w3Output // Cache for backward pass

	// Concatenate w1Output and w3Output for SwiGLU
	swigluInput, err := f.engine.Concat(ctx, []*tensor.TensorNumeric[T]{w1Output, w3Output}, len(w1Output.Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU: %w", err)
	}

	// SwiGLU Activation
	swiGLUOutput, err := f.swiglu.Forward(ctx, swigluInput)
	if err != nil {
		return nil, err
	}

	f.swiGLUOutput = swiGLUOutput // Cache for backward pass

	// Linear Layer 2
	w2Output, err := f.w2.Forward(ctx, swiGLUOutput)
	if err != nil {
		return nil, err
	}

	f.w2Output = w2Output // Cache for backward pass
	f.outputShape = w2Output.Shape()

	return w2Output, nil
}

// Backward computes the backward pass of the FFN.
func (f *FFN[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Backward through W2
	dSwiGLUOutput, err := f.w2.Backward(ctx, mode, dOut, f.swiGLUOutput)
	if err != nil {
		return nil, err
	}

	dSwiGLUOutputTensor := dSwiGLUOutput[0]

	// Concatenate w1Output and w3Output to reconstruct swigluInput for backward
	swigluInput, err := f.engine.Concat(ctx, []*tensor.TensorNumeric[T]{f.w1Output, f.w3Output}, len(f.w1Output.Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU backward: %w", err)
	}

	// Backward through SwiGLU
	dSwiGLUInputs, err := f.swiglu.Backward(ctx, mode, dSwiGLUOutputTensor, swigluInput)
	if err != nil {
		return nil, err
	}

	dSwiGLUInputTensor := dSwiGLUInputs[0]

	// Split the gradient back for w1 and w3
	splitGrads, err := f.engine.Split(ctx, dSwiGLUInputTensor, 2, len(dSwiGLUInputTensor.Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to split gradients for w1 and w3: %w", err)
	}

	dW1Output, dW3Output := splitGrads[0], splitGrads[1]

	// Backward through W1
	dInputW1, err := f.w1.Backward(ctx, mode, dW1Output, f.inputTensor)
	if err != nil {
		return nil, err
	}

	dInputW1Tensor := dInputW1[0]

	// Backward through W3
	dInputW3, err := f.w3.Backward(ctx, mode, dW3Output, f.inputTensor)
	if err != nil {
		return nil, err
	}

	dInputW3Tensor := dInputW3[0]

	// Sum gradients for the input tensor
	dInputTotal, err := f.engine.Add(ctx, dInputW1Tensor, dInputW3Tensor)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInputTotal}, nil
}

// Engine returns the compute engine of the FFN layer.
func (f *FFN[T]) Engine() compute.Engine[T] {
	return f.engine
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*FFN[float32])(nil)
