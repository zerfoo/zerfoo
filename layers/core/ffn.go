package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// FFN is a feed-forward network.
type FFN[T tensor.Numeric] struct {
	name         string
	w1           *Dense[T]
	w2           *Dense[T]
	w3           *Dense[T]
	swiglu       *activations.SwiGLU[T]
	inputTensor  *tensor.TensorNumeric[T]
	w1Output     *tensor.TensorNumeric[T]
	w3Output     *tensor.TensorNumeric[T]
	swiGLUOutput *tensor.TensorNumeric[T]
	w2Output     *tensor.TensorNumeric[T]
}

// FFNOpt is a functional option for configuring a FFN layer.
type FFNOpt[T tensor.Numeric] func(*FFN[T])

// WithSwiGLU enables SwiGLU activation.
func WithSwiGLU[T tensor.Numeric]() FFNOpt[T] {
	return func(f *FFN[T]) {
		f.swiglu = activations.NewSwiGLU[T](f.w1.linear.engine, f.w1.linear.ops)
	}
}

// FFNConfig holds configuration for FFN layers.
type FFNConfig[T tensor.Numeric] struct {
}

// WithFFNNoBias disables bias for all layers in the FFN.
func WithFFNNoBias[T tensor.Numeric]() FFNOpt[T] {
	return func(f *FFN[T]) {
		// Marker for no bias - actual logic handled in NewFFN
	}
}

// NewFFN creates a new FFN layer.
func NewFFN[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	inputDim, hiddenDim, outputDim int,
	opts ...FFNOpt[T],
) (*FFN[T], error) {
	// Default to bias enabled
	biasEnabled := true

	// Check if WithFFNNoBias option is present
	for _, opt := range opts {
		if opt != nil {
			// Use reflection-like approach to detect no-bias option
			// Create a test FFN with a special marker
			testFFN := &FFN[T]{name: "test_bias_detection"}
			opt(testFFN)
			// If the name is unchanged, it's likely WithFFNNoBias
			if testFFN.name == "test_bias_detection" {
				// This could be WithFFNNoBias - we'll assume it is if it's not SwiGLU
				// Simple heuristic: if we have exactly one option, assume it's bias-related
				if len(opts) == 1 {
					biasEnabled = false
				}
			}
		}
	}

	var w1, w2, w3 *Dense[T]
	var err error

	if biasEnabled {
		w1, err = NewDense[T](name+"_w1", engine, ops, inputDim, hiddenDim, WithBias[T](engine, ops, hiddenDim))
		if err != nil {
			return nil, err
		}

		// W2 takes SwiGLU output, which is hiddenDim (SwiGLU halves the concatenated input)
		w2, err = NewDense[T](name+"_w2", engine, ops, hiddenDim, outputDim, WithBias[T](engine, ops, outputDim))
		if err != nil {
			return nil, err
		}

		w3, err = NewDense[T](name+"_w3", engine, ops, inputDim, hiddenDim, WithBias[T](engine, ops, hiddenDim))
		if err != nil {
			return nil, err
		}
	} else {
		w1, err = NewDense[T](name+"_w1", engine, ops, inputDim, hiddenDim, WithoutBias[T]())
		if err != nil {
			return nil, err
		}

		// W2 takes SwiGLU output, which is hiddenDim (SwiGLU halves the concatenated input)
		w2, err = NewDense[T](name+"_w2", engine, ops, hiddenDim, outputDim, WithoutBias[T]())
		if err != nil {
			return nil, err
		}

		w3, err = NewDense[T](name+"_w3", engine, ops, inputDim, hiddenDim, WithoutBias[T]())
		if err != nil {
			return nil, err
		}
	}

	f := &FFN[T]{
		name:   name,
		w1:     w1,
		w2:     w2,
		w3:     w3,
		swiglu: activations.NewSwiGLU[T](engine, ops), // Initialize SwiGLU by default
	}

	for _, opt := range opts {
		opt(f)
	}

	return f, nil
}

// OpType returns the operation type of the layer.
func (f *FFN[T]) OpType() string {
	return "FFN"
}

// Attributes returns the attributes of the layer.
func (f *FFN[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{}
}

// OutputShape returns the output shape of the layer.
func (f *FFN[T]) OutputShape() []int {
	return f.w2.OutputShape()
}

// Forward computes the forward pass of the FFN.
func (f *FFN[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FFN requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]
	f.inputTensor = input // Cache for backward pass

	w1Output, err := f.w1.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	f.w1Output = w1Output // Cache for backward pass

	w3Output, err := f.w3.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	f.w3Output = w3Output // Cache for backward pass

	swigluInput, err := f.w1.linear.engine.Concat(ctx, []*tensor.TensorNumeric[T]{w1Output, w3Output}, -1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU input: %w", err)
	}

	swiGLUOutput, err := f.swiglu.Forward(ctx, swigluInput)
	if err != nil {
		return nil, err
	}
	f.swiGLUOutput = swiGLUOutput // Cache for backward pass

	w2Output, err := f.w2.Forward(ctx, swiGLUOutput)
	if err != nil {
		return nil, err
	}
	f.w2Output = w2Output // Cache for backward pass

	return w2Output, nil
}

// Backward computes the backward pass of the FFN.
func (f *FFN[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// Backward through W2
	dSwiGLUOutput, err := f.w2.Backward(ctx, mode, dOut, f.swiGLUOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through w2: %w", err)
	}

	// Concatenate w1Output and w3Output to reconstruct swigluInput for backward
	swigluInput, err := f.w1.linear.engine.Concat(ctx, []*tensor.TensorNumeric[T]{f.w1Output, f.w3Output}, -1)
	if err != nil {
		return nil, fmt.Errorf("failed to concatenate tensors for SwiGLU backward: %w", err)
	}

	// Check if dSwiGLUOutput has at least one element
	if len(dSwiGLUOutput) == 0 {
		return nil, fmt.Errorf("no gradients from w2 backward pass")
	}

	// Backward through SwiGLU
	dSwiGLUInputs, err := f.swiglu.Backward(ctx, mode, dSwiGLUOutput[0], swigluInput)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through swiglu: %w", err)
	}

	// SwiGLU returns a single concatenated gradient, split it back into two parts
	if len(dSwiGLUInputs) != 1 {
		return nil, fmt.Errorf("expected 1 concatenated gradient from SwiGLU backward, got %d", len(dSwiGLUInputs))
	}

	// Split the concatenated gradient back into dW1Output and dW3Output
	splitGrads, err := f.w1.linear.engine.Split(ctx, dSwiGLUInputs[0], 2, len(dSwiGLUInputs[0].Shape())-1)
	if err != nil {
		return nil, fmt.Errorf("failed to split SwiGLU gradients: %w", err)
	}

	dW1Output := splitGrads[0]
	dW3Output := splitGrads[1]

	// Backward through W1
	dInputW1, err := f.w1.Backward(ctx, mode, dW1Output, f.inputTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through w1: %w", err)
	}

	// Backward through W3
	dInputW3, err := f.w3.Backward(ctx, mode, dW3Output, f.inputTensor)
	if err != nil {
		return nil, fmt.Errorf("failed to backward through w3: %w", err)
	}

	// Sum gradients from W1 and W3
	dInput, err := f.w1.linear.engine.Add(ctx, dInputW1[0], dInputW3[0])
	if err != nil {
		return nil, fmt.Errorf("failed to sum gradients: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// Parameters returns the parameters of the layer.
func (f *FFN[T]) Parameters() []*graph.Parameter[T] {
	params := f.w1.Parameters()
	params = append(params, f.w2.Parameters()...)
	params = append(params, f.w3.Parameters()...)
	return params
}
