package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// FFN (Feed-Forward Network) implements a two-layer MLP with SwiGLU activation.
type FFN[T tensor.Numeric] struct {
	name    string
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]

	w1 *Dense[T] // First linear layer
	w3 *Dense[T] // Gate linear layer for SwiGLU
	w2 *Dense[T] // Second linear layer

	// Cached tensors for backward pass
	inputTensor *tensor.Tensor[T]
	w1Output    *tensor.Tensor[T]
	w3Output    *tensor.Tensor[T]
	swiGLUOutput *tensor.Tensor[T]
	w2Output    *tensor.Tensor[T]
}

// NewFFN creates a new Feed-Forward Network layer.
// name: The name of the FFN layer.
// engine: The compute engine to use for tensor operations.
// ops: Numeric operations for the given type T.
// inputDim: The input dimension of the FFN.
// hiddenDim: The dimension of the hidden layer.
// outputDim: The output dimension of the FFN.
func NewFFN[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], inputDim, hiddenDim, outputDim int) (*FFN[T], error) {
	// W1: inputDim -> hiddenDim
	w1, err := NewDense[T](name+"_w1", engine, ops, inputDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create W1 dense layer: %w", err)
	}

	// W3: inputDim -> hiddenDim (for the gate in SwiGLU)
	w3, err := NewDense[T](name+"_w3", engine, ops, inputDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create W3 dense layer: %w", err)
	}

	// W2: hiddenDim -> outputDim
	w2, err := NewDense[T](name+"_w2", engine, ops, hiddenDim, outputDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create W2 dense layer: %w", err)
	}

	return &FFN[T]{
		name:   name,
		engine: engine,
		ops:    ops,
		w1:     w1,
		w3:     w3,
		w2:     w2,
	}, nil
}

// OutputShape returns the output shape of the FFN layer.
// Input shape is (batch_size, input_dim). Output shape is (batch_size, output_dim).
func (ffn *FFN[T]) OutputShape(inputShapes ...[]int) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("FFN: expected 1 input shape, got %d", len(inputShapes))
	}
	inputShape := inputShapes[0]
	if len(inputShape) != 2 {
		return nil, fmt.Errorf("expected 2D tensor (batch, input_dim) for FFN, got %v", inputShape)
	}

	return []int{inputShape[0], ffn.w2.linear.OutputShape()[1]}, nil
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

	// SwiGLU Activation: (W1_output * Swish(W3_output))
	swiglu := activations.NewSwiGLU[T](ffn.engine, ffn.ops)
	swiGLUOutput, err := swiglu.Forward(ctx, ffn.w1Output, ffn.w3Output)
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

	return w2Output, nil
}

// Backward computes the backward pass of the FFN.
func (ffn *FFN[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	// Backward through W2
	dSwiGLUOutput, err := ffn.w2.Backward(ctx, dOut)
	if err != nil {
		return nil, err
	}
	dSwiGLUOutputTensor := dSwiGLUOutput[0]

	// Backward through SwiGLU
	swiglu := activations.NewSwiGLU[T](ffn.engine, ffn.ops)
	dSwiGLUInputs, err := swiglu.Backward(ctx, dSwiGLUOutputTensor, ffn.w1Output, ffn.w3Output)
	if err != nil {
		return nil, err
	}
	dW1Output, dW3Output := dSwiGLUInputs[0], dSwiGLUInputs[1]

	// Backward through W1
	dInputW1, err := ffn.w1.Backward(ctx, dW1Output)
	if err != nil {
		return nil, err
	}
	dInputW1Tensor := dInputW1[0]

	// Backward through W3
	dInputW3, err := ffn.w3.Backward(ctx, dW3Output)
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
