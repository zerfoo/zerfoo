// Package normalization provides various normalization layers for neural networks.
package normalization

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// LayerNormalization implements the Layer Normalization operation.
type LayerNormalization[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	epsilon T // Small constant to avoid division by zero

	// Trainable parameters
	gamma *graph.Parameter[T] // Scale parameter
	beta  *graph.Parameter[T] // Shift parameter

	// Cached tensors for backward pass
	inputShape  []int
	mean        *tensor.TensorNumeric[T]
	variance    *tensor.TensorNumeric[T]
	normedInput *tensor.TensorNumeric[T] // (input - mean) / sqrt(variance + epsilon)
	outputShape []int
}

// LayerNormalizationOptions holds configuration options for LayerNormalization layers.
type LayerNormalizationOptions[T tensor.Numeric] struct {
	Epsilon T // Small constant to avoid division by zero
}

// LayerNormalizationOption is a functional option for configuring LayerNormalization layers.
type LayerNormalizationOption[T tensor.Numeric] func(*LayerNormalizationOptions[T])

// WithLayerNormEpsilon sets the epsilon parameter for LayerNormalization.
func WithLayerNormEpsilon[T tensor.Numeric](epsilon T) LayerNormalizationOption[T] {
	return func(opts *LayerNormalizationOptions[T]) {
		opts.Epsilon = epsilon
	}
}

// NewLayerNormalization creates a new LayerNormalization layer.
// featureDim: The dimension over which to normalize (typically the last dimension).
func NewLayerNormalization[T tensor.Numeric](engine compute.Engine[T], featureDim int, options ...LayerNormalizationOption[T]) (*LayerNormalization[T], error) {
	// Apply functional options
	opts := &LayerNormalizationOptions[T]{
		Epsilon: engine.Ops().FromFloat64(1e-5), // Default epsilon value
	}
	for _, option := range options {
		option(opts)
	}

	// Initialize gamma (scale) and beta (shift) parameters
	// They should have shape (featureDim,)
	gammaTensor, err := tensor.New[T]([]int{featureDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create gamma tensor: %w", err)
	}
	// Initialize gamma to ones
	if err := engine.Fill(context.Background(), gammaTensor, engine.Ops().FromFloat64(1.0)); err != nil { // Assuming Fill is available
		return nil, fmt.Errorf("failed to fill gamma tensor: %w", err)
	}

	gamma, err := graph.NewParameter[T]("gamma", gammaTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("failed to create gamma parameter: %w", err)
	}

	betaTensor, err := tensor.New[T]([]int{featureDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create beta tensor: %w", err)
	}
	// Initialize beta to zeros
	if err := engine.Fill(context.Background(), betaTensor, engine.Ops().FromFloat64(0.0)); err != nil {
		return nil, fmt.Errorf("failed to fill beta tensor: %w", err)
	}

	beta, err := graph.NewParameter[T]("beta", betaTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("failed to create beta parameter: %w", err)
	}

	return &LayerNormalization[T]{
		engine:  engine,
		epsilon: opts.Epsilon,
		gamma:   gamma,
		beta:    beta,
	}, nil
}

// OutputShape returns the output shape, which is the same as the input shape.
func (ln *LayerNormalization[T]) OutputShape() []int {
	return ln.outputShape
}

// Parameters returns the trainable gamma and beta parameters.
func (ln *LayerNormalization[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{ln.gamma, ln.beta}
}

// Forward computes the Layer Normalization.
func (ln *LayerNormalization[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LayerNormalization: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	input := inputs[0]
	ln.inputShape = input.Shape() // Cache input shape for backward
	ln.outputShape = input.Shape()

	// Calculate mean along the last dimension
	// KeepDims=true to maintain original dimensions for broadcasting
	sum, err := ln.engine.ReduceSum(ctx, input, len(input.Shape())-1, true, nil)
	if err != nil {
		return nil, err
	}

	featureSize := ln.engine.Ops().FromFloat64(float64(input.Shape()[len(input.Shape())-1]))

	mean, err := ln.engine.DivScalar(ctx, sum, featureSize, nil) // Assuming ReduceMean is available
	if err != nil {
		return nil, err
	}

	ln.mean = mean // Cache for backward

	// Calculate variance
	// (input - mean)
	inputMinusMean, err := ln.engine.Sub(ctx, input, mean, nil)
	if err != nil {
		return nil, err
	}

	// (input - mean)^2
	squaredDiff, err := ln.engine.Mul(ctx, inputMinusMean, inputMinusMean, nil)
	if err != nil {
		return nil, err
	}

	// Mean of squared_diff (variance)
	sumSquaredDiff, err := ln.engine.ReduceSum(ctx, squaredDiff, len(input.Shape())-1, true, nil)
	if err != nil {
		return nil, err
	}
	// featureSize already defined above
	variance, err := ln.engine.DivScalar(ctx, sumSquaredDiff, featureSize, nil)
	if err != nil {
		return nil, err
	}

	ln.variance = variance // Cache for backward

	// sqrt(variance + epsilon)
	variancePlusEpsilon, err := ln.engine.AddScalar(ctx, variance, ln.epsilon, nil) // Assuming AddScalar is available
	if err != nil {
		return nil, err
	}

	stdDev, err := ln.engine.Sqrt(ctx, variancePlusEpsilon, nil) // Assuming Sqrt is available
	if err != nil {
		return nil, err
	}

	// Normalized input: (input - mean) / stdDev
	normedInput, err := ln.engine.Div(ctx, inputMinusMean, stdDev, nil)
	if err != nil {
		return nil, err
	}

	ln.normedInput = normedInput // Cache for backward

	// Scale and shift: normedInput * gamma + beta
	scaled, err := ln.engine.Mul(ctx, normedInput, ln.gamma.Value, nil)
	if err != nil {
		return nil, err
	}

	output, err := ln.engine.Add(ctx, scaled, ln.beta.Value, nil)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for LayerNormalization.
func (ln *LayerNormalization[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LayerNormalization: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	// Gradients for gamma and beta
	// dL/dgamma = sum(dOut * normedInput) along the normalization axis
	dOutMulNormedInput, err := ln.engine.Mul(ctx, dOut, ln.normedInput, nil)
	if err != nil {
		return nil, err
	}

	dGamma, err := ln.engine.ReduceSum(ctx, dOutMulNormedInput, len(ln.inputShape)-1, false, nil)
	if err != nil {
		return nil, err
	}

	if err := ln.gamma.AddGradient(dGamma); err != nil {
		return nil, err
	}

	// dL/dbeta = sum(dOut) along the normalization axis
	dBeta, err := ln.engine.ReduceSum(ctx, dOut, len(ln.inputShape)-1, false, nil)
	if err != nil {
		return nil, err
	}

	if err := ln.beta.AddGradient(dBeta); err != nil {
		return nil, err
	}

	// Gradient for input (dL/dx)
	// This derivation follows the standard backpropagation for Layer Normalization.
	// N is the size of the feature dimension (last dimension of inputShape)
	N := ln.engine.Ops().FromFloat64(float64(ln.inputShape[len(ln.inputShape)-1]))

	// dL/d_normed_input = dOut * gamma
	dLdNormedInput, err := ln.engine.Mul(ctx, dOut, ln.gamma.Value, nil)
	if err != nil {
		return nil, err
	}

	// input - mean
	inputMinusMean, err := ln.engine.Sub(ctx, inputs[0], ln.mean, nil)
	if err != nil {
		return nil, err
	}

	// stdDev = sqrt(variance + epsilon)
	variancePlusEpsilon, err := ln.engine.AddScalar(ctx, ln.variance, ln.epsilon, nil)
	if err != nil {
		return nil, err
	}

	stdDev, err := ln.engine.Sqrt(ctx, variancePlusEpsilon, nil)
	if err != nil {
		return nil, err
	}

	// dL/d_variance_term = sum(dL/d_normed_input * (input - mean)) along the feature dimension
	mulResult, err := ln.engine.Mul(ctx, dLdNormedInput, inputMinusMean, nil)
	if err != nil {
		return nil, err
	}

	dLdVarianceTerm, err := ln.engine.ReduceSum(ctx, mulResult, len(ln.inputShape)-1, true, nil)
	if err != nil {
		return nil, err
	}

	// dL/d_mean_term = sum(dL/d_normed_input) along the feature dimension
	dLdMeanTerm, err := ln.engine.ReduceSum(ctx, dLdNormedInput, len(ln.inputShape)-1, true, nil)
	if err != nil {
		return nil, err
	}

	// Term 1: dLdNormedInput / stdDev
	term1, err := ln.engine.Div(ctx, dLdNormedInput, stdDev, nil)
	if err != nil {
		return nil, err
	}

	// Term 2: (input - mean) * dLdVarianceTerm / (N * stdDev^3)
	stdDevSquared, err := ln.engine.Mul(ctx, stdDev, stdDev, nil)
	if err != nil {
		return nil, err
	}

	stdDevCubed, err := ln.engine.Mul(ctx, stdDevSquared, stdDev, nil)
	if err != nil {
		return nil, err
	}

	term2Numerator, err := ln.engine.Mul(ctx, inputMinusMean, dLdVarianceTerm, nil)
	if err != nil {
		return nil, err
	}

	term2Denominator, err := ln.engine.MulScalar(ctx, stdDevCubed, N, nil)
	if err != nil {
		return nil, err
	}

	term2, err := ln.engine.Div(ctx, term2Numerator, term2Denominator, nil)
	if err != nil {
		return nil, err
	}

	// Term 3: dLdMeanTerm / N
	term3, err := ln.engine.DivScalar(ctx, dLdMeanTerm, N, nil)
	if err != nil {
		return nil, err
	}

	// dL/dx = term1 - term2 - term3
	dInput, err := ln.engine.Sub(ctx, term1, term2, nil)
	if err != nil {
		return nil, err
	}

	dInput, err = ln.engine.Sub(ctx, dInput, term3, nil)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// OpType returns the operation type of the LayerNormalization layer.
func (ln *LayerNormalization[T]) OpType() string {
	return "LayerNormalization"
}

// Attributes returns the attributes of the LayerNormalization layer.
func (ln *LayerNormalization[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"epsilon": ln.epsilon}
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*LayerNormalization[float32])(nil)
