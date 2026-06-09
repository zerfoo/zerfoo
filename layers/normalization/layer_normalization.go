// Package normalization provides various normalization layers for neural networks.
package normalization

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// LayerNormalization implements the Layer Normalization operation.
//
// When T is float32 or sub-float32 (float16, float8) AND the engine is
// a CPU engine, Backward runs all per-element arithmetic in float64 and
// casts the result back to T. Parameter storage and gradient outputs stay
// at T. This removes the catastrophic-cancellation risk in (input - mean)
// when activations drift far from zero, and protects the stdDev^3
// division chain from float32 rounding noise. The GPU engine path
// preserves the original T-only computation because GPU kernels batch
// these ops and maintain intermediate precision better than naive
// per-element float32 math; upgrading GPU kernels to mixed precision
// is a follow-up once native kernels exist.
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

	useMixedBackward bool // Run Backward per-element in float64 on CPU.
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
		engine:           engine,
		epsilon:          opts.Epsilon,
		gamma:            gamma,
		beta:             beta,
		useMixedBackward: shouldUseMixedPrecisionBackward[T](engine),
	}, nil
}

// NewLayerNormalizationFromParams creates a LayerNormalization layer from
// existing gamma (weight) and beta (bias) parameters. This is used for
// constructing layers from pre-loaded GGUF tensors during model loading.
func NewLayerNormalizationFromParams[T tensor.Numeric](
	engine compute.Engine[T],
	epsilon T,
	gamma *graph.Parameter[T],
	beta *graph.Parameter[T],
) *LayerNormalization[T] {
	return &LayerNormalization[T]{
		engine:           engine,
		epsilon:          epsilon,
		gamma:            gamma,
		beta:             beta,
		useMixedBackward: shouldUseMixedPrecisionBackward[T](engine),
	}
}

// shouldUseMixedPrecisionBackward returns true when LayerNorm.Backward
// should run per-element arithmetic in float64 instead of T. True only
// when T is float32 or below AND the engine is a CPU engine. GPU engines
// keep the current all-T path so their CUDA-native kernels remain in
// charge of precision.
func shouldUseMixedPrecisionBackward[T tensor.Numeric](_ compute.Engine[T]) bool {
	// Enabled for the low-precision float types on ALL engines (CPU and GPU).
	// LayerNorm.Backward divides by the per-row stddev; in float32 an early-
	// training gradient spike overflows (the GPU "CrossAsset cliff": gamma/beta
	// grads -> ~1e12, cascading to ~1e34/NaN, while CPU's float64 backward
	// absorbed it). Running the per-element backward in float64 fixes both.
	// Previously gated to *compute.CPUEngine only, which left GPU float32
	// LayerNorm backward overflowing to NaN.
	var zero T
	switch any(zero).(type) {
	case float64:
		return false
	case float32, float16.Float16, float16.BFloat16, float8.Float8:
		return true
	default:
		return false
	}
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

	if ln.useMixedBackward {
		return ln.backwardMixedCPU(dOut, inputs[0])
	}

	// Gradients for gamma and beta
	// dL/dgamma = sum(dOut * normedInput) along the normalization axis
	dOutMulNormedInput, err := ln.engine.Mul(ctx, dOut, ln.normedInput, nil)
	if err != nil {
		return nil, err
	}

	dGamma := dOutMulNormedInput
	for i := 0; i < len(ln.inputShape)-1; i++ {
		dGamma, err = ln.engine.ReduceSum(ctx, dGamma, 0, false, nil)
		if err != nil {
			return nil, err
		}
	}

	if err := ln.gamma.AddGradient(dGamma); err != nil {
		return nil, err
	}

	// dL/dbeta = sum(dOut) along the batch dimensions (all non-feature dims)
	dBeta := dOut
	for i := 0; i < len(ln.inputShape)-1; i++ {
		dBeta, err = ln.engine.ReduceSum(ctx, dBeta, 0, false, nil)
		if err != nil {
			return nil, err
		}
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

	// Term 3: dLdMeanTerm / (N * stdDev)
	nTimesStd, err := ln.engine.MulScalar(ctx, stdDev, N, nil)
	if err != nil {
		return nil, err
	}

	term3, err := ln.engine.Div(ctx, dLdMeanTerm, nTimesStd, nil)
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

// backwardMixedCPU runs LayerNormalization.Backward with all per-element
// arithmetic performed in float64, while parameter storage and gradient
// outputs remain at T. Used only when T is float32 or below and the
// engine is a CPU engine (see shouldUseMixedPrecisionBackward).
//
// The critical precision concerns it addresses:
//   - (input - mean) can lose most significant digits to catastrophic
//     cancellation when activations drift far from zero during training.
//     float64 subtraction preserves full precision.
//   - The stdDev^3 division chain amplifies any rounding error; running
//     it in float64 keeps intermediate magnitudes meaningful.
//   - Sum-of-products reductions over the feature axis accumulate error
//     linearly in float32; float64 caps accumulated error below 1e-12.
func (ln *LayerNormalization[T]) backwardMixedCPU(dOut *tensor.TensorNumeric[T], input *tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	ops := ln.engine.Ops()
	nFeat := ln.inputShape[len(ln.inputShape)-1]
	if nFeat <= 0 {
		return nil, fmt.Errorf("LayerNormalization: invalid feature dim %d", nFeat)
	}
	total := 1
	for _, d := range ln.inputShape {
		total *= d
	}
	positions := total / nFeat
	epsilon := lnNumericToFloat64(ln.epsilon)

	inputData := input.Data()
	dOutData := dOut.Data()
	gammaData := ln.gamma.Value.Data()

	dInputTensor, err := tensor.New[T](ln.inputShape, nil)
	if err != nil {
		return nil, err
	}
	dInputData := dInputTensor.Data()

	dGamma64 := make([]float64, nFeat)
	dBeta64 := make([]float64, nFeat)
	nFeatF := float64(nFeat)

	for p := 0; p < positions; p++ {
		base := p * nFeat

		// Recompute mean and variance from the (reliable) forward input in
		// float64 rather than trusting the cached ln.mean / ln.variance /
		// ln.normedInput tensors. On the GPU arena those cached tensors can be
		// overwritten by downstream forward ops before Backward runs -- the
		// cached variance comes back negative, making sigma = sqrt(var+eps) NaN
		// (the residual GPU f32 "CrossAsset cliff"; CPU's GC-kept tensors were
		// immune, masking it). Recomputing here is self-contained and exact.
		var mu float64
		for i := 0; i < nFeat; i++ {
			mu += lnNumericToFloat64(inputData[base+i])
		}
		mu /= nFeatF

		var variance float64
		for i := 0; i < nFeat; i++ {
			d := lnNumericToFloat64(inputData[base+i]) - mu
			variance += d * d
		}
		variance /= nFeatF

		sigma := math.Sqrt(variance + epsilon)
		sigma3 := sigma * sigma * sigma

		var sumDNorm, sumDNormXMu float64
		for i := 0; i < nFeat; i++ {
			gi := lnNumericToFloat64(gammaData[i])
			dOutI := lnNumericToFloat64(dOutData[base+i])
			xMinusMu := lnNumericToFloat64(inputData[base+i]) - mu
			dNorm := dOutI * gi
			sumDNorm += dNorm
			sumDNormXMu += dNorm * xMinusMu
		}

		for i := 0; i < nFeat; i++ {
			gi := lnNumericToFloat64(gammaData[i])
			dOutI := lnNumericToFloat64(dOutData[base+i])
			xMinusMu := lnNumericToFloat64(inputData[base+i]) - mu
			normed := xMinusMu / sigma

			dNorm := dOutI * gi
			term1 := dNorm / sigma
			term2 := xMinusMu * sumDNormXMu / (nFeatF * sigma3)
			term3 := sumDNorm / (nFeatF * sigma)

			dInputData[base+i] = ops.FromFloat64(term1 - term2 - term3)
			dGamma64[i] += dOutI * normed
			dBeta64[i] += dOutI
		}
	}

	dGammaTensor, err := tensor.New[T]([]int{nFeat}, nil)
	if err != nil {
		return nil, err
	}
	dBetaTensor, err := tensor.New[T]([]int{nFeat}, nil)
	if err != nil {
		return nil, err
	}
	dGammaData := dGammaTensor.Data()
	dBetaData := dBetaTensor.Data()
	for i := 0; i < nFeat; i++ {
		dGammaData[i] = ops.FromFloat64(dGamma64[i])
		dBetaData[i] = ops.FromFloat64(dBeta64[i])
	}

	if err := ln.gamma.AddGradient(dGammaTensor); err != nil {
		return nil, err
	}
	if err := ln.beta.AddGradient(dBetaTensor); err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInputTensor}, nil
}

// lnNumericToFloat64 converts a tensor.Numeric value to float64. This is
// duplicated from training/optimizer and training/loss to keep this
// package self-contained. Consolidate to a shared helper if a fourth
// duplicate emerges.
func lnNumericToFloat64[T tensor.Numeric](v T) float64 {
	switch val := any(v).(type) {
	case float32:
		return float64(val)
	case float64:
		return val
	case int:
		return float64(val)
	case int8:
		return float64(val)
	case int16:
		return float64(val)
	case int32:
		return float64(val)
	case int64:
		return float64(val)
	case uint:
		return float64(val)
	case uint8:
		return float64(val)
	case uint32:
		return float64(val)
	case uint64:
		return float64(val)
	case float16.Float16:
		return float64(val.ToFloat32())
	case float16.BFloat16:
		return float64(val.ToFloat32())
	case float8.Float8:
		return val.ToFloat64()
	default:
		return 0
	}
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
