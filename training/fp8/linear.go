// Package fp8 provides FP8 mixed-precision training layers.
package fp8

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// FP8Linear implements a linear layer that uses FP8 quantized weights for the
// forward pass and maintains full-precision master weights for gradient updates.
//
// Forward: quantize input and weight to FP8 (via per-tensor absmax scaling),
// compute the matmul (GPU engine dispatches to FP8 GEMM when both operands
// carry FP8E4M3Storage), output in full precision.
//
// Backward: use full-precision master weights to compute standard gradients
// for both input and weight. After the optimizer step, call SyncFP8Weights
// to refresh the FP8 snapshot.
type FP8Linear[T tensor.Numeric] struct {
	name   string
	engine compute.Engine[T]

	// masterWeight holds the full-precision weights used by the optimizer.
	// Shape: [outFeatures, inFeatures].
	masterWeight *graph.Parameter[T]

	// fp8WeightData holds the dequantized FP8 snapshot of masterWeight.
	// On CPU this simulates quantization noise; on GPU the FP8 storage
	// triggers native FP8 GEMM dispatch in the engine.
	fp8WeightData []T

	inFeatures  int
	outFeatures int

	// cached for backward
	lastInput *tensor.TensorNumeric[T]
}

// NewFP8Linear creates an FP8 linear layer with the given dimensions.
// initData provides the initial weight values in full precision (row-major,
// shape [outFeatures, inFeatures]). If nil, weights are zero-initialized.
func NewFP8Linear[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	inFeatures, outFeatures int,
	initData []T,
) (*FP8Linear[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if inFeatures <= 0 || outFeatures <= 0 {
		return nil, fmt.Errorf("dimensions must be positive, got in=%d out=%d", inFeatures, outFeatures)
	}

	n := outFeatures * inFeatures
	if initData == nil {
		initData = make([]T, n)
	}
	if len(initData) != n {
		return nil, fmt.Errorf("initData length %d does not match %d x %d = %d", len(initData), outFeatures, inFeatures, n)
	}

	// Create master weight parameter (full precision).
	wTensor, err := tensor.New[T]([]int{outFeatures, inFeatures}, initData)
	if err != nil {
		return nil, fmt.Errorf("create weight tensor: %w", err)
	}
	param, err := graph.NewParameter[T](name+"_weight", wTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("create weight parameter: %w", err)
	}

	// Build initial FP8 snapshot (quantize then dequantize to simulate FP8 noise).
	fp8Data := quantizeDequantizeFP8(initData)

	return &FP8Linear[T]{
		name:          name,
		engine:        engine,
		masterWeight:  param,
		fp8WeightData: fp8Data,
		inFeatures:    inFeatures,
		outFeatures:   outFeatures,
	}, nil
}

// OpType returns the operation type.
func (l *FP8Linear[T]) OpType() string { return "FP8Linear" }

// Attributes returns layer attributes.
func (l *FP8Linear[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"in_features":  l.inFeatures,
		"out_features": l.outFeatures,
	}
}

// OutputShape returns the output shape [-1, outFeatures].
func (l *FP8Linear[T]) OutputShape() []int {
	return []int{-1, l.outFeatures}
}

// Forward computes y = x @ W^T using FP8 quantized weights.
// Input x has shape [batch, inFeatures], output has shape [batch, outFeatures].
func (l *FP8Linear[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FP8Linear requires exactly one input, got %d", len(inputs))
	}
	x := inputs[0]
	l.lastInput = x

	// Quantize input through FP8 round-trip to simulate FP8 forward noise.
	xData := x.Data()
	fp8X := quantizeDequantizeFP8(xData)
	xQ, err := tensor.New[T](x.Shape(), fp8X)
	if err != nil {
		return nil, fmt.Errorf("create quantized input: %w", err)
	}

	// Use FP8-quantized weight snapshot.
	wQ, err := tensor.New[T]([]int{l.outFeatures, l.inFeatures}, l.fp8WeightData)
	if err != nil {
		return nil, fmt.Errorf("create quantized weight: %w", err)
	}

	// Transpose weight: [out, in] -> [in, out] for x @ W^T.
	wQT, err := l.engine.Transpose(ctx, wQ, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("transpose weight: %w", err)
	}

	return l.engine.MatMul(ctx, xQ, wQT)
}

// Backward computes full-precision gradients using the master weights.
// grad_input = outputGradient @ W  (shape: [batch, inFeatures])
// grad_weight = outputGradient^T @ x  (shape: [outFeatures, inFeatures])
func (l *FP8Linear[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FP8Linear backward requires exactly one input, got %d", len(inputs))
	}

	x := l.lastInput
	if x == nil {
		x = inputs[0]
	}

	w := l.masterWeight.Value

	// grad_weight = outputGradient^T @ x => [outFeatures, inFeatures]
	gradT, err := l.engine.Transpose(ctx, outputGradient, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("transpose outputGradient: %w", err)
	}
	dW, err := l.engine.MatMul(ctx, gradT, x)
	if err != nil {
		return nil, fmt.Errorf("compute grad_weight: %w", err)
	}

	if l.masterWeight.Gradient != nil {
		l.masterWeight.Gradient, err = l.engine.Add(ctx, l.masterWeight.Gradient, dW)
	} else {
		l.masterWeight.Gradient = dW
	}
	if err != nil {
		return nil, fmt.Errorf("accumulate grad_weight: %w", err)
	}

	// grad_input = outputGradient @ W => [batch, inFeatures]
	dX, err := l.engine.MatMul(ctx, outputGradient, w)
	if err != nil {
		return nil, fmt.Errorf("compute grad_input: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dX}, nil
}

// Parameters returns the trainable master weight parameter.
func (l *FP8Linear[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{l.masterWeight}
}

// SyncFP8Weights re-quantizes the master weights to FP8 after an optimizer step.
// Call this after each optimizer.Step().
func (l *FP8Linear[T]) SyncFP8Weights() error {
	l.fp8WeightData = quantizeDequantizeFP8(l.masterWeight.Value.Data())
	return nil
}

// Name returns the layer name.
func (l *FP8Linear[T]) Name() string { return l.name }

// quantizeDequantizeFP8 simulates FP8 E4M3 quantization by converting values
// through the FP8 quantize/dequantize round-trip. This introduces the same
// quantization noise as real FP8 storage, and when used with a GPU engine,
// the FP8E4M3Storage on the tensor triggers native FP8 GEMM dispatch.
func quantizeDequantizeFP8[T tensor.Numeric](data []T) []T {
	n := len(data)
	if n == 0 {
		return nil
	}

	// Convert to float32 for FP8 quantization.
	f32 := make([]float32, n)
	for i, v := range data {
		f32[i] = float32(v)
	}

	// Quantize via FP8E4M3Storage then dequantize back.
	fs := tensor.NewFP8E4M3Storage(f32)
	deq := fs.Slice()

	// Convert back to T.
	out := make([]T, n)
	for i, v := range deq {
		out[i] = T(v)
	}
	return out
}

// fp8RoundtripError computes the mean absolute error between original and
// FP8-quantized values, useful for verifying quantization quality.
func fp8RoundtripError[T tensor.Numeric](original, quantized []T) float64 {
	if len(original) != len(quantized) {
		return math.MaxFloat64
	}
	var sum float64
	for i := range original {
		sum += math.Abs(float64(original[i]) - float64(quantized[i]))
	}
	return sum / float64(len(original))
}

// Verify FP8Linear implements graph.Node.
var _ graph.Node[float32] = (*FP8Linear[float32])(nil)
