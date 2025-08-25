package features

import (
	"context"
	"fmt"
	"math/cmplx"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
	"gonum.org/v1/gonum/dsp/fourier"
)

// SpectralFingerprint computes a spectral fingerprint of a time series using FFT.
// It is a non-trainable layer.
type SpectralFingerprint[T tensor.Float] struct {
	outputDim int
}

// NewSpectralFingerprint creates a new SpectralFingerprint layer.
func NewSpectralFingerprint[T tensor.Float](outputDim int) *SpectralFingerprint[T] {
	return &SpectralFingerprint[T]{
		outputDim: outputDim,
	}
}

// OutputShape returns the output shape of the layer.
func (s *SpectralFingerprint[T]) OutputShape() []int {
	return []int{1, s.outputDim}
}

// Forward computes the spectral fingerprint.
// Input is expected to be a 1D tensor (or a batch of 1D tensors).
func (s *SpectralFingerprint[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SpectralFingerprint: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	input := inputs[0]
	inputData := input.Data()

	// Convert input to float64 for FFT
	float64Data := make([]float64, len(inputData))
	for i, v := range inputData {
		float64Data[i] = float64(v)
	}

	fft := fourier.NewFFT(len(float64Data))
	coeffs := fft.Coefficients(nil, float64Data)

	fingerprint := make([]T, s.outputDim)
	for i := 1; i <= s.outputDim && i < len(coeffs); i++ {
		fingerprint[i-1] = T(cmplx.Abs(coeffs[i]))
	}

	output, err := tensor.New[T](s.OutputShape(), fingerprint)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	return output, nil
}

// Backward for a non-trainable layer returns a nil gradient for the input.
func (s *SpectralFingerprint[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// The gradient of this operation is not straightforward to compute and may not be useful.
	// For now, we return a zero gradient.
	inputShape := inputs[0].Shape()
	grad, err := tensor.New[T](inputShape, make([]T, inputs[0].Size()))
	if err != nil {
		return nil, fmt.Errorf("failed to create gradient tensor: %w", err)
	}
	return []*tensor.TensorNumeric[T]{grad}, nil
}

// Parameters returns nil as this layer is not trainable.
func (s *SpectralFingerprint[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// OpType returns the operation type of the layer.
func (s *SpectralFingerprint[T]) OpType() string {
	return "SpectralFingerprint"
}

// Attributes returns the attributes of the layer.
func (s *SpectralFingerprint[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"output_dim": s.outputDim,
	}
}

// Ensure SpectralFingerprint implements the graph.Node interface.
var _ graph.Node[float32] = (*SpectralFingerprint[float32])(nil)
