package features

import (
	"context"
	"fmt"
	"math/cmplx"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
	"gonum.org/v1/gonum/dsp/fourier"
)

// SpectralFingerprint is a feature transformation layer that computes the FFT
// of a time-series input and returns the magnitudes of the top K frequencies.
// This layer is non-trainable and is typically used for feature extraction.
// Because it is non-trainable, it does
// not propagate gradients through this transformation (Backward returns nil),
// effectively treating it as a fixed feature extractor.
type SpectralFingerprint[T tensor.Numeric] struct {
	TopK int
}

// NewSpectralFingerprint creates a new SpectralFingerprint layer.
func NewSpectralFingerprint[T tensor.Numeric](topK int) *SpectralFingerprint[T] {
	return &SpectralFingerprint[T]{TopK: topK}
}

// OpType returns the operation type of the layer.
func (s *SpectralFingerprint[T]) OpType() string {
	return "SpectralFingerprint"
}

// Attributes returns the attributes of the layer.
func (s *SpectralFingerprint[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"output_dim": s.TopK}
}

// OutputShape returns the output shape of the layer.
func (s *SpectralFingerprint[T]) OutputShape() []int {
	return []int{s.TopK}
}

// Forward computes the forward pass of the layer.
func (s *SpectralFingerprint[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SpectralFingerprint requires exactly one input, got %d", len(inputs))
	}

	input := inputs[0]
	if input == nil {
		return nil, fmt.Errorf("input tensor cannot be nil")
	}
	inputData := input.Data()

	// Handle empty input
	if len(inputData) == 0 {
		magnitudes := make([]float64, s.TopK)
		outputTensor, err := tensor.New[T]([]int{s.TopK}, fromFloat64[T](magnitudes))
		if err != nil {
			return nil, fmt.Errorf("failed to create output tensor: %w", err)
		}
		return outputTensor, nil
	}

	// Compute FFT
	fft := fourier.NewFFT(len(inputData))
	coeffs := fft.Coefficients(nil, toFloat64(inputData))

	// Extract top K magnitudes
	magnitudes := make([]float64, s.TopK)
	for i := 1; i <= s.TopK && i < len(coeffs); i++ {
		magnitudes[i-1] = cmplx.Abs(coeffs[i])
	}

	// Create output tensor
	outputTensor, err := tensor.New[T]([]int{s.TopK}, fromFloat64[T](magnitudes))
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	return outputTensor, nil
}

// Backward for a non-trainable layer returns a nil gradient for the input.
func (s *SpectralFingerprint[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SpectralFingerprint requires exactly one input, got %d", len(inputs))
	}

	// Return zero gradient tensor with same shape as input
	input := inputs[0]
	inputShape := input.Shape()
	zeroData := make([]T, input.Size())
	zeroGrad, err := tensor.New[T](inputShape, zeroData)
	if err != nil {
		return nil, fmt.Errorf("failed to create zero gradient tensor: %w", err)
	}

	return []*tensor.TensorNumeric[T]{zeroGrad}, nil
}

// Parameters returns no parameters as this layer is not trainable.
func (s *SpectralFingerprint[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

func toFloat64[T tensor.Numeric](data []T) []float64 {
	floatData := make([]float64, len(data))
	for i, v := range data {
		floatData[i] = float64(any(v).(float32)) // This is a hack, assuming T is float32
	}
	return floatData
}

func fromFloat64[T tensor.Numeric](data []float64) []T {
	typedData := make([]T, len(data))
	for i, v := range data {
		typedData[i] = any(float32(v)).(T) // This is a hack, assuming T is float32
	}
	return typedData
}

func init() {
	model.RegisterLayer("SpectralFingerprint", func(engine compute.Engine[float32], ops numeric.Arithmetic[float32], name string, params map[string]*graph.Parameter[float32], attributes map[string]interface{}) (graph.Node[float32], error) {
		topK, ok := attributes["top_k"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'top_k' for SpectralFingerprint")
		}
		return NewSpectralFingerprint[float32](topK), nil
	})
}
