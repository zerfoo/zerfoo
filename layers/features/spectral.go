package features

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// SpectralFingerprint is a feature transformation layer that computes the DFT
// of a time-series input and returns the magnitudes of the top K frequencies.
// This layer is non-trainable and is typically used for feature extraction.
// Because it is non-trainable, it does
// not propagate gradients through this transformation (Backward returns nil),
// effectively treating it as a fixed feature extractor.
//
// Computation uses engine primitives (MatMul, Mul, Add, Sqrt) with precomputed
// DFT basis matrices, making the layer fully traceable by the tracing compiler.
type SpectralFingerprint[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	TopK   int

	// Lazily computed DFT basis matrices for the last seen input length.
	cachedN   int
	cosBasis  *tensor.TensorNumeric[T] // [TopK, N]
	sinBasis  *tensor.TensorNumeric[T] // [TopK, N]
}

// NewSpectralFingerprint creates a new SpectralFingerprint layer.
func NewSpectralFingerprint[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], topK int) *SpectralFingerprint[T] {
	return &SpectralFingerprint[T]{engine: engine, ops: ops, TopK: topK}
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

// ensureBasis precomputes the DFT cosine and sine basis matrices for the given
// number of rows and input length n. Caches the result so it's only recomputed
// when n or rows changes.
func (s *SpectralFingerprint[T]) ensureBasis(rows, n int) error {
	if n == s.cachedN && s.cosBasis != nil {
		return nil
	}

	cosData := make([]T, rows*n)
	sinData := make([]T, rows*n)

	for k := 0; k < rows; k++ {
		freq := k + 1 // bins 1..rows (skip DC)
		for j := 0; j < n; j++ {
			angle := -2 * math.Pi * float64(freq) * float64(j) / float64(n)
			cosData[k*n+j] = s.ops.FromFloat64(math.Cos(angle))
			sinData[k*n+j] = s.ops.FromFloat64(math.Sin(angle))
		}
	}

	var err error
	s.cosBasis, err = tensor.New[T]([]int{rows, n}, cosData)
	if err != nil {
		return fmt.Errorf("create cos basis: %w", err)
	}
	s.sinBasis, err = tensor.New[T]([]int{rows, n}, sinData)
	if err != nil {
		return fmt.Errorf("create sin basis: %w", err)
	}
	s.cachedN = n
	return nil
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

	n := input.Size()

	// Handle empty input
	if n == 0 {
		zeroData := make([]T, s.TopK)
		return tensor.New[T]([]int{s.TopK}, zeroData)
	}

	// Number of usable frequency bins: 1..min(TopK, N/2).
	// For real signals, only N/2 non-DC frequency bins are independent.
	usable := s.TopK
	if n/2 < usable {
		usable = n / 2
	}
	if usable < 0 {
		usable = 0
	}

	// If no usable bins, return zeros.
	if usable == 0 {
		zeroData := make([]T, s.TopK)
		return tensor.New[T]([]int{s.TopK}, zeroData)
	}

	// Ensure input is a column vector [N, 1] for MatMul.
	col, err := s.engine.Reshape(ctx, input, []int{n, 1})
	if err != nil {
		return nil, fmt.Errorf("reshape input: %w", err)
	}

	if err := s.ensureBasis(usable, n); err != nil {
		return nil, err
	}

	// Real part: cosBasis [usable, N] x col [N, 1] = [usable, 1]
	re, err := s.engine.MatMul(ctx, s.cosBasis, col)
	if err != nil {
		return nil, fmt.Errorf("matmul cos: %w", err)
	}

	// Imaginary part: sinBasis [usable, N] x col [N, 1] = [usable, 1]
	im, err := s.engine.MatMul(ctx, s.sinBasis, col)
	if err != nil {
		return nil, fmt.Errorf("matmul sin: %w", err)
	}

	// re² = re * re
	re2, err := s.engine.Mul(ctx, re, re)
	if err != nil {
		return nil, fmt.Errorf("mul re²: %w", err)
	}

	// im² = im * im
	im2, err := s.engine.Mul(ctx, im, im)
	if err != nil {
		return nil, fmt.Errorf("mul im²: %w", err)
	}

	// magnitude² = re² + im²
	mag2, err := s.engine.Add(ctx, re2, im2)
	if err != nil {
		return nil, fmt.Errorf("add mag²: %w", err)
	}

	// magnitude = sqrt(magnitude²)
	mag, err := s.engine.Sqrt(ctx, mag2)
	if err != nil {
		return nil, fmt.Errorf("sqrt: %w", err)
	}

	// Reshape to flat [usable]
	mag, err = s.engine.Reshape(ctx, mag, []int{usable})
	if err != nil {
		return nil, fmt.Errorf("reshape mag: %w", err)
	}

	// Pad with zeros if usable < TopK.
	if usable < s.TopK {
		padData := make([]T, s.TopK-usable)
		pad, pErr := tensor.New[T]([]int{s.TopK - usable}, padData)
		if pErr != nil {
			return nil, fmt.Errorf("create pad: %w", pErr)
		}
		result, cErr := s.engine.Concat(ctx, []*tensor.TensorNumeric[T]{mag, pad}, 0)
		if cErr != nil {
			return nil, fmt.Errorf("concat pad: %w", cErr)
		}
		return result, nil
	}

	// Reshape to [TopK] (flat output)
	result, err := s.engine.Reshape(ctx, mag, []int{s.TopK})
	if err != nil {
		return nil, fmt.Errorf("reshape output: %w", err)
	}

	return result, nil
}

// Backward for a non-trainable layer returns a nil gradient for the input.
func (s *SpectralFingerprint[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
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

func init() {
	model.RegisterLayer("SpectralFingerprint", func(engine compute.Engine[float32], ops numeric.Arithmetic[float32], name string, params map[string]*graph.Parameter[float32], attributes map[string]interface{}) (graph.Node[float32], error) {
		topK, ok := attributes["top_k"].(int)
		if !ok {
			return nil, fmt.Errorf("missing or invalid attribute 'top_k' for SpectralFingerprint")
		}
		return NewSpectralFingerprint[float32](engine, ops, topK), nil
	})
}
