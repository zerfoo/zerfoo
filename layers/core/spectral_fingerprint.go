// Package core provides core neural network layer implementations.
package core

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// SpectralFingerprint computes FFT/DFT magnitude features over a fixed window.
//
// Input shape:  [batch, window]
// Output shape: [batch, topK] where bins 1..topK (non-DC) magnitudes are returned
// (if topK >= window, bins beyond window-1 are returned as zeros).
//
// The layer is stateless and intended primarily for feature engineering. We do
// not propagate gradients through this transformation (Backward returns nil),
// treating it as a non-differentiable pre-processing step.
//
// For generalization to higher ranks or different axes, extend this layer as
// needed. This initial implementation focuses on the common case of a 2D input
// with a single spectral axis equal to `window`.
//
// Note: Computation is performed in the numeric type T using provided ops.
// Cos/Sin are computed in float64 then converted into T using ops.FromFloat64.
// Magnitude is computed as sqrt(re^2 + im^2) in T using ops.
//
// OpType: "SpectralFingerprint"
// Attributes: {"window": int, "top_k": int}
// Parameters: none
//
// Example
//   in:  [N, W]
//   out: [N, K] with out[:, k-1] = |DFT(series)[k]| for k in 1..K
//
// DFT definition used:
//   X[k] = sum_{n=0}^{W-1} x[n] * exp(-j*2*pi*k*n/W)
//   |X[k]| = sqrt(Re^2 + Im^2)
//
// where we explicitly compute cos/sin components.

// SpectralFingerprint implements a layer that extracts the top-K frequency components
// from the input using FFT, creating a fixed-size spectral signature for variable-length inputs.
type SpectralFingerprint[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	window      int
	topK        int
	outputShape []int
}

// NewSpectralFingerprint creates a new SpectralFingerprint layer.
func NewSpectralFingerprint[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], window, topK int) (*SpectralFingerprint[T], error) {
	if window <= 1 {
		return nil, fmt.Errorf("window must be > 1, got %d", window)
	}

	if topK <= 0 {
		return nil, fmt.Errorf("topK must be > 0, got %d", topK)
	}

	return &SpectralFingerprint[T]{
		engine: engine,
		ops:    ops,
		window: window,
		topK:   topK,
	}, nil
}

// OutputShape returns the last computed output shape.
func (s *SpectralFingerprint[T]) OutputShape() []int { return s.outputShape }

// Parameters returns no trainable parameters for SpectralFingerprint.
func (s *SpectralFingerprint[T]) Parameters() []*graph.Parameter[T] { return nil }

// OpType returns the operation type.
func (s *SpectralFingerprint[T]) OpType() string { return "SpectralFingerprint" }

// Attributes returns the attributes of the layer.
func (s *SpectralFingerprint[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"window": s.window,
		"top_k":  s.topK,
	}
}

// Forward computes spectral magnitudes for bins 1..topK for each row in the batch.
// Input must be [batch, window]. If input window dimension is larger than the configured
// window, only the last `window` elements are used. If smaller, an error is returned.
func (s *SpectralFingerprint[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("SpectralFingerprint expects exactly 1 input, got %d", len(inputs))
	}

	in := inputs[0]

	shape := in.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("input must be 2D [batch, window], got shape %v", shape)
	}

	batch := shape[0]

	w := shape[1]
	if w < s.window {
		return nil, fmt.Errorf("input window (%d) smaller than configured window (%d)", w, s.window)
	}

	// We evaluate over the last s.window points of each row
	start := w - s.window
	outShape := []int{batch, s.topK}
	outData := make([]T, batch*s.topK)
	inData := in.Data()

	// Compute DFT magnitudes in type T using ops
	for b := range batch {
		rowStart := b*w + start
		// For k = 1..topK
		for k := 1; k <= s.topK; k++ {
			var re, im T
			// If k >= s.window, magnitude is zero
			if k >= s.window {
				outData[b*s.topK+(k-1)] = s.ops.FromFloat64(0)
				continue
			}

			for n := range s.window {
				angle := -2 * math.Pi * float64(k) * float64(n) / float64(s.window)
				c := s.ops.FromFloat64(math.Cos(angle))
				sn := s.ops.FromFloat64(math.Sin(angle))
				x := inData[rowStart+n]
				re = s.ops.Add(re, s.ops.Mul(x, c))
				im = s.ops.Add(im, s.ops.Mul(x, sn))
			}

			re2 := s.ops.Mul(re, re)
			im2 := s.ops.Mul(im, im)
			mag := s.ops.Sqrt(s.ops.Add(re2, im2))
			outData[b*s.topK+(k-1)] = mag
		}
	}

	out, err := tensor.New[T](outShape, outData)
	if err != nil {
		return nil, err
	}

	s.outputShape = outShape

	return out, nil
}

// Backward returns no gradients (treated as non-differentiable feature transform).
func (s *SpectralFingerprint[T]) Backward(_ context.Context, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Statically assert that the type implements the graph.Node interface.
