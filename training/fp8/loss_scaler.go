// Package fp8 provides FP8 mixed-precision training layers.
package fp8

import "math"

// LossScaler implements dynamic loss scaling for FP8 mixed-precision training.
// It scales the loss before backpropagation to prevent gradient underflow in
// low-precision formats, and dynamically adjusts the scale factor based on
// whether overflow (inf/NaN) is detected in the resulting gradients.
type LossScaler struct {
	Scale        float64
	GrowInterval int // steps between scale doublings (default: 2000)

	stepsSinceGrow int
}

// NewLossScaler creates a LossScaler with the given initial scale factor.
// GrowInterval defaults to 2000.
func NewLossScaler(initialScale float64) *LossScaler {
	if initialScale < 1.0 {
		initialScale = 1.0
	}
	return &LossScaler{
		Scale:        initialScale,
		GrowInterval: 2000,
	}
}

// ScaleLoss returns loss multiplied by the current scale factor.
func (ls *LossScaler) ScaleLoss(loss float64) float64 {
	return loss * ls.Scale
}

// CheckGradients inspects all gradient values for inf or NaN. If any are
// found, it halves the scale (with a floor of 1.0) and returns false.
// Returns true if all gradients are finite.
func (ls *LossScaler) CheckGradients(grads [][]float32) bool {
	for _, g := range grads {
		for _, v := range g {
			if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) {
				ls.Scale /= 2
				if ls.Scale < 1.0 {
					ls.Scale = 1.0
				}
				return false
			}
		}
	}
	return true
}

// Update advances the step counter. If hadOverflow is true, the counter resets.
// After GrowInterval consecutive steps without overflow, the scale is doubled.
func (ls *LossScaler) Update(hadOverflow bool) {
	if hadOverflow {
		ls.stepsSinceGrow = 0
		return
	}
	ls.stepsSinceGrow++
	if ls.stepsSinceGrow >= ls.GrowInterval {
		ls.Scale *= 2
		ls.stepsSinceGrow = 0
	}
}

// UnscaleGradients divides all gradient values by the current scale factor,
// reversing the effect of ScaleLoss on the gradient magnitudes.
func (ls *LossScaler) UnscaleGradients(grads [][]float32) {
	invScale := float32(1.0 / ls.Scale)
	for i := range grads {
		for j := range grads[i] {
			grads[i][j] *= invScale
		}
	}
}
