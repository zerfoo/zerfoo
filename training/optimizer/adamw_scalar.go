package optimizer

import "math"

// AdamWStateF64 holds first and second moment estimates for scalar AdamW updates.
type AdamWStateF64 struct {
	M []float64 // first moment
	V []float64 // second moment
}

// NewAdamWStateF64 creates a new scalar AdamW state for nParams parameters.
func NewAdamWStateF64(nParams int) *AdamWStateF64 {
	return &AdamWStateF64{
		M: make([]float64, nParams),
		V: make([]float64, nParams),
	}
}

// AdamWUpdateF64 applies one AdamW parameter update step on raw float64 slices.
// t is the global step count (1-indexed for bias correction).
func AdamWUpdateF64(params []*float64, grads []float64, state *AdamWStateF64, lr, beta1, beta2, epsilon, weightDecay, t float64) {
	bc1 := 1 - math.Pow(beta1, t)
	bc2 := 1 - math.Pow(beta2, t)

	for i := range params {
		state.M[i] = beta1*state.M[i] + (1-beta1)*grads[i]
		state.V[i] = beta2*state.V[i] + (1-beta2)*grads[i]*grads[i]
		mHat := state.M[i] / bc1
		vHat := state.V[i] / bc2
		*params[i] = *params[i] - lr*(mHat/(math.Sqrt(vHat)+epsilon)+weightDecay*(*params[i]))
	}
}

// ClipGradientsF64 applies global norm gradient clipping in-place.
// If the L2 norm of grads exceeds maxNorm, all gradients are scaled down
// proportionally. A maxNorm of 0 disables clipping.
func ClipGradientsF64(grads []float64, maxNorm float64) {
	if maxNorm <= 0 {
		return
	}
	norm := 0.0
	for _, g := range grads {
		norm += g * g
	}
	norm = math.Sqrt(norm)
	if norm > maxNorm {
		scale := maxNorm / norm
		for i := range grads {
			grads[i] *= scale
		}
	}
}
