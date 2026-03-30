package timeseries

import "math"

// adamStateF32 holds first and second moment estimates for AdamW in float32.
type adamStateF32 struct {
	m []float32
	v []float32
}

// clipGradientsF32 clips a gradient vector by L2 norm in-place.
func clipGradientsF32(grad []float32, maxNorm float64) {
	if maxNorm <= 0 {
		return
	}
	var norm float64
	for _, g := range grad {
		norm += float64(g) * float64(g)
	}
	norm = math.Sqrt(norm)
	if norm > maxNorm {
		scale := float32(maxNorm / norm)
		for i := range grad {
			grad[i] *= scale
		}
	}
}

// adamWUpdateF32 applies one AdamW step in-place.
func adamWUpdateF32(params, grads []float32, state *adamStateF32, beta1, beta2, eps, lr, wd float32, t int) {
	bc1 := float32(1.0) - float32(math.Pow(float64(beta1), float64(t)))
	bc2 := float32(1.0) - float32(math.Pow(float64(beta2), float64(t)))

	for i := range params {
		state.m[i] = beta1*state.m[i] + (1-beta1)*grads[i]
		state.v[i] = beta2*state.v[i] + (1-beta2)*grads[i]*grads[i]
		mHat := state.m[i] / bc1
		vHat := state.v[i] / bc2
		params[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + wd*params[i])
	}
}
