package timeseries

import "math"

// clipGradients applies global norm gradient clipping in-place.
// If the L2 norm of grads exceeds maxNorm, all gradients are scaled down
// proportionally. A maxNorm of 0 disables clipping.
func clipGradients(grads []float64, maxNorm float64) {
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

// adamWState holds first and second moment estimates for AdamW.
type adamWState struct {
	m []float64 // first moment
	v []float64 // second moment
}

// newAdamWState creates a new AdamW state for nParams parameters.
func newAdamWState(nParams int) *adamWState {
	return &adamWState{
		m: make([]float64, nParams),
		v: make([]float64, nParams),
	}
}

// adamWUpdate applies one AdamW parameter update step.
// t is the global step count (1-indexed for bias correction).
func adamWUpdate(params []*float64, grads []float64, state *adamWState, lr float64, config TrainConfig, t float64) {
	bc1 := 1 - math.Pow(config.Beta1, t)
	bc2 := 1 - math.Pow(config.Beta2, t)

	for i := range params {
		state.m[i] = config.Beta1*state.m[i] + (1-config.Beta1)*grads[i]
		state.v[i] = config.Beta2*state.v[i] + (1-config.Beta2)*grads[i]*grads[i]
		mHat := state.m[i] / bc1
		vHat := state.v[i] / bc2
		// AdamW: weight decay applied to param directly, not through gradient.
		*params[i] = *params[i] - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*params[i]))
	}
}

// mseLossFlat computes MSE loss and gradient for flat prediction/target vectors.
// Returns (loss, dPred) where dPred[i] = 2*(pred[i]-target[i])/n.
func mseLossFlat(pred, target []float64) (float64, []float64) {
	n := len(pred)
	loss := 0.0
	dPred := make([]float64, n)
	for i := range pred {
		diff := pred[i] - target[i]
		loss += diff * diff
		dPred[i] = 2 * diff / float64(n)
	}
	loss /= float64(n)
	return loss, dPred
}

// warmupLR returns the effective learning rate for the given epoch,
// applying linear warmup over the first warmupEpochs epochs.
func warmupLR(baseLR float64, epoch, warmupEpochs int) float64 {
	if warmupEpochs <= 0 {
		return baseLR
	}
	scale := float64(epoch+1) / float64(warmupEpochs)
	if scale > 1.0 {
		scale = 1.0
	}
	return baseLR * scale
}
