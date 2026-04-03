package crossasset

import "math"

// cpuAdamState holds per-parameter first and second moment estimates for AdamW.
type cpuAdamState struct {
	params []adamSlice // one per parameter slice
}

type adamSlice struct {
	m []float64 // first moment
	v []float64 // second moment
}

// newAdamState allocates AdamW state for all model parameters.
func newAdamState(model *Model) *cpuAdamState {
	var slices []adamSlice

	// Head.
	slices = append(slices, newAdamSlice(len(model.headW)))
	slices = append(slices, newAdamSlice(len(model.headB)))

	// Layers.
	for _, l := range model.layers {
		slices = append(slices,
			newAdamSlice(len(l.qW)),
			newAdamSlice(len(l.kW)),
			newAdamSlice(len(l.vW)),
			newAdamSlice(len(l.outW)),
			newAdamSlice(len(l.lnGamma)),
			newAdamSlice(len(l.lnBeta)),
			newAdamSlice(len(l.ffnW1)),
			newAdamSlice(len(l.ffnB1)),
			newAdamSlice(len(l.ffnW2)),
			newAdamSlice(len(l.ffnB2)),
			newAdamSlice(len(l.ffnGamma)),
			newAdamSlice(len(l.ffnBeta)),
		)
	}

	// Input projections.
	for s := range model.inputW {
		slices = append(slices, newAdamSlice(len(model.inputW[s])))
		slices = append(slices, newAdamSlice(len(model.inputB[s])))
	}

	return &cpuAdamState{params: slices}
}

func newAdamSlice(n int) adamSlice {
	return adamSlice{m: make([]float64, n), v: make([]float64, n)}
}

// adamWUpdateAll applies AdamW to all model parameters.
func adamWUpdateAll(
	lr float64, step int,
	model *Model,
	dHeadW, dHeadB []float64,
	dLayers []layer,
	dInputW, dInputB [][]float64,
	state *cpuAdamState,
) {
	const (
		beta1       = 0.9
		beta2       = 0.999
		eps         = 1e-8
		weightDecay = 0.01
	)

	idx := 0
	adamW := func(params, grads []float64, s *adamSlice) {
		// Bias correction.
		bc1 := 1.0 - math.Pow(beta1, float64(step))
		bc2 := 1.0 - math.Pow(beta2, float64(step))

		for i := range params {
			g := grads[i]
			// Gradient clipping.
			if g > 1.0 {
				g = 1.0
			} else if g < -1.0 {
				g = -1.0
			}

			s.m[i] = beta1*s.m[i] + (1-beta1)*g
			s.v[i] = beta2*s.v[i] + (1-beta2)*g*g

			mHat := s.m[i] / bc1
			vHat := s.v[i] / bc2

			// AdamW: weight decay applied to param directly, not through gradient.
			params[i] -= lr * (mHat/(math.Sqrt(vHat)+eps) + weightDecay*params[i])
		}
	}

	// Head.
	adamW(model.headW, dHeadW, &state.params[idx])
	idx++
	adamW(model.headB, dHeadB, &state.params[idx])
	idx++

	// Layers.
	for li := range model.layers {
		l := &model.layers[li]
		dl := &dLayers[li]
		adamW(l.qW, dl.qW, &state.params[idx])
		idx++
		adamW(l.kW, dl.kW, &state.params[idx])
		idx++
		adamW(l.vW, dl.vW, &state.params[idx])
		idx++
		adamW(l.outW, dl.outW, &state.params[idx])
		idx++
		adamW(l.lnGamma, dl.lnGamma, &state.params[idx])
		idx++
		adamW(l.lnBeta, dl.lnBeta, &state.params[idx])
		idx++
		adamW(l.ffnW1, dl.ffnW1, &state.params[idx])
		idx++
		adamW(l.ffnB1, dl.ffnB1, &state.params[idx])
		idx++
		adamW(l.ffnW2, dl.ffnW2, &state.params[idx])
		idx++
		adamW(l.ffnB2, dl.ffnB2, &state.params[idx])
		idx++
		adamW(l.ffnGamma, dl.ffnGamma, &state.params[idx])
		idx++
		adamW(l.ffnBeta, dl.ffnBeta, &state.params[idx])
		idx++
	}

	// Input projections.
	for s := range model.inputW {
		adamW(model.inputW[s], dInputW[s], &state.params[idx])
		idx++
		adamW(model.inputB[s], dInputB[s], &state.params[idx])
		idx++
	}
}
