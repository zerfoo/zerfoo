package timeseries

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestTimeMixer_Backward_GradientCheck(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    8,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   2,
		HiddenSize:  4,
		NumLayers:   1,
	}
	m := NewTimeMixer(cfg)

	rng := rand.New(rand.NewPCG(2026, 407))
	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = rng.NormFloat64() * 0.5
		}
	}

	target := make([][]float64, cfg.NumFeatures)
	for f := range target {
		target[f] = make([]float64, cfg.OutputLen)
		for i := range target[f] {
			target[f][i] = rng.NormFloat64() * 0.5
		}
	}

	// Compute analytical gradients.
	msOut, cache := m.forwardWithCache(input)
	pred := m.predict(msOut)

	// MSE loss and dL/d(pred).
	nElem := float64(cfg.NumFeatures * cfg.OutputLen)
	dPred := make([][]float64, cfg.NumFeatures)
	for f := 0; f < cfg.NumFeatures; f++ {
		dPred[f] = make([]float64, cfg.OutputLen)
		for i := 0; i < cfg.OutputLen; i++ {
			diff := pred[f][i] - target[f][i]
			dPred[f][i] = 2.0 * diff / nElem
		}
	}

	// Convert dPred to dScales.
	// pred[f][i] = avg over scales of trend[s][f][srcIdx]
	// dL/d(trend[s][f][srcIdx]) += dPred[f][i] / numScales
	numScales := cfg.NumScales
	dScales := make([]scaleDecomposition, numScales)
	for s := 0; s < numScales; s++ {
		dScales[s] = scaleDecomposition{
			trend:    make([][]float64, cfg.NumFeatures),
			seasonal: make([][]float64, cfg.NumFeatures),
		}
		for f := 0; f < cfg.NumFeatures; f++ {
			dScales[s].trend[f] = make([]float64, cfg.InputLen)
			dScales[s].seasonal[f] = make([]float64, cfg.InputLen)
		}
	}

	outLen := cfg.OutputLen
	for f := 0; f < cfg.NumFeatures; f++ {
		for i := 0; i < outLen; i++ {
			srcIdx := cfg.InputLen - outLen + i
			if srcIdx < 0 {
				srcIdx = 0
			}
			for s := 0; s < numScales; s++ {
				dScales[s].trend[f][srcIdx] += dPred[f][i] / float64(numScales)
			}
		}
	}

	grads := newTimeMixerGrads(m)
	m.backward(dScales, cache, &grads)
	analyticalGrads := grads.collectGrads(m)

	// Numerical gradient check via central finite differences.
	params := m.FlatParams()
	nParams := len(params)
	if len(analyticalGrads) != nParams {
		t.Fatalf("grad length mismatch: analytical=%d, params=%d", len(analyticalGrads), nParams)
	}

	// Loss function for finite differences.
	lossFn := func() float64 {
		out, _ := m.Forward(input)
		p := m.predict(&out.MultiScaleOutput)
		loss := 0.0
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.OutputLen; i++ {
				diff := p[f][i] - target[f][i]
				loss += diff * diff
			}
		}
		return loss / nElem
	}

	eps := 1e-5
	maxRelErr := 0.0
	failCount := 0

	for pi := 0; pi < nParams; pi++ {
		orig := *params[pi]

		*params[pi] = orig + eps
		lossPlus := lossFn()

		*params[pi] = orig - eps
		lossMinus := lossFn()

		*params[pi] = orig

		numerical := (lossPlus - lossMinus) / (2.0 * eps)
		analytical := analyticalGrads[pi]

		denom := math.Max(math.Abs(numerical), math.Abs(analytical))
		if math.Abs(analytical) < 1e-12 && math.Abs(numerical) < 1e-6 {
			continue
		}
		if denom < 1e-10 {
			continue
		}
		relErr := math.Abs(analytical-numerical) / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > 1e-3 {
			failCount++
			if failCount <= 5 {
				t.Errorf("param[%d]: analytical=%.8e, numerical=%.8e, relErr=%.4e",
					pi, analytical, numerical, relErr)
			}
		}
	}

	if failCount > 0 {
		t.Errorf("%d/%d parameters exceed 0.1%% relative error", failCount, nParams)
	}
	t.Logf("gradient check: %d params, maxRelErr=%.4e, failures=%d", nParams, maxRelErr, failCount)
}

func TestTimeMixer_Backward_LossReduction(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    8,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   2,
		HiddenSize:  4,
		NumLayers:   1,
	}
	m := NewTimeMixer(cfg)

	rng := rand.New(rand.NewPCG(2026, 407))
	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = rng.NormFloat64() * 0.5
		}
	}

	target := make([][]float64, cfg.NumFeatures)
	for f := range target {
		target[f] = make([]float64, cfg.OutputLen)
		for i := range target[f] {
			target[f][i] = rng.NormFloat64() * 0.5
		}
	}

	nElem := float64(cfg.NumFeatures * cfg.OutputLen)
	numScales := cfg.NumScales

	computeLoss := func() float64 {
		out, _ := m.Forward(input)
		p := m.predict(&out.MultiScaleOutput)
		loss := 0.0
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.OutputLen; i++ {
				diff := p[f][i] - target[f][i]
				loss += diff * diff
			}
		}
		return loss / nElem
	}

	initialLoss := computeLoss()
	lr := 0.01
	steps := 50

	for step := 0; step < steps; step++ {
		msOut, cache := m.forwardWithCache(input)
		pred := m.predict(msOut)

		// Compute dScales from MSE loss.
		outLen := cfg.OutputLen
		dScales := make([]scaleDecomposition, numScales)
		for s := 0; s < numScales; s++ {
			dScales[s] = scaleDecomposition{
				trend:    make([][]float64, cfg.NumFeatures),
				seasonal: make([][]float64, cfg.NumFeatures),
			}
			for f := 0; f < cfg.NumFeatures; f++ {
				dScales[s].trend[f] = make([]float64, cfg.InputLen)
				dScales[s].seasonal[f] = make([]float64, cfg.InputLen)
			}
		}

		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < outLen; i++ {
				srcIdx := cfg.InputLen - outLen + i
				if srcIdx < 0 {
					srcIdx = 0
				}
				diff := pred[f][i] - target[f][i]
				dPred := 2.0 * diff / nElem
				for s := 0; s < numScales; s++ {
					dScales[s].trend[f][srcIdx] += dPred / float64(numScales)
				}
			}
		}

		grads := newTimeMixerGrads(m)
		m.backward(dScales, cache, &grads)
		analyticalGrads := grads.collectGrads(m)

		// SGD update.
		params := m.FlatParams()
		for pi := range params {
			*params[pi] -= lr * analyticalGrads[pi]
		}
	}

	finalLoss := computeLoss()
	if finalLoss >= initialLoss {
		t.Errorf("loss did not decrease: initial=%.6f, final=%.6f", initialLoss, finalLoss)
	}
	t.Logf("loss reduction: %.6f -> %.6f (%.1f%%)", initialLoss, finalLoss, 100*(1-finalLoss/initialLoss))
}

func TestTimeMixer_Backward_MultiLayer(t *testing.T) {
	// HiddenSize=16: with smaller HiddenSize (e.g. 4), the mixing MLP's hidden
	// ReLU neurons regularly saturate (preReLU < 0 for all positions of some
	// neurons). The analytical gradient correctly reports 0 for those dead
	// neurons' biases, but a finite-difference perturbation of size eps can
	// "wake" them and produce a non-zero numerical gradient, causing spurious
	// failures that look like a backward bug. With HiddenSize=16 the dead-
	// neuron pattern is rare enough that the analytical and numerical
	// gradients agree across all tested seeds. See issue #351 for the full
	// investigation.
	cfg := TimeMixerConfig{
		InputLen:    8,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   3,
		HiddenSize:  16,
		NumLayers:   2,
	}
	// Deterministic RNG so this gradient check is reproducible across runs.
	initRNG := rand.New(rand.NewPCG(2026, 407))
	m := NewTimeMixer(cfg, WithTimeMixerRNG(initRNG))

	rng := rand.New(rand.NewPCG(2026, 407))
	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = rng.NormFloat64() * 0.5
		}
	}

	target := make([][]float64, cfg.NumFeatures)
	for f := range target {
		target[f] = make([]float64, cfg.OutputLen)
		for i := range target[f] {
			target[f][i] = rng.NormFloat64() * 0.5
		}
	}

	nElem := float64(cfg.NumFeatures * cfg.OutputLen)
	numScales := cfg.NumScales

	// Compute analytical gradients.
	msOut, cache := m.forwardWithCache(input)
	pred := m.predict(msOut)

	outLen := cfg.OutputLen
	dScales := make([]scaleDecomposition, numScales)
	for s := 0; s < numScales; s++ {
		dScales[s] = scaleDecomposition{
			trend:    make([][]float64, cfg.NumFeatures),
			seasonal: make([][]float64, cfg.NumFeatures),
		}
		for f := 0; f < cfg.NumFeatures; f++ {
			dScales[s].trend[f] = make([]float64, cfg.InputLen)
			dScales[s].seasonal[f] = make([]float64, cfg.InputLen)
		}
	}

	for f := 0; f < cfg.NumFeatures; f++ {
		for i := 0; i < outLen; i++ {
			srcIdx := cfg.InputLen - outLen + i
			if srcIdx < 0 {
				srcIdx = 0
			}
			diff := pred[f][i] - target[f][i]
			dPred := 2.0 * diff / nElem
			for s := 0; s < numScales; s++ {
				dScales[s].trend[f][srcIdx] += dPred / float64(numScales)
			}
		}
	}

	grads := newTimeMixerGrads(m)
	m.backward(dScales, cache, &grads)
	analyticalGrads := grads.collectGrads(m)

	// Numerical gradient check.
	params := m.FlatParams()
	nParams := len(params)
	if len(analyticalGrads) != nParams {
		t.Fatalf("grad length mismatch: analytical=%d, params=%d", len(analyticalGrads), nParams)
	}

	lossFn := func() float64 {
		out, _ := m.Forward(input)
		p := m.predict(&out.MultiScaleOutput)
		loss := 0.0
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.OutputLen; i++ {
				diff := p[f][i] - target[f][i]
				loss += diff * diff
			}
		}
		return loss / nElem
	}

	eps := 1e-5
	maxRelErr := 0.0
	failCount := 0

	for pi := 0; pi < nParams; pi++ {
		orig := *params[pi]

		*params[pi] = orig + eps
		lossPlus := lossFn()

		*params[pi] = orig - eps
		lossMinus := lossFn()

		*params[pi] = orig

		numerical := (lossPlus - lossMinus) / (2.0 * eps)
		analytical := analyticalGrads[pi]

		denom := math.Max(math.Abs(numerical), math.Abs(analytical))
		if math.Abs(analytical) < 1e-12 && math.Abs(numerical) < 1e-6 {
			continue
		}
		if denom < 1e-10 {
			continue
		}
		relErr := math.Abs(analytical-numerical) / denom
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > 1e-3 {
			failCount++
			if failCount <= 5 {
				t.Errorf("param[%d]: analytical=%.8e, numerical=%.8e, relErr=%.4e",
					pi, analytical, numerical, relErr)
			}
		}
	}

	if failCount > 0 {
		t.Errorf("%d/%d parameters exceed 0.1%% relative error", failCount, nParams)
	}
	t.Logf("gradient check (multi-layer): %d params, maxRelErr=%.4e, failures=%d", nParams, maxRelErr, failCount)
}

func TestTimeMixer_ForwardWithCache_Parity(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    8,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   3,
		HiddenSize:  4,
		NumLayers:   2,
	}
	m := NewTimeMixer(cfg)

	rng := rand.New(rand.NewPCG(2026, 407))
	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = rng.NormFloat64() * 0.5
		}
	}

	// Forward without cache.
	out1, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Forward with cache.
	out2, _ := m.forwardWithCache(input)

	// Should produce identical results.
	for s := range out1.Scales {
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.InputLen; i++ {
				if math.Abs(out1.Scales[s].trend[f][i]-out2.Scales[s].trend[f][i]) > 1e-12 {
					t.Errorf("scale %d feature %d index %d: trend mismatch: %.15e vs %.15e",
						s, f, i, out1.Scales[s].trend[f][i], out2.Scales[s].trend[f][i])
				}
				if math.Abs(out1.Scales[s].seasonal[f][i]-out2.Scales[s].seasonal[f][i]) > 1e-12 {
					t.Errorf("scale %d feature %d index %d: seasonal mismatch: %.15e vs %.15e",
						s, f, i, out1.Scales[s].seasonal[f][i], out2.Scales[s].seasonal[f][i])
				}
			}
		}
	}
}
