package timeseries

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The bespoke finite-difference gradient checks that used to live here
// (TestTimeMixer_Backward_GradientCheck, TestTimeMixer_Backward_MultiLayer)
// were migrated to ztensor's shared gradcheck harness; see
// TestTimeseriesBackward_Gradcheck in gradcheck_test.go (plan T1.6).

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
