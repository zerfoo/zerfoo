package timeseries

import (
	"math"
	"testing"
)

// TestAllBackends_OverfitTinyDataset verifies that every backend can memorise
// a tiny dataset where the labels are a deterministic function of the inputs
// (per-channel mean, scaled to [0, 0.1]). If a backend cannot drive the loss
// well below its initial value it likely has a broken gradient path.
func TestAllBackends_OverfitTinyDataset(t *testing.T) {
	const (
		channels  = 5
		inputLen  = 24
		outputLen = 6
		nSamples  = 5
		epochs    = 200
	)

	windows := makeSyntheticWindows(nSamples, channels, inputLen)

	for _, bc := range allBackends() {
		t.Run(bc.name, func(t *testing.T) {
			// Build labels as a simple function of inputs so the mapping is
			// learnable. Scale labels into [0, 0.1] to keep targets small and
			// avoid architectures whose output range is narrow from exploding.
			totalLabels := bc.labelsLen(nSamples)
			labels := make([]float64, totalLabels)

			if totalLabels == nSamples*channels*outputLen {
				// Per-sample, per-channel labels.
				idx := 0
				for i := 0; i < nSamples; i++ {
					for c := 0; c < channels; c++ {
						mean := channelMean(windows[i][c]) * 0.1
						for o := 0; o < outputLen; o++ {
							labels[idx] = mean
							idx++
						}
					}
				}
			} else {
				// Per-sample labels (NHiTS / PatchTST style).
				idx := 0
				for i := 0; i < nSamples; i++ {
					mean := sampleMean(windows[i]) * 0.1
					for o := 0; o < outputLen; o++ {
						labels[idx] = mean
						idx++
					}
				}
			}

			cfg := TrainConfig{
				Epochs:       epochs,
				LR:           1e-3,
				WeightDecay:  0,
				GradClip:     1.0,
				WarmupEpochs: 5,
				Beta1:        0.9,
				Beta2:        0.999,
				Epsilon:      1e-8,
			}

			result, _, _, err := bc.train(windows, labels, cfg)
			if err != nil {
				t.Fatalf("TrainWindowed: %v", err)
			}

			if len(result.LossHistory) < 2 {
				t.Fatal("LossHistory has fewer than 2 entries")
			}

			firstLoss := result.LossHistory[0]
			finalLoss := result.LossHistory[len(result.LossHistory)-1]
			if math.IsNaN(finalLoss) || math.IsInf(finalLoss, 0) {
				t.Fatalf("final loss is not finite: %v", finalLoss)
			}

			// The model must reduce loss by at least 50 %.
			if finalLoss >= firstLoss*0.5 {
				t.Errorf("failed to overfit: first loss = %.6f, final loss = %.6f (want at least 50%% reduction)", firstLoss, finalLoss)
			}
		})
	}
}

// TestAllBackends_NumericalStability trains each backend on multi-scale data
// (features spanning 10 orders of magnitude) and asserts that the loss remains
// finite and decreases. This catches missing normalisation or overflow bugs.
func TestAllBackends_NumericalStability(t *testing.T) {
	const (
		channels  = 5
		inputLen  = 24
		outputLen = 6
		nSamples  = 20
		epochs    = 10
	)

	for _, bc := range allBackends() {
		t.Run(bc.name, func(t *testing.T) {
			samples := nSamples
			if bc.samples > 0 && bc.samples < samples {
				samples = bc.samples
			}

			outputDim := bc.labelsLen(1) // labels per sample
			windows, labels := makeMultiScaleWindows(samples, channels, inputLen, outputDim)
			// Trim labels to exact expected length.
			expectedLen := bc.labelsLen(samples)
			if len(labels) > expectedLen {
				labels = labels[:expectedLen]
			}

			cfg := TrainConfig{
				Epochs:       epochs,
				LR:           1e-4, // conservative LR for stability
				WeightDecay:  1e-4,
				GradClip:     1.0,
				WarmupEpochs: 2,
				Beta1:        0.9,
				Beta2:        0.999,
				Epsilon:      1e-8,
			}

			result, _, _, err := bc.train(windows, labels, cfg)
			if err != nil {
				t.Fatalf("TrainWindowed: %v", err)
			}

			if len(result.LossHistory) < 2 {
				t.Fatalf("LossHistory has %d entries, want >= 2", len(result.LossHistory))
			}

			// Assert all losses are finite.
			for i, loss := range result.LossHistory {
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					t.Fatalf("LossHistory[%d] = %v, want finite", i, loss)
				}
			}

			// Assert loss decreases (first vs last).
			first := result.LossHistory[0]
			last := result.LossHistory[len(result.LossHistory)-1]
			if last >= first {
				t.Errorf("loss did not decrease on multi-scale data: first=%.6f last=%.6f", first, last)
			}
		})
	}
}

// TestAllBackends_PredictionDeterminism verifies that running PredictWindowed
// twice on the same trained model with the same input produces bit-identical
// output. Non-determinism would indicate uninitialised memory or accidental
// randomness in the forward pass.
func TestAllBackends_PredictionDeterminism(t *testing.T) {
	const (
		channels  = 5
		inputLen  = 24
		outputLen = 6
	)

	for _, bc := range allBackends() {
		t.Run(bc.name, func(t *testing.T) {
			nSamples := bc.samples
			if nSamples == 0 {
				nSamples = 20
			}
			epochs := bc.epochs
			if epochs == 0 {
				epochs = 30
			}

			windows := makeSyntheticWindows(nSamples, channels, inputLen)
			labels := makeSyntheticLabels(bc.labelsLen(nSamples))

			cfg := TrainConfig{
				Epochs:       epochs,
				LR:           1e-3,
				WeightDecay:  1e-4,
				GradClip:     1.0,
				WarmupEpochs: 3,
				Beta1:        0.9,
				Beta2:        0.999,
				Epsilon:      1e-8,
			}

			_, predFn, _, err := bc.train(windows, labels, cfg)
			if err != nil {
				t.Fatalf("TrainWindowed: %v", err)
			}

			testWindows := windows[:3]

			preds1, err := predFn(testWindows)
			if err != nil {
				t.Fatalf("PredictWindowed (run 1): %v", err)
			}
			preds2, err := predFn(testWindows)
			if err != nil {
				t.Fatalf("PredictWindowed (run 2): %v", err)
			}

			if len(preds1) != len(preds2) {
				t.Fatalf("prediction lengths differ: %d vs %d", len(preds1), len(preds2))
			}

			for i := range preds1 {
				if preds1[i] != preds2[i] {
					t.Errorf("predictions differ at index %d: %.15e vs %.15e", i, preds1[i], preds2[i])
					break
				}
			}
		})
	}
}

// TestAllBackends_MiniBatchConsistency verifies that training with mini-batches
// still converges (loss decreases). Both full-batch and mini-batch runs must
// show decreasing loss. This catches batch-indexing or gradient-accumulation bugs
// that only manifest with BatchSize > 0.
func TestAllBackends_MiniBatchConsistency(t *testing.T) {
	const (
		channels  = 5
		inputLen  = 24
		outputLen = 6
		nSamples  = 20
		epochs    = 30
	)

	for _, bc := range allBackends() {
		t.Run(bc.name, func(t *testing.T) {
			windows := makeSyntheticWindows(nSamples, channels, inputLen)
			labels := makeSyntheticLabels(bc.labelsLen(nSamples))

			baseCfg := TrainConfig{
				LR:           1e-3,
				WeightDecay:  1e-4,
				GradClip:     1.0,
				WarmupEpochs: 3,
				Beta1:        0.9,
				Beta2:        0.999,
				Epsilon:      1e-8,
				Epochs:       epochs,
			}

			// Full-batch training.
			fullCfg := baseCfg
			fullCfg.BatchSize = 0

			fullResult, _, _, err := bc.train(windows, labels, fullCfg)
			if err != nil {
				t.Fatalf("TrainWindowed (full-batch): %v", err)
			}

			if len(fullResult.LossHistory) < 2 {
				t.Fatalf("full-batch LossHistory has %d entries, want >= 2", len(fullResult.LossHistory))
			}
			fullFirst := fullResult.LossHistory[0]
			fullLast := fullResult.LossHistory[len(fullResult.LossHistory)-1]
			if fullLast >= fullFirst {
				t.Errorf("full-batch loss did not decrease: first=%.6f last=%.6f", fullFirst, fullLast)
			}

			// Mini-batch training (batch size = 5).
			miniBatchCfg := baseCfg
			miniBatchCfg.BatchSize = 5

			miniResult, _, _, err := bc.train(windows, labels, miniBatchCfg)
			if err != nil {
				t.Fatalf("TrainWindowed (mini-batch): %v", err)
			}

			if len(miniResult.LossHistory) < 2 {
				t.Fatalf("mini-batch LossHistory has %d entries, want >= 2", len(miniResult.LossHistory))
			}
			miniFirst := miniResult.LossHistory[0]
			miniLast := miniResult.LossHistory[len(miniResult.LossHistory)-1]
			if miniLast >= miniFirst {
				t.Errorf("mini-batch loss did not decrease: first=%.6f last=%.6f", miniFirst, miniLast)
			}
		})
	}
}

// TestMamba_Issue158_LargeConfig is a regression test for issue #158:
// Mamba panics with nil pointer dereference during TrainWindowed with
// larger configs (DModel=64, DState=16, DConv=4, NLayers=2).
// Root cause was uncentered [0,1) weight initialization in core.NewLinear
// causing activation explosion through the SSM recurrence.
func TestMamba_Issue158_LargeConfig(t *testing.T) {
	configs := []struct {
		name   string
		config MambaConfig
	}{
		{"DModel64_NLayers2", MambaConfig{Channels: 5, InputLen: 10, OutputLen: 1, DModel: 64, DState: 16, DConv: 4, ExpandFactor: 2, NLayers: 2}},
		{"DModel32_NLayers3", MambaConfig{Channels: 5, InputLen: 10, OutputLen: 1, DModel: 32, DState: 8, DConv: 4, ExpandFactor: 2, NLayers: 3}},
	}

	for _, tc := range configs {
		t.Run(tc.name, func(t *testing.T) {
			m, err := NewMamba(tc.config, nil, nil)
			if err != nil {
				t.Fatalf("NewMamba: %v", err)
			}

			nSamples := 20
			windows := makeSyntheticWindows(nSamples, tc.config.Channels, tc.config.InputLen)
			labels := makeSyntheticLabels(nSamples * tc.config.Channels * tc.config.OutputLen)

			result, err := m.TrainWindowed(windows, labels, TrainConfig{
				Epochs: 5, LR: 1e-3, GradClip: 1.0,
			})
			if err != nil {
				t.Fatalf("TrainWindowed: %v", err)
			}

			// Loss must be finite.
			if math.IsNaN(result.FinalLoss) || math.IsInf(result.FinalLoss, 0) {
				t.Fatalf("final loss is not finite: %v", result.FinalLoss)
			}

			// Loss must decrease.
			if result.FinalLoss >= result.LossHistory[0] {
				t.Errorf("loss did not decrease: first=%.6f last=%.6f",
					result.LossHistory[0], result.FinalLoss)
			}

			// Predictions must be finite.
			preds, err := m.PredictWindowed("", windows[:3])
			if err != nil {
				t.Fatalf("PredictWindowed: %v", err)
			}
			for i, v := range preds {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("prediction[%d] is not finite: %v", i, v)
				}
			}
		})
	}
}

// channelMean returns the arithmetic mean of a single channel's time series.
func channelMean(ts []float64) float64 {
	if len(ts) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range ts {
		sum += v
	}
	return sum / float64(len(ts))
}

// sampleMean returns the arithmetic mean across all channels and time steps.
func sampleMean(chans [][]float64) float64 {
	sum := 0.0
	n := 0
	for _, ts := range chans {
		for _, v := range ts {
			sum += v
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}
