package timeseries

import (
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/zerfoo/ztensor/numeric"
)

// makeSyntheticWindows generates sine-wave + noise training windows.
// Returns [nSamples][channels][inputLen].
func makeSyntheticWindows(nSamples, channels, inputLen int) [][][]float64 {
	rng := rand.New(rand.NewPCG(42, 0))
	windows := make([][][]float64, nSamples)
	for i := range windows {
		windows[i] = make([][]float64, channels)
		for c := range windows[i] {
			windows[i][c] = make([]float64, inputLen)
			for t := range windows[i][c] {
				windows[i][c][t] = math.Sin(float64(t+i)*0.3+float64(c)*0.5) + rng.Float64()*0.1
			}
		}
	}
	return windows
}

// makeSyntheticLabels generates random labels of the given total length.
func makeSyntheticLabels(n int) []float64 {
	rng := rand.New(rand.NewPCG(99, 0))
	labels := make([]float64, n)
	for i := range labels {
		labels[i] = rng.Float64()
	}
	return labels
}

// backendCase describes how to construct, train, predict, and save a single
// timeseries backend so the table-driven tests can treat them uniformly.
type backendCase struct {
	name     string
	samples  int // training samples (0 = default 40)
	epochs   int // training epochs (0 = default 50)

	// labelsLen returns the expected total label count for nSamples.
	labelsLen func(nSamples int) int

	// predLen returns the expected prediction length for nSamples.
	predLen func(nSamples int) int

	// train creates a fresh model, trains it, and returns
	// (trainResult, predictFn, saveFn). predictFn and saveFn close over the
	// trained model so the caller does not need to know its concrete type.
	train func(windows [][][]float64, labels []float64, cfg TrainConfig) (
		*TrainResult,
		func(windows [][][]float64) ([]float64, error), // predictFn
		func(path string) error,                        // saveFn
		error,
	)

	// freshPredict creates a brand-new (untrained) model and returns its
	// predict function, plus a load-and-predict function for round-trip tests.
	freshPredict func() (
		func(windows [][][]float64) ([]float64, error),         // predictFn (no load)
		func(path string, w [][][]float64) ([]float64, error),  // predictFromPath
		error,
	)
}

func allBackends() []backendCase {
	engine, ops := newTestEngine()

	const (
		channels  = 5
		inputLen  = 24
		outputLen = 6
	)

	return []backendCase{
		{
			name:      "DLinear",
			labelsLen: func(n int) int { return n * channels * outputLen },
			predLen:   func(n int) int { return n * channels * outputLen },
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, func([][][]float64) ([]float64, error), func(string) error, error) {
				m, err := NewDLinear(inputLen, outputLen, channels, 5)
				if err != nil {
					return nil, nil, nil, err
				}
				res, err := m.TrainWindowed(w, l, cfg)
				if err != nil {
					return nil, nil, nil, err
				}
				return res,
					func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string) error { return m.SaveWeights(p) },
					nil
			},
			freshPredict: func() (func([][][]float64) ([]float64, error), func(string, [][][]float64) ([]float64, error), error) {
				m, err := NewDLinear(inputLen, outputLen, channels, 5)
				if err != nil {
					return nil, nil, err
				}
				return func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string, win [][][]float64) ([]float64, error) {
						m2, err := NewDLinear(inputLen, outputLen, channels, 5)
						if err != nil {
							return nil, err
						}
						return m2.PredictWindowed(p, win)
					}, nil
			},
		},
		{
			name:    "NHiTS",
			samples: 15,
			epochs:  20,
			labelsLen: func(n int) int { return n * outputLen },
			predLen:   func(n int) int { return n * outputLen },
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, func([][][]float64) ([]float64, error), func(string) error, error) {
				m, err := NewNHiTS(NHiTSConfig{
					InputLength: inputLen, OutputLength: outputLen, Channels: channels,
					PoolKernels: []int{2, 4}, HiddenSize: 16, NumMLPLayers: 2,
				}, engine, ops)
				if err != nil {
					return nil, nil, nil, err
				}
				res, err := m.TrainWindowed(w, l, cfg)
				if err != nil {
					return nil, nil, nil, err
				}
				return res,
					func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string) error { return m.Save(p) },
					nil
			},
			freshPredict: func() (func([][][]float64) ([]float64, error), func(string, [][][]float64) ([]float64, error), error) {
				m, err := NewNHiTS(NHiTSConfig{
					InputLength: inputLen, OutputLength: outputLen, Channels: channels,
					PoolKernels: []int{2, 4}, HiddenSize: 16, NumMLPLayers: 2,
				}, engine, ops)
				if err != nil {
					return nil, nil, err
				}
				return func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string, win [][][]float64) ([]float64, error) {
						m2, err := NewNHiTS(NHiTSConfig{
							InputLength: inputLen, OutputLength: outputLen, Channels: channels,
							PoolKernels: []int{2, 4}, HiddenSize: 16, NumMLPLayers: 2,
						}, engine, ops)
						if err != nil {
							return nil, err
						}
						return m2.PredictWindowed(p, win)
					}, nil
			},
		},
		{
			name:      "FreTS",
			labelsLen: func(n int) int { return n * channels * outputLen },
			predLen:   func(n int) int { return n * channels * outputLen },
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, func([][][]float64) ([]float64, error), func(string) error, error) {
				m, err := NewFreTS(FreTSConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					TopK: 4, HiddenSize: 16,
				})
				if err != nil {
					return nil, nil, nil, err
				}
				res, err := m.TrainWindowed(w, l, cfg)
				if err != nil {
					return nil, nil, nil, err
				}
				return res,
					func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string) error { return m.SaveWeights(p) },
					nil
			},
			freshPredict: func() (func([][][]float64) ([]float64, error), func(string, [][][]float64) ([]float64, error), error) {
				m, err := NewFreTS(FreTSConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					TopK: 4, HiddenSize: 16,
				})
				if err != nil {
					return nil, nil, err
				}
				return func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string, win [][][]float64) ([]float64, error) {
						m2, err := NewFreTS(FreTSConfig{
							Channels: channels, InputLen: inputLen, OutputLen: outputLen,
							TopK: 4, HiddenSize: 16,
						})
						if err != nil {
							return nil, err
						}
						return m2.PredictWindowed(p, win)
					}, nil
			},
		},
		{
			name:      "ITransformer",
			labelsLen: func(n int) int { return n * channels * outputLen },
			predLen:   func(n int) int { return n * channels * outputLen },
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, func([][][]float64) ([]float64, error), func(string) error, error) {
				m, err := NewITransformer(ITransformerConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
				}, nil, nil)
				if err != nil {
					return nil, nil, nil, err
				}
				res, err := m.TrainWindowed(w, l, cfg)
				if err != nil {
					return nil, nil, nil, err
				}
				return res,
					func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string) error { return m.Save(p) },
					nil
			},
			freshPredict: func() (func([][][]float64) ([]float64, error), func(string, [][][]float64) ([]float64, error), error) {
				m, err := NewITransformer(ITransformerConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
				}, nil, nil)
				if err != nil {
					return nil, nil, err
				}
				return func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string, win [][][]float64) ([]float64, error) {
						m2, err := NewITransformer(ITransformerConfig{
							Channels: channels, InputLen: inputLen, OutputLen: outputLen,
							DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
						}, nil, nil)
						if err != nil {
							return nil, err
						}
						return m2.PredictWindowed(p, win)
					}, nil
			},
		},
		{
			name:    "Mamba",
			samples: 10,
			epochs:  15,
			labelsLen: func(n int) int { return n * channels * outputLen },
			predLen:   func(n int) int { return n * channels * outputLen },
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, func([][][]float64) ([]float64, error), func(string) error, error) {
				m, err := NewMamba(MambaConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1,
				}, engine, ops)
				if err != nil {
					return nil, nil, nil, err
				}
				res, err := m.TrainWindowed(w, l, cfg)
				if err != nil {
					return nil, nil, nil, err
				}
				return res,
					func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string) error { return m.SaveWeights(p) },
					nil
			},
			freshPredict: func() (func([][][]float64) ([]float64, error), func(string, [][][]float64) ([]float64, error), error) {
				m, err := NewMamba(MambaConfig{
					Channels: channels, InputLen: inputLen, OutputLen: outputLen,
					DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1,
				}, engine, ops)
				if err != nil {
					return nil, nil, err
				}
				return func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string, win [][][]float64) ([]float64, error) {
						m2, err := NewMamba(MambaConfig{
							Channels: channels, InputLen: inputLen, OutputLen: outputLen,
							DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1,
						}, engine, ops)
						if err != nil {
							return nil, err
						}
						return m2.PredictWindowed(p, win)
					}, nil
			},
		},
		{
			name:    "CfC",
			samples: 10,
			epochs:  20,
			labelsLen: func(n int) int { return n * channels * outputLen },
			predLen:   func(n int) int { return n * channels * outputLen },
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, func([][][]float64) ([]float64, error), func(string) error, error) {
				m, err := NewCfC(CfCConfig{
					InputSize: channels, HiddenSize: 16, OutputSize: channels,
					NumLayers: 1, OutputLen: outputLen,
				}, WithCfCEngine(engine, ops))
				if err != nil {
					return nil, nil, nil, err
				}
				res, err := m.TrainWindowed(w, l, cfg)
				if err != nil {
					return nil, nil, nil, err
				}
				return res,
					func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string) error { return m.SaveWeights(p) },
					nil
			},
			freshPredict: func() (func([][][]float64) ([]float64, error), func(string, [][][]float64) ([]float64, error), error) {
				m, err := NewCfC(CfCConfig{
					InputSize: channels, HiddenSize: 16, OutputSize: channels,
					NumLayers: 1, OutputLen: outputLen,
				}, WithCfCEngine(engine, ops))
				if err != nil {
					return nil, nil, err
				}
				return func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string, win [][][]float64) ([]float64, error) {
						m2, err := NewCfC(CfCConfig{
							InputSize: channels, HiddenSize: 16, OutputSize: channels,
							NumLayers: 1, OutputLen: outputLen,
						}, WithCfCEngine(engine, ops))
						if err != nil {
							return nil, err
						}
						return m2.PredictWindowed(p, win)
					}, nil
			},
		},
		{
			name:    "PatchTST",
			samples: 4,
			epochs:  3,
			labelsLen: func(n int) int { return n * outputLen },
			predLen:   func(n int) int { return n * outputLen },
			train: func(w [][][]float64, l []float64, cfg TrainConfig) (*TrainResult, func([][][]float64) ([]float64, error), func(string) error, error) {
				// Use nil engine for CPU path — engine path is too slow for CI.
				m, err := NewPatchTST(PatchTSTConfig{
					InputLength: inputLen, PatchLength: 8, Stride: 4,
					DModel: 16, NHeads: 2, NLayers: 1,
					OutputDim: outputLen, ChannelIndependent: true,
				}, nil, numeric.Float32Ops{})
				if err != nil {
					return nil, nil, nil, err
				}
				res, err := m.TrainWindowed(w, l, cfg)
				if err != nil {
					return nil, nil, nil, err
				}
				return res,
					func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string) error { return m.SaveWeights(p) },
					nil
			},
			freshPredict: func() (func([][][]float64) ([]float64, error), func(string, [][][]float64) ([]float64, error), error) {
				m, err := NewPatchTST(PatchTSTConfig{
					InputLength: inputLen, PatchLength: 8, Stride: 4,
					DModel: 16, NHeads: 2, NLayers: 1,
					OutputDim: outputLen, ChannelIndependent: true,
				}, nil, numeric.Float32Ops{})
				if err != nil {
					return nil, nil, err
				}
				return func(win [][][]float64) ([]float64, error) { return m.PredictWindowed("", win) },
					func(p string, win [][][]float64) ([]float64, error) {
						m2, err := NewPatchTST(PatchTSTConfig{
							InputLength: inputLen, PatchLength: 8, Stride: 4,
							DModel: 16, NHeads: 2, NLayers: 1,
							OutputDim: outputLen, ChannelIndependent: true,
						}, nil, numeric.Float32Ops{})
						if err != nil {
							return nil, err
						}
						return m2.PredictWindowed(p, win)
					}, nil
			},
		},
	}
}

func TestAllBackends_LearnAndPredict(t *testing.T) {
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

			// 1. Get predictions from a fresh (untrained) model.
			freshPredFn, _, err := bc.freshPredict()
			if err != nil {
				t.Fatalf("freshPredict setup: %v", err)
			}
			testWindows := windows[:3]
			predsBefore, err := freshPredFn(testWindows)
			if err != nil {
				t.Fatalf("PredictWindowed (before training): %v", err)
			}

			// 2. Train.
			labels := makeSyntheticLabels(bc.labelsLen(nSamples))
			result, trainedPredFn, _, err := bc.train(windows, labels, cfg)
			if err != nil {
				t.Fatalf("TrainWindowed: %v", err)
			}

			// 3. Verify loss decreases.
			if len(result.LossHistory) < 2 {
				t.Fatalf("LossHistory has %d entries, want >= 2", len(result.LossHistory))
			}
			firstLoss := result.LossHistory[0]
			lastLoss := result.LossHistory[len(result.LossHistory)-1]
			if lastLoss >= firstLoss {
				t.Errorf("loss did not decrease: first=%.6f last=%.6f", firstLoss, lastLoss)
			}

			// 4. Get predictions after training.
			predsAfter, err := trainedPredFn(testWindows)
			if err != nil {
				t.Fatalf("PredictWindowed (after training): %v", err)
			}

			// 5. Verify prediction output length.
			expectedLen := bc.predLen(len(testWindows))
			if len(predsAfter) != expectedLen {
				t.Fatalf("prediction length = %d, want %d", len(predsAfter), expectedLen)
			}
			if len(predsBefore) != expectedLen {
				t.Fatalf("pre-training prediction length = %d, want %d", len(predsBefore), expectedLen)
			}

			// 6. Verify all predictions are finite.
			for i, v := range predsAfter {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Errorf("predsAfter[%d] = %v, want finite", i, v)
					break
				}
			}
			for i, v := range predsBefore {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Errorf("predsBefore[%d] = %v, want finite", i, v)
					break
				}
			}

			// 7. Verify training changed the model (predictions differ).
			allSame := true
			for i := range predsAfter {
				if predsAfter[i] != predsBefore[i] {
					allSame = false
					break
				}
			}
			if allSame {
				t.Error("predictions before and after training are identical; training had no effect")
			}
		})
	}
}

func TestAllBackends_SaveLoadRoundTrip(t *testing.T) {
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

			labels := makeSyntheticLabels(bc.labelsLen(nSamples))

			// Train.
			_, trainedPredFn, saveFn, err := bc.train(windows, labels, cfg)
			if err != nil {
				t.Fatalf("TrainWindowed: %v", err)
			}

			// Save to temp dir.
			dir := t.TempDir()
			path := filepath.Join(dir, bc.name+".json")
			if err := saveFn(path); err != nil {
				t.Fatalf("Save: %v", err)
			}

			// Predict with trained model.
			testWindows := windows[:3]
			preds1, err := trainedPredFn(testWindows)
			if err != nil {
				t.Fatalf("PredictWindowed (trained): %v", err)
			}

			// Load into fresh model and predict.
			_, loadPredFn, err := bc.freshPredict()
			if err != nil {
				t.Fatalf("freshPredict setup: %v", err)
			}
			preds2, err := loadPredFn(path, testWindows)
			if err != nil {
				t.Fatalf("PredictWindowed (loaded): %v", err)
			}

			// Lengths must match.
			if len(preds1) != len(preds2) {
				t.Fatalf("prediction lengths differ: %d vs %d", len(preds1), len(preds2))
			}

			// Values must match within tolerance.
			for i := range preds1 {
				diff := math.Abs(preds1[i] - preds2[i])
				if diff > 1e-9 {
					t.Errorf("preds1[%d]=%.10f != preds2[%d]=%.10f (diff=%.2e)",
						i, preds1[i], i, preds2[i], diff)
					break
				}
			}
		})
	}
}
