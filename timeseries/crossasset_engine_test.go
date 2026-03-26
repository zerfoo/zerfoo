package timeseries

import (
	"math"
	"math/rand/v2"
	"testing"
)

func defaultCrossAssetConfig() CrossAssetConfig {
	return CrossAssetConfig{
		NSources:          4,
		FeaturesPerSource: 8,
		DModel:            16,
		NHeads:            4,
		NLayers:           2,
		LearningRate:      0.01,
		Epochs:            30,
		BatchSize:         10,
	}
}

func TestNewCrossAsset_Validation(t *testing.T) {
	tests := []struct {
		name    string
		config  CrossAssetConfig
		wantErr bool
	}{
		{
			name:    "valid config",
			config:  defaultCrossAssetConfig(),
			wantErr: false,
		},
		{
			name:    "zero sources",
			config:  CrossAssetConfig{NSources: 0, FeaturesPerSource: 8, DModel: 16, NHeads: 4, NLayers: 2},
			wantErr: true,
		},
		{
			name:    "zero features",
			config:  CrossAssetConfig{NSources: 4, FeaturesPerSource: 0, DModel: 16, NHeads: 4, NLayers: 2},
			wantErr: true,
		},
		{
			name:    "DModel not divisible by NHeads",
			config:  CrossAssetConfig{NSources: 4, FeaturesPerSource: 8, DModel: 15, NHeads: 4, NLayers: 2},
			wantErr: true,
		},
		{
			name:    "zero layers",
			config:  CrossAssetConfig{NSources: 4, FeaturesPerSource: 8, DModel: 16, NHeads: 4, NLayers: 0},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewCrossAsset(tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewCrossAsset() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestCrossAsset_TrainWindowed_Convergence(t *testing.T) {
	cfg := defaultCrossAssetConfig()
	ca, err := NewCrossAsset(cfg)
	if err != nil {
		t.Fatalf("NewCrossAsset: %v", err)
	}

	nSamples := 40
	rng := rand.New(rand.NewPCG(42, 0))
	windows, labels := makeCrossAssetData(nSamples, cfg.NSources, cfg.FeaturesPerSource, rng)

	result, err := ca.TrainWindowed(windows, labels, TrainConfig{
		Epochs: 30,
		LR:     0.01,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if len(result.LossHistory) != 30 {
		t.Fatalf("loss history length = %d, want 30", len(result.LossHistory))
	}

	// Loss should be finite.
	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	// Loss should decrease over training (comparing first third average to last third).
	firstThird := 0.0
	lastThird := 0.0
	n := len(result.LossHistory)
	third := n / 3
	for i := 0; i < third; i++ {
		firstThird += result.LossHistory[i]
		lastThird += result.LossHistory[n-1-i]
	}
	firstThird /= float64(third)
	lastThird /= float64(third)

	if lastThird >= firstThird {
		t.Errorf("loss did not decrease: first_third_avg=%.6f, last_third_avg=%.6f", firstThird, lastThird)
	}

	t.Logf("training: first_loss=%.6f, final_loss=%.6f", result.LossHistory[0], result.FinalLoss)
}

func TestCrossAsset_PredictWindowed_Shape(t *testing.T) {
	cfg := defaultCrossAssetConfig()
	ca, err := NewCrossAsset(cfg)
	if err != nil {
		t.Fatalf("NewCrossAsset: %v", err)
	}

	nSamples := 5
	rng := rand.New(rand.NewPCG(99, 0))
	windows, _ := makeCrossAssetData(nSamples, cfg.NSources, cfg.FeaturesPerSource, rng)

	preds, err := ca.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	expected := nSamples * cfg.NSources * 3
	if len(preds) != expected {
		t.Fatalf("PredictWindowed returned %d values, want %d", len(preds), expected)
	}

	// Each triplet should be valid probabilities summing to ~1.
	for i := 0; i < nSamples*cfg.NSources; i++ {
		base := i * 3
		sum := preds[base] + preds[base+1] + preds[base+2]
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("sample/source %d: probabilities sum to %f, want 1.0", i, sum)
		}
		for j := 0; j < 3; j++ {
			if preds[base+j] < 0 || preds[base+j] > 1 {
				t.Errorf("sample/source %d class %d: probability %f out of [0,1]", i, j, preds[base+j])
			}
		}
	}
}

func TestCrossAsset_PredictWindowed_AfterTraining(t *testing.T) {
	cfg := defaultCrossAssetConfig()
	ca, err := NewCrossAsset(cfg)
	if err != nil {
		t.Fatalf("NewCrossAsset: %v", err)
	}

	nSamples := 30
	rng := rand.New(rand.NewPCG(7, 0))
	windows, labels := makeCrossAssetData(nSamples, cfg.NSources, cfg.FeaturesPerSource, rng)

	_, err = ca.TrainWindowed(windows, labels, TrainConfig{
		Epochs: 20,
		LR:     0.01,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Predict on training data.
	preds, err := ca.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	expected := nSamples * cfg.NSources * 3
	if len(preds) != expected {
		t.Fatalf("PredictWindowed returned %d values, want %d", len(preds), expected)
	}

	// All values should be finite.
	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("pred[%d] = %v, want finite", i, v)
		}
	}
}

func TestCrossAsset_TrainWindowed_EmptyInput(t *testing.T) {
	cfg := defaultCrossAssetConfig()
	ca, err := NewCrossAsset(cfg)
	if err != nil {
		t.Fatalf("NewCrossAsset: %v", err)
	}

	_, err = ca.TrainWindowed(nil, nil, TrainConfig{Epochs: 1})
	if err == nil {
		t.Fatal("expected error for empty training set")
	}
}

func TestCrossAsset_TrainWindowed_LabelMismatch(t *testing.T) {
	cfg := defaultCrossAssetConfig()
	ca, err := NewCrossAsset(cfg)
	if err != nil {
		t.Fatalf("NewCrossAsset: %v", err)
	}

	rng := rand.New(rand.NewPCG(1, 0))
	windows, _ := makeCrossAssetData(5, cfg.NSources, cfg.FeaturesPerSource, rng)

	// Wrong label count.
	_, err = ca.TrainWindowed(windows, []float64{0, 1}, TrainConfig{Epochs: 1})
	if err == nil {
		t.Fatal("expected error for mismatched label count")
	}
}

func TestCrossAsset_PredictWindowed_EmptyInput(t *testing.T) {
	cfg := defaultCrossAssetConfig()
	ca, err := NewCrossAsset(cfg)
	if err != nil {
		t.Fatalf("NewCrossAsset: %v", err)
	}

	_, err = ca.PredictWindowed("", nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestCrossAsset_DirectionToProbs(t *testing.T) {
	tests := []struct {
		dir  int
		conf float64
	}{
		{0, 0.8},
		{1, 0.6},
		{2, 0.5},
		{0, 1.0},
	}

	for _, tt := range tests {
		probs := directionToProbs(tt.dir, tt.conf)
		sum := probs[0] + probs[1] + probs[2]
		if math.Abs(sum-1.0) > 1e-10 {
			t.Errorf("dir=%d conf=%.2f: probs sum to %f, want 1.0", tt.dir, tt.conf, sum)
		}
		if probs[tt.dir] != tt.conf {
			t.Errorf("dir=%d conf=%.2f: probs[dir]=%f, want %f", tt.dir, tt.conf, probs[tt.dir], tt.conf)
		}
	}
}

// makeCrossAssetData generates synthetic cross-asset training data.
// Returns windows [nSamples][nSources][featuresPerSource] and labels [nSamples * nSources]
// where labels are direction classes in {0, 1, 2}.
func makeCrossAssetData(nSamples, nSources, featuresPerSource int, rng *rand.Rand) ([][][]float64, []float64) {
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*nSources)

	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, nSources)
		for s := 0; s < nSources; s++ {
			windows[i][s] = make([]float64, featuresPerSource)
			for f := 0; f < featuresPerSource; f++ {
				windows[i][s][f] = rng.NormFloat64() * 0.5
			}
			// Assign label based on mean of features (deterministic given features).
			mean := 0.0
			for _, v := range windows[i][s] {
				mean += v
			}
			mean /= float64(featuresPerSource)
			var label int
			if mean > 0.1 {
				label = 0 // Long
			} else if mean < -0.1 {
				label = 1 // Short
			} else {
				label = 2 // Flat
			}
			labels[i*nSources+s] = float64(label)
		}
	}

	return windows, labels
}
