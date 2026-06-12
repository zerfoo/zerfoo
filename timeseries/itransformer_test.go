package timeseries

import (
	"math"
	"testing"
)

func TestNewITransformer_Validation(t *testing.T) {
	tests := []struct {
		name    string
		config  ITransformerConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: ITransformerConfig{
				Channels: 3, InputLen: 24, OutputLen: 12,
				DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
			},
		},
		{
			name: "zero channels",
			config: ITransformerConfig{
				Channels: 0, InputLen: 24, OutputLen: 12,
				DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
			},
			wantErr: true,
		},
		{
			name: "zero input length",
			config: ITransformerConfig{
				Channels: 3, InputLen: 0, OutputLen: 12,
				DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
			},
			wantErr: true,
		},
		{
			name: "zero output length",
			config: ITransformerConfig{
				Channels: 3, InputLen: 24, OutputLen: 0,
				DModel: 16, DFF: 32, NHeads: 2, NLayers: 1,
			},
			wantErr: true,
		},
		{
			name: "dmodel not divisible by nheads",
			config: ITransformerConfig{
				Channels: 3, InputLen: 24, OutputLen: 12,
				DModel: 15, DFF: 32, NHeads: 2, NLayers: 1,
			},
			wantErr: true,
		},
		{
			name: "zero dmodel",
			config: ITransformerConfig{
				Channels: 3, InputLen: 24, OutputLen: 12,
				DModel: 0, DFF: 32, NHeads: 2, NLayers: 1,
			},
			wantErr: true,
		},
		{
			name: "zero dff",
			config: ITransformerConfig{
				Channels: 3, InputLen: 24, OutputLen: 12,
				DModel: 16, DFF: 0, NHeads: 2, NLayers: 1,
			},
			wantErr: true,
		},
		{
			name: "zero nheads",
			config: ITransformerConfig{
				Channels: 3, InputLen: 24, OutputLen: 12,
				DModel: 16, DFF: 32, NHeads: 0, NLayers: 1,
			},
			wantErr: true,
		},
		{
			name: "zero nlayers",
			config: ITransformerConfig{
				Channels: 3, InputLen: 24, OutputLen: 12,
				DModel: 16, DFF: 32, NHeads: 2, NLayers: 0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewITransformer(tt.config, nil, nil)
			if tt.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestITransformer_Convergence(t *testing.T) {
	config := ITransformerConfig{
		Channels:  3,
		InputLen:  8,
		OutputLen: 4,
		DModel:    8,
		DFF:       16,
		NHeads:    2,
		NLayers:   1,
	}

	m, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	// Generate synthetic multivariate data: 3 channels with different patterns.
	nSamples := 20
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.Channels*config.OutputLen)

	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[s][c] = make([]float64, config.InputLen)
			for i := 0; i < config.InputLen; i++ {
				// Channel 0: linear, Channel 1: sinusoidal, Channel 2: constant + noise
				switch c {
				case 0:
					windows[s][c][i] = 0.1 * float64(s+i)
				case 1:
					windows[s][c][i] = math.Sin(0.5 * float64(s+i))
				case 2:
					windows[s][c][i] = 1.0 + 0.01*float64(s)
				}
			}
		}
		for c := 0; c < config.Channels; c++ {
			for o := 0; o < config.OutputLen; o++ {
				idx := s*config.Channels*config.OutputLen + c*config.OutputLen + o
				switch c {
				case 0:
					labels[idx] = 0.1 * float64(s+config.InputLen+o)
				case 1:
					labels[idx] = math.Sin(0.5 * float64(s+config.InputLen+o))
				case 2:
					labels[idx] = 1.0 + 0.01*float64(s)
				}
			}
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       30,
		LR:           1e-3,
		GradClip:     1.0,
		WarmupEpochs: 5,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if len(result.LossHistory) != 30 {
		t.Fatalf("loss history length = %d, want 30", len(result.LossHistory))
	}

	// Loss should decrease.
	if result.LossHistory[29] >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: epoch 0 = %v, epoch 29 = %v",
			result.LossHistory[0], result.LossHistory[29])
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	t.Logf("convergence: loss[0]=%.6f -> loss[29]=%.6f", result.LossHistory[0], result.LossHistory[29])
}

func TestITransformer_MultiChannel(t *testing.T) {
	config := ITransformerConfig{
		Channels:  4,
		InputLen:  12,
		OutputLen: 4,
		DModel:    8,
		DFF:       16,
		NHeads:    2,
		NLayers:   1,
	}

	m, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	// Single sample, 4 channels.
	input := make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		input[c] = make([]float64, config.InputLen)
		for i := 0; i < config.InputLen; i++ {
			input[c][i] = float64(c*100 + i)
		}
	}

	pred := m.forward(input)
	if len(pred) != config.Channels {
		t.Fatalf("forward returned %d channels, want %d", len(pred), config.Channels)
	}
	for c := 0; c < config.Channels; c++ {
		if len(pred[c]) != config.OutputLen {
			t.Fatalf("channel %d output length = %d, want %d", c, len(pred[c]), config.OutputLen)
		}
		for i, v := range pred[c] {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("channel %d output[%d] = %v, want finite", c, i, v)
			}
		}
	}
}

func TestITransformer_SaveLoad(t *testing.T) {
	config := ITransformerConfig{
		Channels:  2,
		InputLen:  8,
		OutputLen: 4,
		DModel:    8,
		DFF:       16,
		NHeads:    2,
		NLayers:   1,
	}

	m1, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	// Train briefly to set normMeans/normStds.
	nSamples := 10
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.Channels*config.OutputLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[s][c] = make([]float64, config.InputLen)
			for i := 0; i < config.InputLen; i++ {
				windows[s][c][i] = float64(s+i+c) * 0.1
			}
		}
		for i := range config.Channels * config.OutputLen {
			labels[s*config.Channels*config.OutputLen+i] = float64(s) * 0.05
		}
	}

	_, err = m1.TrainWindowed(windows, labels, TrainConfig{
		Epochs: 3,
		LR:     1e-3,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Save.
	dir := t.TempDir()
	path := dir + "/itransformer.json"
	if err := m1.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Predict with original model.
	preds1, err := m1.PredictWindowed("", windows[:3])
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	// Load into new model and predict.
	m2, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	preds2, err := m2.PredictWindowed(path, windows[:3])
	if err != nil {
		t.Fatalf("PredictWindowed with load: %v", err)
	}

	// Predictions should match.
	if len(preds1) != len(preds2) {
		t.Fatalf("prediction lengths differ: %d vs %d", len(preds1), len(preds2))
	}
	for i := range preds1 {
		if math.Abs(preds1[i]-preds2[i]) > 1e-6 {
			t.Errorf("loaded model prediction[%d] = %v, want %v", i, preds2[i], preds1[i])
		}
	}
}

func TestITransformer_SingleChannel(t *testing.T) {
	// Edge case: single channel should degenerate to simple forecasting.
	config := ITransformerConfig{
		Channels:  1,
		InputLen:  8,
		OutputLen: 4,
		DModel:    8,
		DFF:       16,
		NHeads:    2,
		NLayers:   1,
	}

	m, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	// Forward should work with a single channel.
	input := [][]float64{make([]float64, config.InputLen)}
	for i := range input[0] {
		input[0][i] = float64(i) * 0.1
	}

	pred := m.forward(input)
	if len(pred) != 1 {
		t.Fatalf("forward returned %d channels, want 1", len(pred))
	}
	if len(pred[0]) != config.OutputLen {
		t.Fatalf("output length = %d, want %d", len(pred[0]), config.OutputLen)
	}
	for i, v := range pred[0] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("output[%d] = %v, want finite", i, v)
		}
	}

	// Train + predict should also work.
	nSamples := 10
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.OutputLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = [][]float64{make([]float64, config.InputLen)}
		for i := 0; i < config.InputLen; i++ {
			windows[s][0][i] = 0.1 * float64(s+i)
		}
		for o := 0; o < config.OutputLen; o++ {
			labels[s*config.OutputLen+o] = 0.1 * float64(s+config.InputLen+o)
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   10,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}
	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	preds, err := m.PredictWindowed("", windows[:3])
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}
	if len(preds) != 3*config.OutputLen {
		t.Fatalf("predictions length = %d, want %d", len(preds), 3*config.OutputLen)
	}
	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("prediction[%d] = %v, want finite", i, v)
		}
	}
}

func TestITransformer_TrainWindowed_Empty(t *testing.T) {
	config := ITransformerConfig{
		Channels: 2, InputLen: 8, OutputLen: 4,
		DModel: 8, DFF: 16, NHeads: 2, NLayers: 1,
	}
	m, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	_, err = m.TrainWindowed(nil, nil, TrainConfig{Epochs: 5})
	if err == nil {
		t.Fatal("expected error for empty training set")
	}
}

func TestITransformer_PredictWindowed_Empty(t *testing.T) {
	config := ITransformerConfig{
		Channels: 2, InputLen: 8, OutputLen: 4,
		DModel: 8, DFF: 16, NHeads: 2, NLayers: 1,
	}
	m, err := NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	_, err = m.PredictWindowed("", nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

// The bespoke finite-difference spot check that used to live here
// (TestITransformer_GradientCheck) was migrated to ztensor's shared gradcheck
// harness; see TestTimeseriesBackward_Gradcheck in gradcheck_test.go (plan T1.6).
