package timeseries

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMamba_NewValidation(t *testing.T) {
	tests := []struct {
		name   string
		config MambaConfig
		errMsg string
	}{
		{
			name:   "zero channels",
			config: MambaConfig{Channels: 0, InputLen: 24, OutputLen: 12, DModel: 8, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
			errMsg: "Channels must be positive",
		},
		{
			name:   "zero input len",
			config: MambaConfig{Channels: 1, InputLen: 0, OutputLen: 12, DModel: 8, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
			errMsg: "InputLen must be positive",
		},
		{
			name:   "zero output len",
			config: MambaConfig{Channels: 1, InputLen: 24, OutputLen: 0, DModel: 8, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
			errMsg: "OutputLen must be positive",
		},
		{
			name:   "zero dmodel",
			config: MambaConfig{Channels: 1, InputLen: 24, OutputLen: 12, DModel: 0, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
			errMsg: "DModel must be positive",
		},
		{
			name:   "zero nlayers",
			config: MambaConfig{Channels: 1, InputLen: 24, OutputLen: 12, DModel: 8, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 0},
			errMsg: "NLayers must be positive",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewMamba(tt.config)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tt.errMsg)
			}
		})
	}
}

func TestMamba_ForwardOutputShape(t *testing.T) {
	config := MambaConfig{
		Channels:     2,
		InputLen:     16,
		OutputLen:    8,
		DModel:       4,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	input := make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		input[c] = make([]float64, config.InputLen)
		for i := range input[c] {
			input[c][i] = float64(i) * 0.1
		}
	}

	output := m.forward(input)
	if len(output) != config.Channels {
		t.Fatalf("expected %d channels, got %d", config.Channels, len(output))
	}
	for c := 0; c < config.Channels; c++ {
		if len(output[c]) != config.OutputLen {
			t.Fatalf("channel %d: expected length %d, got %d", c, config.OutputLen, len(output[c]))
		}
	}
}

func TestMamba_Convergence(t *testing.T) {
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	// Simple synthetic data: output = mean of input repeated.
	nSamples := 10
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.Channels*config.OutputLen)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[i][c] = make([]float64, config.InputLen)
			sum := 0.0
			for t := 0; t < config.InputLen; t++ {
				v := float64(i+1) * 0.5 * float64(t+1)
				windows[i][c][t] = v
				sum += v
			}
			mean := sum / float64(config.InputLen)
			for o := 0; o < config.OutputLen; o++ {
				labels[i*config.Channels*config.OutputLen+c*config.OutputLen+o] = mean
			}
		}
	}

	tc := TrainConfig{
		Epochs:       30,
		LR:           1e-3,
		WeightDecay:  0,
		GradClip:     1.0,
		WarmupEpochs: 3,
	}

	result, err := m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Loss should decrease from first to last epoch.
	if result.LossHistory[len(result.LossHistory)-1] >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: first=%.6f last=%.6f",
			result.LossHistory[0], result.LossHistory[len(result.LossHistory)-1])
	}

	if !isFinite(result.FinalLoss) {
		t.Errorf("final loss is not finite: %v", result.FinalLoss)
	}
}

func TestMamba_LongSequence(t *testing.T) {
	// Verify inputLen=512 works without excessive memory (O(L) not O(L^2)).
	config := MambaConfig{
		Channels:     1,
		InputLen:     512,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        4,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	input := make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		input[c] = make([]float64, config.InputLen)
		for i := range input[c] {
			input[c][i] = math.Sin(float64(i) * 0.01)
		}
	}

	output := m.forward(input)
	if len(output) != config.Channels {
		t.Fatalf("expected %d channels, got %d", config.Channels, len(output))
	}
	for c := 0; c < config.Channels; c++ {
		if len(output[c]) != config.OutputLen {
			t.Fatalf("channel %d: expected length %d, got %d", c, config.OutputLen, len(output[c]))
		}
		for _, v := range output[c] {
			if !isFinite(v) {
				t.Fatalf("channel %d: non-finite output value: %v", c, v)
			}
		}
	}
}

func TestMamba_SaveLoadRoundTrip(t *testing.T) {
	config := MambaConfig{
		Channels:     2,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m1, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	// Train briefly to set normMeans/normStds.
	nSamples := 5
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.Channels*config.OutputLen)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[i][c] = make([]float64, config.InputLen)
			for t := 0; t < config.InputLen; t++ {
				windows[i][c][t] = float64(i*config.InputLen+t) * 0.1
			}
			for o := 0; o < config.OutputLen; o++ {
				labels[i*config.Channels*config.OutputLen+c*config.OutputLen+o] = float64(i) * 0.5
			}
		}
	}
	_, err = m1.TrainWindowed(windows, labels, TrainConfig{Epochs: 2, LR: 1e-3})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Save.
	dir := t.TempDir()
	path := filepath.Join(dir, "mamba_weights.json")
	if err := m1.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	// Load into new model.
	m2, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}
	if err := m2.loadWeights(path); err != nil {
		t.Fatalf("loadWeights: %v", err)
	}

	// Verify predictions match.
	testInput := make([][][]float64, 1)
	testInput[0] = make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		testInput[0][c] = make([]float64, config.InputLen)
		for t := 0; t < config.InputLen; t++ {
			testInput[0][c][t] = float64(t) * 0.2
		}
	}

	pred1, err := m1.PredictWindowed("", testInput)
	if err != nil {
		t.Fatalf("PredictWindowed m1: %v", err)
	}
	pred2, err := m2.PredictWindowed("", testInput)
	if err != nil {
		t.Fatalf("PredictWindowed m2: %v", err)
	}

	if len(pred1) != len(pred2) {
		t.Fatalf("prediction length mismatch: %d vs %d", len(pred1), len(pred2))
	}
	for i := range pred1 {
		if math.Abs(pred1[i]-pred2[i]) > 1e-10 {
			t.Errorf("prediction[%d] mismatch: %.10f vs %.10f", i, pred1[i], pred2[i])
		}
	}
}

func TestMamba_MultiChannel(t *testing.T) {
	config := MambaConfig{
		Channels:     3,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	nSamples := 8
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.Channels*config.OutputLen)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[i][c] = make([]float64, config.InputLen)
			for t := 0; t < config.InputLen; t++ {
				windows[i][c][t] = float64(c+1) * math.Sin(float64(t)*0.5+float64(i)*0.3)
			}
			for o := 0; o < config.OutputLen; o++ {
				labels[i*config.Channels*config.OutputLen+c*config.OutputLen+o] = float64(c+1) * 0.5
			}
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       20,
		LR:           1e-3,
		GradClip:     1.0,
		WarmupEpochs: 2,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Errorf("final loss is not finite: %v", result.FinalLoss)
	}

	// Predict.
	preds, err := m.PredictWindowed("", windows[:1])
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}
	expectedLen := config.Channels * config.OutputLen
	if len(preds) != expectedLen {
		t.Errorf("expected %d predictions, got %d", expectedLen, len(preds))
	}
}

func TestMamba_PredictWindowed_LoadFromPath(t *testing.T) {
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "mamba.json")
	if err := m.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	m2, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	input := make([][][]float64, 1)
	input[0] = make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		input[0][c] = make([]float64, config.InputLen)
		for t := range input[0][c] {
			input[0][c][t] = float64(t)
		}
	}

	_, err = m2.PredictWindowed(path, input)
	if err != nil {
		t.Fatalf("PredictWindowed with path: %v", err)
	}
}

func TestMamba_PredictWindowed_BadPath(t *testing.T) {
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	input := make([][][]float64, 1)
	input[0] = make([][]float64, config.Channels)
	input[0][0] = make([]float64, config.InputLen)

	_, err = m.PredictWindowed("/nonexistent/path.json", input)
	if err == nil {
		t.Fatal("expected error for bad path, got nil")
	}
}

func TestMamba_EmptyInput(t *testing.T) {
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	_, err = m.TrainWindowed(nil, nil, TrainConfig{Epochs: 1})
	if err == nil {
		t.Fatal("expected error for empty training set")
	}

	_, err = m.PredictWindowed("", nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestMamba_ParamCount(t *testing.T) {
	config := MambaConfig{
		Channels:     2,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	params := m.flatParams()
	count := m.paramCount()
	if len(params) != count {
		t.Errorf("paramCount()=%d but flatParams() returned %d pointers", count, len(params))
	}
}

func TestMamba_SaveWeightsCreatesFile(t *testing.T) {
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       4,
		DState:       2,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "test_weights.json")
	if err := m.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("saved file is empty")
	}
}
