package timeseries

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMamba_NewValidation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name   string
		config MambaConfig
	}{
		{
			name:   "zero channels",
			config: MambaConfig{Channels: 0, InputLen: 24, OutputLen: 12, DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
		},
		{
			name:   "zero input len",
			config: MambaConfig{Channels: 1, InputLen: 0, OutputLen: 12, DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
		},
		{
			name:   "zero output len",
			config: MambaConfig{Channels: 1, InputLen: 24, OutputLen: 0, DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
		},
		{
			name:   "zero dmodel",
			config: MambaConfig{Channels: 1, InputLen: 24, OutputLen: 12, DModel: 0, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 1},
		},
		{
			name:   "zero nlayers",
			config: MambaConfig{Channels: 1, InputLen: 24, OutputLen: 12, DModel: 16, DState: 4, DConv: 2, ExpandFactor: 2, NLayers: 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewMamba(tt.config, engine, ops)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestMamba_ForwardOutputShape(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     2,
		InputLen:     16,
		OutputLen:    8,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
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

	preds, err := m.PredictWindowed("", [][][]float64{input})
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	expectedLen := config.Channels * config.OutputLen
	if len(preds) != expectedLen {
		t.Fatalf("expected %d predictions, got %d", expectedLen, len(preds))
	}
}

func TestMamba_Convergence(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
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
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     1,
		InputLen:     512,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        4,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
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

	preds, err := m.PredictWindowed("", [][][]float64{input})
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	for i, v := range preds {
		if !isFinite(v) {
			t.Fatalf("prediction[%d] is not finite: %v", i, v)
		}
	}
}

func TestMamba_SaveLoadRoundTrip(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     2,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m1, err := NewMamba(config, engine, ops)
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
	m2, err := NewMamba(config, engine, ops)
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
		if math.Abs(pred1[i]-pred2[i]) > 1e-4 {
			t.Errorf("prediction[%d] mismatch: %.6f vs %.6f", i, pred1[i], pred2[i])
		}
	}
}

func TestMamba_MultiChannel(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     3,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
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
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "mamba.json")
	if err := m.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	m2, err := NewMamba(config, engine, ops)
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
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
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
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
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

func TestMamba_TrainWindowed_NilEngine(t *testing.T) {
	// Create Mamba with nil engine — TrainWindowed should auto-create CPUEngine.
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, nil, nil)
	if err != nil {
		t.Fatalf("NewMamba with nil engine: %v", err)
	}

	// Simple synthetic data.
	nSamples := 6
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*config.Channels*config.OutputLen)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[i][c] = make([]float64, config.InputLen)
			for ti := 0; ti < config.InputLen; ti++ {
				windows[i][c][ti] = float64(i+1) * 0.3 * float64(ti+1)
			}
			for o := 0; o < config.OutputLen; o++ {
				labels[i*config.Channels*config.OutputLen+c*config.OutputLen+o] = float64(i+1) * 0.5
			}
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   10,
		LR:       1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed with nil engine: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Errorf("final loss is not finite: %v", result.FinalLoss)
	}

	// Loss should decrease.
	if result.LossHistory[len(result.LossHistory)-1] >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: first=%.6f last=%.6f",
			result.LossHistory[0], result.LossHistory[len(result.LossHistory)-1])
	}

	// Engine should still be nil on the original struct (no side effects).
	if m.engine != nil {
		t.Error("expected engine to remain nil after training")
	}
}

func TestMamba_SaveWeightsCreatesFile(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
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

func TestMamba_BatchedForwardParity(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     2,
		InputLen:     16,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      2,
	}
	m, err := NewMamba(config, engine, ops)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	batch := 5
	windows := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		windows[b] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[b][c] = make([]float64, config.InputLen)
			for i := 0; i < config.InputLen; i++ {
				windows[b][c][i] = math.Sin(float64(b*100+c*10+i) * 0.1)
			}
		}
	}

	ctx := context.Background()
	outDim := config.Channels * config.OutputLen

	singleResults := make([]float32, 0, batch*outDim)
	for b := 0; b < batch; b++ {
		pred, _, err := m.forward(ctx, windows[b])
		if err != nil {
			t.Fatalf("forward sample %d: %v", b, err)
		}
		singleResults = append(singleResults, pred...)
	}

	batchResults, err := m.forwardBatch(ctx, windows)
	if err != nil {
		t.Fatalf("forwardBatch: %v", err)
	}

	if len(batchResults) != len(singleResults) {
		t.Fatalf("length mismatch: single=%d batch=%d", len(singleResults), len(batchResults))
	}

	for i := range singleResults {
		diff := math.Abs(float64(singleResults[i] - batchResults[i]))
		if diff > 1e-4 {
			t.Errorf("output[%d] mismatch: single=%.6f batch=%.6f diff=%.6e",
				i, singleResults[i], batchResults[i], diff)
		}
	}
}

func TestMamba_BatchedForwardBatchSize1(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     1,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	windows := make([][][]float64, 1)
	windows[0] = make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		windows[0][c] = make([]float64, config.InputLen)
		for i := range windows[0][c] {
			windows[0][c][i] = float64(i) * 0.2
		}
	}

	ctx := context.Background()

	singlePred, _, err := m.forward(ctx, windows[0])
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	batchPred, err := m.forwardBatch(ctx, windows)
	if err != nil {
		t.Fatalf("forwardBatch: %v", err)
	}

	if len(singlePred) != len(batchPred) {
		t.Fatalf("length mismatch: %d vs %d", len(singlePred), len(batchPred))
	}

	for i := range singlePred {
		diff := math.Abs(float64(singlePred[i] - batchPred[i]))
		if diff > 1e-5 {
			t.Errorf("output[%d] mismatch: single=%.6f batch=%.6f", i, singlePred[i], batchPred[i])
		}
	}
}

func TestMamba_PredictWindowed_UsesBatchedPath(t *testing.T) {
	engine, ops := newTestEngine()
	config := MambaConfig{
		Channels:     2,
		InputLen:     8,
		OutputLen:    4,
		DModel:       16,
		DState:       4,
		DConv:        2,
		ExpandFactor: 2,
		NLayers:      1,
	}
	m, err := NewMamba(config, engine, ops)
	if err != nil {
		t.Fatalf("NewMamba: %v", err)
	}

	batch := 3
	windows := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		windows[b] = make([][]float64, config.Channels)
		for c := 0; c < config.Channels; c++ {
			windows[b][c] = make([]float64, config.InputLen)
			for i := range windows[b][c] {
				windows[b][c][i] = float64(b*10+c*5+i) * 0.1
			}
		}
	}

	preds, err := m.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	expectedLen := batch * config.Channels * config.OutputLen
	if len(preds) != expectedLen {
		t.Fatalf("expected %d predictions, got %d", expectedLen, len(preds))
	}

	for i, v := range preds {
		if !isFinite(v) {
			t.Errorf("prediction[%d] is not finite: %v", i, v)
		}
	}
}

