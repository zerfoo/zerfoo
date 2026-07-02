package timeseries

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNewCfC_Validation(t *testing.T) {
	tests := []struct {
		name    string
		config  CfCConfig
		wantErr bool
	}{
		{"valid", CfCConfig{InputSize: 3, HiddenSize: 8, OutputSize: 1, NumLayers: 2, OutputLen: 4}, false},
		{"single layer", CfCConfig{InputSize: 1, HiddenSize: 4, OutputSize: 1, NumLayers: 1, OutputLen: 1}, false},
		{"zero InputSize", CfCConfig{InputSize: 0, HiddenSize: 8, OutputSize: 1, NumLayers: 1, OutputLen: 4}, true},
		{"zero HiddenSize", CfCConfig{InputSize: 3, HiddenSize: 0, OutputSize: 1, NumLayers: 1, OutputLen: 4}, true},
		{"zero OutputSize", CfCConfig{InputSize: 3, HiddenSize: 8, OutputSize: 0, NumLayers: 1, OutputLen: 4}, true},
		{"zero NumLayers", CfCConfig{InputSize: 3, HiddenSize: 8, OutputSize: 1, NumLayers: 0, OutputLen: 4}, true},
		{"zero OutputLen", CfCConfig{InputSize: 3, HiddenSize: 8, OutputSize: 1, NumLayers: 1, OutputLen: 0}, true},
		{"negative InputSize", CfCConfig{InputSize: -1, HiddenSize: 8, OutputSize: 1, NumLayers: 1, OutputLen: 4}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewCfC(tt.config)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if m == nil {
				t.Fatal("expected non-nil model")
			}
		})
	}
}

func TestCfC_ForwardOutputShape(t *testing.T) {
	config := CfCConfig{InputSize: 3, HiddenSize: 8, OutputSize: 2, NumLayers: 2, OutputLen: 4}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Input: [seqLen=5][inputSize=3].
	input := make([][]float64, 5)
	for t := range input {
		input[t] = []float64{0.1, 0.2, 0.3}
	}

	out := m.forward(input)
	expectedLen := config.OutputSize * config.OutputLen
	if len(out) != expectedLen {
		t.Fatalf("output length = %d, want %d", len(out), expectedLen)
	}
	for i, v := range out {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("output[%d] = %v, want finite", i, v)
		}
	}
}

func TestCfC_MultiLayerForward(t *testing.T) {
	for _, numLayers := range []int{1, 2, 3} {
		t.Run("layers_"+string(rune('0'+numLayers)), func(t *testing.T) {
			config := CfCConfig{InputSize: 2, HiddenSize: 4, OutputSize: 1, NumLayers: numLayers, OutputLen: 3}
			m, err := NewCfC(config)
			if err != nil {
				t.Fatalf("NewCfC: %v", err)
			}

			input := make([][]float64, 4)
			for i := range input {
				input[i] = []float64{float64(i) * 0.1, float64(i) * 0.2}
			}

			out := m.forward(input)
			if len(out) != 3 {
				t.Fatalf("output length = %d, want 3", len(out))
			}
			for i, v := range out {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Errorf("output[%d] = %v, want finite", i, v)
				}
			}
		})
	}
}

func TestCfC_HiddenStateResetBetweenWindows(t *testing.T) {
	config := CfCConfig{InputSize: 1, HiddenSize: 4, OutputSize: 1, NumLayers: 1, OutputLen: 2}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Two identical windows should produce identical outputs
	// because hidden state resets between windows.
	w := [][]float64{{1.0}, {2.0}, {3.0}}
	out1 := m.forward(w)
	// Run a different window first to "dirty" any global state (there shouldn't be any).
	_ = m.forward([][]float64{{99.0}, {-99.0}})
	out2 := m.forward(w)

	for i := range out1 {
		if out1[i] != out2[i] {
			t.Errorf("output[%d]: first=%v, second=%v — hidden state not reset", i, out1[i], out2[i])
		}
	}
}

func TestCfC_GradientVerification(t *testing.T) {
	config := CfCConfig{InputSize: 2, HiddenSize: 3, OutputSize: 1, NumLayers: 1, OutputLen: 1}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Use small weights for numerical stability.
	params := m.FlatParams()
	for _, p := range params {
		*p *= 0.1
	}

	input := [][]float64{{0.5, -0.3}, {0.2, 0.7}, {-0.1, 0.4}}
	eps := 1e-5
	outDim := config.OutputSize * config.OutputLen

	// Compute analytical gradients using vector-Jacobian product.
	// Use unit upstream gradient for each output to recover full Jacobian row.
	pred := m.forward(input)

	maxErr := 0.0
	nChecked := 0
	for oi := 0; oi < outDim; oi++ {
		dLoss := make([]float64, outDim)
		dLoss[oi] = 1.0
		analytical := m.backwardSample(input, dLoss)

		for pi, p := range params {
			orig := *p

			*p = orig + eps
			outPlus := m.forward(input)
			*p = orig - eps
			outMinus := m.forward(input)
			*p = orig

			numerical := (outPlus[oi] - outMinus[oi]) / (2 * eps)

			absDiff := math.Abs(numerical - analytical[pi])
			denom := math.Max(math.Abs(numerical)+math.Abs(analytical[pi]), 1e-8)
			relErr := absDiff / denom

			if relErr > maxErr {
				maxErr = relErr
			}
			nChecked++

			if relErr > 1e-3 && absDiff > 1e-6 {
				t.Errorf("param %d, output %d: analytical=%v, numerical=%v, relErr=%v",
					pi, oi, analytical[pi], numerical, relErr)
			}
		}
	}

	_ = pred
	t.Logf("gradient check: %d comparisons, max relative error = %.2e", nChecked, maxErr)
}

func TestCfC_TrainConvergence(t *testing.T) {
	config := CfCConfig{InputSize: 1, HiddenSize: 8, OutputSize: 1, NumLayers: 1, OutputLen: 4}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Generate synthetic linear ramp data.
	nSamples := 30
	inputLen := 8
	windows := make([][][]float64, nSamples)
	outDim := config.OutputSize * config.OutputLen
	labels := make([]float64, nSamples*outDim)

	for s := 0; s < nSamples; s++ {
		offset := float64(s) * 0.3
		windows[s] = make([][]float64, 1) // 1 channel
		windows[s][0] = make([]float64, inputLen)
		for i := 0; i < inputLen; i++ {
			windows[s][0][i] = offset + float64(i)*0.1
		}
		for o := 0; o < config.OutputLen; o++ {
			labels[s*outDim+o] = offset + float64(inputLen+o)*0.1
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:      80,
		LR:          1e-3,
		WeightDecay: 1e-5,
		GradClip:    1.0,
		Beta1:       0.9,
		Beta2:       0.999,
		Epsilon:     1e-8,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if len(result.LossHistory) != 80 {
		t.Fatalf("loss history length = %d, want 80", len(result.LossHistory))
	}

	firstLoss := result.LossHistory[0]
	lastLoss := result.FinalLoss
	if lastLoss >= firstLoss {
		t.Errorf("loss did not decrease: first=%v, last=%v", firstLoss, lastLoss)
	}

	t.Logf("convergence: first_loss=%.6f final_loss=%.6f ratio=%.4f", firstLoss, lastLoss, lastLoss/firstLoss)
}

func TestCfC_SaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cfc.json")

	config := CfCConfig{InputSize: 2, HiddenSize: 4, OutputSize: 1, NumLayers: 2, OutputLen: 3}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	if err := m.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	m2, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Two windows for prediction.
	testWindows := [][][]float64{
		{{1, 2, 3, 4}, {5, 6, 7, 8}},
		{{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}},
	}

	preds1, err := m.PredictWindowed("", testWindows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	preds2, err := m2.PredictWindowed(path, testWindows)
	if err != nil {
		t.Fatalf("PredictWindowed with load: %v", err)
	}

	for i := range preds1 {
		if preds1[i] != preds2[i] {
			t.Errorf("loaded model prediction[%d] = %v, want %v", i, preds2[i], preds1[i])
		}
	}
}

func TestCfC_EmptyInput(t *testing.T) {
	config := CfCConfig{InputSize: 1, HiddenSize: 4, OutputSize: 1, NumLayers: 1, OutputLen: 2}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	_, err = m.TrainWindowed(nil, nil, TrainConfig{Epochs: 10})
	if err == nil {
		t.Fatal("expected error for empty training set")
	}

	_, err = m.PredictWindowed("", nil)
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestCfC_LabelMismatch(t *testing.T) {
	config := CfCConfig{InputSize: 2, HiddenSize: 4, OutputSize: 1, NumLayers: 1, OutputLen: 3}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	windows := [][][]float64{
		{{1, 2, 3}, {4, 5, 6}},
	}
	// Wrong number of labels (should be 1*1*3 = 3).
	labels := []float64{1, 2}

	_, err = m.TrainWindowed(windows, labels, TrainConfig{Epochs: 10})
	if err == nil {
		t.Fatal("expected error for label count mismatch")
	}
}

func TestCfC_ParamCount(t *testing.T) {
	config := CfCConfig{InputSize: 2, HiddenSize: 3, OutputSize: 1, NumLayers: 1, OutputLen: 2}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Layer 0: Wh=3*3=9, Wx=2*3=6, Bh=3, Wtau=(2+3)*3=15, Btau=3 → 36
	// OutW: 3*2=6, OutB: 2 → 8
	// Total: 44
	want := 44
	got := m.paramCount()
	if got != want {
		t.Errorf("paramCount = %d, want %d", got, want)
	}

	params := m.FlatParams()
	if len(params) != want {
		t.Errorf("flatParams length = %d, want %d", len(params), want)
	}
}

func TestCfC_PredictWindowed_BadModelPath(t *testing.T) {
	config := CfCConfig{InputSize: 1, HiddenSize: 4, OutputSize: 1, NumLayers: 1, OutputLen: 2}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	_, err = m.PredictWindowed("/nonexistent/path.json", [][][]float64{{{1, 2, 3}}})
	if err == nil {
		t.Fatal("expected error for nonexistent model path")
	}
}

func TestCfC_SaveWeightsCreatesFile(t *testing.T) {
	config := CfCConfig{InputSize: 1, HiddenSize: 4, OutputSize: 1, NumLayers: 1, OutputLen: 2}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "weights.json")
	if err := m.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("weights file not created: %v", err)
	}
}

func TestCfC_TrainWindowed_Engine(t *testing.T) {
	config := CfCConfig{InputSize: 1, HiddenSize: 8, OutputSize: 1, NumLayers: 1, OutputLen: 4}
	engine, ops := newTestEngine()
	m, err := NewCfC(config, WithCfCEngine(engine, ops))
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Generate synthetic linear ramp data.
	nSamples := 30
	inputLen := 8
	windows := make([][][]float64, nSamples)
	outDim := config.OutputSize * config.OutputLen
	labels := make([]float64, nSamples*outDim)

	for s := 0; s < nSamples; s++ {
		offset := float64(s) * 0.3
		windows[s] = make([][]float64, 1) // 1 channel
		windows[s][0] = make([]float64, inputLen)
		for i := 0; i < inputLen; i++ {
			windows[s][0][i] = offset + float64(i)*0.1
		}
		for o := 0; o < config.OutputLen; o++ {
			labels[s*outDim+o] = offset + float64(inputLen+o)*0.1
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:      80,
		LR:          1e-3,
		WeightDecay: 1e-5,
		GradClip:    1.0,
		Beta1:       0.9,
		Beta2:       0.999,
		Epsilon:     1e-8,
	})
	if err != nil {
		t.Fatalf("TrainWindowed (engine): %v", err)
	}

	if len(result.LossHistory) != 80 {
		t.Fatalf("loss history length = %d, want 80", len(result.LossHistory))
	}

	firstLoss := result.LossHistory[0]
	lastLoss := result.FinalLoss
	if lastLoss >= firstLoss {
		t.Errorf("engine loss did not decrease: first=%v, last=%v", firstLoss, lastLoss)
	}

	// Verify all weights are finite.
	assertFiniteWeights(t, m.FlatParams())
	t.Logf("engine convergence: first_loss=%.6f final_loss=%.6f ratio=%.4f", firstLoss, lastLoss, lastLoss/firstLoss)
}

func TestCfC_TrainWindowed_MultiScale(t *testing.T) {
	// Issue #121: training on data with features spanning 10 orders of magnitude
	// previously produced NaN/Inf weights. Normalization should prevent this.
	nChannels := 5
	inputLen := 8
	outputLen := 1
	config := CfCConfig{
		InputSize:  nChannels,
		HiddenSize: 16,
		NumLayers:  1,
		OutputSize: 1,
		OutputLen:  outputLen,
	}

	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	windows, labels := makeMultiScaleWindows(200, nChannels, inputLen, outputLen)

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       20,
		LR:           1e-3,
		GradClip:     1.0,
		WarmupEpochs: 5,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	// Verify all weights are finite.
	assertFiniteWeights(t, m.FlatParams())
	t.Logf("multi-scale training: final_loss=%.6f (20 epochs, 5 channels, 200 samples)", result.FinalLoss)
}

func TestCfC_PredictWindowed_NormalizationApplied(t *testing.T) {
	nChannels := 3
	inputLen := 8
	m, err := NewCfC(CfCConfig{
		InputSize:  nChannels,
		HiddenSize: 8,
		OutputSize: 2,
		NumLayers:  1,
		OutputLen:  3,
	})
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	windows, labels := makeMultiScaleWindows(100, nChannels, inputLen, 2*3)

	_, err = m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:       10,
		LR:           1e-3,
		GradClip:     1.0,
		WarmupEpochs: 3,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if m.normMeans == nil || m.normStds == nil {
		t.Fatal("normMeans/normStds not stored after training")
	}

	preds, err := m.PredictWindowed("", windows[:5])
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("prediction[%d] = %v, want finite", i, v)
		}
	}
}

func TestCfC_ForwardBatch_Parity(t *testing.T) {
	config := CfCConfig{InputSize: 3, HiddenSize: 8, OutputSize: 2, NumLayers: 2, OutputLen: 4}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Create batch of 7 samples with 3 channels, inputLen=5.
	batchSize := 7
	nChannels := 3
	inputLen := 5
	windows := make([][][]float64, batchSize)
	for s := 0; s < batchSize; s++ {
		windows[s] = make([][]float64, nChannels)
		for c := 0; c < nChannels; c++ {
			windows[s][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				windows[s][c][i] = float64(s*100+c*10+i) * 0.01
			}
		}
	}

	// Per-sample forward pass (reference).
	outDim := config.OutputSize * config.OutputLen
	perSamplePreds := make([][]float64, batchSize)
	for s := 0; s < batchSize; s++ {
		seqInput := transposeWindow(windows[s])
		pred := m.forward(seqInput)
		perSamplePreds[s] = pred
	}

	// Batched forward pass.
	batchPreds := m.ForwardBatch(windows)

	if len(batchPreds) != batchSize {
		t.Fatalf("batched preds length = %d, want %d", len(batchPreds), batchSize)
	}

	const tol = 1e-12
	for s := 0; s < batchSize; s++ {
		if len(batchPreds[s]) != outDim {
			t.Fatalf("sample %d: batched pred length = %d, want %d", s, len(batchPreds[s]), outDim)
		}
		for j := 0; j < outDim; j++ {
			diff := math.Abs(batchPreds[s][j] - perSamplePreds[s][j])
			if diff > tol {
				t.Errorf("sample %d output[%d]: batched=%.15f, per-sample=%.15f, diff=%.4e",
					s, j, batchPreds[s][j], perSamplePreds[s][j], diff)
			}
		}
	}

	t.Logf("parity check: %d samples, %d channels, %d layers — all predictions match within %.0e",
		batchSize, nChannels, config.NumLayers, tol)
}

func TestCfC_ForwardBatch_EmptyBatch(t *testing.T) {
	config := CfCConfig{InputSize: 2, HiddenSize: 4, OutputSize: 1, NumLayers: 1, OutputLen: 3}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	out := m.ForwardBatch(nil)
	if out != nil {
		t.Fatalf("expected nil for empty batch, got %v", out)
	}
}

func TestCfC_ForwardBatch_SingleSample(t *testing.T) {
	config := CfCConfig{InputSize: 2, HiddenSize: 6, OutputSize: 1, NumLayers: 1, OutputLen: 2}
	m, err := NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	windows := [][][]float64{
		{{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}},
	}

	seqInput := transposeWindow(windows[0])
	singlePred := m.forward(seqInput)
	batchPreds := m.ForwardBatch(windows)

	if len(batchPreds) != 1 {
		t.Fatalf("batch preds length = %d, want 1", len(batchPreds))
	}
	for j := range singlePred {
		if batchPreds[0][j] != singlePred[j] {
			t.Errorf("output[%d]: batched=%v, single=%v", j, batchPreds[0][j], singlePred[j])
		}
	}
}
