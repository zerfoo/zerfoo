package timeseries

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNewDLinear_Validation(t *testing.T) {
	tests := []struct {
		name       string
		inputLen   int
		outputLen  int
		channels   int
		kernelSize int
		wantErr    bool
	}{
		{"valid", 24, 12, 3, 5, false},
		{"single channel", 10, 5, 1, 3, false},
		{"zero inputLen", 0, 5, 1, 3, true},
		{"zero outputLen", 10, 0, 1, 3, true},
		{"zero channels", 10, 5, 0, 3, true},
		{"zero kernelSize", 10, 5, 1, 0, true},
		{"even kernelSize", 10, 5, 1, 4, true},
		{"negative inputLen", -1, 5, 1, 3, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewDLinear(tt.inputLen, tt.outputLen, tt.channels, tt.kernelSize)
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
			if m.config.InputLen != tt.inputLen {
				t.Errorf("inputLen = %d, want %d", m.config.InputLen, tt.inputLen)
			}
			if m.config.OutputLen != tt.outputLen {
				t.Errorf("outputLen = %d, want %d", m.config.OutputLen, tt.outputLen)
			}
			if m.config.Channels != tt.channels {
				t.Errorf("channels = %d, want %d", m.config.Channels, tt.channels)
			}
		})
	}
}

func TestMovingAverage(t *testing.T) {
	t.Run("kernel 3", func(t *testing.T) {
		x := []float64{1, 2, 3, 4, 5}
		got := movingAverage(x, 3)

		// With edge padding:
		// i=0: avg(x[0], x[0], x[1]) = (1+1+2)/3
		// i=1: avg(x[0], x[1], x[2]) = (1+2+3)/3 = 2
		// i=2: avg(x[1], x[2], x[3]) = (2+3+4)/3 = 3
		// i=3: avg(x[2], x[3], x[4]) = (3+4+5)/3 = 4
		// i=4: avg(x[3], x[4], x[4]) = (4+5+5)/3
		want := []float64{4.0 / 3.0, 2.0, 3.0, 4.0, 14.0 / 3.0}

		if len(got) != len(want) {
			t.Fatalf("length = %d, want %d", len(got), len(want))
		}
		for i := range want {
			if math.Abs(got[i]-want[i]) > 1e-10 {
				t.Errorf("movingAverage[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("kernel 1 is identity", func(t *testing.T) {
		x := []float64{3, 1, 4, 1, 5}
		got := movingAverage(x, 1)
		for i := range x {
			if got[i] != x[i] {
				t.Errorf("[%d] = %v, want %v", i, got[i], x[i])
			}
		}
	})
}

func TestDLinear_Decomposition(t *testing.T) {
	m, err := NewDLinear(10, 5, 2, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	input := make([][]float64, 2)
	for c := 0; c < 2; c++ {
		input[c] = make([]float64, 10)
		for i := 0; i < 10; i++ {
			input[c][i] = float64(i) + float64(c)*10
		}
	}

	trend, seasonal := m.decompose(input)

	if len(trend) != 2 || len(seasonal) != 2 {
		t.Fatalf("decompose returned wrong channel count")
	}

	// trend + seasonal should equal original.
	for c := 0; c < 2; c++ {
		for i := 0; i < 10; i++ {
			recon := trend[c][i] + seasonal[c][i]
			if math.Abs(recon-input[c][i]) > 1e-10 {
				t.Errorf("channel %d pos %d: trend+seasonal = %v, want %v", c, i, recon, input[c][i])
			}
		}
	}
}

func TestDLinear_ForwardOutputShape(t *testing.T) {
	m, err := NewDLinear(10, 5, 3, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	input := make([][]float64, 3)
	for c := 0; c < 3; c++ {
		input[c] = make([]float64, 10)
		for i := range input[c] {
			input[c][i] = float64(i) * 0.1
		}
	}

	out := m.forward(input)
	if len(out) != 3 {
		t.Fatalf("output channels = %d, want 3", len(out))
	}
	for c, ch := range out {
		if len(ch) != 5 {
			t.Errorf("channel %d length = %d, want 5", c, len(ch))
		}
		for i, v := range ch {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("output[%d][%d] = %v, want finite", c, i, v)
			}
		}
	}
}

func TestDLinear_TrainConvergence(t *testing.T) {
	// Train on synthetic sine wave and verify loss decreases.
	inputLen := 24
	outputLen := 12
	channels := 1
	nSamples := 50

	m, err := NewDLinear(inputLen, outputLen, channels, 5)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	// Generate sine wave windows.
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*channels*outputLen)

	for s := 0; s < nSamples; s++ {
		offset := float64(s) * 0.5
		windows[s] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			windows[s][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				windows[s][c][i] = math.Sin(2*math.Pi*float64(i+int(offset))/24.0) + float64(c)*0.1
			}
		}
		for c := 0; c < channels; c++ {
			for o := 0; o < outputLen; o++ {
				labels[s*channels*outputLen+c*outputLen+o] = math.Sin(2*math.Pi*float64(inputLen+o+int(offset))/24.0) + float64(c)*0.1
			}
		}
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:      50,
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

	if len(result.LossHistory) != 50 {
		t.Fatalf("loss history length = %d, want 50", len(result.LossHistory))
	}

	// Loss should decrease over training.
	firstLoss := result.LossHistory[0]
	lastLoss := result.FinalLoss
	if lastLoss >= firstLoss {
		t.Errorf("loss did not decrease: first=%v, last=%v", firstLoss, lastLoss)
	}

	// Final loss should be much lower than initial.
	if lastLoss > firstLoss*0.9 {
		t.Errorf("loss decreased insufficiently: first=%v, last=%v (ratio=%v)", firstLoss, lastLoss, lastLoss/firstLoss)
	}
}

func TestDLinear_TrainThenPredict(t *testing.T) {
	inputLen := 12
	outputLen := 6
	channels := 2
	nSamples := 30

	m, err := NewDLinear(inputLen, outputLen, channels, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*channels*outputLen)

	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			windows[s][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				windows[s][c][i] = float64(i)*0.1 + float64(c)
			}
		}
		for c := 0; c < channels; c++ {
			for o := 0; o < outputLen; o++ {
				labels[s*channels*outputLen+c*outputLen+o] = float64(inputLen+o)*0.1 + float64(c)
			}
		}
	}

	_, err = m.TrainWindowed(windows, labels, TrainConfig{
		Epochs: 30,
		LR:     1e-3,
	})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Predict on training data.
	preds, err := m.PredictWindowed("", windows[:5])
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	expectedLen := 5 * channels * outputLen
	if len(preds) != expectedLen {
		t.Fatalf("predictions length = %d, want %d", len(preds), expectedLen)
	}

	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("prediction[%d] = %v, want finite", i, v)
			break
		}
	}
}

func TestDLinear_SaveLoadWeights(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dlinear.json")

	m, err := NewDLinear(10, 5, 2, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	if err := m.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	m2, err := NewDLinear(10, 5, 2, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	preds1, err := m.PredictWindowed("", [][][]float64{{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}}})
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	preds2, err := m2.PredictWindowed(path, [][][]float64{{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}}})
	if err != nil {
		t.Fatalf("PredictWindowed with load: %v", err)
	}

	for i := range preds1 {
		if preds1[i] != preds2[i] {
			t.Errorf("loaded model prediction[%d] = %v, want %v", i, preds2[i], preds1[i])
		}
	}
}

func TestDLinear_SingleChannel(t *testing.T) {
	m, err := NewDLinear(8, 4, 1, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	windows := [][][]float64{
		{{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	labels := []float64{9, 10, 11, 12}

	_, err = m.TrainWindowed(windows, labels, TrainConfig{Epochs: 10, LR: 1e-3})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	preds, err := m.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}
	if len(preds) != 4 {
		t.Fatalf("predictions length = %d, want 4", len(preds))
	}
}

func TestDLinear_VeryShortWindow(t *testing.T) {
	// Minimum viable: inputLen=1, outputLen=1, channels=1, kernelSize=1.
	m, err := NewDLinear(1, 1, 1, 1)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	windows := [][][]float64{{{5.0}}, {{10.0}}}
	labels := []float64{6.0, 11.0}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{Epochs: 50, LR: 1e-2})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if math.IsNaN(result.FinalLoss) || math.IsInf(result.FinalLoss, 0) {
		t.Errorf("final loss is not finite: %v", result.FinalLoss)
	}
}

func TestDLinear_EmptyInput(t *testing.T) {
	m, err := NewDLinear(10, 5, 1, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
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

func TestDLinear_PredictWindowed_BadModelPath(t *testing.T) {
	m, err := NewDLinear(10, 5, 1, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	_, err = m.PredictWindowed("/nonexistent/path.json", [][][]float64{{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}})
	if err == nil {
		t.Fatal("expected error for nonexistent model path")
	}
}

func TestDLinear_ParamCount(t *testing.T) {
	m, err := NewDLinear(10, 5, 3, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	// Per channel: 2 * (5*10 + 5) = 2 * 55 = 110
	// 3 channels: 330
	want := 330
	got := m.paramCount()
	if got != want {
		t.Errorf("paramCount = %d, want %d", got, want)
	}

	// Verify flatParams matches.
	params := m.FlatParams()
	if len(params) != want {
		t.Errorf("flatParams length = %d, want %d", len(params), want)
	}
}

func TestTrainResult_ModelPathAndMetrics(t *testing.T) {
	m, err := NewDLinear(8, 4, 1, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	windows := [][][]float64{
		{{1, 2, 3, 4, 5, 6, 7, 8}},
		{{2, 3, 4, 5, 6, 7, 8, 9}},
	}
	labels := []float64{9, 10, 11, 12, 10, 11, 12, 13}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{Epochs: 5, LR: 1e-3})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// ModelPath should be empty since TrainWindowed does not save weights.
	if result.ModelPath != "" {
		t.Errorf("ModelPath = %q, want empty string", result.ModelPath)
	}

	// Metrics should contain at least "mse".
	if result.Metrics == nil {
		t.Fatal("Metrics is nil, want non-nil map")
	}
	mse, ok := result.Metrics["mse"]
	if !ok {
		t.Fatal("Metrics missing 'mse' key")
	}
	if mse != result.FinalLoss {
		t.Errorf("Metrics['mse'] = %v, want %v (FinalLoss)", mse, result.FinalLoss)
	}
}

func TestDLinear_TrainWindowed_LabelMismatch(t *testing.T) {
	m, err := NewDLinear(10, 5, 2, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	windows := [][][]float64{
		{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {10, 9, 8, 7, 6, 5, 4, 3, 2, 1}},
	}
	// Wrong number of labels (should be 1*2*5 = 10).
	labels := []float64{1, 2, 3}

	_, err = m.TrainWindowed(windows, labels, TrainConfig{Epochs: 10})
	if err == nil {
		t.Fatal("expected error for label count mismatch")
	}
}

func TestDLinear_SaveWeightsPath(t *testing.T) {
	m, err := NewDLinear(5, 3, 1, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
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

func TestIsFinite(t *testing.T) {
	tests := []struct {
		name string
		v    float64
		want bool
	}{
		{"zero", 0, true},
		{"positive", 3.14, true},
		{"negative", -1.5, true},
		{"nan", math.NaN(), false},
		{"pos_inf", math.Inf(1), false},
		{"neg_inf", math.Inf(-1), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isFinite(tt.v); got != tt.want {
				t.Errorf("isFinite(%v) = %v, want %v", tt.v, got, tt.want)
			}
		})
	}
}

func TestNormalizeWindows(t *testing.T) {
	// 3 samples, 2 channels, 4 timesteps. Channel 0 has values ~100,
	// channel 1 has values ~0.001. After normalization both should be ~O(1).
	windows := [][][]float64{
		{{100, 200, 300, 400}, {0.001, 0.002, 0.003, 0.004}},
		{{110, 210, 310, 410}, {0.0015, 0.0025, 0.0035, 0.0045}},
		{{90, 190, 290, 390}, {0.0005, 0.0015, 0.0025, 0.0035}},
	}
	out, means, stds := normalizeWindows(windows)

	// Check dimensions.
	if len(out) != 3 {
		t.Fatalf("len(out) = %d, want 3", len(out))
	}
	if len(means) != 2 || len(stds) != 2 {
		t.Fatalf("means/stds channels = %d/%d, want 2/2", len(means), len(stds))
	}

	// Verify normalized values are roughly zero-mean unit-variance.
	for c := 0; c < 2; c++ {
		for ts := 0; ts < 4; ts++ {
			var sum, sumSq float64
			for i := 0; i < 3; i++ {
				sum += out[i][c][ts]
				sumSq += out[i][c][ts] * out[i][c][ts]
			}
			mean := sum / 3.0
			if math.Abs(mean) > 1e-6 {
				t.Errorf("channel %d timestep %d: normalized mean = %v, want ~0", c, ts, mean)
			}
		}
	}

	// All outputs must be finite.
	for i := range out {
		for c := range out[i] {
			for ts := range out[i][c] {
				if !isFinite(out[i][c][ts]) {
					t.Errorf("out[%d][%d][%d] = %v, want finite", i, c, ts, out[i][c][ts])
				}
			}
		}
	}
}

func TestNormalizeWindows_MultiScale(t *testing.T) {
	// Simulate features spanning 10 orders of magnitude (like issue #121).
	// 100 samples, 5 channels with scales: 1e-4, 1e-1, 1e2, 1e4, 1e6.
	scales := []float64{1e-4, 1e-1, 1e2, 1e4, 1e6}
	nSamples := 100
	inputLen := 10
	windows := make([][][]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		windows[i] = make([][]float64, len(scales))
		for c, scale := range scales {
			windows[i][c] = make([]float64, inputLen)
			for ts := 0; ts < inputLen; ts++ {
				windows[i][c][ts] = scale * (1.0 + 0.1*float64(i%10))
			}
		}
	}

	out, means, stds := normalizeWindows(windows)

	if len(out) != nSamples {
		t.Fatalf("len(out) = %d, want %d", len(out), nSamples)
	}
	if len(means) != len(scales) {
		t.Fatalf("len(means) = %d, want %d", len(means), len(scales))
	}

	// Every normalized value must be finite and within a reasonable range.
	for i := range out {
		for c := range out[i] {
			for ts := range out[i][c] {
				v := out[i][c][ts]
				if !isFinite(v) {
					t.Fatalf("out[%d][%d][%d] = %v, want finite", i, c, ts, v)
				}
				if math.Abs(v) > 100 {
					t.Errorf("out[%d][%d][%d] = %v, unexpectedly large after normalization", i, c, ts, v)
				}
			}
		}
	}

	// Stds should be positive for all channels.
	for c := range stds {
		for ts := range stds[c] {
			if stds[c][ts] <= 0 {
				t.Errorf("stds[%d][%d] = %v, want positive", c, ts, stds[c][ts])
			}
		}
	}
}

func TestNormalizeWindows_Empty(t *testing.T) {
	out, means, stds := normalizeWindows(nil)
	if out != nil {
		t.Errorf("expected nil output for nil input")
	}
	if means != nil || stds != nil {
		t.Errorf("expected nil means/stds for nil input")
	}
}

func TestDLinear_TrainWindowed_MultiScale(t *testing.T) {
	// Issue #121: training on data with features spanning 10 orders of magnitude
	// previously produced NaN/Inf weights. Normalization should prevent this.
	nChannels := 5
	inputLen := 10
	outputLen := 3
	m, err := NewDLinear(inputLen, outputLen, nChannels, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	windows, labels := makeMultiScaleWindows(500, nChannels, inputLen, nChannels*outputLen)

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
	t.Logf("multi-scale training: final_loss=%.6f (20 epochs, 5 channels, 500 samples)", result.FinalLoss)
}

func TestDLinear_PredictWindowed_NormalizationApplied(t *testing.T) {
	inputLen := 10
	outputLen := 3
	nChannels := 3
	m, err := NewDLinear(inputLen, outputLen, nChannels, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	windows, labels := makeMultiScaleWindows(100, nChannels, inputLen, nChannels*outputLen)

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
