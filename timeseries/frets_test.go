package timeseries

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNewFreTS_Validation(t *testing.T) {
	tests := []struct {
		name    string
		config  FreTSConfig
		wantErr bool
	}{
		{"valid", FreTSConfig{Channels: 2, InputLen: 24, OutputLen: 12, TopK: 4, HiddenSize: 16}, false},
		{"single channel", FreTSConfig{Channels: 1, InputLen: 10, OutputLen: 5, TopK: 3, HiddenSize: 8}, false},
		{"zero Channels", FreTSConfig{Channels: 0, InputLen: 24, OutputLen: 12, TopK: 4, HiddenSize: 16}, true},
		{"zero InputLen", FreTSConfig{Channels: 2, InputLen: 0, OutputLen: 12, TopK: 4, HiddenSize: 16}, true},
		{"zero OutputLen", FreTSConfig{Channels: 2, InputLen: 24, OutputLen: 0, TopK: 4, HiddenSize: 16}, true},
		{"zero TopK", FreTSConfig{Channels: 2, InputLen: 24, OutputLen: 12, TopK: 0, HiddenSize: 16}, true},
		{"zero HiddenSize", FreTSConfig{Channels: 2, InputLen: 24, OutputLen: 12, TopK: 4, HiddenSize: 0}, true},
		{"negative Channels", FreTSConfig{Channels: -1, InputLen: 24, OutputLen: 12, TopK: 4, HiddenSize: 16}, true},
		{"TopK exceeds max freqs", FreTSConfig{Channels: 2, InputLen: 4, OutputLen: 2, TopK: 4, HiddenSize: 8}, true}, // max freqs = 4/2+1 = 3
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewFreTS(tt.config)
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

func TestFreTS_ForwardOutputShape(t *testing.T) {
	config := FreTSConfig{Channels: 3, InputLen: 16, OutputLen: 8, TopK: 4, HiddenSize: 16}
	m, err := NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS: %v", err)
	}

	input := make([][]float64, 3)
	for c := 0; c < 3; c++ {
		input[c] = make([]float64, 16)
		for i := range input[c] {
			input[c][i] = float64(i) * 0.1
		}
	}

	out := m.forward(input)
	if len(out) != 3 {
		t.Fatalf("output channels = %d, want 3", len(out))
	}
	for c, ch := range out {
		if len(ch) != 8 {
			t.Errorf("channel %d length = %d, want 8", c, len(ch))
		}
		for i, v := range ch {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("output[%d][%d] = %v, want finite", c, i, v)
			}
		}
	}
}

func TestFreTS_DFTRoundTrip(t *testing.T) {
	// Verify DFT -> IDFT round-trip.
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	coeffs := dft(x)
	reconstructed := idft(coeffs, len(x))

	for i := range x {
		if math.Abs(reconstructed[i]-x[i]) > 1e-10 {
			t.Errorf("reconstructed[%d] = %v, want %v", i, reconstructed[i], x[i])
		}
	}

	// Odd length.
	y := []float64{1, 3, 5, 7, 9}
	coeffsY := dft(y)
	reconY := idft(coeffsY, len(y))
	for i := range y {
		if math.Abs(reconY[i]-y[i]) > 1e-10 {
			t.Errorf("reconstructed[%d] = %v, want %v", i, reconY[i], y[i])
		}
	}
}

func TestFreTS_Convergence(t *testing.T) {
	// FreTS should learn sinusoidal patterns well since it operates in frequency domain.
	inputLen := 32
	outputLen := 8
	channels := 1
	nSamples := 50

	config := FreTSConfig{
		Channels:   channels,
		InputLen:   inputLen,
		OutputLen:  outputLen,
		TopK:       4,
		HiddenSize: 16,
	}
	m, err := NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS: %v", err)
	}

	// Generate synthetic sinusoidal data.
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*channels*outputLen)

	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, channels)
		windows[s][0] = make([]float64, inputLen)
		offset := float64(s) * 0.5
		for i := 0; i < inputLen; i++ {
			windows[s][0][i] = math.Sin(2*math.Pi*float64(i)/16.0 + offset)
		}
		for o := 0; o < outputLen; o++ {
			labels[s*outputLen+o] = math.Sin(2*math.Pi*float64(inputLen+o)/16.0 + offset)
		}
	}

	tc := DefaultTrainConfig()
	tc.Epochs = 50
	tc.LR = 1e-3
	tc.GradClip = 1.0

	result, err := m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Loss should decrease.
	if result.LossHistory[len(result.LossHistory)-1] >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: first=%v, last=%v",
			result.LossHistory[0], result.LossHistory[len(result.LossHistory)-1])
	}

	// Final loss should be finite.
	if !isFinite(result.FinalLoss) {
		t.Errorf("final loss is not finite: %v", result.FinalLoss)
	}
}

func TestFreTS_NaNProtection(t *testing.T) {
	// Multi-scale data: one channel with large values, one with tiny values.
	inputLen := 16
	outputLen := 4
	channels := 2
	nSamples := 20

	config := FreTSConfig{
		Channels:   channels,
		InputLen:   inputLen,
		OutputLen:  outputLen,
		TopK:       3,
		HiddenSize: 8,
	}
	m, err := NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS: %v", err)
	}

	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*channels*outputLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, channels)
		// Channel 0: large scale.
		windows[s][0] = make([]float64, inputLen)
		for i := 0; i < inputLen; i++ {
			windows[s][0][i] = 1e6 * math.Sin(float64(i)*0.5+float64(s))
		}
		// Channel 1: tiny scale.
		windows[s][1] = make([]float64, inputLen)
		for i := 0; i < inputLen; i++ {
			windows[s][1][i] = 1e-6 * math.Cos(float64(i)*0.3+float64(s))
		}
		for c := 0; c < channels; c++ {
			for o := 0; o < outputLen; o++ {
				labels[s*channels*outputLen+c*outputLen+o] = windows[s][c][inputLen-1]
			}
		}
	}

	tc := DefaultTrainConfig()
	tc.Epochs = 30
	tc.LR = 1e-4
	tc.GradClip = 1.0

	result, err := m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// No NaN/Inf in predictions.
	preds, err := m.PredictWindowed("", windows)
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}
	for i, v := range preds {
		if !isFinite(v) {
			t.Errorf("prediction[%d] = %v, want finite", i, v)
		}
	}

	if !isFinite(result.FinalLoss) {
		t.Errorf("final loss is not finite: %v", result.FinalLoss)
	}
}

func TestFreTS_SaveLoadRoundTrip(t *testing.T) {
	inputLen := 16
	outputLen := 4
	channels := 2

	config := FreTSConfig{
		Channels:   channels,
		InputLen:   inputLen,
		OutputLen:  outputLen,
		TopK:       3,
		HiddenSize: 8,
	}

	m, err := NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS: %v", err)
	}

	// Generate some training data.
	nSamples := 20
	windows := make([][][]float64, nSamples)
	labels := make([]float64, nSamples*channels*outputLen)
	for s := 0; s < nSamples; s++ {
		windows[s] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			windows[s][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				windows[s][c][i] = math.Sin(float64(i)*0.5 + float64(s)*0.1 + float64(c))
			}
		}
		for c := 0; c < channels; c++ {
			for o := 0; o < outputLen; o++ {
				labels[s*channels*outputLen+c*outputLen+o] = math.Sin(float64(inputLen+o)*0.5 + float64(s)*0.1 + float64(c))
			}
		}
	}

	tc := DefaultTrainConfig()
	tc.Epochs = 10
	tc.LR = 1e-3

	_, err = m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	// Save.
	dir := t.TempDir()
	path := filepath.Join(dir, "frets.json")
	if err := m.SaveWeights(path); err != nil {
		t.Fatalf("SaveWeights: %v", err)
	}

	// Verify file exists.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("saved file not found: %v", err)
	}

	// Get predictions from original model.
	testWindows := windows[:3]
	pred1, err := m.PredictWindowed("", testWindows)
	if err != nil {
		t.Fatalf("PredictWindowed (original): %v", err)
	}

	// Load into a new model.
	m2, err := NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS (2): %v", err)
	}

	pred2, err := m2.PredictWindowed(path, testWindows)
	if err != nil {
		t.Fatalf("PredictWindowed (loaded): %v", err)
	}

	// Predictions should match.
	if len(pred1) != len(pred2) {
		t.Fatalf("prediction lengths differ: %d vs %d", len(pred1), len(pred2))
	}
	for i := range pred1 {
		if math.Abs(pred1[i]-pred2[i]) > 1e-10 {
			t.Errorf("prediction[%d] mismatch: original=%v, loaded=%v", i, pred1[i], pred2[i])
		}
	}
}

func TestFreTS_SingleChannel(t *testing.T) {
	config := FreTSConfig{
		Channels:   1,
		InputLen:   16,
		OutputLen:  4,
		TopK:       3,
		HiddenSize: 8,
	}
	m, err := NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS: %v", err)
	}

	input := make([][]float64, 1)
	input[0] = make([]float64, 16)
	for i := range input[0] {
		input[0][i] = math.Sin(float64(i) * 0.5)
	}

	out := m.forward(input)
	if len(out) != 1 {
		t.Fatalf("output channels = %d, want 1", len(out))
	}
	if len(out[0]) != 4 {
		t.Fatalf("output length = %d, want 4", len(out[0]))
	}
	for i, v := range out[0] {
		if !isFinite(v) {
			t.Errorf("output[0][%d] = %v, want finite", i, v)
		}
	}
}

func TestFreTS_SingleSample(t *testing.T) {
	config := FreTSConfig{
		Channels:   2,
		InputLen:   16,
		OutputLen:  4,
		TopK:       3,
		HiddenSize: 8,
	}
	m, err := NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS: %v", err)
	}

	// Train with a single sample.
	windows := make([][][]float64, 1)
	windows[0] = make([][]float64, 2)
	for c := 0; c < 2; c++ {
		windows[0][c] = make([]float64, 16)
		for i := 0; i < 16; i++ {
			windows[0][c][i] = float64(i) * 0.1
		}
	}
	labels := make([]float64, 2*4)
	for i := range labels {
		labels[i] = float64(i) * 0.1
	}

	tc := DefaultTrainConfig()
	tc.Epochs = 5
	tc.LR = 1e-3

	result, err := m.TrainWindowed(windows, labels, tc)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}
	if !isFinite(result.FinalLoss) {
		t.Errorf("final loss is not finite: %v", result.FinalLoss)
	}
}
