package timeseries

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestFreTS_TrainWindowed_Engine(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
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
	m, err := NewFreTS(config, WithFreTSEngine(eng, numeric.Float32Ops{}))
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
		t.Fatalf("TrainWindowed (engine): %v", err)
	}

	if len(result.LossHistory) != 50 {
		t.Fatalf("loss history length = %d, want 50", len(result.LossHistory))
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	// Loss should decrease over training.
	firstLoss := result.LossHistory[0]
	lastLoss := result.FinalLoss
	if lastLoss >= firstLoss {
		t.Errorf("loss did not decrease: first=%v, last=%v", firstLoss, lastLoss)
	}

	t.Logf("engine training: first_loss=%.6f, final_loss=%.6f", firstLoss, lastLoss)
}

func TestFreTS_ForwardBatch_MatchesSampleBySample(t *testing.T) {
	inputLen := 32
	outputLen := 8
	channels := 3
	batch := 10

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

	// Build input: [batch][channels][inputLen].
	inputs := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		inputs[b] = make([][]float64, channels)
		for c := 0; c < channels; c++ {
			inputs[b][c] = make([]float64, inputLen)
			for i := 0; i < inputLen; i++ {
				inputs[b][c][i] = math.Sin(float64(b*channels*inputLen+c*inputLen+i)*0.3) + float64(c)*0.5
			}
		}
	}

	// Sample-by-sample reference.
	want := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		want[b] = m.forward(inputs[b])
	}

	// Batched forward.
	got := m.forwardBatch(inputs)

	if len(got) != batch {
		t.Fatalf("forwardBatch returned %d samples, want %d", len(got), batch)
	}
	for b := 0; b < batch; b++ {
		if len(got[b]) != channels {
			t.Fatalf("sample %d: got %d channels, want %d", b, len(got[b]), channels)
		}
		for c := 0; c < channels; c++ {
			if len(got[b][c]) != outputLen {
				t.Fatalf("sample %d channel %d: got len %d, want %d", b, c, len(got[b][c]), outputLen)
			}
			for o := 0; o < outputLen; o++ {
				diff := math.Abs(got[b][c][o] - want[b][c][o])
				if diff > 1e-9 {
					t.Errorf("sample %d channel %d output %d: got %v, want %v (diff %v)",
						b, c, o, got[b][c][o], want[b][c][o], diff)
				}
			}
		}
	}
}

func TestFreTS_TrainWindowed_NilEngine_Unchanged(t *testing.T) {
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
	if m.engine != nil {
		t.Fatal("expected nil engine by default")
	}

	windows := make([][][]float64, 1)
	windows[0] = make([][]float64, 1)
	windows[0][0] = make([]float64, 16)
	for i := range windows[0][0] {
		windows[0][0][i] = float64(i) * 0.1
	}
	labels := make([]float64, 4)
	for i := range labels {
		labels[i] = float64(i) * 0.1
	}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{Epochs: 5, LR: 1e-3})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}
	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}
}
