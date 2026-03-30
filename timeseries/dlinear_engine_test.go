package timeseries

import (
	"math"
	"testing"
	"time"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestDLinear_TrainWindowed_Engine_Convergence(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	inputLen := 24
	outputLen := 12
	channels := 1
	nSamples := 50

	m, err := NewDLinear(inputLen, outputLen, channels, 5, WithEngine(eng))
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	// Generate sine wave windows (same as TestDLinear_TrainConvergence).
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
		t.Fatalf("TrainWindowed (engine): %v", err)
	}

	if len(result.LossHistory) != 50 {
		t.Fatalf("loss history length = %d, want 50", len(result.LossHistory))
	}

	// Loss should be finite.
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

func TestDLinear_TrainWindowed_NilEngine_Unchanged(t *testing.T) {
	// Verify nil engine uses the CPU path (existing behavior).
	m, err := NewDLinear(12, 6, 1, 3)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}
	if m.engine != nil {
		t.Fatal("expected nil engine by default")
	}

	windows := [][][]float64{
		{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
	}
	labels := []float64{13, 14, 15, 16, 17, 18}

	result, err := m.TrainWindowed(windows, labels, TrainConfig{Epochs: 5, LR: 1e-3})
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}
	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}
}

func TestDLinear_TrainWindowed_Engine_MultiChannel(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	inputLen := 10
	outputLen := 3
	channels := 3
	nSamples := 30

	m, err := NewDLinear(inputLen, outputLen, channels, 3, WithEngine(eng))
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

	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:  30,
		LR:      1e-3,
		GradClip: 1.0,
	})
	if err != nil {
		t.Fatalf("TrainWindowed (engine, multi-channel): %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}

	// Verify predictions are finite after training.
	preds, err := m.PredictWindowed("", windows[:5])
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}
	for i, v := range preds {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("prediction[%d] = %v, want finite", i, v)
			break
		}
	}

	t.Logf("engine multi-channel training: final_loss=%.6f", result.FinalLoss)
}

// TestDLinear_TrainWindowed_Engine_Issue172 reproduces the scenario from issue
// #172: 500 samples, 20 channels, 3 epochs. The batched engine path should
// complete in well under 1 second on CPU (the old per-sample path took ~3.7s
// on GPU due to allocation overhead).
func TestDLinear_TrainWindowed_Engine_Issue172(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	inputLen := 24
	outputLen := 12
	channels := 20
	nSamples := 500

	m, err := NewDLinear(inputLen, outputLen, channels, 5, WithEngine(eng))
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

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

	start := time.Now()
	result, err := m.TrainWindowed(windows, labels, TrainConfig{
		Epochs:   3,
		LR:       1e-3,
		GradClip: 1.0,
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("TrainWindowed: %v", err)
	}

	if !isFinite(result.FinalLoss) {
		t.Fatalf("final loss = %v, want finite", result.FinalLoss)
	}
	if result.FinalLoss >= result.LossHistory[0] {
		t.Errorf("loss did not decrease: first=%v, last=%v", result.LossHistory[0], result.FinalLoss)
	}

	t.Logf("issue #172 scenario: 500 samples × 20 channels × 3 epochs = %.3fs, final_loss=%.6f", elapsed.Seconds(), result.FinalLoss)
	if elapsed > 2*time.Second {
		t.Errorf("training took %v, expected <2s (batched engine should eliminate per-sample allocation overhead)", elapsed)
	}
}

func TestDLinear_ForwardBatch_MatchesSampleBySample(t *testing.T) {
	inputLen := 12
	outputLen := 6
	channels := 3
	batch := 10

	m, err := NewDLinear(inputLen, outputLen, channels, 5)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
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
				if diff > 1e-5 {
					t.Errorf("sample %d channel %d output %d: got %v, want %v (diff %v)",
						b, c, o, got[b][c][o], want[b][c][o], diff)
				}
			}
		}
	}
}
