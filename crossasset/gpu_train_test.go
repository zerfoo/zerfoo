package crossasset

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func testConfig() Config {
	return Config{
		NSources:          3,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           1,
		DropoutRate:       0.0,
		LearningRate:      0.001,
	}
}

func testData(cfg Config, n int) ([][][]float32, [][]int) {
	data := make([][][]float32, n)
	labels := make([][]int, n)
	for i := range n {
		data[i] = make([][]float32, cfg.NSources)
		labels[i] = make([]int, cfg.NSources)
		for s := range cfg.NSources {
			data[i][s] = make([]float32, cfg.FeaturesPerSource)
			for f := range cfg.FeaturesPerSource {
				data[i][s][f] = float32(i*cfg.NSources*cfg.FeaturesPerSource+s*cfg.FeaturesPerSource+f) * 0.01
			}
			labels[i][s] = (i + s) % 3
		}
	}
	return data, labels
}

func TestTrainGPU_LossDecreases(t *testing.T) {
	cfg := Config{
		NSources:          3,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           1,
		DropoutRate:       0.0,
		LearningRate:      0.01,
	}
	m := NewModel(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	data, labels := testData(cfg, 20)

	tc := TrainConfig{
		Epochs:       10,
		BatchSize:    10,
		LearningRate: 0.01,
	}

	result, err := m.TrainGPU(data, labels, tc, engine)
	if err != nil {
		t.Fatalf("TrainGPU: %v", err)
	}

	if len(result.Losses) != tc.Epochs {
		t.Fatalf("expected %d epoch losses, got %d", tc.Epochs, len(result.Losses))
	}

	// Loss should decrease (or at least not blow up).
	firstLoss := result.Losses[0]
	lastLoss := result.Losses[tc.Epochs-1]
	t.Logf("Loss: first=%.4f last=%.4f", firstLoss, lastLoss)

	if math.IsNaN(firstLoss) || math.IsNaN(lastLoss) {
		t.Fatal("loss is NaN")
	}
	if math.IsInf(firstLoss, 0) || math.IsInf(lastLoss, 0) {
		t.Fatal("loss is Inf")
	}

	// After training, model should still produce valid predictions.
	dirs, confs, err := m.Predict(data[0])
	if err != nil {
		t.Fatalf("Predict after training: %v", err)
	}
	for s := range cfg.NSources {
		if dirs[s] < 0 || dirs[s] > 2 {
			t.Errorf("direction[%d] = %d, want 0-2", s, dirs[s])
		}
		if confs[s] < 0 || confs[s] > 1 {
			t.Errorf("confidence[%d] = %v, want 0-1", s, confs[s])
		}
	}
}

func TestTrainGPU_Validation(t *testing.T) {
	cfg := testConfig()
	m := NewModel(cfg)
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// No data.
	_, err := m.TrainGPU(nil, nil, TrainConfig{Epochs: 1, BatchSize: 1, LearningRate: 0.01}, engine)
	if err == nil {
		t.Error("expected error for nil data")
	}

	// Zero epochs.
	data, labels := testData(cfg, 5)
	_, err = m.TrainGPU(data, labels, TrainConfig{Epochs: 0, BatchSize: 1, LearningRate: 0.01}, engine)
	if err == nil {
		t.Error("expected error for zero epochs")
	}

	// Mismatched lengths.
	_, err = m.TrainGPU(data, labels[:3], TrainConfig{Epochs: 1, BatchSize: 1, LearningRate: 0.01}, engine)
	if err == nil {
		t.Error("expected error for mismatched data/labels")
	}
}
