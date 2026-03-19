package crossasset

import (
	"math"
	"testing"
)

func defaultConfig() Config {
	return Config{
		NSources:          4,
		FeaturesPerSource: 8,
		DModel:            16,
		NHeads:            4,
		NLayers:           2,
		DropoutRate:        0.0,
		LearningRate:      0.001,
	}
}

func TestCrossAsset_Forward(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	features := make([][]float64, cfg.NSources)
	for i := range features {
		features[i] = make([]float64, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float64(i*cfg.FeaturesPerSource+j) * 0.1
		}
	}

	output, err := m.Forward(features)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Check output shape.
	if len(output) != cfg.NSources {
		t.Fatalf("expected %d sources in output, got %d", cfg.NSources, len(output))
	}
	for i, o := range output {
		if len(o) != cfg.DModel {
			t.Fatalf("source %d: expected %d dims, got %d", i, cfg.DModel, len(o))
		}
	}

	// Verify outputs are non-zero.
	for i, o := range output {
		allZero := true
		for _, v := range o {
			if v != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Errorf("source %d: output is all zeros", i)
		}
	}

	// Verify outputs are finite.
	for i, o := range output {
		for j, v := range o {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("source %d, dim %d: non-finite value %v", i, j, v)
			}
		}
	}
}

func TestCrossAsset_Forward_InputValidation(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	t.Run("wrong number of sources", func(t *testing.T) {
		features := make([][]float64, cfg.NSources+1)
		for i := range features {
			features[i] = make([]float64, cfg.FeaturesPerSource)
		}
		_, err := m.Forward(features)
		if err == nil {
			t.Fatal("expected error for wrong number of sources")
		}
	})

	t.Run("wrong features per source", func(t *testing.T) {
		features := make([][]float64, cfg.NSources)
		for i := range features {
			features[i] = make([]float64, cfg.FeaturesPerSource+1)
		}
		_, err := m.Forward(features)
		if err == nil {
			t.Fatal("expected error for wrong features per source")
		}
	})
}

func TestCrossAsset_AttentionWeights(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	features := make([][]float64, cfg.NSources)
	for i := range features {
		features[i] = make([]float64, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float64(i*cfg.FeaturesPerSource+j) * 0.1
		}
	}

	attn, err := m.AttentionWeights(features)
	if err != nil {
		t.Fatalf("AttentionWeights: %v", err)
	}

	// Check shape.
	if len(attn) != cfg.NSources {
		t.Fatalf("expected %d rows, got %d", cfg.NSources, len(attn))
	}
	for i, row := range attn {
		if len(row) != cfg.NSources {
			t.Fatalf("row %d: expected %d cols, got %d", i, cfg.NSources, len(row))
		}
	}

	// Verify weights sum to 1 across attended sources (j dimension).
	for i, row := range attn {
		sum := 0.0
		for _, w := range row {
			sum += w
		}
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("row %d: attention weights sum to %f, expected 1.0", i, sum)
		}
	}

	// Verify all weights are non-negative.
	for i, row := range attn {
		for j, w := range row {
			if w < 0 {
				t.Errorf("attn[%d][%d] = %f, expected non-negative", i, j, w)
			}
		}
	}

	// Verify weights are finite.
	for i, row := range attn {
		for j, w := range row {
			if math.IsNaN(w) || math.IsInf(w, 0) {
				t.Errorf("attn[%d][%d] = %v, expected finite", i, j, w)
			}
		}
	}
}

func TestCrossAsset_Predict(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	features := make([][]float64, cfg.NSources)
	for i := range features {
		features[i] = make([]float64, cfg.FeaturesPerSource)
		for j := range features[i] {
			features[i][j] = float64(i+j) * 0.1
		}
	}

	dirs, confs, err := m.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if len(dirs) != cfg.NSources {
		t.Fatalf("expected %d directions, got %d", cfg.NSources, len(dirs))
	}
	if len(confs) != cfg.NSources {
		t.Fatalf("expected %d confidences, got %d", cfg.NSources, len(confs))
	}

	for i, d := range dirs {
		if d < 0 || d > 2 {
			t.Errorf("source %d: direction %d out of range [0, 2]", i, d)
		}
	}
	for i, c := range confs {
		if c < 0 || c > 1 {
			t.Errorf("source %d: confidence %f out of range [0, 1]", i, c)
		}
	}
}

func TestCrossAsset_Train(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	nSamples := 20
	data := make([][][]float64, nSamples)
	labels := make([][]int, nSamples)
	for i := 0; i < nSamples; i++ {
		data[i] = make([][]float64, cfg.NSources)
		labels[i] = make([]int, cfg.NSources)
		for s := 0; s < cfg.NSources; s++ {
			data[i][s] = make([]float64, cfg.FeaturesPerSource)
			for f := 0; f < cfg.FeaturesPerSource; f++ {
				data[i][s][f] = float64(i+s+f) * 0.01
			}
			labels[i][s] = i % 3
		}
	}

	tc := TrainConfig{
		Epochs:       5,
		BatchSize:    10,
		LearningRate: 0.01,
	}

	err := m.Train(data, labels, tc)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	// Verify model can still predict after training.
	dirs, confs, err := m.Predict(data[0])
	if err != nil {
		t.Fatalf("Predict after train: %v", err)
	}
	if len(dirs) != cfg.NSources {
		t.Fatalf("expected %d directions, got %d", cfg.NSources, len(dirs))
	}
	if len(confs) != cfg.NSources {
		t.Fatalf("expected %d confidences, got %d", cfg.NSources, len(confs))
	}
}

func TestCrossAsset_Train_Validation(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	t.Run("no data", func(t *testing.T) {
		err := m.Train(nil, nil, TrainConfig{Epochs: 1})
		if err == nil {
			t.Fatal("expected error for no data")
		}
	})

	t.Run("mismatched lengths", func(t *testing.T) {
		data := make([][][]float64, 2)
		labels := make([][]int, 3)
		err := m.Train(data, labels, TrainConfig{Epochs: 1})
		if err == nil {
			t.Fatal("expected error for mismatched lengths")
		}
	})

	t.Run("zero epochs", func(t *testing.T) {
		data := make([][][]float64, 1)
		labels := make([][]int, 1)
		err := m.Train(data, labels, TrainConfig{Epochs: 0})
		if err == nil {
			t.Fatal("expected error for zero epochs")
		}
	})
}

func TestCrossAsset_DifferentInputsProduceDifferentOutputs(t *testing.T) {
	cfg := defaultConfig()
	m := NewModel(cfg)

	feat1 := make([][]float64, cfg.NSources)
	feat2 := make([][]float64, cfg.NSources)
	for i := 0; i < cfg.NSources; i++ {
		feat1[i] = make([]float64, cfg.FeaturesPerSource)
		feat2[i] = make([]float64, cfg.FeaturesPerSource)
		for j := 0; j < cfg.FeaturesPerSource; j++ {
			feat1[i][j] = float64(j) * 0.1
			feat2[i][j] = float64(j) * 0.5
		}
	}

	out1, err := m.Forward(feat1)
	if err != nil {
		t.Fatalf("Forward feat1: %v", err)
	}
	out2, err := m.Forward(feat2)
	if err != nil {
		t.Fatalf("Forward feat2: %v", err)
	}

	same := true
	for s := 0; s < cfg.NSources; s++ {
		for d := 0; d < cfg.DModel; d++ {
			if out1[s][d] != out2[s][d] {
				same = false
				break
			}
		}
	}
	if same {
		t.Error("different inputs produced identical outputs")
	}
}
