package tabular

import (
	"math"
	"testing"
)

func TestEnsemble_CombinesModels(t *testing.T) {
	engine, ops := newTestEngine()

	// Create two sub-models with different configurations.
	m1, err := NewModel(ModelConfig{
		InputDim:    4,
		HiddenDims:  []int{8},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel m1: %v", err)
	}

	m2, err := NewModel(ModelConfig{
		InputDim:    4,
		HiddenDims:  []int{8, 4},
		DropoutRate: 0.0,
		Activation:  ActivationGELU,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel m2: %v", err)
	}

	treeOutputDim := 3
	treePreds := func(features []float64) []float64 {
		// Simple mock: return normalized feature sums.
		sum := 0.0
		for _, f := range features {
			sum += f
		}
		return []float64{sum * 0.1, sum * 0.2, sum * 0.3}
	}

	ens := NewEnsemble([]*Model{m1, m2}, treePreds)

	tests := []struct {
		name     string
		features []float64
	}{
		{
			name:     "positive features",
			features: []float64{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:     "negative features",
			features: []float64{-1.0, -2.0, -3.0, -4.0},
		},
		{
			name:     "zero features",
			features: []float64{0.0, 0.0, 0.0, 0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputs, err := ens.collectSubOutputs(tt.features)
			if err != nil {
				t.Fatalf("collectSubOutputs: %v", err)
			}

			// 2 models * 4 outputs (3 one-hot + 1 confidence) + 3 tree outputs = 11
			wantDim := 2*4 + treeOutputDim
			if len(outputs) != wantDim {
				t.Errorf("output dim = %d, want %d", len(outputs), wantDim)
			}

			// Verify each sub-model output block has valid one-hot + confidence.
			for modelIdx := 0; modelIdx < 2; modelIdx++ {
				base := modelIdx * 4
				oneHotSum := outputs[base] + outputs[base+1] + outputs[base+2]
				if math.Abs(oneHotSum-1.0) > 1e-6 {
					t.Errorf("model %d one-hot sum = %f, want 1.0", modelIdx, oneHotSum)
				}
				conf := outputs[base+3]
				if conf <= 0 || conf > 1 {
					t.Errorf("model %d confidence = %f, want (0, 1]", modelIdx, conf)
				}
			}
		})
	}
}

func TestEnsemble_MetaLearnerConverges(t *testing.T) {
	engine, ops := newTestEngine()

	// Create two sub-models.
	m1, err := NewModel(ModelConfig{
		InputDim:    2,
		HiddenDims:  []int{8},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel m1: %v", err)
	}

	m2, err := NewModel(ModelConfig{
		InputDim:    2,
		HiddenDims:  []int{8},
		DropoutRate: 0.0,
		Activation:  ActivationGELU,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel m2: %v", err)
	}

	ens := NewEnsemble([]*Model{m1, m2}, nil)

	// Generate synthetic sub-model outputs with clear class separation.
	// 2 models * 4 = 8 dimensional input to meta-learner.
	numSamples := 60
	subOutputs := make([][]float64, numSamples)
	labels := make([]int, numSamples)

	for i := 0; i < numSamples; i++ {
		row := make([]float64, 8)
		cls := i % 3
		labels[i] = cls

		// Create separable patterns per class.
		switch cls {
		case 0: // Long
			row[0], row[1], row[2], row[3] = 1, 0, 0, 0.9
			row[4], row[5], row[6], row[7] = 1, 0, 0, 0.8
		case 1: // Short
			row[0], row[1], row[2], row[3] = 0, 1, 0, 0.85
			row[4], row[5], row[6], row[7] = 0, 1, 0, 0.75
		case 2: // Flat
			row[0], row[1], row[2], row[3] = 0, 0, 1, 0.7
			row[4], row[5], row[6], row[7] = 0, 0, 1, 0.65
		}

		// Add small noise.
		noise := float64(i%5) * 0.01
		for j := range row {
			row[j] += noise
		}

		subOutputs[i] = row
	}

	config := EnsembleTrainConfig{
		Epochs:       500,
		BatchSize:    numSamples,
		LearningRate: 0.05,
		HiddenDims:   []int{16},
		Activation:   ActivationReLU,
	}

	err = ens.TrainEnsemble(subOutputs, labels, config, engine, ops)
	if err != nil {
		t.Fatalf("TrainEnsemble: %v", err)
	}

	if ens.metaLearner == nil {
		t.Fatal("meta-learner should be set after training")
	}

	// Verify meta-learner can classify the training patterns.
	correct := 0
	for i := 0; i < numSamples; i++ {
		dir, _, err := ens.metaLearner.Predict(subOutputs[i])
		if err != nil {
			t.Fatalf("meta-learner Predict: %v", err)
		}
		if int(dir) == labels[i] {
			correct++
		}
	}

	acc := float64(correct) / float64(numSamples)
	if acc < 0.8 {
		t.Errorf("meta-learner accuracy = %.2f, want >= 0.80", acc)
	}
}

func TestEnsemble_Predict(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{
		InputDim:    2,
		HiddenDims:  []int{8},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treePreds := func(features []float64) []float64 {
		return []float64{0.5, 0.3, 0.2}
	}

	ens := NewEnsemble([]*Model{m1}, treePreds)

	// Predict without training meta-learner should fail.
	_, _, err = ens.Predict([]float64{1.0, 2.0})
	if err == nil {
		t.Fatal("expected error for untrained meta-learner")
	}

	// Generate sub-model outputs and train.
	// 1 model * 4 + 3 tree outputs = 7 dimensional input.
	numSamples := 30
	subOutputs := make([][]float64, numSamples)
	labels := make([]int, numSamples)

	for i := 0; i < numSamples; i++ {
		cls := i % 3
		labels[i] = cls

		row := make([]float64, 7)
		switch cls {
		case 0:
			row = []float64{1, 0, 0, 0.9, 0.8, 0.1, 0.1}
		case 1:
			row = []float64{0, 1, 0, 0.85, 0.1, 0.8, 0.1}
		case 2:
			row = []float64{0, 0, 1, 0.7, 0.1, 0.1, 0.8}
		}

		subOutputs[i] = row
	}

	config := EnsembleTrainConfig{
		Epochs:       300,
		BatchSize:    numSamples,
		LearningRate: 0.05,
		HiddenDims:   []int{16},
		Activation:   ActivationReLU,
	}

	err = ens.TrainEnsemble(subOutputs, labels, config, engine, ops)
	if err != nil {
		t.Fatalf("TrainEnsemble: %v", err)
	}

	// Now Predict should work end-to-end.
	tests := []struct {
		name     string
		features []float64
	}{
		{
			name:     "positive features",
			features: []float64{1.0, 2.0},
		},
		{
			name:     "negative features",
			features: []float64{-1.0, -2.0},
		},
		{
			name:     "zero features",
			features: []float64{0.0, 0.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir, conf, err := ens.Predict(tt.features)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}

			if dir < Long || dir > Flat {
				t.Errorf("direction %d is not in [Long, Short, Flat]", dir)
			}

			if conf <= 0 || conf > 1 {
				t.Errorf("confidence %f is not in (0, 1]", conf)
			}
		})
	}

	// Verify determinism: same input should produce same output.
	features := []float64{1.0, 2.0}
	dir1, conf1, err := ens.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	for i := 0; i < 5; i++ {
		dir, conf, err := ens.Predict(features)
		if err != nil {
			t.Fatalf("Predict iteration %d: %v", i, err)
		}
		if dir != dir1 {
			t.Errorf("iteration %d: direction %v != %v", i, dir, dir1)
		}
		if conf != conf1 {
			t.Errorf("iteration %d: confidence %f != %f", i, conf, conf1)
		}
	}
}

func TestEnsemble_TrainErrors(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{
		InputDim:    2,
		HiddenDims:  []int{4},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	ens := NewEnsemble([]*Model{m1}, nil)

	tests := []struct {
		name       string
		subOutputs [][]float64
		labels     []int
		config     EnsembleTrainConfig
	}{
		{
			name:       "no data",
			subOutputs: nil,
			labels:     nil,
			config:     EnsembleTrainConfig{Epochs: 10},
		},
		{
			name:       "mismatched lengths",
			subOutputs: [][]float64{{1, 2, 3, 4}},
			labels:     []int{0, 1},
			config:     EnsembleTrainConfig{Epochs: 10},
		},
		{
			name:       "zero epochs",
			subOutputs: [][]float64{{1, 2, 3, 4}},
			labels:     []int{0},
			config:     EnsembleTrainConfig{Epochs: 0},
		},
		{
			name:       "label out of range",
			subOutputs: [][]float64{{1, 2, 3, 4}},
			labels:     []int{5},
			config:     EnsembleTrainConfig{Epochs: 10},
		},
		{
			name:       "inconsistent feature count",
			subOutputs: [][]float64{{1, 2, 3, 4}, {1, 2}},
			labels:     []int{0, 1},
			config:     EnsembleTrainConfig{Epochs: 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ens.TrainEnsemble(tt.subOutputs, tt.labels, tt.config, engine, ops)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestEnsemble_GenerateSubModelOutputs(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{
		InputDim:    3,
		HiddenDims:  []int{6},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treePreds := func(features []float64) []float64 {
		return []float64{0.1, 0.2}
	}

	ens := NewEnsemble([]*Model{m1}, treePreds)

	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	outputs, err := ens.GenerateSubModelOutputs(data)
	if err != nil {
		t.Fatalf("GenerateSubModelOutputs: %v", err)
	}

	if len(outputs) != len(data) {
		t.Fatalf("got %d output rows, want %d", len(outputs), len(data))
	}

	// 1 model * 4 + 2 tree outputs = 6
	wantDim := 1*4 + 2
	for i, row := range outputs {
		if len(row) != wantDim {
			t.Errorf("row %d: dim = %d, want %d", i, len(row), wantDim)
		}
	}
}
