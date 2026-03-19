package tabular

import (
	"math"
	"testing"
)

func TestNewEnsemble_Valid(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{InputDim: 4, HiddenDims: []int{8}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	m2, err := NewModel(ModelConfig{InputDim: 4, HiddenDims: []int{8}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treeFn := func(features []float64) []float64 {
		return []float64{0.5, 0.3, 0.2}
	}

	ens, err := NewEnsemble([]*Model{m1, m2}, treeFn, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}
	if ens == nil {
		t.Fatal("expected non-nil ensemble")
	}
	if len(ens.models) != 2 {
		t.Errorf("expected 2 models, got %d", len(ens.models))
	}
}

func TestNewEnsemble_Errors(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name   string
		models []*Model
		treeFn func([]float64) []float64
	}{
		{
			name:   "nil models",
			models: nil,
			treeFn: func(f []float64) []float64 { return []float64{0.5} },
		},
		{
			name:   "empty models",
			models: []*Model{},
			treeFn: func(f []float64) []float64 { return []float64{0.5} },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewEnsemble(tt.models, tt.treeFn, engine, ops)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestEnsemble_CombinesModels(t *testing.T) {
	engine, ops := newTestEngine()

	// Create two models with different architectures.
	m1, err := NewModel(ModelConfig{InputDim: 4, HiddenDims: []int{8, 4}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel m1: %v", err)
	}
	m2, err := NewModel(ModelConfig{InputDim: 4, HiddenDims: []int{6}, Activation: ActivationGELU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel m2: %v", err)
	}

	treeOutputDim := 3
	treeFn := func(features []float64) []float64 {
		// Simulate tree ensemble outputs (e.g., class probabilities).
		return []float64{0.6, 0.2, 0.2}
	}

	ens, err := NewEnsemble([]*Model{m1, m2}, treeFn, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}

	// Generate synthetic training data for the meta-learner.
	// Each sub-model output is 3 (softmax over Long/Short/Flat).
	// Total meta-learner input = 2*3 + treeOutputDim = 9.
	metaInputDim := 2*3 + treeOutputDim
	nSamples := 200
	subModelOutputs := make([][]float64, nSamples)
	labels := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		subModelOutputs[i] = make([]float64, metaInputDim)
		label := i % 3
		labels[i] = label
		// Make the outputs correlate with the label.
		for j := 0; j < metaInputDim; j++ {
			if j%3 == label {
				subModelOutputs[i][j] = 0.8
			} else {
				subModelOutputs[i][j] = 0.1
			}
		}
	}

	err = ens.TrainMetaLearner(subModelOutputs, labels, TrainConfig{
		Epochs:       50,
		LearningRate: 0.01,
		BatchSize:    32,
	}, MetaLearnerConfig{HiddenDims: []int{16, 8}})
	if err != nil {
		t.Fatalf("TrainMetaLearner: %v", err)
	}

	// Verify the meta-learner was trained.
	if ens.metaLearner == nil {
		t.Fatal("meta-learner should not be nil after training")
	}
}

func TestEnsemble_MetaLearnerConverges(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{InputDim: 2, HiddenDims: []int{4}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treeFn := func(features []float64) []float64 {
		return []float64{0.5, 0.3, 0.2}
	}

	ens, err := NewEnsemble([]*Model{m1}, treeFn, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}

	// meta-learner input dim = 1*3 + 3 = 6
	nSamples := 300
	subModelOutputs := make([][]float64, nSamples)
	labels := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		label := i % 3
		labels[i] = label
		subModelOutputs[i] = make([]float64, 6)
		// Strong signal: the label-corresponding position is high.
		for j := 0; j < 6; j++ {
			if j%3 == label {
				subModelOutputs[i][j] = 0.9
			} else {
				subModelOutputs[i][j] = 0.05
			}
		}
	}

	err = ens.TrainMetaLearner(subModelOutputs, labels, TrainConfig{
		Epochs:       100,
		LearningRate: 0.01,
		BatchSize:    32,
	}, MetaLearnerConfig{HiddenDims: []int{8}})
	if err != nil {
		t.Fatalf("TrainMetaLearner: %v", err)
	}

	// Verify convergence: predict on training data and check accuracy.
	correct := 0
	for i := 0; i < nSamples; i++ {
		dir, _, err := ens.predictFromOutputs(subModelOutputs[i])
		if err != nil {
			t.Fatalf("predictFromOutputs: %v", err)
		}
		if int(dir) == labels[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(nSamples)
	if accuracy < 0.8 {
		t.Errorf("meta-learner accuracy %.2f < 0.80 threshold", accuracy)
	}
}

func TestEnsemble_Predict(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{InputDim: 3, HiddenDims: []int{8, 4}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	m2, err := NewModel(ModelConfig{InputDim: 3, HiddenDims: []int{6}, Activation: ActivationGELU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treeFn := func(features []float64) []float64 {
		return []float64{0.4, 0.3, 0.3}
	}

	ens, err := NewEnsemble([]*Model{m1, m2}, treeFn, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}

	// Train meta-learner. Input dim = 2*3 + 3 = 9.
	nSamples := 200
	subModelOutputs := make([][]float64, nSamples)
	labels := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		label := i % 3
		labels[i] = label
		subModelOutputs[i] = make([]float64, 9)
		for j := 0; j < 9; j++ {
			if j%3 == label {
				subModelOutputs[i][j] = 0.85
			} else {
				subModelOutputs[i][j] = 0.075
			}
		}
	}

	err = ens.TrainMetaLearner(subModelOutputs, labels, TrainConfig{
		Epochs:       80,
		LearningRate: 0.01,
		BatchSize:    32,
	}, MetaLearnerConfig{HiddenDims: []int{8}})
	if err != nil {
		t.Fatalf("TrainMetaLearner: %v", err)
	}

	// Run end-to-end predict.
	features := []float64{1.0, 2.0, 3.0}
	dir, conf, err := ens.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	// Direction must be valid.
	if dir < Long || dir > Flat {
		t.Errorf("direction %d is not in [Long, Short, Flat]", dir)
	}

	// Confidence must be in (0, 1].
	if conf <= 0 || conf > 1 {
		t.Errorf("confidence %f is not in (0, 1]", conf)
	}
}

func TestEnsemble_Predict_Deterministic(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{InputDim: 2, HiddenDims: []int{4}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treeFn := func(features []float64) []float64 {
		return []float64{0.5, 0.3, 0.2}
	}

	ens, err := NewEnsemble([]*Model{m1}, treeFn, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}

	// Train meta-learner.
	nSamples := 100
	subModelOutputs := make([][]float64, nSamples)
	labels := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		labels[i] = i % 3
		subModelOutputs[i] = make([]float64, 6)
		for j := range subModelOutputs[i] {
			subModelOutputs[i][j] = 0.33
		}
	}

	err = ens.TrainMetaLearner(subModelOutputs, labels, TrainConfig{
		Epochs:       20,
		LearningRate: 0.01,
		BatchSize:    32,
	}, MetaLearnerConfig{HiddenDims: []int{4}})
	if err != nil {
		t.Fatalf("TrainMetaLearner: %v", err)
	}

	features := []float64{1.0, -1.0}
	dir1, conf1, err := ens.Predict(features)
	if err != nil {
		t.Fatalf("Predict 1: %v", err)
	}

	for i := 0; i < 10; i++ {
		dir, conf, err := ens.Predict(features)
		if err != nil {
			t.Fatalf("Predict %d: %v", i+2, err)
		}
		if dir != dir1 {
			t.Errorf("iteration %d: direction %v != %v", i+2, dir, dir1)
		}
		if math.Abs(conf-conf1) > 1e-6 {
			t.Errorf("iteration %d: confidence %f != %f", i+2, conf, conf1)
		}
	}
}

func TestEnsemble_Predict_NoMetaLearner(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{InputDim: 2, HiddenDims: []int{4}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treeFn := func(features []float64) []float64 {
		return []float64{0.5, 0.3, 0.2}
	}

	ens, err := NewEnsemble([]*Model{m1}, treeFn, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}

	// Predict without training meta-learner should error.
	_, _, err = ens.Predict([]float64{1.0, 2.0})
	if err == nil {
		t.Fatal("expected error when predicting without trained meta-learner")
	}
}

func TestEnsemble_TrainMetaLearner_Errors(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{InputDim: 2, HiddenDims: []int{4}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	treeFn := func(features []float64) []float64 {
		return []float64{0.5}
	}

	ens, err := NewEnsemble([]*Model{m1}, treeFn, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}

	tests := []struct {
		name    string
		outputs [][]float64
		labels  []int
		config  TrainConfig
		mlConf  MetaLearnerConfig
	}{
		{
			name:    "empty outputs",
			outputs: nil,
			labels:  nil,
			config:  TrainConfig{Epochs: 10, LearningRate: 0.01},
			mlConf:  MetaLearnerConfig{HiddenDims: []int{4}},
		},
		{
			name:    "mismatched lengths",
			outputs: [][]float64{{0.5, 0.3, 0.2, 0.1}},
			labels:  []int{0, 1},
			config:  TrainConfig{Epochs: 10, LearningRate: 0.01},
			mlConf:  MetaLearnerConfig{HiddenDims: []int{4}},
		},
		{
			name:    "invalid label",
			outputs: [][]float64{{0.5, 0.3, 0.2, 0.1}},
			labels:  []int{5},
			config:  TrainConfig{Epochs: 10, LearningRate: 0.01},
			mlConf:  MetaLearnerConfig{HiddenDims: []int{4}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ens.TrainMetaLearner(tt.outputs, tt.labels, tt.config, tt.mlConf)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestEnsemble_NilTreePredictions(t *testing.T) {
	engine, ops := newTestEngine()

	m1, err := NewModel(ModelConfig{InputDim: 3, HiddenDims: []int{4}, Activation: ActivationReLU}, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	// nil treePredictions is valid — ensemble uses only MLP sub-models.
	ens, err := NewEnsemble([]*Model{m1}, nil, engine, ops)
	if err != nil {
		t.Fatalf("NewEnsemble: %v", err)
	}

	// Train meta-learner with just model outputs (dim = 1*3 = 3).
	nSamples := 100
	subModelOutputs := make([][]float64, nSamples)
	labels := make([]int, nSamples)
	for i := 0; i < nSamples; i++ {
		label := i % 3
		labels[i] = label
		subModelOutputs[i] = make([]float64, 3)
		for j := 0; j < 3; j++ {
			if j == label {
				subModelOutputs[i][j] = 0.9
			} else {
				subModelOutputs[i][j] = 0.05
			}
		}
	}

	err = ens.TrainMetaLearner(subModelOutputs, labels, TrainConfig{
		Epochs:       50,
		LearningRate: 0.01,
		BatchSize:    32,
	}, MetaLearnerConfig{HiddenDims: []int{4}})
	if err != nil {
		t.Fatalf("TrainMetaLearner: %v", err)
	}

	features := []float64{1.0, 2.0, 3.0}
	dir, conf, err := ens.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	if dir < Long || dir > Flat {
		t.Errorf("direction %d is not valid", dir)
	}
	if conf <= 0 || conf > 1 {
		t.Errorf("confidence %f is not in (0, 1]", conf)
	}
}
