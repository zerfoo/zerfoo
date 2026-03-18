package tabular

import (
	"testing"
)

func TestTrain_Convergence(t *testing.T) {
	engine, ops := newTestEngine()

	// XOR problem: two binary inputs, labels in {0, 1}.
	// We use only 2 classes (Long=0, Short=1) for XOR.
	// Duplicate data for robust training.
	data := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}
	// XOR: 0^0=0, 0^1=1, 1^0=1, 1^1=0
	labels := []int{
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
	}

	mc := ModelConfig{
		HiddenDims:  []int{32, 16},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}

	tc := TrainConfig{
		Epochs:       1000,
		BatchSize:    16,
		LearningRate: 0.05,
		WeightDecay:  0.0,
	}

	model, err := Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	// Verify the model learned XOR. All 4 patterns should be correct.
	tests := []struct {
		features []float64
		want     Direction
	}{
		{[]float64{0, 0}, Long},  // 0 XOR 0 = 0 -> Long
		{[]float64{0, 1}, Short}, // 0 XOR 1 = 1 -> Short
		{[]float64{1, 0}, Short}, // 1 XOR 0 = 1 -> Short
		{[]float64{1, 1}, Long},  // 1 XOR 1 = 0 -> Long
	}

	correct := 0
	for _, tt := range tests {
		dir, _, err := model.Predict(tt.features)
		if err != nil {
			t.Fatalf("Predict(%v): %v", tt.features, err)
		}
		if dir == tt.want {
			correct++
		}
	}

	// Require at least 3 out of 4 correct (75% accuracy on XOR).
	if correct < 3 {
		t.Errorf("XOR convergence: got %d/4 correct, want >= 3", correct)
	}
}

func TestTrain_Validation(t *testing.T) {
	engine, ops := newTestEngine()

	// Simple linearly separable data.
	data := [][]float64{
		{0, 0}, {0.1, 0.1}, {0.2, 0.1},
		{1, 1}, {0.9, 1.1}, {1.1, 0.9},
	}
	labels := []int{0, 0, 0, 1, 1, 1}

	mc := ModelConfig{
		HiddenDims:  []int{8},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}

	tc := TrainConfig{
		Epochs:          100,
		BatchSize:       6,
		LearningRate:    0.01,
		ValidationSplit: 0.33,
	}

	model, err := Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		t.Fatalf("Train with validation split: %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}
}

func TestTrain_ErrorCases(t *testing.T) {
	engine, ops := newTestEngine()

	mc := ModelConfig{
		HiddenDims:  []int{8},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}

	tests := []struct {
		name    string
		data    [][]float64
		labels  []int
		config  TrainConfig
		wantErr bool
	}{
		{
			name:    "no data",
			data:    nil,
			labels:  nil,
			config:  TrainConfig{Epochs: 10, BatchSize: 1},
			wantErr: true,
		},
		{
			name:    "mismatched lengths",
			data:    [][]float64{{1, 2}},
			labels:  []int{0, 1},
			config:  TrainConfig{Epochs: 10, BatchSize: 1},
			wantErr: true,
		},
		{
			name:    "zero epochs",
			data:    [][]float64{{1, 2}},
			labels:  []int{0},
			config:  TrainConfig{Epochs: 0, BatchSize: 1},
			wantErr: true,
		},
		{
			name:    "invalid validation split",
			data:    [][]float64{{1, 2}},
			labels:  []int{0},
			config:  TrainConfig{Epochs: 1, BatchSize: 1, ValidationSplit: 1.0},
			wantErr: true,
		},
		{
			name:    "label out of range",
			data:    [][]float64{{1, 2}},
			labels:  []int{5},
			config:  TrainConfig{Epochs: 1, BatchSize: 1},
			wantErr: true,
		},
		{
			name:    "inconsistent feature count",
			data:    [][]float64{{1, 2}, {3}},
			labels:  []int{0, 1},
			config:  TrainConfig{Epochs: 1, BatchSize: 1},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Train(tt.data, tt.labels, tt.config, mc, engine, ops)
			if tt.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestTrain_GELUActivation(t *testing.T) {
	engine, ops := newTestEngine()

	// Simple two-class problem with GELU activation.
	data := make([][]float64, 20)
	labels := make([]int, 20)
	for i := 0; i < 20; i++ {
		if i < 10 {
			data[i] = []float64{float64(i) * 0.1, 0.0}
			labels[i] = 0
		} else {
			data[i] = []float64{0.0, float64(i) * 0.1}
			labels[i] = 1
		}
	}

	mc := ModelConfig{
		HiddenDims:  []int{8},
		DropoutRate: 0.0,
		Activation:  ActivationGELU,
	}

	tc := TrainConfig{
		Epochs:       200,
		BatchSize:    20,
		LearningRate: 0.01,
	}

	model, err := Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		t.Fatalf("Train with GELU: %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}
}
