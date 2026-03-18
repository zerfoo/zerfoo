package tabular

import (
	"testing"
)

func TestSAINT_Forward(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name     string
		config   SAINTConfig
		features []float64
		wantErr  bool
	}{
		{
			name: "basic forward pass",
			config: SAINTConfig{
				NumFeatures:          4,
				DModel:               8,
				NHeads:               2,
				NLayers:              1,
				InterSampleAttention: false,
			},
			features: []float64{1.0, 2.0, 3.0, 4.0},
		},
		{
			name: "multiple layers",
			config: SAINTConfig{
				NumFeatures:          3,
				DModel:               6,
				NHeads:               2,
				NLayers:              2,
				InterSampleAttention: false,
			},
			features: []float64{0.5, -0.5, 1.0},
		},
		{
			name: "with intersample attention disabled single sample",
			config: SAINTConfig{
				NumFeatures:          2,
				DModel:               4,
				NHeads:               2,
				NLayers:              1,
				InterSampleAttention: true,
			},
			features: []float64{1.0, -1.0},
		},
		{
			name: "zero features",
			config: SAINTConfig{
				NumFeatures:          3,
				DModel:               6,
				NHeads:               3,
				NLayers:              1,
				InterSampleAttention: false,
			},
			features: []float64{0.0, 0.0, 0.0},
		},
		{
			name: "wrong feature count",
			config: SAINTConfig{
				NumFeatures:          4,
				DModel:               8,
				NHeads:               2,
				NLayers:              1,
				InterSampleAttention: false,
			},
			features: []float64{1.0, 2.0},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := NewSAINT(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewSAINT: %v", err)
			}

			dir, conf, err := model.Predict(tt.features)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
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
}

func TestSAINT_ForwardConsistency(t *testing.T) {
	engine, ops := newTestEngine()

	config := SAINTConfig{
		NumFeatures:          4,
		DModel:               8,
		NHeads:               2,
		NLayers:              2,
		InterSampleAttention: false,
	}

	model, err := NewSAINT(config, engine, ops)
	if err != nil {
		t.Fatalf("NewSAINT: %v", err)
	}

	features := []float64{1.0, -2.0, 3.0, -4.0}

	dir1, conf1, err := model.Predict(features)
	if err != nil {
		t.Fatalf("Predict 1: %v", err)
	}

	for i := 0; i < 5; i++ {
		dir, conf, err := model.Predict(features)
		if err != nil {
			t.Fatalf("Predict %d: %v", i+2, err)
		}
		if dir != dir1 {
			t.Errorf("iteration %d: direction %v != %v", i+2, dir, dir1)
		}
		if conf != conf1 {
			t.Errorf("iteration %d: confidence %f != %f", i+2, conf, conf1)
		}
	}
}

func TestNewSAINT_Validation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  SAINTConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: SAINTConfig{
				NumFeatures:          4,
				DModel:               8,
				NHeads:               2,
				NLayers:              1,
				InterSampleAttention: true,
			},
		},
		{
			name: "zero num features",
			config: SAINTConfig{
				NumFeatures: 0,
				DModel:      8,
				NHeads:      2,
				NLayers:     1,
			},
			wantErr: true,
		},
		{
			name: "zero d_model",
			config: SAINTConfig{
				NumFeatures: 4,
				DModel:      0,
				NHeads:      2,
				NLayers:     1,
			},
			wantErr: true,
		},
		{
			name: "zero heads",
			config: SAINTConfig{
				NumFeatures: 4,
				DModel:      8,
				NHeads:      0,
				NLayers:     1,
			},
			wantErr: true,
		},
		{
			name: "d_model not divisible by heads",
			config: SAINTConfig{
				NumFeatures: 4,
				DModel:      7,
				NHeads:      2,
				NLayers:     1,
			},
			wantErr: true,
		},
		{
			name: "zero layers",
			config: SAINTConfig{
				NumFeatures: 4,
				DModel:      8,
				NHeads:      2,
				NLayers:     0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewSAINT(tt.config, engine, ops)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestSAINT_IntersampleAttention(t *testing.T) {
	engine, ops := newTestEngine()

	// Create two models: one with intersample attention, one without.
	configWithInter := SAINTConfig{
		NumFeatures:          3,
		DModel:               6,
		NHeads:               2,
		NLayers:              1,
		InterSampleAttention: true,
	}
	configWithoutInter := SAINTConfig{
		NumFeatures:          3,
		DModel:               6,
		NHeads:               2,
		NLayers:              1,
		InterSampleAttention: false,
	}

	modelWith, err := NewSAINT(configWithInter, engine, ops)
	if err != nil {
		t.Fatalf("NewSAINT (with inter): %v", err)
	}
	modelWithout, err := NewSAINT(configWithoutInter, engine, ops)
	if err != nil {
		t.Fatalf("NewSAINT (without inter): %v", err)
	}

	// Test single-sample: intersample attention should be a no-op.
	features := []float64{1.0, 2.0, 3.0}

	dir1, conf1, err := modelWith.Predict(features)
	if err != nil {
		t.Fatalf("Predict (with inter, single): %v", err)
	}
	if dir1 < Long || dir1 > Flat {
		t.Errorf("direction %d out of range", dir1)
	}
	if conf1 <= 0 || conf1 > 1 {
		t.Errorf("confidence %f out of range", conf1)
	}

	// Test batch prediction with intersample attention.
	batch := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}

	dirsWithInter, confsWithInter, err := modelWith.PredictBatch(batch)
	if err != nil {
		t.Fatalf("PredictBatch (with inter): %v", err)
	}
	if len(dirsWithInter) != 3 {
		t.Fatalf("expected 3 directions, got %d", len(dirsWithInter))
	}
	if len(confsWithInter) != 3 {
		t.Fatalf("expected 3 confidences, got %d", len(confsWithInter))
	}

	dirsWithout, confsWithout, err := modelWithout.PredictBatch(batch)
	if err != nil {
		t.Fatalf("PredictBatch (without inter): %v", err)
	}
	if len(dirsWithout) != 3 {
		t.Fatalf("expected 3 directions, got %d", len(dirsWithout))
	}

	// Verify all outputs are valid.
	for i := 0; i < 3; i++ {
		if dirsWithInter[i] < Long || dirsWithInter[i] > Flat {
			t.Errorf("batch with inter: direction[%d] = %d out of range", i, dirsWithInter[i])
		}
		if confsWithInter[i] <= 0 || confsWithInter[i] > 1 {
			t.Errorf("batch with inter: confidence[%d] = %f out of range", i, confsWithInter[i])
		}
		if dirsWithout[i] < Long || dirsWithout[i] > Flat {
			t.Errorf("batch without inter: direction[%d] = %d out of range", i, dirsWithout[i])
		}
		if confsWithout[i] <= 0 || confsWithout[i] > 1 {
			t.Errorf("batch without inter: confidence[%d] = %f out of range", i, confsWithout[i])
		}
	}

	// Verify batch consistency: running the same batch twice produces same results.
	dirsAgain, confsAgain, err := modelWith.PredictBatch(batch)
	if err != nil {
		t.Fatalf("PredictBatch repeat: %v", err)
	}
	for i := 0; i < 3; i++ {
		if dirsAgain[i] != dirsWithInter[i] {
			t.Errorf("batch repeat: direction[%d] = %v, want %v", i, dirsAgain[i], dirsWithInter[i])
		}
		if confsAgain[i] != confsWithInter[i] {
			t.Errorf("batch repeat: confidence[%d] = %f, want %f", i, confsAgain[i], confsWithInter[i])
		}
	}
}

func TestSAINT_Train(t *testing.T) {
	engine, ops := newTestEngine()

	// Simple classification task: feature patterns map to classes.
	// Class 0 (Long): high first feature, low second
	// Class 1 (Short): low first feature, high second
	data := make([][]float64, 40)
	labels := make([]int, 40)
	for i := 0; i < 40; i++ {
		if i < 20 {
			data[i] = []float64{float64(i)*0.1 + 0.5, float64(i)*0.01}
			labels[i] = 0 // Long
		} else {
			data[i] = []float64{float64(i-20)*0.01, float64(i-20)*0.1 + 0.5}
			labels[i] = 1 // Short
		}
	}

	saintConfig := SAINTConfig{
		NumFeatures:          2,
		DModel:               8,
		NHeads:               2,
		NLayers:              1,
		InterSampleAttention: false,
	}

	tc := TrainConfig{
		Epochs:       200,
		BatchSize:    40,
		LearningRate: 0.01,
	}

	model, err := TrainSAINT(data, labels, tc, saintConfig, engine, ops)
	if err != nil {
		t.Fatalf("TrainSAINT: %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	// Verify the model learned something — at least better than random (33%).
	correct := 0
	total := len(data)
	for i, d := range data {
		dir, _, err := model.Predict(d)
		if err != nil {
			t.Fatalf("Predict(%v): %v", d, err)
		}
		if int(dir) == labels[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(total)
	if accuracy < 0.5 {
		t.Errorf("training accuracy %.1f%% is below 50%% threshold", accuracy*100)
	}
}
