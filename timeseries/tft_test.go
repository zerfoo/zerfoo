package timeseries

import (
	"math"
	"testing"
)

func TestTFT_Forward(t *testing.T) {
	engine, ops := newTestEngine()

	config := TFTConfig{
		NumStaticFeatures: 3,
		NumTimeFeatures:   4,
		DModel:            8,
		NHeads:            2,
		NHorizons:         3,
		Quantiles:         []float64{0.1, 0.5, 0.9},
	}

	m, err := NewTFT(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTFT: %v", err)
	}

	staticFeatures := []float64{1.0, 2.0, 3.0}
	timeFeatures := [][]float64{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
		{0.9, 1.0, 1.1, 1.2},
	}

	result, err := m.Predict(staticFeatures, timeFeatures)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if len(result) != config.NHorizons {
		t.Fatalf("expected %d horizons, got %d", config.NHorizons, len(result))
	}
	for h, row := range result {
		if len(row) != len(config.Quantiles) {
			t.Errorf("horizon %d: expected %d quantiles, got %d", h, len(config.Quantiles), len(row))
		}
		for q, val := range row {
			if math.IsNaN(val) || math.IsInf(val, 0) {
				t.Errorf("horizon %d, quantile %d: got %v", h, q, val)
			}
		}
	}
}

func TestTFT_Forward_Deterministic(t *testing.T) {
	engine, ops := newTestEngine()

	config := TFTConfig{
		NumStaticFeatures: 2,
		NumTimeFeatures:   3,
		DModel:            4,
		NHeads:            2,
		NHorizons:         2,
		Quantiles:         []float64{0.5},
	}

	m, err := NewTFT(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTFT: %v", err)
	}

	staticFeatures := []float64{1.0, -1.0}
	timeFeatures := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
	}

	result1, err := m.Predict(staticFeatures, timeFeatures)
	if err != nil {
		t.Fatalf("Predict 1: %v", err)
	}

	// Same model, same input => same output.
	for i := 0; i < 5; i++ {
		result2, err := m.Predict(staticFeatures, timeFeatures)
		if err != nil {
			t.Fatalf("Predict %d: %v", i+2, err)
		}
		for h := range result1 {
			for q := range result1[h] {
				if result1[h][q] != result2[h][q] {
					t.Errorf("non-deterministic: horizon %d quantile %d: %f != %f",
						h, q, result1[h][q], result2[h][q])
				}
			}
		}
	}
}

func TestTFT_Forward_Errors(t *testing.T) {
	engine, ops := newTestEngine()

	config := TFTConfig{
		NumStaticFeatures: 2,
		NumTimeFeatures:   3,
		DModel:            4,
		NHeads:            2,
		NHorizons:         2,
		Quantiles:         []float64{0.5},
	}

	m, err := NewTFT(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTFT: %v", err)
	}

	tests := []struct {
		name    string
		static  []float64
		time    [][]float64
		wantErr bool
	}{
		{
			name:    "wrong static feature count",
			static:  []float64{1.0},
			time:    [][]float64{{0.1, 0.2, 0.3}},
			wantErr: true,
		},
		{
			name:    "empty time features",
			static:  []float64{1.0, 2.0},
			time:    [][]float64{},
			wantErr: true,
		},
		{
			name:    "wrong time feature count",
			static:  []float64{1.0, 2.0},
			time:    [][]float64{{0.1, 0.2}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := m.Predict(tt.static, tt.time)
			if tt.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestTFT_VariableSelection(t *testing.T) {
	engine, ops := newTestEngine()

	config := TFTConfig{
		NumStaticFeatures: 4,
		NumTimeFeatures:   3,
		DModel:            8,
		NHeads:            2,
		NHorizons:         1,
		Quantiles:         []float64{0.5},
	}

	m, err := NewTFT(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTFT: %v", err)
	}

	// Test static variable selection weights.
	staticWeights, err := m.VariableSelectionWeights("static", []float64{1.0, 2.0, 3.0, 4.0})
	if err != nil {
		t.Fatalf("VariableSelectionWeights static: %v", err)
	}

	if len(staticWeights) != config.NumStaticFeatures {
		t.Fatalf("expected %d static weights, got %d", config.NumStaticFeatures, len(staticWeights))
	}

	// Weights must sum to ~1.0 (softmax output).
	var sum float64
	for _, w := range staticWeights {
		if w < 0 || w > 1 {
			t.Errorf("variable weight %f not in [0, 1]", w)
		}
		sum += w
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Errorf("static variable weights sum to %f, expected ~1.0", sum)
	}

	// Test time variable selection weights.
	timeWeights, err := m.VariableSelectionWeights("time", []float64{0.5, 1.5, 2.5})
	if err != nil {
		t.Fatalf("VariableSelectionWeights time: %v", err)
	}

	if len(timeWeights) != config.NumTimeFeatures {
		t.Fatalf("expected %d time weights, got %d", config.NumTimeFeatures, len(timeWeights))
	}

	sum = 0
	for _, w := range timeWeights {
		if w < 0 || w > 1 {
			t.Errorf("variable weight %f not in [0, 1]", w)
		}
		sum += w
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Errorf("time variable weights sum to %f, expected ~1.0", sum)
	}

	// Test error on invalid feature type.
	_, err = m.VariableSelectionWeights("invalid", []float64{1.0})
	if err == nil {
		t.Fatal("expected error for invalid feature type")
	}

	// Test error on wrong feature count.
	_, err = m.VariableSelectionWeights("static", []float64{1.0})
	if err == nil {
		t.Fatal("expected error for wrong feature count")
	}
}

func TestTFT_MultiHorizon(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name      string
		nHorizons int
		quantiles []float64
	}{
		{
			name:      "single horizon single quantile",
			nHorizons: 1,
			quantiles: []float64{0.5},
		},
		{
			name:      "multi horizon multi quantile",
			nHorizons: 5,
			quantiles: []float64{0.1, 0.25, 0.5, 0.75, 0.9},
		},
		{
			name:      "many horizons",
			nHorizons: 12,
			quantiles: []float64{0.1, 0.5, 0.9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            2,
				NHorizons:         tt.nHorizons,
				Quantiles:         tt.quantiles,
			}

			m, err := NewTFT(config, engine, ops)
			if err != nil {
				t.Fatalf("NewTFT: %v", err)
			}

			staticFeatures := []float64{1.0, 2.0}
			timeFeatures := [][]float64{
				{0.1, 0.2, 0.3},
				{0.4, 0.5, 0.6},
				{0.7, 0.8, 0.9},
			}

			result, err := m.Predict(staticFeatures, timeFeatures)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}

			// Check output shape.
			if len(result) != tt.nHorizons {
				t.Fatalf("expected %d horizons, got %d", tt.nHorizons, len(result))
			}
			for h, row := range result {
				if len(row) != len(tt.quantiles) {
					t.Errorf("horizon %d: expected %d quantiles, got %d", h, len(tt.quantiles), len(row))
				}
				for q, val := range row {
					if math.IsNaN(val) || math.IsInf(val, 0) {
						t.Errorf("horizon %d, quantile %d: got %v", h, q, val)
					}
				}
			}
		})
	}
}

func TestNewTFT_ConfigValidation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  TFTConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{0.1, 0.5, 0.9},
			},
		},
		{
			name: "zero static features",
			config: TFTConfig{
				NumStaticFeatures: 0,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{0.5},
			},
			wantErr: true,
		},
		{
			name: "zero time features",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   0,
				DModel:            4,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{0.5},
			},
			wantErr: true,
		},
		{
			name: "zero d_model",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            0,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{0.5},
			},
			wantErr: true,
		},
		{
			name: "d_model not divisible by n_heads",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            5,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{0.5},
			},
			wantErr: true,
		},
		{
			name: "zero n_heads",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            0,
				NHorizons:         3,
				Quantiles:         []float64{0.5},
			},
			wantErr: true,
		},
		{
			name: "zero n_horizons",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            2,
				NHorizons:         0,
				Quantiles:         []float64{0.5},
			},
			wantErr: true,
		},
		{
			name: "empty quantiles",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{},
			},
			wantErr: true,
		},
		{
			name: "quantile out of range",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{0.5, 1.0},
			},
			wantErr: true,
		},
		{
			name: "quantile zero",
			config: TFTConfig{
				NumStaticFeatures: 2,
				NumTimeFeatures:   3,
				DModel:            4,
				NHeads:            2,
				NHorizons:         3,
				Quantiles:         []float64{0.0, 0.5},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewTFT(tt.config, engine, ops)
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

func TestQuantileLoss(t *testing.T) {
	quantiles := []float64{0.1, 0.5, 0.9}

	tests := []struct {
		name      string
		predicted [][]float64
		targets   []float64
		wantErr   bool
	}{
		{
			name: "perfect predictions at median",
			predicted: [][]float64{
				{1.0, 1.0, 1.0},
			},
			targets: []float64{1.0},
		},
		{
			name: "under-prediction",
			predicted: [][]float64{
				{0.0, 0.0, 0.0},
			},
			targets: []float64{1.0},
		},
		{
			name: "over-prediction",
			predicted: [][]float64{
				{2.0, 2.0, 2.0},
			},
			targets: []float64{1.0},
		},
		{
			name: "multi-horizon",
			predicted: [][]float64{
				{0.8, 1.0, 1.2},
				{1.8, 2.0, 2.2},
			},
			targets: []float64{1.0, 2.0},
		},
		{
			name:      "empty predicted",
			predicted: [][]float64{},
			targets:   []float64{},
			wantErr:   true,
		},
		{
			name: "mismatched horizons",
			predicted: [][]float64{
				{1.0, 1.0, 1.0},
			},
			targets: []float64{1.0, 2.0},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loss, err := QuantileLoss(tt.predicted, tt.targets, quantiles)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Loss must be non-negative.
			if loss < 0 {
				t.Errorf("loss %f is negative", loss)
			}

			// Perfect predictions should have zero loss.
			if tt.name == "perfect predictions at median" && loss != 0 {
				t.Errorf("expected zero loss for perfect predictions, got %f", loss)
			}
		})
	}
}

func TestQuantileLoss_Asymmetry(t *testing.T) {
	// For quantile 0.9: under-prediction should be penalized more than over-prediction.
	// For quantile 0.1: over-prediction should be penalized more than under-prediction.
	quantiles := []float64{0.1}

	// Under-prediction: predicted=0, target=1, loss = 0.1 * 1 = 0.1
	underLoss, err := QuantileLoss([][]float64{{0.0}}, []float64{1.0}, quantiles)
	if err != nil {
		t.Fatalf("QuantileLoss: %v", err)
	}

	// Over-prediction: predicted=2, target=1, loss = (0.1 - 1) * (-1) = 0.9
	overLoss, err := QuantileLoss([][]float64{{2.0}}, []float64{1.0}, quantiles)
	if err != nil {
		t.Fatalf("QuantileLoss: %v", err)
	}

	// For q=0.1, over-prediction should be penalized more.
	if overLoss <= underLoss {
		t.Errorf("for q=0.1, over-prediction loss (%f) should exceed under-prediction loss (%f)",
			overLoss, underLoss)
	}
}
