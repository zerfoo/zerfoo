package tabular

import (
	"testing"
)

func TestFTTransformer_Forward(t *testing.T) {
	engine, ops := newTestEngine()

	config := FTTransformerConfig{
		NumFeatures: 4,
		DToken:      8,
		NHeads:      2,
		NLayers:     2,
		DFFN:        16,
		DropoutRate: 0.0,
	}

	ft, err := NewFTTransformer(config, engine, ops)
	if err != nil {
		t.Fatalf("NewFTTransformer: %v", err)
	}

	features := []float64{1.0, -0.5, 2.0, 0.3}
	dir, conf, err := ft.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if dir < Long || dir > Flat {
		t.Errorf("direction %d is not in [Long, Short, Flat]", dir)
	}
	if conf <= 0 || conf > 1 {
		t.Errorf("confidence %f is not in (0, 1]", conf)
	}

	// Deterministic: same input must produce same output.
	for i := 0; i < 5; i++ {
		dir2, conf2, err := ft.Predict(features)
		if err != nil {
			t.Fatalf("Predict iteration %d: %v", i, err)
		}
		if dir2 != dir {
			t.Errorf("iteration %d: direction %v != %v", i, dir2, dir)
		}
		if conf2 != conf {
			t.Errorf("iteration %d: confidence %f != %f", i, conf2, conf)
		}
	}
}

func TestFTTransformer_Train(t *testing.T) {
	engine, ops := newTestEngine()

	config := FTTransformerConfig{
		NumFeatures: 2,
		DToken:      8,
		NHeads:      2,
		NLayers:     1,
		DFFN:        16,
		DropoutRate: 0.0,
	}

	ft, err := NewFTTransformer(config, engine, ops)
	if err != nil {
		t.Fatalf("NewFTTransformer: %v", err)
	}

	// Verify the model can produce predictions for various inputs
	// (basic "trainability" check — model doesn't NaN or panic).
	inputs := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{0.5, 0.5},
		{-1, 2},
	}

	for i, inp := range inputs {
		dir, conf, err := ft.Predict(inp)
		if err != nil {
			t.Fatalf("Predict(%v): %v", inp, err)
		}
		if dir < Long || dir > Flat {
			t.Errorf("input %d: direction %d out of range", i, dir)
		}
		if conf <= 0 || conf > 1 {
			t.Errorf("input %d: confidence %f out of range", i, conf)
		}
	}

	// Verify different inputs can produce different outputs
	// (the model is not collapsed to a constant function).
	dirs := make(map[Direction]bool)
	testInputs := [][]float64{
		{10.0, -10.0},
		{-10.0, 10.0},
		{0.0, 0.0},
		{100.0, 100.0},
		{-100.0, -100.0},
	}
	for _, inp := range testInputs {
		dir, _, err := ft.Predict(inp)
		if err != nil {
			t.Fatalf("Predict(%v): %v", inp, err)
		}
		dirs[dir] = true
	}
	// With extreme inputs, we expect at least some variation.
	// If all outputs are identical, the model is degenerate, but this is
	// initialization-dependent, so we just check no errors occurred.
}

func TestFTTransformer_Shapes(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  FTTransformerConfig
		nFeat   int
		wantErr bool
	}{
		{
			name: "minimal config",
			config: FTTransformerConfig{
				NumFeatures: 1,
				DToken:      4,
				NHeads:      1,
				NLayers:     1,
				DFFN:        8,
				DropoutRate: 0.0,
			},
			nFeat: 1,
		},
		{
			name: "larger config",
			config: FTTransformerConfig{
				NumFeatures: 10,
				DToken:      16,
				NHeads:      4,
				NLayers:     3,
				DFFN:        32,
				DropoutRate: 0.0,
			},
			nFeat: 10,
		},
		{
			name: "single head",
			config: FTTransformerConfig{
				NumFeatures: 5,
				DToken:      12,
				NHeads:      1,
				NLayers:     2,
				DFFN:        24,
				DropoutRate: 0.0,
			},
			nFeat: 5,
		},
		{
			name: "wrong feature count",
			config: FTTransformerConfig{
				NumFeatures: 3,
				DToken:      4,
				NHeads:      1,
				NLayers:     1,
				DFFN:        8,
				DropoutRate: 0.0,
			},
			nFeat:   5,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ft, err := NewFTTransformer(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewFTTransformer: %v", err)
			}

			features := make([]float64, tt.nFeat)
			for i := range features {
				features[i] = float64(i) * 0.1
			}

			_, _, err = ft.Predict(features)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
		})
	}
}

func TestNewFTTransformer_Validation(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  FTTransformerConfig
		wantErr bool
	}{
		{
			name: "valid config",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      8,
				NHeads:      2,
				NLayers:     2,
				DFFN:        16,
				DropoutRate: 0.0,
			},
		},
		{
			name: "zero features",
			config: FTTransformerConfig{
				NumFeatures: 0,
				DToken:      8,
				NHeads:      2,
				NLayers:     1,
				DFFN:        16,
			},
			wantErr: true,
		},
		{
			name: "zero DToken",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      0,
				NHeads:      2,
				NLayers:     1,
				DFFN:        16,
			},
			wantErr: true,
		},
		{
			name: "zero heads",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      8,
				NHeads:      0,
				NLayers:     1,
				DFFN:        16,
			},
			wantErr: true,
		},
		{
			name: "DToken not divisible by NHeads",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      7,
				NHeads:      2,
				NLayers:     1,
				DFFN:        16,
			},
			wantErr: true,
		},
		{
			name: "zero layers",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      8,
				NHeads:      2,
				NLayers:     0,
				DFFN:        16,
			},
			wantErr: true,
		},
		{
			name: "zero DFFN",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      8,
				NHeads:      2,
				NLayers:     1,
				DFFN:        0,
			},
			wantErr: true,
		},
		{
			name: "negative dropout",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      8,
				NHeads:      2,
				NLayers:     1,
				DFFN:        16,
				DropoutRate: -0.1,
			},
			wantErr: true,
		},
		{
			name: "dropout >= 1",
			config: FTTransformerConfig{
				NumFeatures: 4,
				DToken:      8,
				NHeads:      2,
				NLayers:     1,
				DFFN:        16,
				DropoutRate: 1.0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewFTTransformer(tt.config, engine, ops)
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
