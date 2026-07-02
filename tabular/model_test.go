package tabular

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func newTestEngine() (compute.Engine[float32], numeric.Arithmetic[float32]) {
	ops := numeric.Float32Ops{}
	return compute.NewCPUEngine[float32](ops), ops
}

func TestNewModel(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		config  ModelConfig
		wantErr bool
	}{
		{
			name: "valid single hidden layer",
			config: ModelConfig{
				InputDim:    4,
				HiddenDims:  []int{8},
				DropoutRate: 0.1,
				Activation:  ActivationReLU,
			},
		},
		{
			name: "valid multiple hidden layers",
			config: ModelConfig{
				InputDim:    10,
				HiddenDims:  []int{32, 16, 8},
				DropoutRate: 0.0,
				Activation:  ActivationGELU,
			},
		},
		{
			name: "zero input dim",
			config: ModelConfig{
				InputDim:   0,
				HiddenDims: []int{8},
			},
			wantErr: true,
		},
		{
			name: "negative input dim",
			config: ModelConfig{
				InputDim:   -1,
				HiddenDims: []int{8},
			},
			wantErr: true,
		},
		{
			name: "empty hidden dims",
			config: ModelConfig{
				InputDim:   4,
				HiddenDims: []int{},
			},
			wantErr: true,
		},
		{
			name: "zero hidden dim",
			config: ModelConfig{
				InputDim:   4,
				HiddenDims: []int{8, 0, 4},
			},
			wantErr: true,
		},
		{
			name: "negative dropout rate",
			config: ModelConfig{
				InputDim:    4,
				HiddenDims:  []int{8},
				DropoutRate: -0.1,
			},
			wantErr: true,
		},
		{
			name: "dropout rate >= 1",
			config: ModelConfig{
				InputDim:    4,
				HiddenDims:  []int{8},
				DropoutRate: 1.0,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewModel(tt.config, engine, ops)
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
			if len(m.layers) != len(tt.config.HiddenDims) {
				t.Errorf("expected %d layers, got %d", len(tt.config.HiddenDims), len(m.layers))
			}
		})
	}
}

func TestPredict_ThreeClasses(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name       string
		config     ModelConfig
		features   []float64
		wantErr    bool
		errFeature bool // error is about feature count mismatch
	}{
		{
			name: "basic prediction returns valid direction",
			config: ModelConfig{
				InputDim:    4,
				HiddenDims:  []int{8, 4},
				DropoutRate: 0.0,
				Activation:  ActivationReLU,
			},
			features: []float64{1.0, 2.0, 3.0, 4.0},
		},
		{
			name: "GELU activation returns valid direction",
			config: ModelConfig{
				InputDim:    3,
				HiddenDims:  []int{6},
				DropoutRate: 0.0,
				Activation:  ActivationGELU,
			},
			features: []float64{0.5, -0.5, 1.0},
		},
		{
			name: "zero features returns valid direction",
			config: ModelConfig{
				InputDim:    2,
				HiddenDims:  []int{4},
				DropoutRate: 0.0,
				Activation:  ActivationReLU,
			},
			features: []float64{0.0, 0.0},
		},
		{
			name: "wrong feature count",
			config: ModelConfig{
				InputDim:    4,
				HiddenDims:  []int{8},
				DropoutRate: 0.0,
				Activation:  ActivationReLU,
			},
			features:   []float64{1.0, 2.0},
			wantErr:    true,
			errFeature: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewModel(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewModel: %v", err)
			}

			dir, conf, err := m.Predict(tt.features)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}

			// Direction must be one of Long, Short, Flat.
			if dir < Long || dir > Flat {
				t.Errorf("direction %d is not in [Long, Short, Flat]", dir)
			}

			// Confidence must be in (0, 1] (softmax output, at least 1/3 for uniform).
			if conf <= 0 || conf > 1 {
				t.Errorf("confidence %f is not in (0, 1]", conf)
			}
		})
	}
}

func TestPredict_BatchConsistency(t *testing.T) {
	engine, ops := newTestEngine()

	config := ModelConfig{
		InputDim:    5,
		HiddenDims:  []int{10, 8},
		DropoutRate: 0.0,
		Activation:  ActivationReLU,
	}

	m, err := NewModel(config, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	features := []float64{1.0, -2.0, 3.0, -4.0, 5.0}

	// Run prediction multiple times — same model, same input, must get same output.
	dir1, conf1, err := m.Predict(features)
	if err != nil {
		t.Fatalf("Predict 1: %v", err)
	}

	for i := 0; i < 10; i++ {
		dir, conf, err := m.Predict(features)
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

func TestDirection_String(t *testing.T) {
	tests := []struct {
		dir  Direction
		want string
	}{
		{Long, "Long"},
		{Short, "Short"},
		{Flat, "Flat"},
		{Direction(99), "Direction(99)"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.dir.String(); got != tt.want {
				t.Errorf("Direction.String() = %q, want %q", got, tt.want)
			}
		})
	}
}
