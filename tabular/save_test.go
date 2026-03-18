package tabular

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSave(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name   string
		config ModelConfig
	}{
		{
			name: "single hidden layer",
			config: ModelConfig{
				InputDim:    4,
				HiddenDims:  []int{8},
				DropoutRate: 0.1,
				Activation:  ActivationReLU,
			},
		},
		{
			name: "multiple hidden layers",
			config: ModelConfig{
				InputDim:    10,
				HiddenDims:  []int{32, 16, 8},
				DropoutRate: 0.0,
				Activation:  ActivationGELU,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewModel(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewModel: %v", err)
			}

			path := filepath.Join(t.TempDir(), "model.ztab")
			if err := Save(m, path); err != nil {
				t.Fatalf("Save: %v", err)
			}

			info, err := os.Stat(path)
			if err != nil {
				t.Fatalf("Stat: %v", err)
			}
			if info.Size() == 0 {
				t.Fatal("saved file is empty")
			}
		})
	}
}

func TestLoad(t *testing.T) {
	engine, ops := newTestEngine()

	config := ModelConfig{
		InputDim:    4,
		HiddenDims:  []int{8, 4},
		DropoutRate: 0.1,
		Activation:  ActivationReLU,
	}

	m, err := NewModel(config, engine, ops)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	path := filepath.Join(t.TempDir(), "model.ztab")
	if err := Save(m, path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	loaded, err := Load(path, engine, ops)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Verify config was restored.
	if loaded.config.InputDim != config.InputDim {
		t.Errorf("InputDim = %d, want %d", loaded.config.InputDim, config.InputDim)
	}
	if len(loaded.config.HiddenDims) != len(config.HiddenDims) {
		t.Fatalf("HiddenDims length = %d, want %d", len(loaded.config.HiddenDims), len(config.HiddenDims))
	}
	for i, h := range loaded.config.HiddenDims {
		if h != config.HiddenDims[i] {
			t.Errorf("HiddenDims[%d] = %d, want %d", i, h, config.HiddenDims[i])
		}
	}
	if loaded.config.Activation != config.Activation {
		t.Errorf("Activation = %d, want %d", loaded.config.Activation, config.Activation)
	}

	// Verify layer count.
	if len(loaded.layers) != len(m.layers) {
		t.Errorf("layer count = %d, want %d", len(loaded.layers), len(m.layers))
	}
}

func TestRoundTrip(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name     string
		config   ModelConfig
		features []float64
	}{
		{
			name: "single layer ReLU",
			config: ModelConfig{
				InputDim:    3,
				HiddenDims:  []int{6},
				DropoutRate: 0.0,
				Activation:  ActivationReLU,
			},
			features: []float64{1.0, -0.5, 2.0},
		},
		{
			name: "multi layer GELU",
			config: ModelConfig{
				InputDim:    5,
				HiddenDims:  []int{16, 8, 4},
				DropoutRate: 0.2,
				Activation:  ActivationGELU,
			},
			features: []float64{0.1, 0.2, 0.3, 0.4, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original, err := NewModel(tt.config, engine, ops)
			if err != nil {
				t.Fatalf("NewModel: %v", err)
			}

			// Get original prediction.
			origDir, origConf, err := original.Predict(tt.features)
			if err != nil {
				t.Fatalf("Predict (original): %v", err)
			}

			// Save and load.
			path := filepath.Join(t.TempDir(), "model.ztab")
			if err := Save(original, path); err != nil {
				t.Fatalf("Save: %v", err)
			}
			loaded, err := Load(path, engine, ops)
			if err != nil {
				t.Fatalf("Load: %v", err)
			}

			// Get loaded prediction.
			loadDir, loadConf, err := loaded.Predict(tt.features)
			if err != nil {
				t.Fatalf("Predict (loaded): %v", err)
			}

			if origDir != loadDir {
				t.Errorf("direction mismatch: original=%v, loaded=%v", origDir, loadDir)
			}
			if diff := math.Abs(origConf - loadConf); diff > 1e-7 {
				t.Errorf("confidence mismatch: original=%v, loaded=%v, diff=%v", origConf, loadConf, diff)
			}
		})
	}
}

func TestLoad_InvalidFile(t *testing.T) {
	engine, ops := newTestEngine()

	tests := []struct {
		name    string
		content []byte
	}{
		{
			name:    "empty file",
			content: []byte{},
		},
		{
			name:    "wrong magic",
			content: []byte("XYZW\x01\x00\x00\x00\x02\x00\x00\x00{}"),
		},
		{
			name:    "truncated header",
			content: []byte("ZT"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "bad.ztab")
			if err := os.WriteFile(path, tt.content, 0o644); err != nil {
				t.Fatalf("WriteFile: %v", err)
			}

			_, err := Load(path, engine, ops)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
		})
	}
}

func TestLoad_VersionMismatch(t *testing.T) {
	engine, ops := newTestEngine()

	// Build a file with correct magic but wrong version (99).
	path := filepath.Join(t.TempDir(), "badversion.ztab")
	data := []byte{
		'Z', 'T', 'A', 'B', // magic
		99, 0, 0, 0, // version 99 (little-endian)
		2, 0, 0, 0, // config length 2
		'{', '}', // minimal JSON
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, err := Load(path, engine, ops)
	if err == nil {
		t.Fatal("expected error for version mismatch, got nil")
	}
}
