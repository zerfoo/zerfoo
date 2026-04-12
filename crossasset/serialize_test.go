package crossasset

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoad_RoundTrip(t *testing.T) {
	configs := []Config{
		{NSources: 4, FeaturesPerSource: 3, DModel: 16, NHeads: 4, NLayers: 2, LearningRate: 1e-3},
		{NSources: 2, FeaturesPerSource: 5, DModel: 8, NHeads: 2, NLayers: 1, LearningRate: 1e-3},
		{NSources: 6, FeaturesPerSource: 2, DModel: 32, NHeads: 8, NLayers: 3, LearningRate: 1e-3},
	}

	for _, cfg := range configs {
		t.Run("", func(t *testing.T) {
			// Train a model.
			m := NewModel(cfg)
			data, labels := makeTrainData(cfg, 20)
			if err := m.Train(data, labels, TrainConfig{Epochs: 2, BatchSize: 10, LearningRate: 1e-4}); err != nil {
				t.Fatalf("Train: %v", err)
			}

			// Predict before save.
			input := data[0]
			dirsBefore, confsBefore, err := m.Predict(input)
			if err != nil {
				t.Fatalf("Predict before save: %v", err)
			}

			// Save.
			path := filepath.Join(t.TempDir(), "model.zcam")
			if err := m.Save(path); err != nil {
				t.Fatalf("Save: %v", err)
			}

			// Verify file exists and is non-empty.
			info, err := os.Stat(path)
			if err != nil {
				t.Fatalf("Stat: %v", err)
			}
			if info.Size() == 0 {
				t.Fatal("saved file is empty")
			}

			// Load.
			loaded, err := LoadModel(path)
			if err != nil {
				t.Fatalf("LoadModel: %v", err)
			}

			// Predict after load.
			dirsAfter, confsAfter, err := loaded.Predict(input)
			if err != nil {
				t.Fatalf("Predict after load: %v", err)
			}

			// Verify bitwise equality.
			if len(dirsBefore) != len(dirsAfter) {
				t.Fatalf("directions length mismatch: %d vs %d", len(dirsBefore), len(dirsAfter))
			}
			for i := range dirsBefore {
				if dirsBefore[i] != dirsAfter[i] {
					t.Errorf("direction[%d]: before=%d after=%d", i, dirsBefore[i], dirsAfter[i])
				}
				if confsBefore[i] != confsAfter[i] {
					t.Errorf("confidence[%d]: before=%f after=%f", i, confsBefore[i], confsAfter[i])
				}
			}

			// Verify config round-tripped.
			if loaded.config != m.config {
				t.Errorf("config mismatch: %+v vs %+v", m.config, loaded.config)
			}
		})
	}
}

func TestLoadModel_InvalidMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.zcam")
	if err := os.WriteFile(path, []byte("BADMxxxxxx"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadModel(path)
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestLoadModel_Truncated(t *testing.T) {
	// Save a valid model then truncate the file.
	cfg := Config{NSources: 2, FeaturesPerSource: 3, DModel: 8, NHeads: 2, NLayers: 1, LearningRate: 1e-3}
	m := NewModel(cfg)
	path := filepath.Join(t.TempDir(), "trunc.zcam")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	// Truncate to half the file size.
	info, _ := os.Stat(path)
	if err := os.Truncate(path, info.Size()/2); err != nil {
		t.Fatal(err)
	}
	_, err := LoadModel(path)
	if err == nil {
		t.Fatal("expected error for truncated file")
	}
}

func TestLoadModel_FileNotFound(t *testing.T) {
	_, err := LoadModel("/nonexistent/path/model.zcam")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

// makeTrainData generates synthetic training data for a given config.
func makeTrainData(cfg Config, nSamples int) ([][][]float32, [][]int) {
	data := make([][][]float32, nSamples)
	labels := make([][]int, nSamples)
	for i := 0; i < nSamples; i++ {
		data[i] = make([][]float32, cfg.NSources)
		labels[i] = make([]int, cfg.NSources)
		for s := 0; s < cfg.NSources; s++ {
			data[i][s] = make([]float32, cfg.FeaturesPerSource)
			for f := 0; f < cfg.FeaturesPerSource; f++ {
				data[i][s][f] = float32(i*cfg.NSources*cfg.FeaturesPerSource+s*cfg.FeaturesPerSource+f) * 0.01
			}
			labels[i][s] = (i + s) % 3 // cycle through Long/Short/Flat
		}
	}
	return data, labels
}
