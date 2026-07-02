package generate

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWithEAGLE(t *testing.T) {
	t.Run("sets eagle weights path", func(t *testing.T) {
		var opts generatorOptions
		WithEAGLE("/tmp/eagle-head.gguf")(&opts)
		if opts.eagleWeightsPath != "/tmp/eagle-head.gguf" {
			t.Errorf("eagleWeightsPath = %q, want %q", opts.eagleWeightsPath, "/tmp/eagle-head.gguf")
		}
	})

	t.Run("empty path leaves option unset", func(t *testing.T) {
		var opts generatorOptions
		WithEAGLE("")(&opts)
		if opts.eagleWeightsPath != "" {
			t.Errorf("eagleWeightsPath = %q, want empty", opts.eagleWeightsPath)
		}
	})
}

func TestEAGLEEnabled(t *testing.T) {
	cfg := ModelConfig{VocabSize: 32000, MaxSeqLen: 2048, EOSTokenID: 2, BOSTokenID: 1, NumLayers: 12}

	t.Run("disabled when path is empty", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg)
		if gen.EAGLEEnabled() {
			t.Error("expected EAGLEEnabled=false with no option set")
		}
		if gen.EAGLEWeightsPath() != "" {
			t.Error("expected empty EAGLEWeightsPath")
		}
	})

	t.Run("disabled when file does not exist", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithEAGLE("/nonexistent/eagle.gguf"))
		if gen.EAGLEEnabled() {
			t.Error("expected EAGLEEnabled=false for missing file")
		}
		if gen.EAGLEWeightsPath() != "/nonexistent/eagle.gguf" {
			t.Errorf("EAGLEWeightsPath = %q, want %q", gen.EAGLEWeightsPath(), "/nonexistent/eagle.gguf")
		}
	})

	t.Run("enabled when file exists", func(t *testing.T) {
		tmp := filepath.Join(t.TempDir(), "eagle-head.gguf")
		if err := os.WriteFile(tmp, []byte("fake"), 0644); err != nil {
			t.Fatal(err)
		}
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithEAGLE(tmp))
		if !gen.EAGLEEnabled() {
			t.Error("expected EAGLEEnabled=true when weights file exists")
		}
		if gen.EAGLEWeightsPath() != tmp {
			t.Errorf("EAGLEWeightsPath = %q, want %q", gen.EAGLEWeightsPath(), tmp)
		}
	})

	t.Run("falls back to vanilla when path set but file missing", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithEAGLE("/tmp/does-not-exist-eagle.gguf"))
		// Generator should still be created successfully — fallback to vanilla.
		if gen == nil {
			t.Fatal("expected non-nil generator even with missing EAGLE weights")
		}
		if gen.EAGLEEnabled() {
			t.Error("expected EAGLEEnabled=false as fallback for missing weights")
		}
	})
}
