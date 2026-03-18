// Package multimodal provides GGUF metadata loading for vision and
// multimodal models.
package multimodal

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/model/gguf"
)

// MultiModalConfig holds vision encoder and projector parameters loaded
// from GGUF metadata.
type MultiModalConfig struct {
	EncoderType      string    // vision.encoder.type ("siglip", "clip")
	HiddenSize       int       // vision.hidden_size
	PatchSize        int       // vision.patch_size
	ImageSize        int       // vision.image_size
	NumHeads         int       // vision.attention.head_count
	NumLayers        int       // vision.block_count
	ProjectorWeights []float32 // mm.projector.weight (flattened tensor)
}

// LoadMultiModalConfig reads GGUF from r and extracts multimodal config.
func LoadMultiModalConfig(r io.ReadSeeker) (*MultiModalConfig, error) {
	gf, err := gguf.Parse(r)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}
	return MultiModalConfigFromMetadata(gf.Metadata)
}

// LoadMultiModalConfigFromFile opens a GGUF file at path and loads
// multimodal config from it.
func LoadMultiModalConfigFromFile(path string) (*MultiModalConfig, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()
	return LoadMultiModalConfig(f)
}

// MultiModalConfigFromMetadata extracts a MultiModalConfig from a
// pre-parsed GGUF metadata map.
func MultiModalConfigFromMetadata(metadata map[string]any) (*MultiModalConfig, error) {
	cfg := &MultiModalConfig{}

	// Required: encoder type.
	encType, ok := getMetaString(metadata, "vision.encoder.type")
	if !ok {
		return nil, fmt.Errorf("missing required metadata key %q", "vision.encoder.type")
	}
	if encType != "siglip" && encType != "clip" {
		return nil, fmt.Errorf("unsupported vision encoder type %q (expected \"siglip\" or \"clip\")", encType)
	}
	cfg.EncoderType = encType

	// Required: hidden size.
	hiddenSize, ok := getMetaInt(metadata, "vision.hidden_size")
	if !ok {
		return nil, fmt.Errorf("missing required metadata key %q", "vision.hidden_size")
	}
	cfg.HiddenSize = hiddenSize

	// Optional with defaults.
	if v, ok := getMetaInt(metadata, "vision.patch_size"); ok {
		cfg.PatchSize = v
	} else {
		cfg.PatchSize = 14
	}

	if v, ok := getMetaInt(metadata, "vision.image_size"); ok {
		cfg.ImageSize = v
	} else {
		cfg.ImageSize = 224
	}

	if v, ok := getMetaInt(metadata, "vision.attention.head_count"); ok {
		cfg.NumHeads = v
	} else {
		cfg.NumHeads = 12
	}

	if v, ok := getMetaInt(metadata, "vision.block_count"); ok {
		cfg.NumLayers = v
	} else {
		cfg.NumLayers = 12
	}

	// Optional: projector weights (flattened float32 slice).
	if w, ok := getMetaFloat32Slice(metadata, "mm.projector.weight"); ok {
		cfg.ProjectorWeights = w
	}

	return cfg, nil
}

// getMetaString extracts a string value from GGUF metadata.
func getMetaString(meta map[string]any, key string) (string, bool) {
	v, ok := meta[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// getMetaInt extracts an integer value from GGUF metadata. GGUF stores
// integer metadata as uint32, so we accept several numeric types.
func getMetaInt(meta map[string]any, key string) (int, bool) {
	v, ok := meta[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case uint32:
		return int(n), true
	case int:
		return n, true
	case int64:
		return int(n), true
	case uint64:
		return int(n), true
	case float64:
		return int(n), true
	case float32:
		return int(n), true
	default:
		return 0, false
	}
}

// getMetaFloat32Slice extracts a []float32 value from GGUF metadata.
func getMetaFloat32Slice(meta map[string]any, key string) ([]float32, bool) {
	v, ok := meta[key]
	if !ok {
		return nil, false
	}
	s, ok := v.([]float32)
	return s, ok
}
