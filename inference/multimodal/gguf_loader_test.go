package multimodal

import (
	"strings"
	"testing"
)

func TestMultiModalConfigFromMetadata(t *testing.T) {
	meta := map[string]any{
		"vision.encoder.type":        "siglip",
		"vision.hidden_size":         uint32(1152),
		"vision.patch_size":          uint32(14),
		"vision.image_size":          uint32(384),
		"vision.attention.head_count": uint32(16),
		"vision.block_count":         uint32(27),
		"mm.projector.weight":        []float32{0.1, 0.2, 0.3},
	}

	cfg, err := MultiModalConfigFromMetadata(meta)
	if err != nil {
		t.Fatalf("MultiModalConfigFromMetadata() error = %v", err)
	}

	if cfg.EncoderType != "siglip" {
		t.Errorf("EncoderType = %q, want %q", cfg.EncoderType, "siglip")
	}
	if cfg.HiddenSize != 1152 {
		t.Errorf("HiddenSize = %d, want %d", cfg.HiddenSize, 1152)
	}
	if cfg.PatchSize != 14 {
		t.Errorf("PatchSize = %d, want %d", cfg.PatchSize, 14)
	}
	if cfg.ImageSize != 384 {
		t.Errorf("ImageSize = %d, want %d", cfg.ImageSize, 384)
	}
	if cfg.NumHeads != 16 {
		t.Errorf("NumHeads = %d, want %d", cfg.NumHeads, 16)
	}
	if cfg.NumLayers != 27 {
		t.Errorf("NumLayers = %d, want %d", cfg.NumLayers, 27)
	}
	if len(cfg.ProjectorWeights) != 3 {
		t.Errorf("ProjectorWeights length = %d, want %d", len(cfg.ProjectorWeights), 3)
	}
}

func TestMultiModalConfigDefaults(t *testing.T) {
	meta := map[string]any{
		"vision.encoder.type": "clip",
		"vision.hidden_size":  uint32(768),
	}

	cfg, err := MultiModalConfigFromMetadata(meta)
	if err != nil {
		t.Fatalf("MultiModalConfigFromMetadata() error = %v", err)
	}

	if cfg.EncoderType != "clip" {
		t.Errorf("EncoderType = %q, want %q", cfg.EncoderType, "clip")
	}
	if cfg.HiddenSize != 768 {
		t.Errorf("HiddenSize = %d, want %d", cfg.HiddenSize, 768)
	}
	if cfg.PatchSize != 14 {
		t.Errorf("PatchSize = %d, want default %d", cfg.PatchSize, 14)
	}
	if cfg.ImageSize != 224 {
		t.Errorf("ImageSize = %d, want default %d", cfg.ImageSize, 224)
	}
	if cfg.NumHeads != 12 {
		t.Errorf("NumHeads = %d, want default %d", cfg.NumHeads, 12)
	}
	if cfg.NumLayers != 12 {
		t.Errorf("NumLayers = %d, want default %d", cfg.NumLayers, 12)
	}
	if cfg.ProjectorWeights != nil {
		t.Errorf("ProjectorWeights = %v, want nil when absent", cfg.ProjectorWeights)
	}
}

func TestMultiModalConfigEncoderType(t *testing.T) {
	tests := []struct {
		name      string
		encType   string
		wantErr   bool
		errSubstr string
	}{
		{name: "siglip", encType: "siglip", wantErr: false},
		{name: "clip", encType: "clip", wantErr: false},
		{name: "unknown", encType: "resnet", wantErr: true, errSubstr: "unsupported"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta := map[string]any{
				"vision.encoder.type": tt.encType,
				"vision.hidden_size":  uint32(768),
			}
			cfg, err := MultiModalConfigFromMetadata(meta)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.errSubstr) {
					t.Errorf("error %q should contain %q", err.Error(), tt.errSubstr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if cfg.EncoderType != tt.encType {
				t.Errorf("EncoderType = %q, want %q", cfg.EncoderType, tt.encType)
			}
		})
	}
}

func TestMultiModalConfigMissingRequired(t *testing.T) {
	tests := []struct {
		name    string
		meta    map[string]any
		wantKey string
	}{
		{
			name:    "missing encoder type",
			meta:    map[string]any{"vision.hidden_size": uint32(768)},
			wantKey: "vision.encoder.type",
		},
		{
			name:    "missing hidden size",
			meta:    map[string]any{"vision.encoder.type": "siglip"},
			wantKey: "vision.hidden_size",
		},
		{
			name:    "empty metadata",
			meta:    map[string]any{},
			wantKey: "vision.encoder.type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := MultiModalConfigFromMetadata(tt.meta)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantKey) {
				t.Errorf("error %q should mention key %q", err.Error(), tt.wantKey)
			}
		})
	}
}

func TestLoadMultiModalConfigFromFileNotFound(t *testing.T) {
	_, err := LoadMultiModalConfigFromFile("/nonexistent/path/model.gguf")
	if err == nil {
		t.Fatal("expected error for nonexistent file, got nil")
	}
	if !strings.Contains(err.Error(), "no such file") {
		t.Errorf("error %q should mention 'no such file'", err.Error())
	}
}
