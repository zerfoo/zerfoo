package timeseries

import (
	"strings"
	"testing"
)

func TestLoadGraniteTimeSeriesConfig(t *testing.T) {
	tests := []struct {
		name    string
		meta    map[string]interface{}
		check   func(t *testing.T, cfg *GraniteTimeSeriesConfig)
		wantErr string
	}{
		{
			name: "TTM model with all keys",
			meta: map[string]interface{}{
				"ts.signal.model_type":        "ttm",
				"ts.signal.context_len":       uint32(512),
				"ts.signal.forecast_len":      uint32(96),
				"ts.signal.num_mixer_layers":  uint32(4),
				"ts.signal.channel_mixing":    true,
				"ts.signal.patch_len":         uint32(16),
				"ts.signal.num_patches":       uint32(32),
				"ts.signal.input_features":    uint32(3),
			},
			check: func(t *testing.T, cfg *GraniteTimeSeriesConfig) {
				if cfg.ModelType != "ttm" {
					t.Errorf("ModelType = %q, want %q", cfg.ModelType, "ttm")
				}
				if cfg.ContextLen != 512 {
					t.Errorf("ContextLen = %d, want 512", cfg.ContextLen)
				}
				if cfg.ForecastLen != 96 {
					t.Errorf("ForecastLen = %d, want 96", cfg.ForecastLen)
				}
				if cfg.NumMixerLayers != 4 {
					t.Errorf("NumMixerLayers = %d, want 4", cfg.NumMixerLayers)
				}
				if !cfg.ChannelMixing {
					t.Error("ChannelMixing = false, want true")
				}
				if cfg.PatchLen != 16 {
					t.Errorf("PatchLen = %d, want 16", cfg.PatchLen)
				}
				if cfg.NumPatches != 32 {
					t.Errorf("NumPatches = %d, want 32", cfg.NumPatches)
				}
				// Embedded base config should have loaded input_features
				if cfg.InputFeatures != 3 {
					t.Errorf("InputFeatures = %d, want 3", cfg.InputFeatures)
				}
				// Default scale_factor should be 1.0
				if cfg.ScaleFactor != 1.0 {
					t.Errorf("ScaleFactor = %f, want 1.0", cfg.ScaleFactor)
				}
			},
		},
		{
			name: "FlowState model",
			meta: map[string]interface{}{
				"ts.signal.model_type":      "flowstate",
				"ts.signal.context_len":     uint32(1024),
				"ts.signal.forecast_len":    uint32(192),
				"ts.signal.scale_factor":    float32(2.5),
				"ts.signal.num_ssm_layers":  uint32(8),
			},
			check: func(t *testing.T, cfg *GraniteTimeSeriesConfig) {
				if cfg.ModelType != "flowstate" {
					t.Errorf("ModelType = %q, want %q", cfg.ModelType, "flowstate")
				}
				if cfg.ContextLen != 1024 {
					t.Errorf("ContextLen = %d, want 1024", cfg.ContextLen)
				}
				if cfg.ForecastLen != 192 {
					t.Errorf("ForecastLen = %d, want 192", cfg.ForecastLen)
				}
				if cfg.ScaleFactor != 2.5 {
					t.Errorf("ScaleFactor = %f, want 2.5", cfg.ScaleFactor)
				}
				if cfg.NumSSMLayers != 8 {
					t.Errorf("NumSSMLayers = %d, want 8", cfg.NumSSMLayers)
				}
			},
		},
		{
			name: "TSPulse model",
			meta: map[string]interface{}{
				"ts.signal.model_type":    "tspulse",
				"ts.signal.context_len":   uint32(1536),
				"ts.signal.forecast_len":  uint32(336),
				"ts.signal.mask_type":     "hybrid",
				"ts.signal.head_type":     "dualhead",
			},
			check: func(t *testing.T, cfg *GraniteTimeSeriesConfig) {
				if cfg.ModelType != "tspulse" {
					t.Errorf("ModelType = %q, want %q", cfg.ModelType, "tspulse")
				}
				if cfg.ContextLen != 1536 {
					t.Errorf("ContextLen = %d, want 1536", cfg.ContextLen)
				}
				if cfg.ForecastLen != 336 {
					t.Errorf("ForecastLen = %d, want 336", cfg.ForecastLen)
				}
				if cfg.MaskType != "hybrid" {
					t.Errorf("MaskType = %q, want %q", cfg.MaskType, "hybrid")
				}
				if cfg.HeadType != "dualhead" {
					t.Errorf("HeadType = %q, want %q", cfg.HeadType, "dualhead")
				}
			},
		},
		{
			name: "missing optional keys use defaults",
			meta: map[string]interface{}{
				"ts.signal.model_type": "ttm",
			},
			check: func(t *testing.T, cfg *GraniteTimeSeriesConfig) {
				if cfg.ModelType != "ttm" {
					t.Errorf("ModelType = %q, want %q", cfg.ModelType, "ttm")
				}
				if cfg.ContextLen != 0 {
					t.Errorf("ContextLen = %d, want 0", cfg.ContextLen)
				}
				if cfg.ForecastLen != 0 {
					t.Errorf("ForecastLen = %d, want 0", cfg.ForecastLen)
				}
				if cfg.ChannelMixing {
					t.Error("ChannelMixing = true, want false")
				}
				if cfg.ScaleFactor != 1.0 {
					t.Errorf("ScaleFactor = %f, want 1.0", cfg.ScaleFactor)
				}
				if cfg.MaskType != "" {
					t.Errorf("MaskType = %q, want empty", cfg.MaskType)
				}
				if cfg.HeadType != "" {
					t.Errorf("HeadType = %q, want empty", cfg.HeadType)
				}
			},
		},
		{
			name:    "missing required model_type returns error",
			meta:    map[string]interface{}{},
			wantErr: "ts.signal.model_type",
		},
		{
			name: "missing model_type with other keys returns error",
			meta: map[string]interface{}{
				"ts.signal.context_len":  uint32(512),
				"ts.signal.forecast_len": uint32(96),
			},
			wantErr: "ts.signal.model_type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg, err := LoadGraniteTimeSeriesConfig(tt.meta)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("error %q should mention %q", err.Error(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tt.check(t, cfg)
		})
	}
}
