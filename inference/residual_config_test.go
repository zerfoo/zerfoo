package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestResidualConfigDefault(t *testing.T) {
	cfg := DefaultResidualConfig()
	if cfg.Mode != "standard" {
		t.Errorf("DefaultResidualConfig().Mode = %q, want %q", cfg.Mode, "standard")
	}
	if cfg.NumBlocks != 0 {
		t.Errorf("DefaultResidualConfig().NumBlocks = %d, want 0", cfg.NumBlocks)
	}
}

func TestResidualConfigFromGGUF(t *testing.T) {
	tests := []struct {
		name      string
		mode      string
		numBlocks int
		wantMode  string
		wantNum   int
	}{
		{
			name:     "empty mode defaults to standard",
			mode:     "",
			wantMode: "standard",
		},
		{
			name:     "explicit standard",
			mode:     "standard",
			wantMode: "standard",
		},
		{
			name:     "attnres mode",
			mode:     "attnres",
			wantMode: "attnres",
		},
		{
			name:      "block_attnres with explicit blocks",
			mode:      "block_attnres",
			numBlocks: 4,
			wantMode:  "block_attnres",
			wantNum:   4,
		},
		{
			name:     "block_attnres defaults to 8 blocks",
			mode:     "block_attnres",
			wantMode: "block_attnres",
			wantNum:  8,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := ResidualConfigFromGGUF(tc.mode, tc.numBlocks)
			if cfg.Mode != tc.wantMode {
				t.Errorf("Mode = %q, want %q", cfg.Mode, tc.wantMode)
			}
			if cfg.NumBlocks != tc.wantNum {
				t.Errorf("NumBlocks = %d, want %d", cfg.NumBlocks, tc.wantNum)
			}
		})
	}
}

func TestBuildResidualConnection(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	tests := []struct {
		name   string
		config ResidualConfig
	}{
		{
			name:   "standard returns nil",
			config: ResidualConfig{Mode: "standard"},
		},
		{
			name:   "empty mode returns nil",
			config: ResidualConfig{},
		},
		{
			name:   "attnres returns nil placeholder",
			config: ResidualConfig{Mode: "attnres"},
		},
		{
			name:   "block_attnres returns nil placeholder",
			config: ResidualConfig{Mode: "block_attnres", NumBlocks: 8},
		},
		{
			name:   "unknown mode returns nil",
			config: ResidualConfig{Mode: "unknown"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := BuildResidualConnection[float32](tc.config, eng)
			if result != nil {
				t.Errorf("BuildResidualConnection() = %v, want nil", result)
			}
		})
	}
}
