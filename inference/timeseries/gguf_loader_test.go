package timeseries

import (
	"strings"
	"testing"
)

func TestTimeSeriesGGUFLoader(t *testing.T) {
	tests := []struct {
		name string
		meta map[string]interface{}
		want TimeSeriesSignalConfig
	}{
		{
			name: "all fields specified",
			meta: map[string]interface{}{
				"ts.signal.patch_len":      uint32(16),
				"ts.signal.stride":         uint32(8),
				"ts.signal.input_features": uint32(3),
				"ts.signal.hidden_dim":     uint32(256),
				"ts.signal.num_heads":      uint32(4),
				"ts.signal.num_layers":     uint32(12),
				"ts.signal.horizon_len":    uint32(10),
			},
			want: TimeSeriesSignalConfig{
				PatchLen:      16,
				Stride:        8,
				InputFeatures: 3,
				HiddenDim:     256,
				NumHeads:      4,
				NumLayers:     12,
				HorizonLen:    10,
			},
		},
		{
			name: "defaults applied",
			meta: map[string]interface{}{
				"ts.signal.patch_len":      uint32(32),
				"ts.signal.input_features": uint32(7),
			},
			want: TimeSeriesSignalConfig{
				PatchLen:      32,
				Stride:        32, // defaults to PatchLen
				InputFeatures: 7,
				HiddenDim:     128,
				NumHeads:      8,
				NumLayers:     6,
				HorizonLen:    1,
			},
		},
		{
			name: "stride defaults to patch_len when absent",
			meta: map[string]interface{}{
				"ts.signal.patch_len":      uint32(24),
				"ts.signal.input_features": uint32(1),
				"ts.signal.hidden_dim":     uint32(64),
			},
			want: TimeSeriesSignalConfig{
				PatchLen:      24,
				Stride:        24,
				InputFeatures: 1,
				HiddenDim:     64,
				NumHeads:      8,
				NumLayers:     6,
				HorizonLen:    1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := LoadTimeSeriesSignalConfig(tt.meta)
			if err != nil {
				t.Fatalf("LoadTimeSeriesSignalConfig() error = %v", err)
			}
			if got != tt.want {
				t.Errorf("LoadTimeSeriesSignalConfig() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestLoadTimeSeriesSignalConfigMissingRequired(t *testing.T) {
	tests := []struct {
		name    string
		meta    map[string]interface{}
		wantKey string
	}{
		{
			name: "missing patch_len",
			meta: map[string]interface{}{
				"ts.signal.input_features": uint32(3),
			},
			wantKey: "ts.signal.patch_len",
		},
		{
			name: "missing input_features",
			meta: map[string]interface{}{
				"ts.signal.patch_len": uint32(16),
			},
			wantKey: "ts.signal.input_features",
		},
		{
			name:    "empty metadata",
			meta:    map[string]interface{}{},
			wantKey: "ts.signal.patch_len",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadTimeSeriesSignalConfig(tt.meta)
			if err == nil {
				t.Fatal("LoadTimeSeriesSignalConfig() expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantKey) {
				t.Errorf("error %q should mention key %q", err.Error(), tt.wantKey)
			}
		})
	}
}

func TestLoadPatchTSTFromGGUFNotFound(t *testing.T) {
	_, _, err := LoadPatchTSTFromGGUF[float32]("/nonexistent/path/model.gguf", nil, nil)
	if err == nil {
		t.Fatal("LoadPatchTSTFromGGUF() expected error for nonexistent file, got nil")
	}
	if !strings.Contains(err.Error(), "no such file") {
		t.Errorf("error %q should mention 'no such file'", err.Error())
	}
}
