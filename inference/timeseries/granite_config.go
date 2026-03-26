package timeseries

import "fmt"

// GraniteTimeSeriesConfig extends TimeSeriesSignalConfig with Granite-specific
// fields for IBM Granite Time Series models (TTM, FlowState, TSPulse).
type GraniteTimeSeriesConfig struct {
	TimeSeriesSignalConfig // embed existing config

	// Common Granite TS fields
	ModelType   string // ts.signal.model_type — "ttm", "flowstate", "tspulse" (required)
	ContextLen  int    // ts.signal.context_len — context window (512, 1024, 1536)
	ForecastLen int    // ts.signal.forecast_len — prediction horizon

	// TTM-specific
	NumMixerLayers int  // ts.signal.num_mixer_layers — TSMixer backbone depth
	ChannelMixing  bool // ts.signal.channel_mixing — TTM decoder channel mixing enabled
	PatchLen       int  // ts.signal.patch_len — patch length for adaptive patching
	NumPatches     int  // ts.signal.num_patches — number of patches

	// FlowState-specific
	ScaleFactor  float32 // ts.signal.scale_factor — temporal scale for sampling rate adaptation
	NumSSMLayers int     // ts.signal.num_ssm_layers — SSM encoder depth

	// TSPulse-specific
	MaskType string // ts.signal.mask_type — "hybrid" or "block"
	HeadType string // ts.signal.head_type — "allhead" or "dualhead"
}

// LoadGraniteTimeSeriesConfig extracts GraniteTimeSeriesConfig from a GGUF
// metadata map. Only ts.signal.model_type is required; all other fields have
// sensible defaults.
func LoadGraniteTimeSeriesConfig(meta map[string]interface{}) (*GraniteTimeSeriesConfig, error) {
	cfg := &GraniteTimeSeriesConfig{
		ScaleFactor: 1.0,
	}

	// Required: model_type
	modelType, ok := getMetaString(meta, "ts.signal.model_type")
	if !ok {
		return nil, fmt.Errorf("missing required metadata key %q", "ts.signal.model_type")
	}
	cfg.ModelType = modelType

	// Load embedded base config (tolerant — ignore error since base has its
	// own required fields that may not apply to Granite models).
	if base, err := LoadTimeSeriesSignalConfig(meta); err == nil {
		cfg.TimeSeriesSignalConfig = base
	}

	// Common Granite TS fields
	if v, ok := getMetaInt(meta, "ts.signal.context_len"); ok {
		cfg.ContextLen = v
	}
	if v, ok := getMetaInt(meta, "ts.signal.forecast_len"); ok {
		cfg.ForecastLen = v
	}

	// TTM-specific
	if v, ok := getMetaInt(meta, "ts.signal.num_mixer_layers"); ok {
		cfg.NumMixerLayers = v
	}
	if v, ok := getMetaBool(meta, "ts.signal.channel_mixing"); ok {
		cfg.ChannelMixing = v
	}
	if v, ok := getMetaInt(meta, "ts.signal.patch_len"); ok {
		cfg.PatchLen = v
	}
	if v, ok := getMetaInt(meta, "ts.signal.num_patches"); ok {
		cfg.NumPatches = v
	}

	// FlowState-specific
	if v, ok := getMetaFloat32(meta, "ts.signal.scale_factor"); ok {
		cfg.ScaleFactor = v
	}
	if v, ok := getMetaInt(meta, "ts.signal.num_ssm_layers"); ok {
		cfg.NumSSMLayers = v
	}

	// TSPulse-specific
	if v, ok := getMetaString(meta, "ts.signal.mask_type"); ok {
		cfg.MaskType = v
	}
	if v, ok := getMetaString(meta, "ts.signal.head_type"); ok {
		cfg.HeadType = v
	}

	return cfg, nil
}

// getMetaString extracts a string value from GGUF metadata.
func getMetaString(meta map[string]interface{}, key string) (string, bool) {
	v, ok := meta[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// getMetaBool extracts a boolean value from GGUF metadata.
func getMetaBool(meta map[string]interface{}, key string) (bool, bool) {
	v, ok := meta[key]
	if !ok {
		return false, false
	}
	b, ok := v.(bool)
	return b, ok
}

// getMetaFloat32 extracts a float32 value from GGUF metadata.
func getMetaFloat32(meta map[string]interface{}, key string) (float32, bool) {
	v, ok := meta[key]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float32:
		return n, true
	case float64:
		return float32(n), true
	default:
		return 0, false
	}
}
