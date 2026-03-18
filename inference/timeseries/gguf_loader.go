// Package timeseries implements GGUF metadata loading for time-series models.
package timeseries

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// TimeSeriesSignalConfig holds signal processing parameters loaded from
// GGUF metadata for time-series models.
type TimeSeriesSignalConfig struct {
	PatchLen      int // ts.signal.patch_len
	Stride        int // ts.signal.stride (defaults to PatchLen if absent)
	InputFeatures int // ts.signal.input_features
	HiddenDim     int // ts.signal.hidden_dim (default 128)
	NumHeads      int // ts.signal.num_heads (default 8)
	NumLayers     int // ts.signal.num_layers (default 6)
	HorizonLen    int // ts.signal.horizon_len (default 1)
}

// LoadTimeSeriesSignalConfig extracts TimeSeriesSignalConfig from a GGUF
// metadata map. Required keys are ts.signal.patch_len and
// ts.signal.input_features; all others have defaults.
func LoadTimeSeriesSignalConfig(meta map[string]interface{}) (TimeSeriesSignalConfig, error) {
	var cfg TimeSeriesSignalConfig

	patchLen, ok := getMetaInt(meta, "ts.signal.patch_len")
	if !ok {
		return cfg, fmt.Errorf("missing required metadata key %q", "ts.signal.patch_len")
	}
	cfg.PatchLen = patchLen

	inputFeatures, ok := getMetaInt(meta, "ts.signal.input_features")
	if !ok {
		return cfg, fmt.Errorf("missing required metadata key %q", "ts.signal.input_features")
	}
	cfg.InputFeatures = inputFeatures

	if v, ok := getMetaInt(meta, "ts.signal.stride"); ok {
		cfg.Stride = v
	} else {
		cfg.Stride = cfg.PatchLen
	}

	if v, ok := getMetaInt(meta, "ts.signal.hidden_dim"); ok {
		cfg.HiddenDim = v
	} else {
		cfg.HiddenDim = 128
	}

	if v, ok := getMetaInt(meta, "ts.signal.num_heads"); ok {
		cfg.NumHeads = v
	} else {
		cfg.NumHeads = 8
	}

	if v, ok := getMetaInt(meta, "ts.signal.num_layers"); ok {
		cfg.NumLayers = v
	} else {
		cfg.NumLayers = 6
	}

	if v, ok := getMetaInt(meta, "ts.signal.horizon_len"); ok {
		cfg.HorizonLen = v
	} else {
		cfg.HorizonLen = 1
	}

	return cfg, nil
}

// PatchTSTBuilder constructs a PatchTST computation graph from a signal
// config and compute engine. This allows callers to supply their own
// graph builder without creating a circular import.
type PatchTSTBuilder[T tensor.Numeric] func(cfg TimeSeriesSignalConfig, engine compute.Engine[T]) (*graph.Graph[T], error)

// LoadPatchTSTFromGGUF loads a PatchTST model from a GGUF file. It parses
// the file header, extracts ts.signal.* metadata, and delegates graph
// construction to the provided builder function.
func LoadPatchTSTFromGGUF[T tensor.Numeric](path string, engine compute.Engine[T], build PatchTSTBuilder[T]) (*graph.Graph[T], TimeSeriesSignalConfig, error) {
	var zero TimeSeriesSignalConfig

	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, zero, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, zero, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := LoadTimeSeriesSignalConfig(gf.Metadata)
	if err != nil {
		return nil, zero, fmt.Errorf("load time-series signal config: %w", err)
	}

	g, err := build(cfg, engine)
	if err != nil {
		return nil, zero, fmt.Errorf("build PatchTST: %w", err)
	}

	return g, cfg, nil
}

// getMetaInt extracts an integer value from GGUF metadata. GGUF stores
// integer metadata as uint32, so we accept that type and convert.
func getMetaInt(meta map[string]interface{}, key string) (int, bool) {
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
