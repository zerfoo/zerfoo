# Tutorial 4: Time-Series Forecasting

Zerfoo includes three time-series architectures in the `inference/timeseries` package:

| Architecture | Strengths |
|-------------|-----------|
| **TFT** (Temporal Fusion Transformer) | Multi-horizon quantile forecasting with static covariates and LSTM encoder |
| **PatchTST** | Patch-based transformer for multivariate time series; efficient on long sequences |
| **N-BEATS** (via regime detection) | Interpretable basis expansion for univariate series |

All three are built as `graph.Graph[T]` computation graphs using the `ztensor` engine — the same graph compilation and CUDA capture infrastructure used by the LLM inference path.

## Prerequisites

```bash
mkdir ts-forecast && cd ts-forecast
go mod init example.com/ts-forecast
go get github.com/zerfoo/zerfoo@latest
```

## Step 1: Build a PatchTST Graph

PatchTST splits a time series into overlapping patches and processes them with a transformer encoder. It is the simplest architecture to start with.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

func main() {
	engine := compute.NewCPUEngine[float32]()

	cfg := timeseries.PatchTSTConfig{
		PatchLen:  16,  // each patch covers 16 time steps
		Stride:    8,   // patches overlap by 8 steps
		NumLayers: 3,   // transformer encoder depth
		NumHeads:  8,   // attention heads
		DModel:    128, // model dimension (must be divisible by NumHeads)
		Horizon:   24,  // predict 24 steps ahead
		NumVars:   7,   // number of input features (e.g., 7 sensor channels)
	}

	g, err := timeseries.BuildPatchTST[float32](cfg, engine)
	if err != nil {
		log.Fatal("build PatchTST:", err)
	}

	// Input: [batch=1, seq_len=96, num_vars=7]
	// Each element is a float32 sensor reading.
	inputData := make([]float32, 1*96*7)
	// ... fill inputData with your time series ...

	input, err := tensor.New[float32]([]int{1, 96, 7}, inputData)
	if err != nil {
		log.Fatal(err)
	}

	// Run forward pass. Output shape: [1, horizon=24, num_vars=7]
	outputs, err := g.Forward(context.Background(), input)
	if err != nil {
		log.Fatal("forward:", err)
	}

	// outputs[0] has shape [1, 24, 7] — 24-step forecast for each variable.
	forecast := outputs[0].Data()
	fmt.Printf("Forecast horizon: %d steps, %d variables\n", cfg.Horizon, cfg.NumVars)
	fmt.Printf("First predicted value: %.4f\n", forecast[0])
}
```

## Step 2: Build a TFT Graph

The Temporal Fusion Transformer adds static covariates (e.g., asset ID, location) and outputs multiple quantile predictions simultaneously.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

func main() {
	engine := compute.NewCPUEngine[float32]()

	cfg := timeseries.TFTConfig{
		NumStaticFeatures:   4,                         // e.g., [asset_id, sector, region, market_cap]
		NumTemporalFeatures: 10,                        // e.g., OHLCV + technical indicators
		HiddenDim:           64,                        // d_model (divisible by NumHeads)
		NumHeads:            8,
		NumLSTMLayers:       2,
		HorizonLen:          12,                        // 12-step forecast
		Quantiles:           []float32{0.1, 0.5, 0.9}, // 10th, 50th, 90th percentile
	}

	g, err := timeseries.BuildTFT[float32](cfg, engine)
	if err != nil {
		log.Fatal("build TFT:", err)
	}

	// Temporal input: [batch=2, seq_len=48, num_temporal_features=10]
	seqLen := 48
	temporalData := make([]float32, 2*seqLen*cfg.NumTemporalFeatures)
	temporal, err := tensor.New[float32]([]int{2, seqLen, cfg.NumTemporalFeatures}, temporalData)
	if err != nil {
		log.Fatal(err)
	}

	// Static input: [batch=2, num_static_features=4]
	staticData := make([]float32, 2*cfg.NumStaticFeatures)
	static, err := tensor.New[float32]([]int{2, cfg.NumStaticFeatures}, staticData)
	if err != nil {
		log.Fatal(err)
	}

	// TFT takes two inputs: temporal first, static second.
	outputs, err := g.Forward(context.Background(), temporal, static)
	if err != nil {
		log.Fatal("forward:", err)
	}

	// Output shape: [batch=2, horizon=12, num_quantiles=3]
	forecast := outputs[0].Data()
	// forecast[0] = batch 0, step 0, q10
	// forecast[1] = batch 0, step 0, q50
	// forecast[2] = batch 0, step 0, q90
	fmt.Printf("P10=%.4f  P50=%.4f  P90=%.4f\n",
		forecast[0], forecast[1], forecast[2])
}
```

## Step 3: Load a GGUF Time-Series Model

When a GGUF file contains `ts.signal.*` metadata, use the GGUF loader to reconstruct the graph configuration automatically:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

func main() {
	engine := compute.NewCPUEngine[float32]()

	// Parse GGUF metadata.
	meta, err := gguf.ParseMetadata("patchtst-etth1.gguf")
	if err != nil {
		log.Fatal("parse gguf:", err)
	}

	// Extract signal config from GGUF metadata (ts.signal.* keys).
	cfg, err := timeseries.LoadTimeSeriesSignalConfig(meta)
	if err != nil {
		log.Fatal("load signal config:", err)
	}
	// cfg.PatchLen, cfg.InputFeatures, cfg.HorizonLen, etc. are populated.

	// Build the graph using the GGUF-derived config.
	patchCfg := timeseries.PatchTSTConfig{
		PatchLen:  cfg.PatchLen,
		Stride:    cfg.Stride,
		NumLayers: cfg.NumLayers,
		NumHeads:  cfg.NumHeads,
		DModel:    cfg.HiddenDim,
		Horizon:   cfg.HorizonLen,
		NumVars:   cfg.InputFeatures,
	}

	g, err := timeseries.BuildPatchTST[float32](patchCfg, engine)
	if err != nil {
		log.Fatal("build graph:", err)
	}

	seqLen := 96
	inputData := make([]float32, 1*seqLen*cfg.InputFeatures)
	input, _ := tensor.New[float32]([]int{1, seqLen, cfg.InputFeatures}, inputData)

	outputs, err := g.Forward(context.Background(), input)
	if err != nil {
		log.Fatal("forward:", err)
	}

	fmt.Printf("Forecast shape: [1, %d, %d]\n", cfg.HorizonLen, cfg.InputFeatures)
	_ = outputs
}
```

## Step 4: Enable GPU Acceleration

Replace `compute.NewCPUEngine[float32]()` with a CUDA engine. No other code changes:

```go
// CPU (default, no build tags needed):
engine := compute.NewCPUEngine[float32]()

// CUDA (requires -tags cuda at build time and a CUDA-capable GPU at runtime):
engine, err := compute.NewCUDAEngine[float32](0) // device 0
if err != nil {
    log.Fatal(err)
}
```

The computation graph — including PatchTST or TFT — runs identically on both backends.

## Configuration Reference

### PatchTSTConfig

| Field | Description | Typical Value |
|-------|-------------|---------------|
| `PatchLen` | Tokens per patch | 16 |
| `Stride` | Patch stride | 8 |
| `NumLayers` | Encoder depth | 3–6 |
| `NumHeads` | Attention heads | 8 |
| `DModel` | Model dimension | 128–256 |
| `Horizon` | Forecast steps | 24–96 |
| `NumVars` | Input channels | 1–20 |

`DModel` must be divisible by `NumHeads`.

### TFTConfig

| Field | Description | Typical Value |
|-------|-------------|---------------|
| `NumStaticFeatures` | Static covariate count | 1–8 |
| `NumTemporalFeatures` | Time-varying feature count | 4–20 |
| `HiddenDim` | d_model | 64–256 |
| `NumHeads` | Attention heads | 4–8 |
| `NumLSTMLayers` | LSTM encoder layers | 1–3 |
| `HorizonLen` | Forecast steps | 1–24 |
| `Quantiles` | Quantile levels | `[0.1, 0.5, 0.9]` |

## Next Steps

- [Tutorial 5: Edge Deployment](05-edge-deployment.md) — cross-compile and deploy on Raspberry Pi or Jetson
- Read `inference/timeseries/gguf_loader.go` for the full GGUF metadata key reference
