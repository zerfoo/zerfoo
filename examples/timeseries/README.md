# Time-Series Forecasting with N-BEATS

Demonstrates time-series forecasting using the N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) model from the `timeseries` package.

## What this example shows

1. **Model configuration** -- setting up N-BEATS with trend and seasonality stacks
2. **Synthetic data generation** -- creating a time series with trend + seasonal pattern
3. **Forward pass** -- running inference to produce a multi-step forecast
4. **Interpretable decomposition** -- examining per-stack (trend, seasonality) forecast components

## Build

```bash
go build -o ts-forecast ./examples/timeseries/
```

## Run

```bash
./ts-forecast
```

## Expected output

```
=== N-BEATS Time-Series Forecasting Example ===

Model configuration:
  Input length:      24
  Output length:     6
  Stacks:            trend + seasonality
  Blocks per stack:  2
  Hidden dim:        32
  Fourier harmonics: 4

N-BEATS model created successfully.

Generating synthetic data: batch_size=2, seq_len=24
Input (first batch element):
  [0.0000, 5.5000, 9.1603, ...]

--- Forecast (horizon=6) ---
Batch 0 forecast:
  [x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx]
Batch 1 forecast:
  [x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx]

--- Stack Decomposition ---
trend stack (batch 0):
  [x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx]
seasonality stack (batch 0):
  [x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx, x.xxxx]

=== Done ===
```

The actual forecast values depend on the randomly initialized model weights. With training on real data, N-BEATS learns to decompose the signal into meaningful trend and seasonality components.

## How N-BEATS works

N-BEATS uses a stack of blocks with double residual connections:

1. Each block applies FC layers to produce theta parameters
2. Theta is expanded through a **basis function**:
   - **Trend**: polynomial basis (captures linear/quadratic trends)
   - **Seasonality**: Fourier basis (captures periodic patterns)
   - **Generic**: learned linear projections
3. The backcast is subtracted from the input (residual connection)
4. The forecast is accumulated across all blocks

This architecture provides interpretable decomposition -- you can inspect what each stack contributes to the final forecast.

## Key APIs

| Type | Package | Purpose |
|------|---------|---------|
| `timeseries.NBEATSConfig` | `timeseries/` | Model configuration (input/output length, stacks, hidden dim) |
| `timeseries.NBEATS` | `timeseries/` | N-BEATS model with Forward method |
| `timeseries.NBEATSOutput` | `timeseries/` | Forecast + per-stack decomposition |
| `timeseries.StackType` | `timeseries/` | Stack types: StackTrend, StackSeasonality, StackGeneric |
| `timeseries.NBEATSAdapter` | `timeseries/` | Wraps NBEATS to satisfy `training.Model[float32]` for training |

## Also available

- **PatchTST**: Patch Time-Series Transformer for channel-independent forecasting. Use `timeseries.NewPatchTST(config, engine, ops)`.
- **TFT**: Temporal Fusion Transformer with attention-based interpretability.
