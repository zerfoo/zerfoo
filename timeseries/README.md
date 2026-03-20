# timeseries — Time-Series Forecasting Models

This package provides three time-series forecasting architectures built on ztensor:

| Model | Paper | Use Case |
|-------|-------|----------|
| **N-BEATS** | Neural Basis Expansion Analysis | Univariate forecasting with interpretable trend/seasonality decomposition |
| **PatchTST** | Patch Time-Series Transformer | Univariate/multivariate forecasting via patch-based attention |
| **TFT** | Temporal Fusion Transformer | Multi-horizon probabilistic forecasting with static/temporal covariates |

## Training Patterns

### Standalone Training (Recommended)

The timeseries models return rich, domain-specific outputs that are essential for
interpretability — N-BEATS produces per-stack decompositions (trend vs. seasonality),
and TFT provides quantile forecasts with variable importance weights. These outputs
do not fit naturally into the single-tensor `training.Model[T]` interface.

For full control over the training loop and access to all model outputs, use the
models directly:

```go
model, _ := timeseries.NewNBEATS(config, engine, ops)

for epoch := range epochs {
    out, _ := model.Forward(ctx, batchInput)

    // Access decomposition for interpretability loss terms.
    trendForecast := out.StackForecasts[0]
    seasonForecast := out.StackForecasts[1]

    // Compute loss on out.Forecast, backpropagate manually.
}
```

### Trainer Integration (via Adapter)

When `training.Trainer[T]` integration is needed (e.g., to reuse optimizer and
scheduling infrastructure), use the adapter wrappers:

```go
model, _ := timeseries.NewNBEATS(config, engine, ops)
adapter, _ := timeseries.NewNBEATSAdapter(model)

// adapter satisfies training.Model[float32]
trainer := training.NewTrainer(adapter, optimizer, lossFn)
trainer.Train(data)
```

Available adapters:

| Adapter | Wraps | Forward Input | Forward Output |
|---------|-------|--------------|----------------|
| `NBEATSAdapter` | `*NBEATS` | `[batch, inputLen]` | `[batch, outputLen]` (forecast only) |
| `PatchTSTAdapter` | `*PatchTST` | `[batch, inputLen]` or `[batch, channels, inputLen]` | `[batch, outputDim]` or `[batch, channels, outputDim]` |
| `TFTAdapter` | `*TFT` | Two tensors: static `[batch, features]`, time `[batch, seqLen, features]` | `[batch, nHorizons * nQuantiles]` |

**Trade-offs of the adapter approach:**

- The adapters extract only the primary forecast tensor, discarding decomposition
  and interpretability outputs. Use standalone loops when these are needed.
- `Backward()` is not implemented on the adapters. The training infrastructure
  should use engine-level autograd or numerical gradients.
- All adapters expose `Parameters()` for optimizer integration (weight updates,
  learning rate scheduling, gradient clipping).

### When to Use Which

| Scenario | Approach |
|----------|----------|
| Custom loss with decomposition terms | Standalone |
| Quantile loss with variable importance | Standalone |
| Standard MSE/MAE training with existing Trainer | Adapter |
| Hyperparameter search with Trainer infrastructure | Adapter |
