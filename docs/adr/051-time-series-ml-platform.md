# ADR 051: Wolf Time-Series ML Platform

## Status
Accepted

## Date
2026-03-17

## Context
Wolf's autonomous trading system requires ML models that operate on financial
time-series data: price sequences, volume, order book depth, macro indicators,
and sentiment signals. Standard LLM architectures (Decoder-only Transformer)
are not optimized for multivariate time-series with strict causal structure and
non-stationary distributions.

Research (2025-2026) shows that domain-specific time-series models outperform
general foundation models (Chronos, TimeGPT) on highly specialized financial
tasks. Three architectures show promise: PatchTST (patch-based attention),
Temporal Fusion Transformer (TFT, interpretable multi-horizon), and specialized
regime detection networks.

## Decision
Build a Wolf-specific ML platform in a new package: inference/wolf/ (within zerfoo
repo, Wolf integration layer).

Three Model Families:

1. Signal Generator (inference/wolf/signal_model.go):
   - Architecture: PatchTST -- splits time-series into non-overlapping patches,
     treats patches as tokens for a standard Transformer encoder
   - Input: L lookback timesteps across D features (price, volume, technical indicators)
   - Output: H-step ahead return distribution (mean + variance)
   - GGUF format: standard weight tensors + wolf.signal.patch_len, wolf.signal.stride
     metadata fields
   - Training: MSE + NLL loss on held-out returns; 5-fold time-series CV

2. Regime Detector (inference/wolf/regime_model.go):
   - Architecture: Temporal Fusion Transformer -- combines LSTM encoder, variable
     selection networks, attention, and quantile outputs
   - Input: multivariate time-series + static covariates (asset class, sector)
   - Output: regime probability distribution (bull/bear/sideways/volatile)
   - Regime labels: hidden Markov model fitted offline, not human-labeled

3. Risk Estimator (inference/wolf/risk_model.go):
   - Architecture: lightweight MLP over rolling statistics + VaR estimates
   - Input: current regime, position size, volatility regime, correlation matrix
   - Output: position size recommendation + stop-loss level

Feature Store (inference/wolf/features/):
- Offline store: parquet files with precomputed features per asset per timestamp
- Online store: in-memory ring buffer of last 500 timesteps per asset, updated
  by Wolf's market data feed
- Feature registry: YAML schema defining feature names, dtypes, normalization

## Consequences
Positive:
- PatchTST and TFT are proven on financial benchmarks; implementation is well-understood
- GGUF format reuse means all existing inference infrastructure works unchanged
- Wolf feature store decouples data pipeline from model training

Negative:
- Financial time-series are non-stationary; model accuracy degrades as market regime
  changes -- online learning (ADR-052) is required to maintain signal quality
- TFT is complex to implement correctly (variable selection, attention, quantile loss)
- Data leakage in financial ML is a critical risk; feature store must enforce strict
  point-in-time correctness (no future data in any feature)
- Research shows general foundation models are insufficient; Wolf needs its own
  labeled dataset curation pipeline (out of scope for Zerfoo; Wolf team responsibility)
