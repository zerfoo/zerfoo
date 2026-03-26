# Granite Time Series Support Plan

## Context

IBM Granite Time Series is a family of three ultra-lightweight foundation models
for time series tasks, each with fewer than 10M parameters:

| Model | Architecture | Params | Tasks | Key Innovation |
|-------|-------------|--------|-------|----------------|
| **TTM** (TinyTimeMixer) | TSMixer with adaptive patching | ~1M | Forecasting | Channel-independent + channel-mixing; zero-shot and few-shot; exogenous variable support |
| **FlowState** | SSM Encoder + Functional Basis Decoder | ~9M | Forecasting | Timescale-invariant coefficient space; continuous forecasting at any sampling rate |
| **TSPulse** | Dual-space masked reconstruction | ~1M | Anomaly detection, classification, imputation, similarity search | Joint time + frequency domain learning; dual-embedding disentanglement |

All three models are distributed as SafeTensors on HuggingFace under Apache-2.0.
None are available on Ollama -- Ollama only serves Granite LLMs (3.x, 4.0), not
the time series family. This is the key competitive opportunity: Zerfoo can be the
first Go-native runtime for Granite Time Series, offering `go build` simplicity
versus Python-only `granite-tsfm` today.

### Model Distribution

| HuggingFace Repo | Format | Variants |
|-----------------|--------|----------|
| `ibm-granite/granite-timeseries-ttm-r2` | SafeTensors (F32) | Context 512/1024/1536, multiple forecast horizons, frequency prefix tuning |
| `ibm-granite/granite-timeseries-flowstate-r1` | SafeTensors (F32) | Single model, configurable scale_factor per sampling rate |
| `ibm-granite/granite-timeseries-tspulse-r1` | SafeTensors (F32) | 3 variants: hybrid-allhead (anomaly), hybrid-dualhead (imputation/search), block-dualhead (classification) |

### Existing Zerfoo Infrastructure

Zerfoo already has time series support that this plan extends:

- **inference/timeseries/**: PatchTST, TFT, Regime Detection graph builders + GGUF loader
- **timeseries/**: Training backends for DLinear, N-HiTS, CfC, PatchTST, N-BEATS, FReTS, iTransformer, Mamba, TFT
- **layers/timeseries/**: Patch embedding, Variable Selection Network
- **model/gguf/**: GGUF parser with `ts.signal.*` metadata namespace
- **model/huggingface/**: HuggingFace client with download and caching
- **cmd/ts_train/**: CLI for PatchTST training from CSV features

### Goal

Outperform Ollama on Granite Time Series. Since Ollama does not support these
models at all, shipping any working implementation achieves this. The real bar is
outperforming IBM's Python reference (`granite-tsfm`) on inference latency while
matching accuracy, which is achievable given the models are tiny (<10M params)
and Go + CUDA graph capture eliminates Python/PyTorch overhead.

---

## Epics

| Epic | Description | Tasks |
|------|-------------|-------|
| GTS-E1 | SafeTensors-to-GGUF conversion | 3 tasks |
| GTS-E2 | TTM architecture builder + inference | 5 tasks |
| GTS-E3 | FlowState architecture builder + inference | 3 tasks |
| GTS-E4 | TSPulse architecture builder + inference | 3 tasks |
| GTS-E5 | API, CLI, and benchmarks | 4 tasks |

**Total: 18 tasks across 4 waves.**

---

## Wave 1: Foundation (3 agents, parallel)

### GTS-E1: SafeTensors-to-GGUF Conversion

Granite Time Series models ship as SafeTensors, not GGUF. Zerfoo's sole model
format is GGUF (ADR-037). We need a converter that reads SafeTensors weights and
writes GGUF files with the appropriate `ts.signal.*` metadata.

- [ ] GTS-T1.1 Add SafeTensors reader to zonnx
  Owner: ML Eng  Est: 6h
  Files: zonnx/safetensors/reader.go, zonnx/safetensors/reader_test.go
  Description: Implement a SafeTensors file reader that parses the JSON header
  and provides access to named tensors with their dtype and shape. SafeTensors
  uses a simple format: 8-byte little-endian header length, JSON header mapping
  tensor names to {dtype, shape, data_offsets}, followed by raw tensor data.
  Support F32, F16, BF16 dtypes.
  Acceptance:
  - Parse a SafeTensors file downloaded from ibm-granite/granite-timeseries-ttm-r2.
  - Round-trip test: write known tensors, read back, verify exact match.
  - go vet and go test pass.

- [ ] GTS-T1.2 Implement granite-ts-to-gguf converter command
  Owner: ML Eng  Est: 8h
  Files: zonnx/cmd/granite2gguf/main.go
  Description: CLI that reads a Granite Time Series SafeTensors model directory
  (model.safetensors + config.json) and writes a GGUF file. Map HuggingFace
  config.json fields to ts.signal.* GGUF metadata. Handle all three model
  families (TTM, FlowState, TSPulse) by inspecting model_type in config.json.
  Tensor name mapping: convert HuggingFace PyTorch key names to GGUF-compatible
  names (e.g., `backbone.mixer_layers.0.mlp.fc1.weight` -> `blk.0.mlp.fc1.weight`).
  Acceptance:
  - Convert ibm-granite/granite-timeseries-ttm-r2 (512-96 variant) to GGUF.
  - Convert ibm-granite/granite-timeseries-flowstate-r1 to GGUF.
  - Convert ibm-granite/granite-timeseries-tspulse-r1 (hybrid-allhead) to GGUF.
  - All GGUF files loadable by model/gguf.Parse with correct metadata.
  - go vet clean.

- [ ] GTS-T1.3 Extend ts.signal.* metadata namespace for Granite models
  Owner: ML Eng  Est: 3h
  Files: inference/timeseries/gguf_loader.go
  Deps: GTS-T1.2
  Description: Add new GGUF metadata keys required by Granite models that are
  not present in the existing TimeSeriesSignalConfig. New keys:
  - `ts.signal.model_type` (string): "ttm", "flowstate", or "tspulse"
  - `ts.signal.context_len` (uint32): context window (512, 1024, 1536)
  - `ts.signal.forecast_len` (uint32): prediction horizon
  - `ts.signal.num_mixer_layers` (uint32): TTM mixer depth
  - `ts.signal.channel_mixing` (bool): TTM decoder channel mixing
  - `ts.signal.scale_factor` (float32): FlowState temporal scale
  - `ts.signal.mask_type` (string): TSPulse "hybrid" or "block"
  - `ts.signal.head_type` (string): TSPulse "allhead" or "dualhead"
  Add a GraniteTimeSeriesConfig struct and LoadGraniteTimeSeriesConfig function
  that extends the existing pattern in gguf_loader.go.
  Acceptance:
  - LoadGraniteTimeSeriesConfig correctly parses metadata from converted GGUF files.
  - Existing LoadTimeSeriesSignalConfig unchanged (backward compatible).
  - go test ./inference/timeseries/ passes.

---

## Wave 2: Architecture Builders (3 agents, parallel)

### GTS-E2: TTM (TinyTimeMixer) Architecture

TTM uses TSMixer backbone: per-patch linear mixing in both time and feature
dimensions, with adaptive patching for variable context lengths. The decoder
is a smaller TSMixer that maps to forecast horizon. Channel-independent by
default, with optional channel-mixing during fine-tuning.

- [ ] GTS-T2.1 Implement TSMixer backbone layer
  Owner: ML Eng  Est: 6h
  Files: layers/timeseries/tsmixer.go, layers/timeseries/tsmixer_test.go
  Description: Implement the TSMixer block used by TTM. Each block contains:
  (1) time-mixing MLP (linear across the time/patch dimension),
  (2) feature-mixing MLP (linear across the feature dimension),
  (3) LayerNorm after each mixing step, (4) residual connections.
  Use compute.Engine[T] for all operations. Support both channel-independent
  mode (feature-mixing disabled) and channel-mixing mode.
  Acceptance:
  - Forward pass produces correct output shape: [batch, num_patches, d_model].
  - Channel-independent and channel-mixing modes both work.
  - go test with known input/output pairs passes.

- [ ] GTS-T2.2 Implement TTM architecture builder
  Owner: ML Eng  Est: 8h
  Files: inference/timeseries/arch_ttm.go, inference/timeseries/arch_ttm_test.go
  Deps: GTS-T2.1, GTS-T1.3
  Description: Build the full TTM computation graph from GGUF weights:
  1. Input normalization (standard scaling per channel).
  2. Adaptive patching: split input [batch, context_len, channels] into
     patches [batch, num_patches, patch_len * channels].
  3. Patch embedding via linear projection.
  4. N TSMixer backbone blocks (typically 2-4 layers).
  5. Decoder: smaller TSMixer blocks mapping to forecast patches.
  6. Forecast head: linear projection [batch, forecast_len, channels].
  Follow the BuildPatchTST pattern: config struct, validation, graph builder.
  Acceptance:
  - BuildTTM[float32] produces a valid computation graph.
  - Forward pass on random input produces [batch, forecast_len, channels] output.
  - Load weights from converted GGUF file and run inference.
  - go test passes; go vet clean.

- [ ] GTS-T2.3 Implement TTM training backend
  Owner: ML Eng  Est: 6h
  Files: timeseries/ttm.go, timeseries/ttm_engine.go, timeseries/ttm_test.go
  Deps: GTS-T2.1
  Description: Add TTM to the timeseries training package following the existing
  pattern (see patchtst.go, dlinear_engine.go). Implement:
  - TTM struct with TrainWindowed and PredictWindowed methods.
  - TTMEngine with forward/backward pass through TSMixer blocks.
  - Support channel-independent (zero-shot) and channel-mixing (fine-tune) modes.
  - MSE and MAE loss functions (reuse training/loss package).
  Acceptance:
  - TrainWindowed converges on synthetic sinusoidal data (loss decreases).
  - PredictWindowed produces forecasts with correct shape.
  - Fine-tuning from pre-trained weights reduces loss faster than random init.
  - go test -race passes.

- [ ] GTS-T2.4 TTM zero-shot and few-shot inference pipeline
  Owner: ML Eng  Est: 4h
  Files: inference/timeseries/ttm_infer.go, inference/timeseries/ttm_infer_test.go
  Deps: GTS-T2.2
  Description: End-to-end inference function that:
  1. Loads TTM GGUF model.
  2. Accepts raw time series input ([][]float64 or tensor).
  3. Applies instance normalization (standard scaling per channel).
  4. Runs forward pass through the computation graph.
  5. Denormalizes output.
  6. Returns forecast [][]float64.
  Support automatic model variant selection based on context_length and
  prediction_length (mirror granite-tsfm's get_model logic).
  Acceptance:
  - Zero-shot inference on ETTh1 dataset produces reasonable forecasts.
  - Output matches Python granite-tsfm reference within 1e-4 tolerance.
  - Handles multivariate input correctly.

- [ ] GTS-T2.5 TTM exogenous variable support
  Owner: ML Eng  Est: 4h
  Files: inference/timeseries/arch_ttm.go (extend), layers/timeseries/tsmixer.go (extend)
  Deps: GTS-T2.2
  Description: TTM supports exogenous (control) variables and static categorical
  features. Extend the TTM builder to accept:
  - Future exogenous inputs [batch, forecast_len, num_exog] that are concatenated
    with decoder input.
  - Static categorical features [batch, num_static] that are projected and added
    as bias to each mixer block.
  This mirrors TTM's `num_exogenous_channels` and `static_categorical_columns`.
  Acceptance:
  - Inference with exogenous variables produces different forecasts than without.
  - Static categorical features correctly modify mixer block outputs.
  - go test passes.

### GTS-E3: FlowState Architecture

FlowState uses an SSM (State Space Model) encoder with a Functional Basis Decoder.
The SSM encoder processes the input time series, and the FBD maps to a
timescale-invariant coefficient space for continuous forecasting.

- [ ] GTS-T3.1 Implement SSM (State Space Model) layer
  Owner: ML Eng  Est: 8h
  Files: layers/timeseries/ssm.go, layers/timeseries/ssm_test.go
  Description: Implement the S4/S5-style State Space Model layer used by
  FlowState. The SSM layer maps input sequences through a linear state-space
  system: x'(t) = Ax(t) + Bu(t), y(t) = Cx(t) + Du(t). Implement the
  discretized version using ZOH (zero-order hold) for efficient computation.
  Support parallel scan for GPU acceleration.
  Acceptance:
  - Forward pass on [batch, seq_len, d_model] input produces correct output shape.
  - Matches reference SSM implementation output within 1e-5 tolerance.
  - go test passes with known state-space system test vectors.

- [ ] GTS-T3.2 Implement FlowState architecture builder
  Owner: ML Eng  Est: 8h
  Files: inference/timeseries/arch_flowstate.go, inference/timeseries/arch_flowstate_test.go
  Deps: GTS-T3.1, GTS-T1.3
  Description: Build the full FlowState computation graph from GGUF weights:
  1. Input patching and embedding.
  2. SSM encoder blocks (stacked S4 layers with residual connections).
  3. Functional Basis Decoder: projects SSM hidden states to basis function
     coefficients, then evaluates basis functions at target time points.
     The FBD uses Fourier or polynomial basis functions to produce continuous
     forecasts at arbitrary time resolutions.
  4. Scale factor handling: apply `ts.signal.scale_factor` to map between
     input sampling rate and model's base seasonality (24 steps).
  Acceptance:
  - BuildFlowState[float32] produces a valid computation graph.
  - Forward pass with scale_factor=1.0 (hourly) produces [batch, forecast_len, channels].
  - Load weights from converted GGUF file and run inference.
  - go test passes; go vet clean.

- [ ] GTS-T3.3 FlowState continuous forecasting with timescale adaptation
  Owner: ML Eng  Est: 4h
  Files: inference/timeseries/flowstate_infer.go, inference/timeseries/flowstate_infer_test.go
  Deps: GTS-T3.2
  Description: End-to-end FlowState inference with timescale adaptation:
  1. Accept raw time series + sampling rate metadata.
  2. Compute scale_factor from sampling rate (see FlowState docs table:
     15min->0.25, hourly->1.0, daily->3.43, etc.).
  3. Run forward pass through computation graph.
  4. Support variable prediction_length (FlowState can forecast up to
     ~30 seasons, e.g., 2880 steps for quarter-hourly data).
  Acceptance:
  - Inference at different sampling rates (15min, hourly, daily) produces
    appropriately scaled forecasts.
  - Output matches Python reference within 1e-4 tolerance.
  - Forecasts degrade gracefully beyond recommended horizon.

### GTS-E4: TSPulse Architecture

TSPulse uses dual-space masked reconstruction in time and frequency domains,
producing both fine-grained and semantic embeddings.

- [ ] GTS-T4.1 Implement dual-space encoder (time + frequency domain)
  Owner: ML Eng  Est: 8h
  Files: layers/timeseries/dual_space.go, layers/timeseries/dual_space_test.go
  Description: Implement TSPulse's dual-space encoder:
  1. Time-domain path: standard patch embedding + transformer encoder.
  2. Frequency-domain path: FFT of input patches, process in frequency space
     with linear layers, IFFT back.
  3. Dual-embedding disentanglement: produce fine-grained embeddings (for
     anomaly detection/imputation) and semantic embeddings (for classification/
     similarity search) from the fused representation.
  Use compute.Engine[T] for FFT operations (or implement real-valued FFT via
  the existing numeric package).
  Acceptance:
  - Forward pass produces both fine-grained and semantic embeddings.
  - Frequency-domain path correctly transforms and reconstructs.
  - go test with synthetic data passes.

- [ ] GTS-T4.2 Implement TSPulse architecture builder with task heads
  Owner: ML Eng  Est: 8h
  Files: inference/timeseries/arch_tspulse.go, inference/timeseries/arch_tspulse_test.go
  Deps: GTS-T4.1, GTS-T1.3
  Description: Build the full TSPulse computation graph with multiple task heads:
  1. Dual-space encoder (GTS-T4.1).
  2. Reconstruction head (for anomaly detection and imputation): decode
     fine-grained embeddings back to time series, compute reconstruction error.
  3. Classification head: MLP on semantic embeddings -> class logits.
  4. Similarity head: extract semantic embeddings for cosine similarity.
  Support all three TSPulse variants via mask_type and head_type config.
  Multi-head triangulation for anomaly detection: fuse multiple prediction
  streams for robust anomaly scoring.
  Acceptance:
  - BuildTSPulse[float32] produces valid computation graphs for all 3 variants.
  - Anomaly detection: reconstruction error on normal data < anomalous data.
  - Classification: forward pass produces [batch, num_classes] logits.
  - Load weights from converted GGUF files.
  - go test passes; go vet clean.

- [ ] GTS-T4.3 TSPulse multi-task inference pipeline
  Owner: ML Eng  Est: 6h
  Files: inference/timeseries/tspulse_infer.go, inference/timeseries/tspulse_infer_test.go
  Deps: GTS-T4.2
  Description: Unified inference entry point for all TSPulse tasks:
  - AnomalyDetect(series) -> anomaly scores per timestep.
  - Classify(series) -> class probabilities.
  - Impute(series, mask) -> reconstructed series with missing values filled.
  - Embed(series) -> semantic embedding vector for similarity search.
  Handle input length requirements (anomaly needs 3-4x context, classification
  auto-resamples to 512).
  Acceptance:
  - Anomaly detection on synthetic data with injected anomalies achieves >0.8 AUC.
  - Classification on UCR/UEA test split matches Python reference accuracy.
  - Imputation on masked data produces reasonable reconstructions.
  - Similarity search: embedding cosine similarity higher for same-class pairs.

---

## Wave 3: Integration (2 agents, parallel)

### GTS-E5: API, CLI, and Benchmarks

- [ ] GTS-T5.1 CLI commands for Granite Time Series
  Owner: ML Eng  Est: 6h
  Files: cmd/zerfoo/main.go (extend), cmd/zerfoo/granite_ts.go
  Deps: GTS-T2.4, GTS-T3.3, GTS-T4.3
  Description: Add CLI subcommands for Granite Time Series inference:
  - `zerfoo predict --model granite-ttm --data input.csv --horizon 96`
  - `zerfoo predict --model granite-flowstate --data input.csv --horizon 960 --freq hourly`
  - `zerfoo anomaly --model granite-tspulse --data input.csv`
  - `zerfoo classify --model granite-tspulse --data input.csv`
  Input: CSV with timestamp column + value columns. Output: CSV or JSON.
  Auto-download model from HuggingFace if not cached (reuse model/huggingface client).
  Acceptance:
  - All four commands work end-to-end on sample data.
  - `--help` output is clear and complete.
  - Auto-download + convert + cache works on first run.
  - go vet clean.

- [ ] GTS-T5.2 REST API endpoints for time series tasks
  Owner: ML Eng  Est: 6h
  Files: serve/timeseries.go, serve/timeseries_test.go
  Deps: GTS-T2.4, GTS-T3.3, GTS-T4.3
  Description: Add REST API endpoints to the serve package:
  - `POST /api/v1/forecast` -- body: `{model, data[][], horizon, freq?}`
  - `POST /api/v1/anomaly` -- body: `{model, data[][]}`
  - `POST /api/v1/classify` -- body: `{model, data[][]}`
  - `POST /api/v1/embed` -- body: `{model, data[][]}`
  Support batch inference (multiple series in one request). JSON response with
  results + metadata (model, latency_ms, etc.). Wire into existing serve.Server
  router.
  Acceptance:
  - All endpoints return correct JSON responses.
  - Batch of 10 series processed in single request.
  - Error handling for invalid input (wrong shape, missing fields).
  - go test with httptest passes.

- [ ] GTS-T5.3 Accuracy parity tests against Python granite-tsfm
  Owner: ML Eng  Est: 8h
  Files: tests/parity/granite_ts_test.go
  Deps: GTS-T2.4, GTS-T3.3, GTS-T4.3
  Description: Parity tests that verify Zerfoo's Granite TS output matches
  Python granite-tsfm reference. For each model family:
  1. Run Python reference on a fixed input, save output as golden file.
  2. Run Zerfoo on same input, compare within tolerance (1e-4 for F32).
  Test on ETTh1 (forecasting), synthetic anomalies (detection), UCR subset
  (classification). Golden files checked into tests/parity/testdata/.
  Acceptance:
  - TTM forecast MAE within 1e-4 of Python reference on 3 test cases.
  - FlowState forecast within 1e-4 on 3 test cases across sampling rates.
  - TSPulse anomaly scores within 1e-4 on 2 test cases.
  - TSPulse classification logits within 1e-4 on 2 test cases.

---

## Wave 4: Performance (2 agents, parallel)

- [ ] GTS-T5.4 Benchmark suite and performance optimization
  Owner: Kernel Eng  Est: 8h
  Files: tests/benchmark/granite_ts_bench_test.go
  Deps: GTS-T5.3
  Description: Comprehensive benchmarks comparing Zerfoo vs Python granite-tsfm:
  - **Latency**: Single-series inference time (ms) for each model family.
  - **Throughput**: Series/second for batch sizes 1, 8, 32, 128.
  - **Accuracy**: MAE/MSE on ETTh1, Weather, Electricity standard benchmarks.
  - **Memory**: Peak RSS during inference.
  Target: 5-10x latency improvement over Python (these are tiny models where
  Python/PyTorch overhead dominates). Optimize hot paths:
  - Ensure CUDA graph capture works for all three architectures.
  - Profile and fuse sequential operations where possible.
  - Verify quantization to F16 does not degrade accuracy (models are only 1-9M params,
    F32 is fine, but F16 halves memory bandwidth).
  Acceptance:
  - Benchmark results published in tests/benchmark/results/.
  - Zerfoo TTM inference latency < 2ms per series on GPU (batch=1).
  - Zerfoo FlowState inference latency < 5ms per series on GPU (batch=1).
  - Throughput > 10,000 series/sec on GPU (batch=128) for TTM.
  - All accuracy metrics within 1% of Python reference.

---

## Wave Schedule

| Wave | Tasks | Agents | Deps | Est Duration |
|------|-------|--------|------|-------------|
| W1 | GTS-T1.1, GTS-T1.2, GTS-T1.3 | 3 | None | 1 day |
| W2 | GTS-T2.1 through GTS-T2.5, GTS-T3.1 through GTS-T3.3, GTS-T4.1 through GTS-T4.3 | 3 | W1 | 3 days |
| W3 | GTS-T5.1, GTS-T5.2, GTS-T5.3 | 2 | W2 | 2 days |
| W4 | GTS-T5.4 | 2 | W3 | 1 day |

**Total estimated: 7 working days with 3 parallel agents.**

Within each wave, tasks across different epics run in parallel. Within GTS-E2,
GTS-T2.1 must complete before GTS-T2.2; GTS-T2.3 can run in parallel with
GTS-T2.2 (both depend only on GTS-T2.1). Same pattern for GTS-E3 and GTS-E4.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SafeTensors tensor name mapping is non-trivial | Medium | Medium | Inspect actual model files; use granite-tsfm source as reference for layer names |
| FlowState SSM layer requires complex discretization | Medium | High | Start with simple Euler discretization; upgrade to ZOH if accuracy requires it |
| TSPulse FFT in Go may be slow on CPU | Low | Medium | Use CUDA cuFFT via existing GPU bindings; CPU path uses gonum/dsp or hand-rolled |
| Model accuracy does not match Python within tolerance | Medium | Medium | Debug layer by layer; dump intermediate activations from both runtimes |
| HuggingFace model format changes between releases | Low | Low | Pin to specific model revisions in converter |

---

## References

- [TTM Paper (NeurIPS 2024)](https://arxiv.org/pdf/2401.03955)
- [TSPulse Paper](https://arxiv.org/pdf/2505.13033)
- [HuggingFace: granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
- [HuggingFace: granite-timeseries-flowstate-r1](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1)
- [HuggingFace: granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
- [IBM Granite Time Series Docs](https://www.ibm.com/granite/docs/models/time-series)
- [granite-tsfm GitHub](https://github.com/ibm-granite/granite-tsfm)
