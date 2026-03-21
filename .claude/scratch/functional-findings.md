# Functional Correctness Review -- zerfoo Core Framework

**Reviewer**: Principal Engineer
**Date**: 2026-03-21
**Scope**: 5 critical feature traces, testing assessment, code quality
**Test Results**: 1 test failure (`meta/TestMAML_MetaConvergence`), all other packages pass

---

## Feature Trace 1: Model Inference

**Code path**: `cmd/cli/run.go:41` -> `inference.Load()` (`inference/inference.go:184`) -> `LoadFile()` (`inference/load_gguf.go:15`) -> `LoadGGUF()` -> `buildArchGraph()` (`inference/load_gguf.go:141`) -> architecture-specific builder (e.g., `buildLlamaGraph` at `inference/arch_llama.go:36`) -> `generate.Generator.Generate()` / `GenerateStream()` (`generate/generator.go:294` / `generate/stream.go:35`)

### F1-1: Embedding lookup has no bounds check on token IDs (HIGH)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/inference/arch_llama.go:288-295`

The `embeddingLookupNode.Forward()` CPU path indexes into `embData` using `id*hiddenDim` where `id = int(ids[i])`. There is no check that `id >= 0` or `id < vocabSize`. A malformed or out-of-range token ID will cause an index-out-of-bounds panic or silent memory corruption.

The same issue exists in the Q8 dequantize path at line 282-283.

The GPU path (line 247-249) similarly converts to `int(ids[i])` without bounds checking.

**Impact**: Panic on invalid token ID. If the tokenizer produces an ID >= vocabSize (possible with mismatched tokenizer/model), this will crash the server.

**Recommendation**: Add `if id < 0 || id >= vocabSize { return nil, fmt.Errorf(...) }` before the lookup.

### F1-2: RegisterAlias is not thread-safe (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/inference/inference.go:72,84,92`

`modelAliases` is a package-level `map[string]string`. `RegisterAlias()` writes to it, and `ResolveAlias()` reads from it. Both are called without any synchronization. Concurrent calls from multiple goroutines (e.g., an HTTP server registering aliases while serving requests) will cause a data race on the map.

**Recommendation**: Protect with `sync.RWMutex` or use `sync.Map`.

---

## Feature Trace 2: API Serving

**Code path**: `cmd/cli/serve.go:45` -> `serve.NewServer()` (`serve/server.go:89`) -> route registration -> `handleChatCompletions` / `handleCompletions` -> `model.Chat()` / `model.Generate()` -> streaming via `streamChatCompletion()`

### F2-1: `s.unloaded` field is racy (HIGH)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/serve/server.go:31,606,621,638,650`

`Server.unloaded` is a plain `bool` field read by `handleModels` (line 606), `handleModelInfo` (line 621), `handleChatCompletions` (implicit via model access), and written by `handleModelDelete` (line 650). HTTP handlers run concurrently. This is a data race.

**Recommendation**: Use `atomic.Bool` or protect with a `sync.RWMutex`.

### F2-2: Streaming writes error after headers already sent (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/serve/server.go:757,761`

In `streamChatCompletion()`, `w.WriteHeader(http.StatusOK)` is called at line 757. If `w.(http.Flusher)` type assertion fails at line 759, `writeError()` at line 761 attempts to call `w.WriteHeader(http.StatusInternalServerError)`, which is a no-op since headers are already sent. The client receives a 200 status with an error JSON body, which violates the HTTP contract.

Same issue in `streamCompletion()` at lines 797-803.

**Recommendation**: Check for `http.Flusher` support *before* calling `WriteHeader`.

### F2-3: Streaming chat loses message structure (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/serve/server.go:766-769`

`streamChatCompletion()` concatenates all message contents with spaces (`prompt.WriteString(m.Content)`) instead of using the model's chat template via `model.Chat()`. This means system prompts, user/assistant role boundaries, and special tokens are lost during streaming. The non-streaming path correctly uses `s.model.Chat()` (line 453).

The same issue exists in the batched path at lines 441-447.

**Recommendation**: Use `model.formatMessages(messages)` (or equivalent exported function) to format the prompt before streaming.

### F2-4: Model delete does not prevent concurrent inference (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/serve/server.go:649-650`

`handleModelDelete` calls `s.model.Close()` and sets `s.unloaded = true`. But concurrent in-flight requests to `handleChatCompletions` or `handleCompletions` may still be using the model. There is no gate or wait mechanism to drain in-flight requests before closing the model.

**Recommendation**: Add a request counter (e.g., `sync.WaitGroup`) and drain before closing.

---

## Feature Trace 3: Model Pulling

**Code path**: `cmd/cli/pull.go:34` -> `registry.LocalRegistry.Pull()` (`registry/registry.go:78`) -> `pullFunc` -> `pullFromHF()` (`registry/pull.go:71`) -> `downloadFile()` (line 176)

### F3-1: No checksum verification on downloaded GGUF files (HIGH)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/registry/pull.go:71-113`

`pullFromHF()` downloads a GGUF file and returns immediately. There is no SHA-256 or other integrity check on the downloaded content. A corrupted download (network interruption, CDN error) will silently produce a broken model file that may crash at load time or produce garbage output.

The OCI registry path (`registry/oci.go`) does have SHA-256 verification, but the HuggingFace pull path does not.

**Recommendation**: Compute SHA-256 during download and verify against the HuggingFace API-provided hash (available in the `siblings` response or via the `/blob/` endpoint).

### F3-2: No atomic write for downloaded files (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/registry/pull.go:210-229`

`downloadFile()` writes directly to the target path via `os.Create()`. If the download is interrupted (context cancelled, network error mid-stream), a partial file is left on disk. The next `Get()` call will find the directory with a corrupt file and return it as cached.

**Recommendation**: Write to a temporary file, then `os.Rename()` atomically on success. Remove the temp file on failure.

### F3-3: Path traversal defense is incomplete for multi-segment model IDs (LOW)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/registry/registry.go:169-184`

`modelDir()` splits on `/` and takes only 2 parts, but the containment check at line 180 verifies the resolved path stays within `cacheDir`. This is correct, but the `SplitN(modelID, "/", 2)` means a model ID like `org/sub/model` would create `cacheDir/org/sub/model` (via `parts[1]="sub/model"`), which may be unexpected. Not a security issue since the containment check is valid, but it is a usability concern.

---

## Feature Trace 4: Training / Fine-Tuning

**Code path**: `cmd/cli/train.go:52` -> `trainLoop()` (line 300) -> `fsdp.NewShardedModule()` -> manual forward/backward loop -> `fsdp.SaveCheckpoint()`

### F4-1: Training CLI uses synthetic model, not actual GGUF model (HIGH -- functional gap)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/cmd/cli/train.go:240-263`

The `TrainCommand.trainLoop()` creates a synthetic `trainModel` with a single 64-element parameter (line 301-303). The `--config` flag specifying a GGUF model path is parsed but never actually loaded. The `Forward()` method (line 258-263) simply returns the input unchanged. This means the `train` CLI command does not actually fine-tune the specified model -- it runs a synthetic loop for demonstration purposes.

**Impact**: Users attempting to fine-tune a model via `zerfoo train --config model.gguf --data train.jsonl` will get a successful exit but no actual training occurs on their model.

**Recommendation**: Document this limitation clearly or implement actual GGUF model loading + LoRA fine-tuning in the train command.

### F4-2: AdamW optimizer has no gradient clipping or NaN detection (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/training/optimizer/adamw.go:43-139`

The `AdamW.Step()` method does not check for NaN/Inf gradients before updating parameters, nor does it implement gradient clipping. The FP8 loss scaler (`training/fp8/loss_scaler.go`) has NaN detection, but the standard AdamW path does not.

**Recommendation**: Add optional gradient norm clipping and NaN/Inf detection to AdamW.Step(). The FP8 path already has `CheckGradients()` -- reuse that pattern.

---

## Feature Trace 5: Time Series Inference

**Code path**: `inference/timeseries/gguf_loader.go` -> `LoadPatchTSTFromGGUF()` -> builder function -> `inference/timeseries/arch_patchtst.go:BuildPatchTST()` -> graph execution

### F5-1: PatchTST projection head has incorrect shape semantics (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/inference/timeseries/arch_patchtst.go:303-357`

The projection path is convoluted and mathematically questionable:
1. `projected` starts as `[batch * numVars, d_model] @ [d_model, horizon * numVars]` = `[batch * numVars, horizon * numVars]`
2. Reshaped to `[batch, numVars * horizon * numVars]`
3. Reshaped to `[batch, numVars, horizon, numVars]`
4. `ReduceMean` over axis 1 -> `[batch, horizon, numVars]`

The `ReduceMean` over the `numVars` axis averages unrelated variable projections together. The comments in the code (lines 321-332) acknowledge the design confusion: "Let's reconsider the projection design" and "This is a learned mapping, so any shape is fine." This averaging produces mathematically dubious output since each variable's projection bleeds into every other variable's output.

**Impact**: Training may converge to something, but the output quality is degraded by the improper averaging. The model will produce biased predictions for multivariate time series.

**Recommendation**: Use a proper channel-independent projection: `[batch * numVars, d_model] @ [d_model, horizon]` = `[batch * numVars, horizon]`, then reshape to `[batch, numVars, horizon]` and transpose to `[batch, horizon, numVars]`.

### F5-2: Timeseries normalization does not handle NaN input values (LOW)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/timeseries/dlinear.go:60-107`

`normalizeWindows()` computes z-score normalization but does not filter NaN values from the input. If any input value is NaN, the mean and std computations will propagate NaN to all outputs for that channel. The `isFinite()` helper exists (line 52-55) but is not used in `normalizeWindows()`.

**Recommendation**: Filter NaN values when computing mean/std, or return an error if NaN is present in input.

---

## Test Failure

### TF-1: TestMAML_MetaConvergence is flaky (MEDIUM)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/meta/meta_test.go:186`

The test expects meta-loss to decrease by at least 10% after 150 meta-epochs, but the loss actually increased by 3.5% (before=2.337, after=2.420). This is a non-deterministic test that depends on random initialization. With only 150 epochs and random task sampling, convergence is not guaranteed.

**Recommendation**: Either increase epochs, set a fixed random seed for determinism, or use a tolerance that accounts for stochasticity. The underlying MAML implementation may also have a learning rate or gradient computation issue causing divergence.

---

## Testing Assessment

### Packages without tests (critical code):
- `layers/` (root package) -- no test files
- `layers/vision/` -- no test files
- `cmd/bench_tps/`, `cmd/debug-infer/`, `cmd/zerfoo/` -- no test files for CLI entrypoints

### Test coverage gaps:
1. **Inference pipeline end-to-end**: No integration test that loads a real GGUF model and generates output. Parity tests exist but require external model files.
2. **Streaming correctness**: `serve/server.go` streaming handlers are tested for HTTP mechanics but the chat template bypass (F2-3) is not caught by tests.
3. **Download integrity**: `registry/pull.go` tests use mock HTTP servers but do not test partial download recovery.
4. **Model delete during inference**: No concurrent test for `handleModelDelete` + `handleChatCompletions`.

---

## Code Quality Findings

### CQ-1: TODO comments indicate known quality tradeoffs (LOW)

**Files**:
- `/Users/dndungu/Code/zerfoo/zerfoo/model/gguf/loader.go:158` -- Q4_K GEMV optimization TODO
- `/Users/dndungu/Code/zerfoo/zerfoo/model/gguf/loader.go:241` -- Q5_0 native GEMV TODO
- `/Users/dndungu/Code/zerfoo/zerfoo/cmd/cli/finetune_sentiment.go:94` -- GGUF model loading TODO

The Q4_K and Q5_K re-quantization to Q4_0 (lines 155-163, 170-177) trades approximately 1 bit of precision for throughput. This is documented but means Q4_K_M models lose their per-sub-block scale precision advantage.

### CQ-2: Debug logging behind environment variable checks (LOW)

**File**: `/Users/dndungu/Code/zerfoo/zerfoo/generate/generator.go:342,375,436`

Multiple `os.Getenv("ZERFOO_DEBUG_ONNX")` calls inside hot loops (per-token decode). While only active when the env var is set, the string comparison happens on every token. Use a cached bool.

---

## Summary of Findings by Impact

| # | Impact | Finding | Location |
|---|--------|---------|----------|
| F1-1 | HIGH | Embedding lookup no bounds check on token IDs | inference/arch_llama.go:288 |
| F2-1 | HIGH | Server.unloaded is racy (concurrent read/write) | serve/server.go:31 |
| F3-1 | HIGH | No checksum verification on HuggingFace downloads | registry/pull.go:71 |
| F4-1 | HIGH | Train CLI uses synthetic model, ignores --config GGUF | cmd/cli/train.go:300 |
| F1-2 | MEDIUM | RegisterAlias concurrent map write race | inference/inference.go:92 |
| F2-2 | MEDIUM | Streaming writeError after headers already sent | serve/server.go:757-761 |
| F2-3 | MEDIUM | Streaming chat loses message structure/template | serve/server.go:766 |
| F2-4 | MEDIUM | Model delete does not drain in-flight requests | serve/server.go:649 |
| F3-2 | MEDIUM | No atomic write for downloaded files (partial file risk) | registry/pull.go:210 |
| F4-2 | MEDIUM | AdamW has no gradient clipping or NaN guard | training/optimizer/adamw.go:43 |
| F5-1 | MEDIUM | PatchTST projection averages unrelated variables | inference/timeseries/arch_patchtst.go:303 |
| TF-1 | MEDIUM | TestMAML_MetaConvergence is flaky/failing | meta/meta_test.go:186 |
| F3-3 | LOW | Multi-segment model ID handling | registry/registry.go:169 |
| F5-2 | LOW | normalizeWindows does not handle NaN input | timeseries/dlinear.go:60 |
| CQ-1 | LOW | Q4_K/Q5_K precision loss from re-quantization | model/gguf/loader.go:155 |
| CQ-2 | LOW | os.Getenv in hot decode loop | generate/generator.go:342 |
