# E106 Security & Correctness Audit

**Date:** 2026-03-23
**Scope:** inference/, layers/, training/, model/ packages
**Reviewer:** Claude Opus 4.6

---

## E106 Fix Verification

### 1. inference/arch_llama.go -- Embedding lookup bounds check

**Status: PASS**

All three code paths in `embeddingLookupNode.Forward()` have bounds checks:

- **GPU path** (line 252): `if id < 0 || id >= vocabSize` -- returns error
- **Q8 path** (line 289): `if id < 0 || id >= vocabSize` -- returns error
- **float32 path** (line 300): `if id < 0 || id >= vocabSize` -- returns error

All return `fmt.Errorf("token ID %d out of range [0, %d)", id, vocabSize)`.

### 2. inference/inference.go -- RegisterAlias sync.RWMutex + GenerateBatch semaphore

**Status: PASS**

- `modelAliasesMu sync.RWMutex` protects `modelAliases` map (line 82).
- `ResolveAlias()` uses `RLock/RUnlock` (lines 96-98).
- `RegisterAlias()` uses `Lock/Unlock` (lines 107-109).
- `GenerateBatch()` uses a buffered channel semaphore `sem := make(chan struct{}, maxConc)` (line 461) with acquire on line 467 and deferred release on line 468. Default cap is 8 (`defaultMaxBatchConcurrency`).

### 3. layers/core/*.go -- panic() calls removed

**Status: PASS**

Zero panics in non-test production code under `layers/core/`. The only `panic()` is in `layers/core/moe_test.go:689` (test code, acceptable).

### 4. layers/attention/attention_head.go -- panics replaced with error returns

**Status: PASS**

`NewAttentionHead` returns `(*AttentionHead[T], error)` (line 50). Validates `inputDim > 0` (line 57-59) and `headDim > 0` (line 61-63), returning errors instead of panicking. All `Forward()` and `Backward()` methods return errors on invalid input.

### 5. layers/ssm/s4.go -- nil gradient check in Backward

**Status: PASS**

Lines 401-422: Before accumulating gradients, the code checks `if pair.param.Gradient == nil` for all four parameters (aLog, b, c, d) and initializes them with zero tensors if nil.

### 6. training/lora/linear.go -- nil gradient check in Backward

**Status: PASS**

- `B.Gradient` nil check at line 200: `if l.B.Gradient == nil` -- initializes from dB.
- `A.Gradient` nil check at line 224: `if l.A.Gradient == nil` -- initializes from dA.
- Both branches accumulate via `engine.Add()` if non-nil.

### 7. training/optimizer/adamw.go -- gradient clipping + NaN guard

**Status: PASS**

- `guardAndClipGradients()` (lines 236-283) runs before every `Step()` call (line 56).
- NaN check at line 249: returns error with parameter name and index.
- Inf check at line 253: returns error with parameter name and index.
- Global gradient norm clipping at lines 261-279: clips when `globalNorm > a.maxGradNorm`.
- `SetMaxGradNorm()` method at line 49.

### 8. model/gguf/loader.go -- integer overflow checks

**Status: PASS**

Lines 25-33 in `LoadTensors()`:
- Each dimension checked against `math.MaxInt32` (line 27).
- Running product `numElements` checked against `1<<34` (~17 billion, line 31).
- Both return descriptive errors with tensor name.

### 9. model/gguf/parser.go -- tensor/metadata count limits

**Status: PASS**

- Tensor count limit: `tensorCount > 100_000` (line 95) -- returns error.
- Metadata KV count limit: `metadataKVCount > 1_000_000` (line 101) -- returns error.
- String length limit: `length > 1<<20` (line 180, 1 MB max).
- Array length limit: `length > 1<<20` (line 247).

### 10. inference/timeseries/arch_patchtst.go -- channel-independent projection

**Status: PASS**

The `Forward()` method (lines 245-330) processes each variable independently:
- Loop at line 260: `for v := range numVars` extracts each variable separately.
- `extractVariable()` (line 333) extracts a single variable as `[batch, seq_len]`.
- Each variable goes through its own patch embed + encoder pass independently.
- Projection at line 305: reshapes to `[batch*numVars, dModel]`, applies shared projection weight, then reshapes back. This is mathematically equivalent to per-variable projection (no cross-variable mixing in the matmul since each row corresponds to a single variable).

---

## Remaining panic() Calls in Production Code

| File | Line | panic() call | Severity | Notes |
|------|------|-------------|----------|-------|
| `rl/replay.go` | 20 | `panic("rl: ReplayBuffer capacity must be > 0")` | **Medium** | Should return error instead. Constructor panics on invalid input. |
| `rl/replay.go` | 60 | `panic("rl: priorities length must equal ReplayBuffer.Len()")` | **Medium** | Should return `([]Experience, error)`. Runtime panic on mismatched slice length. |
| `layers/reducesum/reducesum.go` | 111 | `panic("ReduceSum layer requires exactly 1 input for backward")` | **High** | In `Backward()` which already returns error -- this should be `return nil, fmt.Errorf(...)`. |
| `security/secrets.go` | 57 | `panic(err)` | **Low** | `MustGet()` pattern is idiomatic Go (similar to `template.Must`). Documented as panic. |
| `inference/registry.go` | 37, 40, 45 | Three `panic()` calls in `RegisterArchitecture()` | **Low** | Init-time registration pattern, idiomatic Go (similar to `http.HandleFunc`). Called only during `init()`. |
| `examples/distributed-training/main.go` | 46, 50, 55, 59 | Four `panic()` calls | **None** | Example code, not production. |
| `cmd/bench_prefix/main.go` | 79 | `panic(...)` | **Low** | Benchmark CLI tool, not library code. |

**Action items:**
- **layers/reducesum/reducesum.go:111** -- Replace `panic()` with `return nil, fmt.Errorf(...)`. This is the most serious remaining panic since it's in a function that already returns error.
- **rl/replay.go:20,60** -- Convert to return `(*ReplayBuffer, error)` and `([]Experience, error)` respectively.

---

## .Data() Calls on GPU Tensors

Reviewed the key production paths for `.Data()` usage on GPU tensors:

| File | Context | Risk |
|------|---------|------|
| `inference/arch_llama.go:170` | `result.Data()` in softcap CPU path | **Safe** -- guarded by `if _, ok := result.GetStorage().(*tensor.GPUStorage[T]); ok` check on line 154. CPU path only reached for non-GPU storage. |
| `inference/arch_llama.go:234,298` | `input.Data()` and `e.weight.Data()` in embedding lookup | **Safe** -- CPU path only, guarded by GPU storage check on line 248. |
| `inference/timeseries/arch_patchtst.go:334` | `x.Data()` in `extractVariable` | **Low risk** -- timeseries models currently CPU-only. Would need GPU path if GPU inference is added. |
| `training/optimizer/adamw.go:247` | `grad.Data()` in NaN guard | **Medium risk** -- if gradients live on GPU, `.Data()` may return stale/empty data. Currently training is CPU-only but this would silently fail to detect NaN on GPU gradients. |
| `layers/ssm/s4.go:304-309` | Multiple `.Data()` calls in Backward | **Low risk** -- SSM models are CPU-only. |
| `training/lora/linear.go` | No `.Data()` calls | **Safe** -- all ops go through engine. |

**Action item:**
- `training/optimizer/adamw.go:247` -- The NaN/Inf guard uses `.Data()` to scan gradient values. If gradients are on GPU, this would need a D2H copy first or an engine-level NaN check. Currently safe because training is CPU-only, but should be addressed before GPU training is enabled.

---

## unsafe.Pointer Usage

All `unsafe.Pointer` usage in production code is in GPU interop paths:
- `generate/gpu_kv_cache.go` -- GPU memory allocation/deallocation via allocator interface. Correct usage pattern (pointer to GPU-allocated memory).
- `internal/cuda/` -- CUDA runtime bindings via purego. Expected and necessary.
- `internal/xblas/` -- SIMD assembly entry points. Expected.
- `internal/gpuapi/` -- GPU runtime abstraction layer. Expected.

No misuse of `unsafe.Pointer` detected (no pointer arithmetic violations, no casting between unrelated types outside of GPU interop).

---

## Resource Leaks

### Goroutine Leaks
- `GenerateBatch()` (inference/inference.go:463-474): All goroutines are tracked via `sync.WaitGroup` with `wg.Wait()` before return. **No leak.**
- `lmHeadNode.Forward()` softcap parallel path (arch_llama.go:181-196): Uses `sync.WaitGroup` with `wg.Wait()`. **No leak.**

### GPU Memory
- `Model.Close()` (inference/inference.go:748-760): Closes engine (which frees GPU handles) and mmap closer. **Correct.**
- Session pool (`acquireSession`/`releaseSession`): Sessions that overflow the pool are discarded (line 422). If sessions hold GPU memory, this relies on GC finalizers. **Minor risk** -- discarded sessions should be explicitly freed if they hold GPU resources.

### File Handles
- `Model.closer` is set for mmap-loaded models and freed in `Close()`. **Correct.**
- `findGGUF()` uses `os.ReadDir()` which doesn't hold file handles. **Safe.**

---

## Summary

| Item | Status |
|------|--------|
| E106-1: Embedding bounds check | PASS |
| E106-2: RegisterAlias mutex + batch semaphore | PASS |
| E106-3: layers/core panics removed | PASS |
| E106-4: AttentionHead error returns | PASS |
| E106-5: S4 nil gradient check | PASS |
| E106-6: LoRA nil gradient check | PASS |
| E106-7: AdamW gradient clipping + NaN guard | PASS |
| E106-8: GGUF loader overflow checks | PASS |
| E106-9: GGUF parser count limits | PASS |
| E106-10: PatchTST channel independence | PASS |

### New Findings Requiring Action

1. **HIGH: `layers/reducesum/reducesum.go:111`** -- `panic()` in `Backward()` should be `return nil, fmt.Errorf(...)`.
2. **MEDIUM: `rl/replay.go:20,60`** -- Two `panic()` calls should be converted to error returns.
3. **MEDIUM: `training/optimizer/adamw.go:247`** -- `.Data()` on gradient tensors will silently skip NaN detection if gradients live on GPU. Needs D2H copy or engine-level check before GPU training is enabled.
4. **LOW: `inference/timeseries/arch_patchtst.go:334`** -- `.Data()` in `extractVariable` needs GPU path before GPU timeseries inference.
5. **LOW: Session pool discard** -- Discarded sessions in `releaseSession()` (inference/inference.go:422) should be explicitly freed if they hold GPU resources.
