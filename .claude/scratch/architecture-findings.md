# Architecture Review: zerfoo ML Framework

**Reviewer**: Chief Software Architect
**Date**: 2026-03-21
**Scope**: 1321 Go files across the zerfoo core framework
**Commit**: d88076a (main)

---

## 1. Design Patterns

### 1.1 Architectural Pattern: Modular Monolith (Well-Structured)

**Impact**: N/A (Positive Finding)

The codebase follows a modular monolith pattern with clear package boundaries. The dependency graph flows downward:

```
cmd/ -> serve/ -> inference/ -> generate/ -> layers/ -> ztensor (external)
                             -> model/
       training/ -> layers/ -> ztensor
       distributed/ -> training/
```

Each package has a doc.go with clear purpose documentation. The Engine[T] interface from ztensor is consistently threaded through layers/, generate/, and inference/ as the central compute abstraction. Functional options (WithX() pattern) are used consistently for configuration across Server, Generator, GQA, Dense, Gateway, etc.

### 1.2 Upward Dependency in layers/attention

**Impact**: Medium
**Affected files**: layers/attention/grouped_query_attention.go

The layers/attention package imports generate for generate.GetCache[T](ctx) and generate.FullBufferProvider[T]. This creates a dependency from a lower-level layer package to a higher-level generation package. The KV cache interface should be defined in a shared package (e.g., layers/attention itself or a cache/ package) and implemented in generate/.

**Recommendation**: Extract Cache and FullBufferProvider interfaces into a package that layers/attention owns, and have generate/ implement them. This follows the Dependency Inversion Principle.

### 1.3 Direct CUDA Imports in layers/

**Impact**: Medium
**Affected files**: layers/attention/grouped_query_attention.go, layers/attention/flash.go, layers/core/matmul_nbits_cuda.go

Layer code directly imports internal/cuda and internal/cuda/kernels, bypassing the Engine[T] abstraction. This means these code paths are CUDA-specific and won't work with ROCm or OpenCL backends.

**Recommendation**: Route all GPU operations through compute.Engine[T] or the GRAL abstraction layer. If kernel-specific fast paths are needed, add them as optional Engine[T] capabilities via interface assertion.

### 1.4 Separation of Concerns

**Impact**: N/A (Positive Finding)

The serve/ package cleanly separates HTTP handling from business logic. serve/server.go handles routing and JSON serialization; inference.Model handles generation; generate.Generator handles token sampling. The Server struct is a thin HTTP adapter over the inference layer.

---

## 2. God Files

### 2.1 Files Over 500 Lines (Production Code Only)

| File | Lines | Assessment |
|------|-------|------------|
| timeseries/patchtst.go | 1400 | Contains model definition, forward pass, training loop, save/load, and Xavier init. Should split into patchtst_model.go, patchtst_train.go, patchtst_io.go. |
| internal/cudnn/cudnn_purego.go | 1179 | Binding file -- acceptable as a single file since it is mechanical. |
| layers/attention/grouped_query_attention.go | 1109 | Core GQA with prefill, decode, backward pass, merged QKV, fused kernels. Large but cohesive. Could split decode fast path into gqa_decode.go. |
| generate/grammar/converter.go | 1107 | JSON Schema to grammar conversion with recursive descent. Cohesive single-purpose. Acceptable. |
| inference/arch_rwkv.go | 973 | Architecture builder. Acceptable -- one file per architecture is a good pattern. |
| layers/ssm/complex_state.go | 949 | Complex SSM state tracking. Could extract scan logic into ssm_scan.go. |
| training/automl/search.go | 939 | AutoML search with multiple architecture evaluators. Should split evaluators into separate files. |
| generate/generator.go | 934 | Core generator with sampling, speculative decode integration. At the boundary -- could split sampling into sampling.go. |
| inference/inference.go | 859 | Model loading, generation dispatch, session pooling. Borderline -- session pool could be extracted. |
| serve/server.go | 848 | Server with routes, handlers, types. Request/response types should move to serve/types.go. |

**Impact**: Medium
**Recommendation**: Split timeseries/patchtst.go (1400 lines) as the highest priority. It mixes model definition, training, I/O, and initialization in a single file. Similar pattern in timeseries/cfc.go (759 lines) and timeseries/nhits.go (744 lines). Extract serve/server.go request/response types into serve/types.go.

---

## 3. Error Handling

### 3.1 Panics in Library Code

**Impact**: High
**Affected files**: layers/core/dense.go:30, layers/core/cast.go:39,62, layers/core/matmul.go:162, layers/core/mul.go:39,55, layers/core/sub.go:70, layers/core/concat.go:42,105, layers/core/unsqueeze.go:41,81, layers/core/reshape.go:77, layers/core/rotary_embedding.go:57,94, layers/attention/attention_head.go:58,62,68,73,78, inference/registry.go:37,40,45

**45 panic() calls** found across production code. The most concerning are in layers/core/ where input validation panics instead of returning errors. For example:

- layers/core/dense.go:30 -- panic inside a functional option: panic(fmt.Sprintf("failed to create bias: %v", err))
- layers/core/matmul.go:162 -- panic on wrong input count: panic("MatMul layer requires exactly 2 inputs")

The inference/registry.go panics are acceptable (follows sql.Register() convention for init-time registration). The layers/core/ panics are not -- they crash the entire server on malformed model graphs.

**Recommendation**: Convert all layers/core/ and layers/attention/ panics to return errors. The Forward() and Backward() methods already return error -- use them. For WithBias() functional option, change the pattern to validate at NewDense() call time.

### 3.2 Bare Error Returns

**Impact**: Low
**Affected files**: 263 instances of return err (without wrapping) across 80 non-test files.

The codebase is generally good about error wrapping (1833 uses of %w out of 3639 fmt.Errorf calls = 50% wrapping rate). The remaining bare returns are concentrated in cmd/ (expected -- top-level commands), registry/oci.go (6 bare returns), training/optimizer/adamw.go (18 bare returns), and internal/cublas/ (16 bare returns).

**Recommendation**: Add wrapping context in training/optimizer/adamw.go and registry/oci.go where errors cross package boundaries.

### 3.3 Swallowed Errors

**Impact**: Low
**Affected files**: 56 instances of _ = across 30 files.

Most are benign (JSON encoding to already-committed HTTP response writer, test code). The one in health/server.go:118 (_ = json.NewEncoder(w).Encode(v)) is the standard Go pattern for writing JSON responses -- acceptable.

---

## 4. Resource Management

### 4.1 Unbounded Goroutine Spawning in GenerateBatch

**Impact**: High
**Affected file**: inference/inference.go:428

GenerateBatch spawns one goroutine per prompt with no concurrency limit. A batch of 1000 prompts will spawn 1000 goroutines, each potentially allocating GPU memory for KV caches. This is called by the batch handler in serve/server.go:108.

**Recommendation**: Use a semaphore (e.g., golang.org/x/sync/errgroup with SetLimit()) or the existing internal/workerpool to cap concurrency. Limit should be tied to available GPU memory / per-sequence KV cache size.

### 4.2 Worker Pool Close() Not Thread-Safe

**Impact**: Medium
**Affected file**: internal/workerpool/pool.go:49-56

The closed field is a plain bool checked and set without synchronization. Concurrent calls to Close() can cause a double-close panic on the tasks channel.

**Recommendation**: Use sync.Once instead:
```go
func (p *Pool) Close() {
    p.once.Do(func() {
        close(p.tasks)
        p.wg.Wait()
    })
}
```

### 4.3 Goroutine Lifecycle Management

**Impact**: N/A (Positive Finding)

The serve/disaggregated/gateway.go properly manages goroutine lifecycle with context.WithCancel, sync.WaitGroup, and a Close() method that cancels context and waits for goroutines. The serve/batcher/scheduler.go also uses proper shutdown patterns.

### 4.4 Defer Close Coverage

**Impact**: N/A (Positive Finding)

419 instances of defer .Close() across 117 files show consistent resource cleanup patterns. File handles, HTTP response bodies, and gRPC connections are properly closed.

---

## 5. API Consistency

### 5.1 Consistent Error Response Format

**Impact**: N/A (Positive Finding)

All serve endpoints use a single writeError() function producing OpenAI-compatible error responses:
```json
{"error": {"message": "..."}}
```

This is consistent across serve/server.go (30+ call sites), serve/audio.go, and serve/repository/handler.go. The inferenceErrorStatus() function properly maps OOM errors to 503.

### 5.2 No Request ID / Correlation Header

**Impact**: Medium
**Affected file**: serve/server.go

The server does not generate or propagate request IDs. The logMiddleware logs method, path, model, latency, and status code, but there is no X-Request-Id header for correlating logs across client-server boundaries. The disaggregated serving path (serve/disaggregated/) does have request_id in its protobuf, but the main HTTP API does not.

**Recommendation**: Add request ID middleware that generates a UUID (or reads from X-Request-Id header) and propagates it via context. Include it in all log entries and return it in the response header.

### 5.3 No Pagination or Filtering on /v1/models

**Impact**: Low
**Affected file**: serve/server.go

The /v1/models endpoint returns all models. Since zerfoo typically serves a single model (or a small number via multi-model manager), this is acceptable. The OpenAI API specification itself does not require pagination for this endpoint.

---

## 6. Observability

### 6.1 Mixed Use of stdlib log and Structured Logger

**Impact**: Medium
**Affected files**: layers/attention/grouped_query_attention.go, generate/generator.go, generate/tensor_cache.go, generate/megakernel.go, inference/load_gguf.go, model/gguf/loader.go, serve/disaggregated/gateway.go

Seven production files import stdlib "log" instead of github.com/zerfoo/ztensor/log. The structured logger is properly used in serve/server.go, health/server.go, distributed/worker_service.go, and most other packages. The stdlib log calls bypass structured logging, losing key-value context and level filtering.

**Recommendation**: Replace all stdlib log imports with the structured logger, passed via constructor or context. Priority: serve/disaggregated/gateway.go (HTTP-facing) and generate/generator.go (hot path).

### 6.2 Health Checks

**Impact**: N/A (Positive Finding)

health/server.go provides proper Kubernetes-compatible /healthz (liveness) and /readyz (readiness) endpoints with pluggable check functions, JSON responses, and correct HTTP status codes (200 vs 503). It also exposes /debug/pprof/ for runtime profiling.

### 6.3 Metrics

**Impact**: N/A (Positive Finding)

The runtime.Collector interface is threaded through Server, Generator, workerService, and other components. The server exposes /metrics endpoint. Speculative decoding reports acceptance rate gauges. The pattern of defaulting to runtime.Nop() prevents nil panics.

### 6.4 No Distributed Tracing

**Impact**: Low

There is no OpenTelemetry or distributed tracing integration. Context propagation exists throughout (all methods accept context.Context), but no spans are created. This matters primarily for the disaggregated serving path where requests cross network boundaries (gateway -> prefill worker -> decode worker).

**Recommendation**: Add optional OpenTelemetry span creation in serve/server.go middleware and serve/disaggregated/gateway.go. The context plumbing is already in place.

---

## 7. Scalability

### 7.1 Continuous Batching Scheduler

**Impact**: N/A (Positive Finding)

serve/batcher/scheduler.go implements a well-designed continuous batching scheduler with ragged batches (zero padding), immediate slot eviction, and pending queue fill. This is the right architecture for high-throughput serving.

### 7.2 Disaggregated Serving

**Impact**: N/A (Positive Finding)

serve/disaggregated/ properly separates prefill and decode phases with gRPC streaming, least-loaded routing, exponential backoff health checks, and clean connection lifecycle management. The Gateway pattern with context.CancelFunc and sync.WaitGroup is correct.

### 7.3 Distributed Training Architecture

**Impact**: N/A (Positive Finding)

distributed/worker_service.go implements AllReduce, Barrier, and Broadcast RPCs with proper session management and metrics collection. The barrierState coordinates across worldSize workers.

### 7.4 Unbounded Batch Size in GenerateBatch

**Impact**: Medium
**Affected file**: inference/inference.go:416

GenerateBatch accepts an unbounded []string of prompts. Combined with the unbounded goroutine spawning (finding 4.1), this can exhaust GPU memory. The serve/server.go batch handler calls this directly.

**Recommendation**: Add a MaxBatchSize configuration to Model or GenerateBatch, rejecting requests that exceed the limit with a clear error message. The continuous batcher in serve/batcher/ already has maxBatchSize -- ensure the legacy batch path also enforces it.

### 7.5 Worker Pool Task Channel Sizing

**Impact**: Low
**Affected file**: internal/workerpool/pool.go:18

The buffered channel size is 4*n where n is the worker count. This is a reasonable heuristic, but for GEMV workloads with thousands of tiles, the Submit() method blocks on channel send while workers drain. The current design (blocking Submit, fan-out, WaitGroup) is correct but could benefit from a steal-based work queue for large tile counts.

**Recommendation**: No immediate action needed. Profile before optimizing.

---

## Summary

### High Impact (Requires Action)

| # | Finding | Files |
|---|---------|-------|
| 3.1 | 45 panic() calls in library code (layers/core/, layers/attention/) | 15+ files in layers/ |
| 4.1 | Unbounded goroutine spawning in GenerateBatch | inference/inference.go |

### Medium Impact (Should Fix)

| # | Finding | Files |
|---|---------|-------|
| 1.2 | Upward dependency: layers/attention imports generate | layers/attention/grouped_query_attention.go |
| 1.3 | Direct CUDA imports bypassing Engine[T] abstraction | 4 files in layers/ |
| 2.1 | timeseries/patchtst.go at 1400 lines needs splitting | timeseries/patchtst.go |
| 4.2 | Worker pool Close() not thread-safe (double-close risk) | internal/workerpool/pool.go |
| 5.2 | No request ID correlation in HTTP API | serve/server.go |
| 6.1 | 7 files use stdlib log instead of structured logger | scattered across generate/, inference/, layers/, serve/ |
| 7.4 | Unbounded batch size in GenerateBatch | inference/inference.go |

### Low Impact (Nice to Fix)

| # | Finding | Files |
|---|---------|-------|
| 3.2 | 263 bare return err without wrapping context | 80 files |
| 5.3 | No pagination on /v1/models | serve/server.go |
| 6.4 | No distributed tracing (OpenTelemetry) | serve/, serve/disaggregated/ |
| 7.5 | Worker pool channel sizing heuristic | internal/workerpool/pool.go |

### Positive Findings

- Clean modular monolith architecture with clear package boundaries
- Consistent functional options pattern across all major types
- Engine[T] abstraction consistently used for compute dispatch
- Proper error wrapping (50% of fmt.Errorf calls use %w)
- Good resource cleanup (419 defer .Close() instances)
- Kubernetes-compatible health checks with readiness probes
- Metrics collection threaded through all major components
- Well-designed continuous batching and disaggregated serving
- Proper goroutine lifecycle management in Gateway and Scheduler
- Graceful shutdown coordinator with reverse-order close
