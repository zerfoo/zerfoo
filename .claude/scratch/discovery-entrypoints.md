# Zerfoo Entry Points Discovery

Audit date: 2026-03-21

---

## 1. CLI Entry Points

### Main binary: cmd/zerfoo/main.go

The CLI uses a custom cli.CommandRegistry (no cobra/flag framework). All commands are registered in main() at cmd/zerfoo/main.go:30-73. Signal handling via cli.SignalContext wraps a shutdown.Coordinator.

| Command | Handler Type | File:Line | Required Args | Input Validation |
|---------|-------------|-----------|---------------|-----------------|
| predict | PredictCommand[float32] | cmd/cli/framework.go:151 | --model-path, --data-path, --output | Manual flag parsing; validates required flags at L295-303 |
| tokenize | TokenizeCommand | cmd/cli/framework.go:555 | --text | Validates --text is non-empty at L593 |
| worker | WorkerCommand | cmd/cli/worker.go:35 | --coordinator-address, --worker-address | Validates required flags at L74-79; parsePositiveInt for --world-size |
| pull | PullCommand | cmd/cli/pull.go:34 | model-id positional | Validates modelID non-empty at L59 |
| list | ListCommand | cmd/cli/list.go:32 | (none) | No required args |
| rm | RmCommand | cmd/cli/rm.go:32 | model-id positional | Validates modelID non-empty at L51 |
| run | RunCommand | cmd/cli/run.go:41 | model-id positional | Validates modelID non-empty at L150; JSON schema validation for --json-schema |
| serve | ServeCommand | cmd/cli/serve.go:45 | model-id positional | Validates modelID at L76; parseGPUList validates --gpus at L147-173 |
| version | VersionCommand | cmd/cli/version.go:39 | (none) | No validation needed |
| automl | AutoMLCommand | cmd/cli/automl.go:75 | --model, --dataset | Validates required flags at L259-264 |
| train | TrainCommand | cmd/cli/train.go:52 | --config, --data | Validates required flags and ranges at L222-232 |
| sentiment | SentimentCommand | cmd/cli/sentiment.go:52 | --model + (--text or --file) | Validates mutual exclusivity at L213-215 |
| finetune-sentiment | FineTuneSentimentCommand | cmd/cli/finetune_sentiment.go:49 | --model, --data | Validates required flags at L243-248 |

Authentication/Authorization: None. CLI commands run locally with the user's OS permissions.

Middleware: None. The CLI framework is a simple dispatch loop at cmd/cli/framework.go:684.

### Standalone binaries

| Binary | File | Purpose |
|--------|------|---------|
| zerfoo-edge | cmd/zerfoo-edge/main.go | Minimal edge/embedded inference (CPU-only, no training/serve/GPU) |
| zerfoo-predict | cmd/zerfoo-predict/main.go | Standalone predict CLI using flag package |
| zerfoo-tokenize | cmd/zerfoo-tokenize/main.go | Standalone tokenize CLI using flag package |
| bench | cmd/bench/main.go | Benchmarking tool |
| bench-compare | cmd/bench-compare/main.go | Benchmark comparison tool |
| bench_tps | cmd/bench_tps/main.go | Tokens-per-second benchmark |
| bench_disagg | cmd/bench_disagg/main.go | Disaggregated serving benchmark |
| bench_prefix | cmd/bench_prefix/main.go | Prefix caching benchmark |
| bench_spec | cmd/bench_spec/main.go | Speculative decoding benchmark |
| bench_mamba | cmd/bench_mamba/main.go | Mamba architecture benchmark |
| bench_batch | cmd/bench_batch/main.go | Batch inference benchmark |
| finetune | cmd/finetune/main.go | Standalone fine-tuning tool |
| ts_train | cmd/ts_train/main.go | Time-series training tool |
| train_distributed | cmd/train_distributed/main.go | Distributed training launcher |
| coverage-gate | cmd/coverage-gate/main.go | CI coverage gate tool |
| deprecation-check | cmd/deprecation-check/main.go | API deprecation checker |
| debug-infer | cmd/debug-infer/main.go | Debug inference tool |

---

## 2. HTTP Routes (Serve Layer)

### Main Server: serve/server.go

Created by serve.NewServer() at serve/server.go:89. Handler returned via s.Handler() at L136 which wraps the mux in logMiddleware.

#### Route Table

| Method | Path | Handler | File:Line | Middleware | Auth | Input Validation |
|--------|------|---------|-----------|------------|------|-----------------|
| POST | /v1/chat/completions | handleChatCompletions | serve/server.go:365 | recoveryMiddleware + logMiddleware | None | JSON decode; validates messages non-empty (L372); validates tools (L378-386); validates tool_choice (L385); validates response_format.json_schema (L403-415) |
| POST | /v1/completions | handleCompletions | serve/server.go:524 | recoveryMiddleware + logMiddleware | None | JSON decode; validates prompt non-empty (L531) |
| POST | /v1/embeddings | handleEmbeddings | serve/server.go:659 | recoveryMiddleware + logMiddleware | None | JSON decode; validates input type (string or []string) (L668-683); validates non-empty (L685) |
| GET | /v1/models | handleModels | serve/server.go:605 | recoveryMiddleware + logMiddleware | None | None |
| GET | /v1/models/{id...} | handleModelInfo | serve/server.go:618 | recoveryMiddleware + logMiddleware | None | Path param id; 404 if not matching loaded model |
| DELETE | /v1/models/{id...} | handleModelDelete | serve/server.go:635 | recoveryMiddleware + logMiddleware | None | Path param id; 404 if not matching; closes model on success |
| POST | /v1/audio/transcriptions | handleAudioTranscriptions | serve/audio.go:31 | recoveryMiddleware + logMiddleware | None | 501 if no transcriber; MaxBytesReader 25MB; multipart form; validates file field; validates size limits (L57-65) |
| POST | /v1/classify | handleClassify | serve/classify.go:68 | recoveryMiddleware + logMiddleware | None | 501 if no classifier; JSON decode; validates input non-empty (L80); validates max batch 256 (L85) |
| GET | /openapi.yaml | handleOpenAPISpec | serve/server.go:745 | recoveryMiddleware + logMiddleware | None | Serves embedded OpenAPI spec |
| GET | /metrics | handleMetrics (closure) | serve/metrics.go:49 | logMiddleware only (no recovery) | None | Prometheus text exposition format |

#### Middleware Stack (applied to all routes)

1. **logMiddleware** (serve/server.go:155): Wraps the entire mux. Logs method, path, model ID, latency, status code. Uses statusRecorder to capture response status.
2. **recoveryMiddleware** (serve/server.go:188): Per-handler. Catches panics, logs them, returns 500. Applied to all routes except /metrics.

Authentication: The base server has NO authentication. Auth is provided by optional middleware layers:
- cloud.TenantRegistry.Middleware() extracts Bearer token from Authorization header.
- cloud.BillingMiddleware() meters token usage per tenant.

#### Streaming Support

- POST /v1/chat/completions with stream: true: SSE via streamChatCompletion (L753)
- POST /v1/completions with stream: true: SSE via streamCompletion (L794)
- Both set Content-Type: text/event-stream, Cache-Control: no-cache, Connection: keep-alive
- Terminal event: data: [DONE]

#### Speculative Decoding

When WithDraftModel is configured, handleCompletions uses s.model.SpeculativeGenerate (L559) with 4 draft tokens.

#### Batch Scheduling

When WithBatchScheduler is configured, both chat and completion handlers route non-streaming requests through s.batch.Submit().

---

### Repository Handler: serve/repository/handler.go

Registered via handler.RegisterRoutes(mux) at serve/repository/handler.go:22.

| Method | Path | Handler | File:Line | Auth | Input Validation |
|--------|------|---------|-----------|------|-----------------|
| GET | /v1/models | handleList | serve/repository/handler.go:35 | None | None |
| GET | /v1/models/{id} | handleGet | serve/repository/handler.go:50 | None | Path param id; 404 on not found |
| POST | /v1/models | handleUpload | serve/repository/handler.go:72 | None | Requires multipart/form-data; 10GB max upload; validates metadata JSON field; validates id and name required; validates file field |
| DELETE | /v1/models/{id} | handleDelete | serve/repository/handler.go:151 | None | Path param id; 404 on not found |

Note: Repository routes overlap with the main serve routes on /v1/models. They would be registered on a separate mux in practice.

---

### Disaggregated Gateway: serve/disaggregated/gateway.go

The Gateway implements http.Handler directly via ServeHTTP at serve/disaggregated/gateway.go:227.

| Method | Path | Handler | File:Line | Auth | Input Validation |
|--------|------|---------|-----------|------|-----------------|
| POST | (any) | Gateway.ServeHTTP | serve/disaggregated/gateway.go:227 | None | Validates POST method (L228); JSON decodes PreFillRequest; checks healthy prefill/decode workers available (L243, L277) |

Flow: HTTP request -> least-loaded prefill worker (gRPC) -> collect KV blocks -> least-loaded decode worker (gRPC) -> stream tokens as SSE.

Health checks: Background goroutines (healthCheckLoop, L339) with exponential backoff monitor gRPC connection state for each worker.

---

### Health Server: health/server.go

Created via health.NewServer(logger) at health/server.go:34. Handler returned via s.Handler() at L50.

| Method | Path | Handler | File:Line | Auth | Input Validation |
|--------|------|---------|-----------|------|-----------------|
| GET | /healthz | handleHealthz | health/server.go:65 | None | Method check (GET only, L66) |
| GET | /readyz | handleReadyz | health/server.go:74 | None | Method check (GET only, L76); runs all registered CheckFuncs; returns 503 if any fail |
| GET | /debug/pprof/ | pprof.Index | health/server.go:56 | None | Standard Go pprof |
| GET | /debug/pprof/cmdline | pprof.Cmdline | health/server.go:57 | None | Standard Go pprof |
| GET | /debug/pprof/profile | pprof.Profile | health/server.go:58 | None | Standard Go pprof |
| GET | /debug/pprof/symbol | pprof.Symbol | health/server.go:59 | None | Standard Go pprof |
| GET | /debug/pprof/trace | pprof.Trace | health/server.go:60 | None | Standard Go pprof |

Readiness checks: Registered via AddReadinessCheck(name, CheckFunc). Built-in checks:
- health.EngineCheck (health/engine_check.go:16): Verifies CPU tensor engine with a small Add operation.
- health.EngineCheckGeneric (health/engine_check.go:47): Generic version using Fill/Zero operations.
- distributed.WorkerNode.healthCheck (distributed/worker_node.go:98): Reports whether the worker node is started.

---

## 3. Cloud/Multi-tenant Middleware

### Tenant Registry: serve/cloud/tenant.go

| Component | File:Line | Purpose |
|-----------|-----------|---------|
| TenantRegistry.Middleware() | serve/cloud/tenant.go:159 | HTTP middleware: extracts Bearer key from Authorization header; looks up tenant; enforces concurrency limit (MaxConcurrentRequests); enforces token rate limit (MaxTokensPerMinute with per-minute reset); injects Tenant into context |
| BillingMiddleware() | serve/cloud/billing.go:87 | HTTP middleware: captures request/response bodies; extracts model from request and usage from response; publishes UsageEvent to UsageRecorder |
| TenantFromContext() | serve/cloud/tenant.go:91 | Context accessor for downstream handlers |

Auth flow:
1. Extract Authorization: Bearer api-key header
2. Look up tenant by API key in TenantRegistry
3. Reject with 401 if missing or unknown
4. Reject with 429 if token budget exhausted or concurrency limit exceeded
5. Inject Tenant into request context

### Resource Manager: serve/cloud/resource_manager.go

LRU-based VRAM budget manager. Not an HTTP handler but controls model loading/eviction. Load(), Touch(), Evict(), Stats(), LoadedModels().

---

## 4. gRPC Services

### Coordinator Service: distributed/coordinator/coordinator.go

Implements pb.CoordinatorServer. Registered at coordinator.go:91 via pb.RegisterCoordinatorServer(server, c).

| RPC | Method | File:Line | Input Validation |
|-----|--------|-----------|-----------------|
| RegisterWorker | Unary | coordinator.go:145 | WorkerId non-empty (L149); checks duplicate registration (L153) |
| UnregisterWorker | Unary | coordinator.go:203 | WorkerId non-empty (L208); checks worker exists (L212) |
| Heartbeat | Unary | coordinator.go:226 | WorkerId non-empty (L231); checks worker exists (L235) |
| StartCheckpoint | Unary | coordinator.go:249 | Epoch range check (L262); creates checkpoint ID from epoch |
| EndCheckpoint | Unary | coordinator.go:277 | WorkerId non-empty (L282); checkpoint ID exists (L286) |

Background: reaper() goroutine (L125) evicts workers that miss heartbeats (configurable timeout, default 30s).

### Distributed Service: distributed/worker_service.go

Implements pb.DistributedServiceServer. Registered via ServerManager.Start() in grpc_strategy.go:129.

| RPC | Method | File:Line | Input Validation |
|-----|--------|-----------|-----------------|
| AllReduce | Bidi streaming | worker_service.go:305 | Session must exist (L309); tensor name non-empty (L324); tensor shape/data validated via validateTensor (L328) |
| Barrier | Unary | worker_service.go:352 | Rank in range [0, worldSize) (L355) |
| Broadcast | Unary | worker_service.go:368 | Name non-empty (L372); tensor validated if present (L374-378) |

Metrics: Each RPC records *_count counter and *_duration_seconds histogram via recordOp (L296).

### Disaggregated PrefillWorker Service: serve/disaggregated/prefill_worker.go

Implements disaggpb.PrefillWorkerServer. Registered via Register(grpc.ServiceRegistrar) at L40.

| RPC | Method | File:Line | Input Validation |
|-----|--------|-----------|-----------------|
| Prefill | Server streaming | prefill_worker.go:48 | Token list non-empty (L53); streams KV blocks per layer, terminal Done=true message |

### Disaggregated DecodeWorker Service: serve/disaggregated/decode_worker.go

Implements disaggpb.DecodeWorkerServer.

| RPC | Method | File:Line | Input Validation |
|-----|--------|-----------|-----------------|
| Decode | Server streaming | decode_worker.go:41 | Nil request check (L43); MaxNewTokens defaults to 1 if <= 0; KV data FP16 length validated (must be even); token_ids non-empty (L75); EOS token ID = 2 |

---

## 5. Agent/Tool Calling Layer

### Agent Adapter: serve/agent/openai_adapter.go

Not a separate HTTP endpoint. Provides helper functions to convert between OpenAI tool format and internal generate/agent types:

| Function | File:Line | Purpose |
|----------|-----------|---------|
| ConvertTools | serve/agent/openai_adapter.go:17 | OpenAI []Tool -> []genagent.ToolDef |
| ToolCallFromAgent | serve/agent/openai_adapter.go:31 | genagent.ToolCall -> serve.ToolCall |
| ToolCallsFromStep | serve/agent/openai_adapter.go:44 | Extract tool calls from AgentStep |
| ResponseFromSession | serve/agent/openai_adapter.go:59 | AgentSession -> ChatCompletionResponse |
| BuildToolCallChunks | serve/agent/openai_adapter.go:144 | Build SSE stream chunks for tool calls |

### Tool Call Detection: serve/tool_calls.go

| Function | File:Line | Purpose |
|----------|-----------|---------|
| DetectToolCall | serve/tool_calls.go:26 | Heuristic detection: checks if model output is JSON matching a tool definition |

### Tool Validation: serve/tools.go

| Function | File:Line | Purpose |
|----------|-----------|---------|
| validateTools | serve/tools.go:85 | Validates tool type is "function", name matches [a-zA-Z0-9_-]{1,64}, parameters are valid JSON |
| validateToolChoice | serve/tools.go:104 | Validates tool_choice consistency with tools array |

---

## 6. Supporting Infrastructure (Non-endpoint)

### Batch Schedulers

| Scheduler | Package | File | Type |
|-----------|---------|------|------|
| BatchScheduler | serve | serve/batch.go | Fixed-size batch with timeout; used by main server |
| Scheduler (continuous) | serve/batcher | serve/batcher/scheduler.go | Continuous batching with ragged batches, zero padding, slot eviction |
| Batcher (adaptive) | serve/adaptive | serve/adaptive/batcher.go | Dynamic batch size based on queue depth and latency EMA |

### Model Version Registry: serve/registry/

bbolt-backed model version store. Not HTTP-exposed directly. Operations: Register, Activate, GetActive, List, Delete.

Supporting components:
- ABRouter (serve/registry/ab_router.go): Deterministic champion/challenger routing via FNV hash
- CanaryController (serve/registry/canary.go): Gradual traffic ramp-up with success rate threshold
- ShadowRunner (serve/registry/shadow.go): Shadow mode inference for challenger evaluation
- MetricsStore (serve/registry/metrics.go): Per-model P50/P95/P99 latency tracking

### Multi-model Manager: serve/multimodel/manager.go

LRU model loading/eviction with GPU memory budget. Not HTTP-exposed directly. Operations: Get, Unload, Loaded, UsedBytes, Close.

### Kubernetes Operator: serve/operator/

CRD reconciler for ZerfooInferenceService. Not an HTTP endpoint -- drives K8s Deployments, Services, HPAs via KubeClient interface. Supports primary + canary deployments with weighted traffic splitting.

---

## 7. Security Summary

| Layer | Authentication | Authorization | Notes |
|-------|---------------|---------------|-------|
| CLI | None | OS-level | Local process |
| Serve HTTP | Optional Bearer token via cloud.TenantRegistry.Middleware | Per-tenant model allow list, concurrency limit, token rate limit | Not enabled by default |
| Health HTTP | None | None | /debug/pprof/* is exposed without auth |
| gRPC Coordinator | Optional TLS via SetServerOptions | None beyond TLS | Worker ID checked for registration |
| gRPC Distributed | Optional TLS via GrpcStrategyConfig.TLS | None | Rank validated |
| gRPC Disaggregated | Insecure by default; DialOptions configurable | None | No auth on gateway HTTP |
| Model Repository | None | None | Upload endpoint accepts any multipart POST |

### Notable Security Observations

1. Health endpoints expose pprof without auth (health/server.go:56-60). The /debug/pprof/profile endpoint can perform CPU profiling and /debug/pprof/trace can capture execution traces -- these should not be exposed in production without auth.

2. Model repository upload has no auth (serve/repository/handler.go:72). Any client can upload models up to 10GB.

3. DELETE /v1/models/{id} on the main server (serve/server.go:635) closes and unloads the model with no auth check. This is a denial-of-service vector.

4. Disaggregated gateway (serve/disaggregated/gateway.go:227) has no auth on the HTTP entry point. Any client can submit inference requests to the prefill/decode pipeline.

5. The base serve layer has no auth by default. The cloud tenant middleware is optional and must be explicitly wired in.
