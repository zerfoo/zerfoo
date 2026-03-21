# Deep Review Report -- zerfoo v1.10.0

**Date**: 2026-03-21
**Commit**: 872dcf1 (v1.10.0)
**Scope**: Full codebase (1321 Go source files)
**Agents deployed**: 10 (5 discovery, 3 security, 1 architecture, 1 functional)

---

## Executive Summary

Zerfoo is an architecturally sound Go ML inference framework with clean modular boundaries, consistent use of the Engine[T] compute abstraction, and production-quality infrastructure (distroless Docker images, Helm charts, multi-cloud marketplace support). However, the **security layer is entirely disconnected from the serving layer**: well-designed authentication, rate limiting, and TLS primitives exist in the `security/` package but are never wired into the default HTTP server. This means **all API endpoints -- including destructive operations like model deletion -- are unauthenticated by default**. The highest-impact architectural recommendation is to integrate the existing `security/` primitives into `serve/server.go` as middleware. The most critical functional bug is a **data race on `Server.unloaded`** that can crash the server under concurrent load.

---

## System Architecture Map

```mermaid
graph TB
    subgraph CLI["CLI (cmd/)"]
        RUN[zerfoo run]
        SERVE[zerfoo serve]
        PULL[zerfoo pull]
        TRAIN[zerfoo train]
        WORKER[zerfoo worker]
    end

    subgraph Serving["HTTP Server (serve/)"]
        MUX[net/http.ServeMux]
        CHAT[/v1/chat/completions]
        COMP[/v1/completions]
        EMB[/v1/embeddings]
        AUDIO[/v1/audio/transcriptions]
        MODELS[/v1/models]
        CLASS[/v1/classify]
        METRICS[/metrics]
        REPO[Repository Handler]
        VISION[Vision Image Fetch]
    end

    subgraph Security["Security (DISCONNECTED)"]
        KEYSTORE[KeyStore - SHA256 hashed keys]
        RATELIM[RateLimiter - per-IP token bucket]
        TLSCFG[TLSConfig - TLS/mTLS]
        IPFILT[IPFilter]
        CORS[CORSPolicy]
    end

    subgraph Inference["Inference Pipeline"]
        LOAD[GGUF Loader]
        ARCH[Architecture Builders<br>Llama/Gemma/Mistral/Qwen/Phi/DeepSeek/Mamba/Whisper]
        GEN[Generator - autoregressive decode]
        KVC[KV Cache - CPU/Paged/GPU]
        SPEC[Speculative Decoding]
        AGENT[Agent/Tool Calling]
        GRAMMAR[Grammar Constraints]
    end

    subgraph Layers["Neural Network Layers"]
        ATT[Attention - GQA/MLA/Flash/Local]
        NORM[Normalization - RMSNorm/LayerNorm]
        FFN[FFN - Dense/SwiGLU/MoE]
        SSM[SSM - Mamba/S4]
        ACT[Activations]
    end

    subgraph Training["Training"]
        TRAINER[Trainer]
        ADAMW[AdamW/SGD]
        LORA[LoRA]
        LOSS[Loss Functions]
        FSDP[FSDP Distributed]
    end

    subgraph GPU["GPU Backends"]
        CUDA[CUDA - purego/dlopen]
        ROCM[ROCm/HIP]
        OCL[OpenCL/CLBlast]
        GRAL[GPU Runtime Abstraction Layer]
    end

    subgraph External["External"]
        HF[HuggingFace Hub]
        OCI[OCI Registry]
        CLOUD_META[Cloud Metadata<br>169.254.169.254]
    end

    subgraph Infra["Infrastructure"]
        HELM[Helm Chart]
        TF[Terraform - GCP]
        CFN[CloudFormation - AWS]
        DOCKER[Distroless Docker]
        MARKET[Marketplace - AWS/Azure/GCP]
    end

    SERVE --> MUX
    MUX --> CHAT & COMP & EMB & AUDIO & MODELS & CLASS & METRICS
    CHAT --> VISION
    VISION -.->|SSRF risk| CLOUD_META
    CHAT & COMP --> GEN
    PULL --> HF & OCI
    LOAD --> ARCH --> GEN
    GEN --> KVC & SPEC & AGENT & GRAMMAR
    GEN --> ATT & NORM & FFN & SSM & ACT
    ATT & FFN --> GRAL --> CUDA & ROCM & OCL
    TRAIN --> TRAINER --> ADAMW & LORA & LOSS
    TRAINER --> FSDP
    WORKER --> FSDP

    Security -.->|NOT WIRED| MUX

    style Security fill:#ff6b6b,stroke:#c92a2a,color:#fff
    style VISION fill:#ffa07a
    style CLOUD_META fill:#ff6b6b,stroke:#c92a2a,color:#fff
```

---

## Critical and High Findings

### [CRITICAL] C-1: Core API Server Has Zero Authentication

- **Location**: `serve/server.go:122-132`, `cmd/cli/serve.go:108-112`
- **Description**: The main `serve.Server` registers all endpoints with no authentication middleware. The `security.KeyStore` (SHA-256 hashed API keys with scopes, expiry, rotation) and `serve/cloud/tenant.go` middleware exist but are never composed into the default serving path.
- **Exploitation**: Anyone with network access to port 8080 can run inference (cost abuse), delete models (DoS), upload models, and exfiltrate embeddings.
- **Fix**:
```go
// serve/server.go -- add auth middleware option
func WithAPIKey(key string) ServerOption {
    return func(s *Server) { s.apiKey = key }
}

func (s *Server) Handler() http.Handler {
    var h http.Handler = s.mux
    if s.apiKey != "" {
        h = s.authMiddleware(h)
    }
    return s.logMiddleware(h)
}

func (s *Server) authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        if r.URL.Path == "/metrics" || r.URL.Path == "/healthz" {
            next.ServeHTTP(w, r)
            return
        }
        auth := r.Header.Get("Authorization")
        const prefix = "Bearer "
        if !strings.HasPrefix(auth, prefix) ||
            subtle.ConstantTimeCompare([]byte(auth[len(prefix):]), []byte(s.apiKey)) != 1 {
            writeError(w, http.StatusUnauthorized, "invalid or missing API key")
            return
        }
        next.ServeHTTP(w, r)
    })
}

// cmd/cli/serve.go -- accept --api-key flag or ZERFOO_API_KEY env var
```

### [CRITICAL] C-2: Tenant ID Via Spoofable HTTP Header

- **Location**: `cloud/server.go:84,92,112`
- **Description**: `CloudServer.authMiddleware` passes tenant identity downstream via `X-Tenant-ID` HTTP header instead of `context.Context`. If any route bypasses auth middleware but passes through rate limiting/billing middleware, the attacker controls the tenant ID -- enabling billing fraud and tenant impersonation.
- **Exploitation**: `X-Tenant-ID: victim-tenant-id` header consumes victim's token budget.
- **Fix**:
```go
// Replace header-based passing with context:
ctx := context.WithValue(r.Context(), tenantKey{}, tenant)
r = r.WithContext(ctx)

// In downstream middleware, read from context:
tenant := tenantFromContext(r.Context())
```

### [HIGH] H-1: Path Traversal in FileSystemRepository

- **Location**: `serve/repository/repository.go:63-73`
- **Description**: Model IDs passed directly to `filepath.Join(r.baseDir, id)` without containment validation. The `DELETE` path calls `os.RemoveAll`, enabling arbitrary directory deletion.
- **Exploitation**: `DELETE /v1/models/../../important-data` deletes arbitrary directories.
- **Fix**:
```go
func (r *FileSystemRepository) modelDir(id string) (string, error) {
    joined := filepath.Join(r.baseDir, id)
    cleaned := filepath.Clean(joined)
    basePrefix := filepath.Clean(r.baseDir) + string(filepath.Separator)
    if !strings.HasPrefix(cleaned+string(filepath.Separator), basePrefix) {
        return "", fmt.Errorf("repository: model ID %q resolves outside base directory", id)
    }
    return cleaned, nil
}
```

### [HIGH] H-2: SSRF via Vision Image Fetch

- **Location**: `serve/vision.go:125-146`
- **Description**: `downloadImage` fetches arbitrary URLs from user chat messages with no restrictions on target host. Uses `http.DefaultClient` with no timeout.
- **Exploitation**: `{"type":"image_url","image_url":{"url":"http://169.254.169.254/latest/meta-data/iam/security-credentials/"}}` steals cloud IAM credentials.
- **Fix**:
```go
func downloadImage(ctx context.Context, rawURL string) ([]byte, error) {
    parsed, err := url.Parse(rawURL)
    if err != nil {
        return nil, err
    }
    host := parsed.Hostname()
    ips, err := net.DefaultResolver.LookupHost(ctx, host)
    if err != nil {
        return nil, fmt.Errorf("resolve host: %w", err)
    }
    for _, ipStr := range ips {
        ip := net.ParseIP(ipStr)
        if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
            return nil, fmt.Errorf("image URL targets a private network address")
        }
    }
    client := &http.Client{Timeout: 30 * time.Second}
    // ... rest of fetch
}
```

### [HIGH] H-3: No Request Body Size Limit on Inference Endpoints

- **Location**: `serve/server.go:367,526,661`
- **Description**: Chat, completion, and embedding endpoints have no `http.MaxBytesReader`. Multi-GB POST bodies cause OOM. Audio endpoint correctly limits body size.
- **Exploitation**: Send multi-GB JSON to `/v1/chat/completions` to OOM the server.
- **Fix**:
```go
// At top of handleChatCompletions, handleCompletions, handleEmbeddings:
r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB
```

### [HIGH] H-4: Server.unloaded Data Race

- **Location**: `serve/server.go:31,606,621,638,650`
- **Description**: Plain `bool` field accessed concurrently by HTTP handlers without synchronization.
- **Exploitation**: Data race under concurrent load; undefined behavior per Go memory model.
- **Fix**: Replace `unloaded bool` with `unloaded atomic.Bool`.

### [HIGH] H-5: Embedding Lookup Has No Bounds Check

- **Location**: `inference/arch_llama.go:288-295`
- **Description**: Token ID used as index into embedding table without bounds checking. Mismatched tokenizer/model causes index-out-of-bounds panic, crashing the server.
- **Fix**:
```go
id := int(ids[i])
if id < 0 || id >= vocabSize {
    return nil, fmt.Errorf("token ID %d out of range [0, %d)", id, vocabSize)
}
```

### [HIGH] H-6: No Checksum Verification on HuggingFace Downloads

- **Location**: `registry/pull.go:71-113`
- **Description**: Downloaded GGUF files are never integrity-checked. Corrupted downloads silently produce broken model files. OCI path has SHA-256 verification; HF path does not.
- **Fix**: Compute SHA-256 during download and verify against HuggingFace API metadata.

### [HIGH] H-7: Helm Deployment Missing Pod SecurityContext

- **Location**: `deploy/helm/zerfoo/templates/deployment.yaml`
- **Description**: No `securityContext` despite enterprise docs recommending `runAsNonRoot`, `readOnlyRootFilesystem`, `allowPrivilegeEscalation: false`.
- **Fix**:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

### [HIGH] H-8: Cloud Run API Gateway Public to allUsers

- **Location**: `infra/terraform/zerfoo-cloud/cloud_run.tf:63-69`
- **Description**: `roles/run.invoker` granted to `allUsers`, making inference endpoints publicly accessible without authentication.
- **Fix**: Replace `allUsers` with authenticated service account.

### [HIGH] H-9: Server Listens on Plain HTTP Only

- **Location**: `cmd/cli/serve.go:121-123`
- **Description**: No TLS flags despite `security.TLSConfig` existing. API keys and inference data transmitted in plaintext.
- **Fix**: Add `--tls-cert`/`--tls-key` flags and use `ListenAndServeTLS`.

### [HIGH] H-10: 45 panic() Calls in Library Code

- **Location**: `layers/core/dense.go:30`, `layers/core/matmul.go:162`, `layers/core/mul.go:39`, `layers/core/cast.go:39`, `layers/core/concat.go:42`, `layers/core/reshape.go:77`, `layers/attention/attention_head.go:58-78`, and 30+ more
- **Description**: Forward/Backward methods panic on input validation instead of returning errors. This crashes the serving process on malformed model graphs.
- **Fix**: Convert all panics in layers/ to error returns. The Forward()/Backward() signatures already return error.

### [HIGH] H-11: Unbounded Goroutine Spawning in GenerateBatch

- **Location**: `inference/inference.go:428`
- **Description**: One goroutine per prompt with no concurrency limit, each allocating GPU KV cache memory.
- **Fix**: Use `errgroup.SetLimit()` or semaphore to cap concurrency.

---

## Security Findings (Medium/Low/Info)

### Injection

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| INJ-1 | Medium | Integer overflow in GGUF tensor size calculation | `model/gguf/loader.go:24-28` |
| INJ-2 | Medium | Unbounded `io.ReadAll` in OCI blob download | `registry/oci.go:343` |
| INJ-3 | Medium | JSON injection in support API error response | `support/api.go:165` |
| INJ-4 | Low | OCI registry URL injection via repository name | `registry/oci.go:208` |

### Authentication & Authorization

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| AUTH-1 | Medium | Distributed worker gRPC uses insecure credentials by default | `serve/disaggregated/gateway.go:18-19` |
| AUTH-2 | Medium | OCI registry credentials stored in plaintext in memory | `registry/oci.go:17-23` |
| AUTH-3 | Medium | Tenant API keys stored as plaintext map keys in serve/cloud | `serve/cloud/tenant.go:98` |
| AUTH-4 | Low | API key lookup uses map-based hash (non-constant-time) | `security/apikey.go:106-112` |
| AUTH-5 | Low | IP spoofing via X-Forwarded-For without trusted proxy config | `security/network.go:167-183` |
| AUTH-6 | Info | Repository upload/delete endpoints have no auth | `serve/repository/handler.go:22-27` |
| AUTH-7 | Info | Well-designed security primitives exist but are disconnected | `security/*.go` |

### Data Exposure

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| DATA-1 | Medium | Raw inference errors leaked to clients | `serve/server.go:456,564,694` |
| DATA-2 | Medium | Raw errors in SSE streaming responses | `serve/server.go:789,822` |
| DATA-3 | Medium | Tenant API keys exposed via Config()/List() methods | `cloud/tenant.go:29,52-58` |
| DATA-4 | Medium | Missing security headers (X-Content-Type-Options, etc.) | `serve/server.go:830` |
| DATA-5 | Medium | CORS wildcard origin allowed when configured | `security/network.go:149` |
| DATA-6 | Low | Streaming errors leak internal details | `serve/server.go:789,823` |
| DATA-7 | Low | Diagnostic log leaks model architecture details | `model/gguf/loader.go:317-321` |
| DATA-8 | Info | Webhook payloads have no HMAC signature | `support/webhook.go:84-117` |

### Business Logic

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| BIZ-1 | Medium | No server-side max_tokens upper bound | `serve/server.go:398-400` |
| BIZ-2 | Low | genproto replace directive may block security updates | `go.mod:32` |

---

## Architectural Findings

### [Impact: High] A-1: Security Layer Disconnected from Serving Layer

- **Affected**: `security/*.go`, `serve/server.go`
- **Description**: The `security/` package contains production-quality building blocks (KeyStore, RateLimiter, IPFilter, CORSPolicy, TLSConfig, IncidentResponder) that are never wired into the default server. The gap between "security library" and "secure server" is the primary risk.
- **Recommendation**: Create `serve.WithSecurity(cfg)` that composes all security middleware. Make the CLI accept security flags or environment variables.

### [Impact: High] A-2: Unbounded Batch Path

- **Affected**: `inference/inference.go:416-428`
- **Description**: `GenerateBatch` accepts unbounded `[]string` prompts and spawns unbounded goroutines. The continuous batcher (`serve/batcher/`) has proper limits; the legacy batch path does not.
- **Recommendation**: Add `MaxBatchSize` enforcement to GenerateBatch.

### [Impact: Medium] A-3: Upward Dependency Violation

- **Affected**: `layers/attention/grouped_query_attention.go`
- **Description**: `layers/attention` imports `generate` for cache interfaces, violating the downward dependency flow.
- **Recommendation**: Extract Cache interface into `layers/attention` package; implement in `generate/`.

### [Impact: Medium] A-4: Direct CUDA Imports Bypass Engine[T]

- **Affected**: `layers/attention/grouped_query_attention.go`, `layers/attention/flash.go`, `layers/core/matmul_nbits_cuda.go`
- **Description**: Layer code directly imports `internal/cuda/kernels`, making these paths CUDA-specific.
- **Recommendation**: Route through Engine[T] capabilities via interface assertion.

### [Impact: Medium] A-5: God File -- timeseries/patchtst.go (1400 lines)

- **Affected**: `timeseries/patchtst.go`, also `timeseries/cfc.go` (759 lines), `timeseries/nhits.go` (744 lines)
- **Description**: Mixes model definition, training loop, save/load, and initialization.
- **Recommendation**: Split into `patchtst_model.go`, `patchtst_train.go`, `patchtst_io.go`.

### [Impact: Medium] A-6: Worker Pool Close() Data Race

- **Affected**: `internal/workerpool/pool.go:49-56`
- **Description**: `closed` bool checked/set without synchronization. Double-close panics.
- **Recommendation**: Use `sync.Once`.

### [Impact: Medium] A-7: No Request ID Correlation

- **Affected**: `serve/server.go`
- **Description**: No `X-Request-Id` header for correlating logs across client-server boundaries.
- **Recommendation**: Add request ID middleware.

### [Impact: Medium] A-8: Mixed stdlib/Structured Logging

- **Affected**: 7 files across `generate/`, `inference/`, `layers/`, `serve/`
- **Description**: stdlib `log` used instead of structured `ztensor/log`.
- **Recommendation**: Replace with structured logger.

### [Impact: Medium] A-9: CPU-Only Operations on GPU Tensors

- **Affected**: SDPA masking, MLA split, SSM scan, MSE loss, AdamW8bit, FP8Linear
- **Description**: Six subsystems call `.Data()` which triggers D2H copies, silently defeating GPU acceleration. The inference hot path correctly uses Engine primitives.
- **Recommendation**: Route through Engine[T] for all training/SSM paths.

---

## Functional Findings

### [Impact: High] F-1: Train CLI Is Non-Functional

- **Affected**: `cmd/cli/train.go:240-300`
- **Description**: `--config` GGUF path is parsed but never loaded. A synthetic 64-element model is used instead. `zerfoo train` does not actually train the specified model.
- **Recommendation**: Implement actual GGUF loading + LoRA fine-tuning, or clearly document as demo-only.

### [Impact: Medium] F-2: Streaming Chat Bypasses Chat Template

- **Affected**: `serve/server.go:766-769`
- **Description**: Streaming path concatenates messages with spaces instead of using the model's chat template. System prompts and role boundaries are lost.
- **Recommendation**: Use `model.formatMessages()` for the streaming prompt.

### [Impact: Medium] F-3: RegisterAlias Concurrent Map Race

- **Affected**: `inference/inference.go:72,84,92`
- **Description**: Package-level map written/read without synchronization.
- **Recommendation**: Use `sync.RWMutex` or `sync.Map`.

### [Impact: Medium] F-4: Non-Atomic Download Writes

- **Affected**: `registry/pull.go:210-229`
- **Description**: Downloads write directly to target path. Interrupted downloads leave partial files cached.
- **Recommendation**: Write to temp file, `os.Rename()` on success.

### [Impact: Medium] F-5: PatchTST Projection Head Averages Unrelated Variables

- **Affected**: `inference/timeseries/arch_patchtst.go:303-357`
- **Description**: `ReduceMean` over the numVars axis averages unrelated variable projections, degrading multivariate prediction quality.
- **Recommendation**: Use channel-independent projection: `[batch*numVars, d_model] @ [d_model, horizon]`.

### [Impact: Medium] F-6: Model Delete Does Not Drain In-Flight Requests

- **Affected**: `serve/server.go:649-650`
- **Description**: `model.Close()` called while concurrent requests may still be using the model.
- **Recommendation**: Add request counter with drain before close.

### [Impact: Medium] F-7: AdamW Has No Gradient Clipping or NaN Guard

- **Affected**: `training/optimizer/adamw.go:43-139`
- **Description**: No NaN/Inf detection before parameter updates. FP8 path has `CheckGradients()` but standard path does not.
- **Recommendation**: Add optional gradient norm clipping and NaN detection.

### [Impact: Medium] F-8: SDPA Causal Masking Modifies CPU Copy Only

- **Affected**: SDPA masking layer (discovered by layers/training agent)
- **Description**: Causal masking modifies a CPU copy of GPU tensor data, leaving GPU copy unmasked during prefill.
- **Recommendation**: Apply masking through Engine[T] to operate on the correct device.

### [Impact: Medium] F-9: S4 Backward Panic on First Call

- **Affected**: S4 layer backward pass
- **Description**: Accesses `Gradient.Data()` without nil check, causing panic on first backward call when gradients are uninitialized.
- **Recommendation**: Add nil check before accessing gradients.

### [Impact: Medium] F-10: LoRA Backward engine.Add(nil, dB)

- **Affected**: LoRA backward pass
- **Description**: Calls `engine.Add(nil, dB)` on first backward when gradients are uninitialized.
- **Recommendation**: Initialize gradients to zero before Add.

### [Impact: Medium] F-11: Flaky Test -- TestMAML_MetaConvergence

- **Affected**: `meta/meta_test.go:186`
- **Description**: Meta-loss increased instead of decreasing after 150 epochs. Non-deterministic.
- **Recommendation**: Set fixed seed or increase tolerance.

---

## Business-Critical Feature Traces

### 1. Model Inference (GGUF Load -> Generate -> Stream)

**Code path**: `cmd/cli/run.go:41` -> `inference.Load()` (`inference/inference.go:184`) -> `LoadFile()` (`inference/load_gguf.go:15`) -> `LoadGGUF()` -> `buildArchGraph()` (`inference/load_gguf.go:141`) -> arch builder (e.g., `inference/arch_llama.go:36`) -> `generate.Generator.Generate()` (`generate/generator.go:294`)

**Issues found**:
- F1-1: Embedding lookup bounds check missing (`arch_llama.go:288`)
- F1-2: RegisterAlias race condition (`inference.go:92`)

**Test coverage**: Parity tests exist but require external model files. No integration test with embedded test model.

### 2. API Serving (HTTP -> Inference -> Stream)

**Code path**: `cmd/cli/serve.go:45` -> `serve.NewServer()` (`serve/server.go:89`) -> `handleChatCompletions` (`serve/server.go:367`) -> `model.Chat()` / `streamChatCompletion()` -> SSE streaming

**Issues found**:
- C-1: No authentication (server.go:122)
- H-4: Server.unloaded data race (server.go:31)
- F2-2: Streaming writeError after headers sent (server.go:757)
- F2-3: Streaming chat loses template (server.go:766)
- F2-4: Model delete doesn't drain requests (server.go:649)

**Test coverage**: HTTP handler tests exist for request/response mechanics. Streaming template bypass not caught. No concurrent delete+inference test.

### 3. Model Pulling (Download + Cache)

**Code path**: `cmd/cli/pull.go:34` -> `registry.LocalRegistry.Pull()` (`registry/registry.go:78`) -> `pullFromHF()` (`registry/pull.go:71`) -> `downloadFile()` (pull.go:176)

**Issues found**:
- H-6: No checksum verification on HF downloads (pull.go:71)
- F-4: Non-atomic writes leave partial files (pull.go:210)

**Test coverage**: Mock HTTP server tests exist. Partial download recovery not tested. Checksum verification not tested (because it doesn't exist).

### 4. Training / Fine-Tuning

**Code path**: `cmd/cli/train.go:52` -> `trainLoop()` (train.go:300) -> synthetic model forward/backward

**Issues found**:
- F-1: Train CLI uses synthetic model, ignores --config GGUF (train.go:240-300)
- F-7: AdamW has no NaN guard (optimizer/adamw.go:43)

**Test coverage**: Training optimizer tests exist for convergence. End-to-end training test uses synthetic data only.

### 5. Time Series Inference

**Code path**: `inference/timeseries/gguf_loader.go` -> `LoadPatchTSTFromGGUF()` -> `BuildPatchTST()` (`arch_patchtst.go`)

**Issues found**:
- F-5: PatchTST projection averages unrelated variables (arch_patchtst.go:303)
- F5-2: normalizeWindows does not handle NaN input (timeseries/dlinear.go:60)

**Test coverage**: Recently improved (issue #121) with NaN detection and normalization fixes. Multi-scale divergence regression tests added.

---

## Positive Observations

1. **Clean modular architecture**: Clear package boundaries with downward dependency flow. Each package has a `doc.go`. The Engine[T] abstraction is consistently threaded through the compute path.

2. **Well-designed security primitives**: The `security/` package has production-quality API key management (SHA-256 hashing, scopes, expiry, rotation), rate limiting, IP filtering, TLS/mTLS, and AES-256-GCM encryption. The problem is integration, not implementation.

3. **Excellent Docker security**: Distroless nonroot base image, multi-stage build, CGO disabled, no shell in runtime. Best-in-class container security posture.

4. **Consistent functional options pattern**: `WithX()` options used consistently across Server, Generator, GQA, Dense, Gateway, and more.

5. **Proper resource cleanup**: 419 instances of `defer .Close()` across 117 files. HTTP response bodies, file handles, and gRPC connections properly managed.

6. **No hardcoded secrets**: No API keys, passwords, or tokens found in source code or committed config files. `.gitignore` excludes `.env`.

7. **Well-designed continuous batching**: `serve/batcher/scheduler.go` implements ragged batching with immediate slot eviction -- the right architecture for high-throughput serving.

8. **Proper Kubernetes integration**: Health checks with `/healthz` (liveness) and `/readyz` (readiness), Prometheus metrics, graceful shutdown coordinator with reverse-order close.

9. **Multi-cloud marketplace support**: AWS, Azure, and GCP marketplace integrations with metering, entitlements, and subscription management.

10. **Constant-time comparison used correctly**: `cloud/tenant.go:170` uses `subtle.ConstantTimeCompare` for API key validation. Azure webhook verification uses `hmac.Equal`.

---

## Statistics

- Files read: ~350 Go source files across 10 agents
- Lines of code analyzed: ~180,000 (estimated from 1321 files)
- Findings by severity: Critical **2**, High **11**, Medium **24**, Low **9**, Info **4**
- Architectural findings: **9**
- Functional findings: **11**

---

## Prioritized Remediation Roadmap

### 1. Fix Immediately (security critical, data loss risk)

| # | Finding | Files to Modify | Change |
|---|---------|----------------|--------|
| 1 | C-1: Wire authentication into serve.Server | `serve/server.go`, `cmd/cli/serve.go` | Add `WithAPIKey()` option and auth middleware; accept `--api-key`/`ZERFOO_API_KEY` |
| 2 | C-2: Replace X-Tenant-ID header with context | `cloud/server.go` | Use `context.WithValue` instead of `r.Header.Set("X-Tenant-ID", ...)` |
| 3 | H-1: Fix path traversal in FileSystemRepository | `serve/repository/repository.go` | Add containment validation to `modelDir()` |
| 4 | H-2: Block SSRF in vision image fetch | `serve/vision.go` | Add private IP blocking, DNS resolution check, timeout |
| 5 | H-4: Fix Server.unloaded data race | `serve/server.go` | Change `unloaded bool` to `unloaded atomic.Bool` |

### 2. Fix This Sprint (high-severity security, major correctness bugs)

| # | Finding | Files to Modify | Change |
|---|---------|----------------|--------|
| 6 | H-3: Add request body size limits | `serve/server.go` | Add `http.MaxBytesReader(w, r.Body, 10<<20)` to chat/completion/embedding handlers |
| 7 | H-5: Add embedding lookup bounds check | `inference/arch_llama.go` | Add `id < 0 || id >= vocabSize` check before index |
| 8 | H-6: Add download checksum verification | `registry/pull.go` | Compute SHA-256 during download, verify against HF API |
| 9 | H-9: Add TLS support to serve CLI | `cmd/cli/serve.go` | Add `--tls-cert`/`--tls-key` flags |
| 10 | H-10: Convert panics to error returns | `layers/core/*.go`, `layers/attention/attention_head.go` | Replace panic() with return fmt.Errorf() |
| 11 | H-11: Cap GenerateBatch concurrency | `inference/inference.go` | Add semaphore/errgroup.SetLimit() |
| 12 | F-3: Fix RegisterAlias race | `inference/inference.go` | Protect with sync.RWMutex |
| 13 | F-4: Atomic download writes | `registry/pull.go` | Write to temp file, os.Rename on success |
| 14 | F2-3: Fix streaming chat template | `serve/server.go` | Use model.formatMessages() for streaming prompt |

### 3. Fix This Quarter (architectural improvements, medium-severity)

| # | Finding | Files to Modify | Change |
|---|---------|----------------|--------|
| 15 | H-7: Add Helm securityContext | `deploy/helm/zerfoo/templates/deployment.yaml` | Add pod security context |
| 16 | H-8: Restrict Cloud Run access | `infra/terraform/zerfoo-cloud/cloud_run.tf` | Replace allUsers with authenticated service account |
| 17 | A-3: Fix upward dependency | `layers/attention/` | Extract Cache interface into layers/attention |
| 18 | A-9: Fix CPU-only ops on GPU tensors | 6 subsystems | Route through Engine[T] |
| 19 | INJ-1: Add GGUF integer overflow checks | `model/gguf/loader.go`, `model/gguf/parser.go` | Add dimension/count bounds |
| 20 | BIZ-1: Cap max_tokens server-side | `serve/server.go` | Enforce upper bound on max_tokens |
| 21 | DATA-1/2: Sanitize error messages | `serve/server.go` | Map internal errors to generic messages |
| 22 | DATA-4: Add security headers | `serve/server.go` | Add X-Content-Type-Options, Cache-Control middleware |
| 23 | A-7: Add request ID correlation | `serve/server.go` | Add X-Request-Id middleware |
| 24 | F-6: Drain in-flight requests on model delete | `serve/server.go` | Add request counter with drain |

### 4. Track as Tech Debt (low-severity, code quality)

| # | Finding | Files to Modify | Change |
|---|---------|----------------|--------|
| 25 | A-5: Split god files | `timeseries/patchtst.go` | Split into model/train/io files |
| 26 | A-6: Fix worker pool Close() race | `internal/workerpool/pool.go` | Use sync.Once |
| 27 | A-8: Standardize logging | 7 files | Replace stdlib log with structured logger |
| 28 | F-1: Document or fix train CLI | `cmd/cli/train.go` | Implement GGUF loading or document as demo |
| 29 | F-5: Fix PatchTST projection | `inference/timeseries/arch_patchtst.go` | Use channel-independent projection |
| 30 | F-7: Add gradient clipping to AdamW | `training/optimizer/adamw.go` | Add NaN detection and gradient norm clipping |
| 31 | F-11: Fix flaky MAML test | `meta/meta_test.go` | Set fixed seed or increase tolerance |
| 32 | CQ-2: Cache ZERFOO_DEBUG_ONNX check | `generate/generator.go` | Use cached bool instead of per-token os.Getenv |
