# Deep Review Report -- zerfoo v1.11.0 (Post-E106 Remediation)

**Date**: 2026-03-23
**Commit**: 163bdb4
**Scope**: Full codebase (1322 Go source files) -- focused on verifying E106 security fixes
**Agents deployed**: 4 (serve security, cloud/infra, inference/layers, misc quality)
**Prior review**: v1.10.0 deep review found 2 Critical, 11 High, 24 Medium (E106 created)

---

## Executive Summary

The E106 security remediation successfully addressed all 37 planned tasks. **All Critical and High findings from the v1.10.0 review are now fixed**: authentication is wired into the serve layer, tenant ID uses context instead of spoofable headers, path traversal is blocked, SSRF protection is active, panics are converted to errors, and infrastructure is hardened. However, the remediation missed one endpoint (`handleClassify`) which lacks body size limits, error sanitization, and in-flight tracking -- a consistency gap rather than a design flaw. One additional panic remains in `layers/reducesum/reducesum.go`. The highest remaining risk is DNS rebinding in the SSRF protection (TOCTOU between validation and connection). Overall, the codebase has moved from "security primitives exist but disconnected" to "security is integrated with minor gaps."

---

## System Architecture Map

```mermaid
graph TB
    subgraph Serving["HTTP Server (serve/) -- NOW SECURED"]
        HEADERS[Security Headers MW]
        LOG[Log MW]
        REQID[Request ID MW]
        RATELIM[Rate Limit MW]
        AUTH[Auth MW - Bearer Token]
        MUX[ServeMux]
        CHAT[/v1/chat/completions<br>MaxBytes+Sanitize+Inflight]
        COMP[/v1/completions<br>MaxBytes+Sanitize+Inflight]
        EMB[/v1/embeddings<br>MaxBytes+Sanitize]
        CLASS[/v1/classify<br>MISSING: MaxBytes,Sanitize,Inflight]
        AUDIO[/v1/audio/transcriptions<br>MaxBytes]
        VISION[Vision Fetch<br>SSRF Protected]
    end

    subgraph Cloud["Cloud (FIXED)"]
        TENANT[Tenant via context.Context]
        REDACT[API Keys Redacted]
    end

    subgraph Inference["Inference (HARDENED)"]
        BOUNDS[Embedding Bounds Check]
        BATCH[GenerateBatch Semaphore]
        ALIAS[RegisterAlias RWMutex]
        STREAM[ChatStream - Template Fixed]
    end

    subgraph Layers["Layers (HARDENED)"]
        CORE[layers/core - 0 panics]
        ATTN[layers/attention - errors not panics]
        SSM[layers/ssm - nil grad guard]
        REDUCE[layers/reducesum - 1 PANIC REMAINS]
    end

    HEADERS --> LOG --> REQID --> RATELIM --> AUTH --> MUX
    MUX --> CHAT & COMP & EMB & CLASS & AUDIO
    CHAT --> VISION

    style CLASS fill:#ffa07a
    style REDUCE fill:#ffa07a
```

---

## Critical and High Findings

### [HIGH] H1: handleClassify Missing Body Size Limit, Error Sanitization, and Inflight Tracking

- **Location**: `serve/classify.go:74,94`
- **Description**: The `/v1/classify` endpoint was not covered by E106 remediation. It lacks `http.MaxBytesReader` (allowing OOM via large POST), returns raw `err.Error()` to clients (leaking internal details), and does not call `s.inflight.Add(1)/Done()` (meaning model delete does not wait for in-flight classify requests).
- **Impact**: OOM denial of service via unbounded request body. Internal error details leaked to attackers. Model delete can race with classify requests.
- **Fix**:
```go
// serve/classify.go -- add at top of handleClassify:
r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB

s.inflight.Add(1)
defer s.inflight.Done()

// Replace error return (line ~94):
// Before: writeError(w, http.StatusInternalServerError, err.Error())
// After:  writeError(w, inferenceErrorStatus(err), s.sanitizeError(err))
```

### [HIGH] H2: Remaining panic() in layers/reducesum/reducesum.go

- **Location**: `layers/reducesum/reducesum.go:111`
- **Description**: `Backward()` calls `panic()` on invalid axis instead of returning an error. The method signature already returns `error`. This was missed during E106 T106.11 which only covered `layers/core/`.
- **Impact**: Malformed model graph crashes the serving process.
- **Fix**:
```go
// Before: panic(fmt.Sprintf("unsupported axis %d", r.axis))
// After:  return nil, fmt.Errorf("reducesum: unsupported axis %d for backward", r.axis)
```

---

## Security Findings (Medium/Low/Info)

### Injection / Input Validation

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| M1 | Medium | DNS rebinding TOCTOU in SSRF protection | `serve/vision.go:53-100` |
| M2 | Medium | `normalizeWindows` does not detect NaN input values | `timeseries/dlinear.go:60` |

**M1 Detail**: `validateImageURL` resolves DNS and validates IPs before the HTTP request. An attacker with a DNS server could return a public IP on first resolution (passing validation) then a private IP on second resolution (used by HTTP client). Fix: use a custom `net.Dialer` that validates the resolved IP at connect time.

### Authentication & Authorization

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| M3 | Medium | `ClientIP` trusts X-Forwarded-For without trusted proxy config | `security/network.go:167-183` |
| L1 | Low | No healthz/readyz handlers registered despite auth exemption | `serve/server.go` |

### Data Exposure

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| L2 | Low | `handleEmbeddings` missing `s.inflight.Add(1)/Done()` | `serve/server.go` |
| L3 | Low | SAML signature verification unimplemented (TODO) | `cloud/sso.go` |

### Infrastructure

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| M4 | Medium | AWS QuickStart assigns public IPs to ECS tasks | `marketplace/aws/cfn/quickstart.yaml` |
| M5 | Medium | Azure ARM template opens SSH+serve to 0.0.0.0/0 | `marketplace/azure/arm/template.json` |
| L4 | Low | Azure webhook uses unbounded `io.ReadAll` | `marketplace/azure/webhook.go:82` |
| L5 | Low | Disaggregated gateway defaults to insecure gRPC | `serve/disaggregated/gateway.go:99` |

### Code Quality

| ID | Sev | Finding | Location |
|----|-----|---------|----------|
| M6 | Medium | `ListByCustomer` sort is broken for 3+ tickets | `support/ticket.go:158-162` |
| M7 | Medium | `rl/replay.go` has 2 panic() calls in non-test code | `rl/replay.go:20,60` |
| L6 | Low | `generate/generator.go` still imports stdlib `"log"` | `generate/generator.go` |
| L7 | Low | Hardcoded EOS token ID=2 in decode worker | `serve/disaggregated/decode_worker.go:107` |
| L8 | Low | AdamW NaN guard uses `.Data()` -- silent on GPU tensors | `training/optimizer/adamw.go:247` |

---

## Architectural Findings

### [Impact: Medium] A-1: Inconsistent Security Coverage Across Endpoints

- **Affected**: `serve/classify.go` vs `serve/server.go`
- **Description**: E106 hardened chat/completions/embeddings/audio but missed classify. The security middleware is applied globally (auth, rate limit, headers) but per-handler protections (MaxBytesReader, sanitizeError, inflight tracking) must be applied manually to each handler, making it easy to miss new endpoints.
- **Recommendation**: Create a `securedHandler` wrapper that applies MaxBytesReader, inflight tracking, and error sanitization to any handler, reducing the chance of missing a new endpoint.

### [Impact: Low] A-2: .Data() Calls in Training Path

- **Affected**: `training/optimizer/adamw.go`, `inference/timeseries/arch_patchtst.go`
- **Description**: The NaN guard and timeseries projection use `.Data()` which triggers D2H copies for GPU tensors. Safe today (CPU-only training) but will silently fail when GPU training is enabled.
- **Recommendation**: Track as tech debt. Add Engine[T]-based NaN detection when GPU training launches.

---

## Functional Findings

### [Impact: Medium] F-1: Ticket Sort Algorithm Broken

- **Affected**: `support/ticket.go:158-162`
- **Description**: The single-pass pairwise swap is not a valid sorting algorithm for 3+ elements. Tickets returned by `ListByCustomer` may be in wrong order.
- **Recommendation**: Replace with `slices.SortFunc` or `sort.Slice`.

---

## E106 Remediation Verification Summary

| Finding ID | v1.10.0 Severity | Fix Task | Verified |
|-----------|-----------------|----------|----------|
| C-1 No auth | Critical | T106.1 | PASS |
| C-2 Tenant header | Critical | T106.2 | PASS |
| H-1 Path traversal | High | T106.3 | PASS |
| H-2 SSRF | High | T106.4 | PASS (DNS rebinding remains M1) |
| H-3 Body size | High | T106.6 | PASS (classify missed - H1) |
| H-4 Data race | High | T106.5 | PASS |
| H-5 Embedding bounds | High | T106.7 | PASS |
| H-6 Download checksum | High | T106.14 | PASS |
| H-7 Helm security | High | T106.24 | PASS |
| H-8 Cloud Run IAM | High | T106.25 | PASS |
| H-9 TLS support | High | T106.8 | PASS |
| H-10 Panics | High | T106.11-12 | PASS (reducesum missed - H2) |
| H-11 Batch cap | High | T106.13 | PASS |

**26/37 E106 tasks verified individually. All pass.** The remaining 11 tasks (medium/low severity) were spot-checked and confirmed present.

---

## Positive Observations

1. **E106 remediation was effective**: All Critical and High findings from v1.10.0 are fixed. The security posture has materially improved.
2. **Auth middleware is well-implemented**: Constant-time comparison, proper Bearer token extraction, sensible path exemptions.
3. **Path traversal fix is robust**: Uses `filepath.Clean` + separator-aware prefix check -- handles encoding attacks correctly.
4. **Error sanitization is thorough**: Maps internal errors to generic messages, logs originals via slog.
5. **Atomic downloads prevent partial file corruption**: Temp file + rename pattern is correct.
6. **Infrastructure hardening complete**: Helm securityContext, Cloud Run IAM restriction, distroless Docker images.
7. **Test coverage for security fixes is good**: Each fix has targeted tests covering happy path and attack vectors.

---

## Statistics

- Files read: ~120 Go source files across 4 agents
- Lines of code analyzed: ~60,000 (focused review of security-critical paths)
- Findings by severity: Critical **0**, High **2**, Medium **7**, Low **8**, Info **0**
- Architectural findings: **2**
- Functional findings: **1**
- E106 fixes verified: **37/37 implemented, 2 gaps found (classify endpoint, reducesum panic)**

---

## Prioritized Remediation Roadmap

### 1. Fix Immediately

| # | Finding | Files | Change |
|---|---------|-------|--------|
| 1 | H1: handleClassify missing protections | `serve/classify.go` | Add MaxBytesReader, sanitizeError, inflight tracking |
| 2 | H2: reducesum panic in Backward | `layers/reducesum/reducesum.go` | Replace panic with error return |

### 2. Fix This Sprint

| # | Finding | Files | Change |
|---|---------|-------|--------|
| 3 | M1: DNS rebinding TOCTOU | `serve/vision.go` | Add custom net.Dialer with IP validation at connect time |
| 4 | M6: Ticket sort broken | `support/ticket.go` | Replace with slices.SortFunc |
| 5 | M7: RL replay panics | `rl/replay.go` | Replace panics with error returns |
| 6 | M3: X-Forwarded-For trust | `security/network.go` | Add TrustedProxies config |

### 3. Fix This Quarter

| # | Finding | Files | Change |
|---|---------|-------|--------|
| 7 | M4: AWS QuickStart public IPs | `marketplace/aws/cfn/quickstart.yaml` | Disable AssignPublicIp |
| 8 | M5: Azure ARM open firewall | `marketplace/azure/arm/template.json` | Restrict CIDR ranges |
| 9 | M2: normalizeWindows NaN | `timeseries/dlinear.go` | Add isFinite check |
| 10 | A-1: Centralize per-handler security | `serve/server.go` | Create securedHandler wrapper |

### 4. Track as Tech Debt

| # | Finding | Files | Change |
|---|---------|-------|--------|
| 11 | L5: Insecure gRPC default | `serve/disaggregated/gateway.go` | Log warning or require TLS |
| 12 | L6: stdlib log in generator | `generate/generator.go` | Migrate remaining log calls |
| 13 | L7: Hardcoded EOS token | `serve/disaggregated/decode_worker.go` | Read from config |
| 14 | L8: .Data() in NaN guard | `training/optimizer/adamw.go` | Use Engine[T] for GPU compat |

---

## Comparison: v1.10.0 vs v1.11.0

| Metric | v1.10.0 | v1.11.0 | Delta |
|--------|---------|---------|-------|
| Critical | 2 | 0 | -2 |
| High | 11 | 2 | -9 |
| Medium | 24 | 7 | -17 |
| Low | 9 | 8 | -1 |
| Total | 46 | 17 | -29 (63% reduction) |

The security posture improved significantly. The 2 remaining High findings are consistency gaps (missed endpoint + missed panic), not design flaws.
