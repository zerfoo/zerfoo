# Security Audit: Cryptography, Data Exposure & Dependencies

**Date:** 2026-03-21
**Scope:** `/Users/dndungu/Code/zerfoo/zerfoo`
**Auditor:** Automated security review

---

## 1. CRYPTOGRAPHY

### 1.1 Encryption (security/encryption.go)

**Status: PASS** -- AES-256-GCM with crypto/rand nonces, 32-byte key requirement enforced by `aes.NewCipher`. No weak algorithms, no hardcoded keys.

### 1.2 TLS Configuration (security/encryption.go, distributed/tlsconfig.go)

**Status: PASS** -- Both TLS implementations default to TLS 1.2 minimum, support mTLS with CA verification, and use `tls.RequireAndVerifyClientCert` when CA is provided.

### 1.3 API Key Management (security/apikey.go)

**FINDING SEC-01: API Key Lookup Uses Map-Based Hash Comparison (Non-Constant-Time)**

- **Severity:** LOW
- **File:** `security/apikey.go:106-112`
- **Description:** `KeyStore.Lookup()` does SHA-256 hash of the raw key then performs a map lookup by the hex-encoded hash. While the SHA-256 step adds computational noise, the map lookup itself is not constant-time. Contrast with `cloud/tenant.go:170` which correctly uses `subtle.ConstantTimeCompare` for API key comparison.
- **Exploitation:** Timing side-channel attack against the hash lookup. Practically very difficult to exploit due to SHA-256 pre-hashing, but inconsistent with the constant-time approach used in `cloud/tenant.go`.
- **Fix:**
  ```go
  // In security/apikey.go, Lookup method:
  func (s *KeyStore) Lookup(rawKey string) *APIKey {
      h := sha256.Sum256([]byte(rawKey))
      hash := hex.EncodeToString(h[:])

      s.mu.RLock()
      defer s.mu.RUnlock()

      // Constant-time scan to prevent timing attacks
      for _, k := range s.keys {
          if subtle.ConstantTimeCompare([]byte(k.Hash), []byte(hash)) == 1 {
              return k
          }
      }
      return nil
  }
  ```

### 1.4 Provenance (provenance/provenance.go)

**Status: PASS** -- Uses SHA-256 hash chains for integrity verification. Deterministic canonicalization of maps via sorted keys. No signing (HMAC/RSA) -- this is integrity-only, not authenticity. Acceptable for the current use case (internal training provenance), but if provenance records are used for compliance or external audit, consider adding digital signatures.

### 1.5 Azure Webhook Signature Verification (marketplace/azure/webhook.go)

**Status: PASS** -- HMAC-SHA256 with `hmac.Equal` for constant-time comparison. Properly rejects empty signatures.

---

## 2. DATA EXPOSURE

### FINDING SEC-02: No Request Body Size Limit on Chat/Completion/Embedding Endpoints

- **Severity:** HIGH
- **File:** `serve/server.go:367`, `serve/server.go:526`, `serve/server.go:661`
- **Description:** The `handleChatCompletions`, `handleCompletions`, and `handleEmbeddings` handlers call `json.NewDecoder(r.Body).Decode()` without any `http.MaxBytesReader` or `io.LimitReader` wrapper. An attacker can send an arbitrarily large JSON body, consuming unbounded server memory. By contrast, `handleAudioTranscriptions` correctly applies `http.MaxBytesReader`.
- **Exploitation:** Send a multi-GB POST body to `/v1/chat/completions` to OOM the server.
- **Fix:**
  ```go
  // Add at the top of each handler, before json.Decode:
  const maxRequestBodySize = 10 * 1024 * 1024 // 10 MB
  r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodySize)
  ```

### FINDING SEC-03: Inference Error Messages Leaked to Clients

- **Severity:** MEDIUM
- **File:** `serve/server.go:456-458`, `serve/server.go:564-566`, `serve/server.go:694-696`, `serve/audio.go:75`
- **Description:** When inference fails, the raw `err.Error()` string is returned to the client. Internal error messages may reveal model architecture details, file paths, GPU memory states, or internal library names (e.g., "cuda: out of memory at offset 0x...").
- **Exploitation:** An attacker can trigger error conditions (e.g., OOM via large prompt) and collect internal diagnostics from error messages.
- **Fix:**
  ```go
  func sanitizeInferenceError(err error) string {
      if isOOMError(err) {
          return "server is temporarily overloaded, please retry"
      }
      return "inference failed"
  }
  // Replace: writeError(w, inferenceErrorStatus(err), err.Error())
  // With:    writeError(w, inferenceErrorStatus(err), sanitizeInferenceError(err))
  ```

### FINDING SEC-04: Streaming Error Messages Leaked to Clients

- **Severity:** MEDIUM
- **File:** `serve/server.go:789`, `serve/server.go:822`
- **Description:** During SSE streaming, if generation fails mid-stream, the raw error is sent as `data: {"error": "<raw error>"}`. This can leak internal state to the client.
- **Exploitation:** Same as SEC-03 but via streaming responses.
- **Fix:** Sanitize the error before including it in the SSE data frame.

### FINDING SEC-05: Tenant API Key Exposed in Config Snapshot

- **Severity:** MEDIUM
- **File:** `cloud/tenant.go:29`, `cloud/tenant.go:52-58`, `cloud/tenant.go:211-219`
- **Description:** `TenantConfig` includes `APIKey string` with `json:"api_key"`. The `Config()` method returns the raw API key, and `List()` returns all tenant configs including raw API keys. If any admin endpoint serializes these to JSON, all tenant API keys are exposed.
- **Exploitation:** An admin API call to list tenants returns every tenant's raw API key.
- **Fix:** Remove `APIKey` from `TenantConfig` or redact it in `Config()`:
  ```go
  func (t *Tenant) Config() TenantConfig {
      return TenantConfig{
          ID:          t.ID,
          APIKey:      "", // never expose raw key
          RateLimit:   t.rateLimit.Load(),
          TokenBudget: t.tokenBudget.Load(),
      }
  }
  ```

### 2.1 Logging

**Status: PASS** -- The `logMiddleware` in `serve/server.go:155-186` logs method, path, model ID, token counts, latency, and status code. No sensitive data (API keys, request bodies, user content) is logged. The `recoveryMiddleware` logs only the panic message, not a full stack trace. Good practice.

### 2.2 Prometheus Metrics (serve/metrics.go)

**Status: PASS** -- Metrics expose only aggregate operational data: request counts, token throughput, latency histograms. No per-request or per-user data is exposed. No model weights or internal state accessible via `/metrics`.

### 2.3 Model Weight Extraction via API

**Status: PASS** -- The embedding endpoint returns computed embeddings, not raw model weights. No endpoints expose raw tensor data. `DELETE /v1/models/:id` only unloads; does not return model data.

---

## 3. DEPENDENCY AUDIT

### 3.1 Direct Dependencies (go.mod)

| Dependency | Version | Assessment |
|-----------|---------|------------|
| `go.etcd.io/bbolt` | v1.4.3 | Current, no known CVEs |
| `golang.org/x/image` | v0.37.0 | Current |
| `gonum.org/v1/gonum` | v0.17.0 | Current |
| `google.golang.org/grpc` | v1.65.0 | Current for its release line |
| `google.golang.org/protobuf` | v1.36.8 | Current |
| `github.com/google/go-cmp` | v0.7.0 | Current |
| `golang.org/x/net` | v0.25.0 | Check for HTTP/2 CVEs |
| `golang.org/x/sys` | v0.29.0 | Current |
| `golang.org/x/text` | v0.35.0 | Current |

### FINDING SEC-06: genproto Replace Directive May Prevent Security Updates

- **Severity:** LOW
- **File:** `go.mod:32`
- **Description:** `replace google.golang.org/genproto => google.golang.org/genproto v0.0.0-20240528184218-531527333157` pins genproto to a specific commit. Combined with the `exclude` directives, this prevents automatic updates. If a security fix is published, it would not be picked up.
- **Fix:** Periodically review whether the replace directive is still needed; if the underlying version conflict is resolved, remove it.

### 3.2 Vendored Code

**Status: PASS** -- No `vendor/` directory present. All dependencies managed via Go modules.

### 3.3 Transitive Dependencies

Notable transitive dependencies (all current): envoy control-plane, opencensus, protobuf. No known CVEs in the listed versions at time of review.

**Recommendation:** Run `govulncheck ./...` regularly in CI to catch newly disclosed vulnerabilities.

---

## 4. INFRASTRUCTURE SECURITY

### FINDING SEC-07: Helm Deployment Template Missing Pod SecurityContext

- **Severity:** HIGH
- **File:** `deploy/helm/zerfoo/templates/deployment.yaml` (entire file)
- **Description:** The Helm deployment template has no `securityContext` at either the pod or container level. The enterprise deployment docs (`docs/enterprise-deployment.md:986-997`) explicitly recommend `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, and `allowPrivilegeEscalation: false`, but these are not implemented in the actual template.
- **Exploitation:** If the container image has a vulnerability, the attacker runs as root with full filesystem write access, making privilege escalation trivial.
- **Fix:** Add to `deploy/helm/zerfoo/templates/deployment.yaml` under the container spec:
  ```yaml
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    readOnlyRootFilesystem: true
    allowPrivilegeEscalation: false
    capabilities:
      drop:
        - ALL
  ```
  And add corresponding values to `deploy/helm/zerfoo/values.yaml`.

### FINDING SEC-08: Cloud Run API Gateway Publicly Accessible to allUsers

- **Severity:** HIGH
- **File:** `infra/terraform/zerfoo-cloud/cloud_run.tf:63-69`
- **Description:** The Cloud Run service IAM binding grants `roles/run.invoker` to `allUsers`, making the API gateway publicly accessible without authentication. This is the inference API, which means anyone on the internet can send inference requests, consuming GPU resources.
- **Exploitation:** Unauthenticated access to inference endpoints enables resource exhaustion (token generation costs compute and GPU time) and data exfiltration if models contain proprietary training data.
- **Fix:** Remove the `allUsers` binding and use IAP, API Gateway, or Cloud Endpoints for authentication:
  ```hcl
  # Replace allUsers with authenticated access
  resource "google_cloud_run_v2_service_iam_member" "authenticated_access" {
    project  = google_cloud_run_v2_service.api_gateway.project
    location = google_cloud_run_v2_service.api_gateway.location
    name     = google_cloud_run_v2_service.api_gateway.name
    role     = "roles/run.invoker"
    member   = "serviceAccount:${google_service_account.api_invoker.email}"
  }
  ```

### 4.1 AWS CloudFormation (deploy/aws/cloudformation.yaml)

**Status: PASS (with notes)**
- ECS tasks use Fargate (no direct host access)
- Tasks have private IPs only (`AssignPublicIp: DISABLED`)
- Security groups properly scoped (ALB -> task only on 8080)
- HTTP->HTTPS redirect when certificate is provided
- IAM policies appropriately scoped (Marketplace metering + CloudWatch only)
- Note: IAM policies use `Resource: '*'` for metering and CloudWatch, which is standard for these services as they don't support resource-level permissions

### 4.2 Dockerfile (deploy/aws/Dockerfile)

**Status: PASS** -- Uses distroless nonroot base image (`gcr.io/distroless/static-debian12:nonroot`), multi-stage build, CGO disabled, no shell available in runtime image.

### 4.3 GCS Storage (infra/terraform/zerfoo-cloud/storage.tf)

**Status: PASS** -- Uniform bucket-level access enabled, versioning enabled, Workload Identity for pod access (no static credentials), least-privilege `roles/storage.objectViewer` for model reader.

---

## 5. BUSINESS LOGIC

### FINDING SEC-09: No MaxTokens Upper Bound on Inference Requests

- **Severity:** MEDIUM
- **File:** `serve/server.go:398-400`
- **Description:** The `max_tokens` parameter from the client request is passed directly to the inference engine without any server-side upper bound. A client can request `max_tokens: 1000000` to force extended GPU computation.
- **Exploitation:** Resource exhaustion via arbitrarily large token generation requests. Even with rate limiting at the cloud/tenant level, a single request can monopolize GPU resources for an extended period.
- **Fix:**
  ```go
  const serverMaxTokens = 8192 // or model-specific context length

  if req.MaxTokens != nil {
      n := *req.MaxTokens
      if n > serverMaxTokens {
          n = serverMaxTokens
      }
      opts = append(opts, inference.WithMaxTokens(n))
  }
  ```

### FINDING SEC-10: Vision Endpoint SSRF via Image URL Fetching

- **Severity:** MEDIUM
- **File:** `serve/vision.go:125-146`
- **Description:** `downloadImage` uses `http.DefaultClient` to fetch arbitrary HTTP(S) URLs provided in the request. There is no validation of the URL target, allowing Server-Side Request Forgery (SSRF). An attacker can use the server to probe internal networks, access cloud metadata endpoints (e.g., `http://169.254.169.254/latest/meta-data/`), or access internal services.
- **Exploitation:** Send a chat completion request with an image URL pointing to `http://169.254.169.254/latest/meta-data/iam/security-credentials/` to extract IAM credentials from the instance metadata service.
- **Fix:**
  ```go
  import "net"

  func isBlockedHost(host string) bool {
      ips, err := net.LookupIP(host)
      if err != nil {
          return true // block on resolution failure
      }
      for _, ip := range ips {
          if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() {
              return true
          }
      }
      return false
  }

  // In downloadImage, before http.DefaultClient.Do:
  parsedURL, err := url.Parse(rawURL)
  if err != nil || isBlockedHost(parsedURL.Hostname()) {
      return nil, fmt.Errorf("blocked URL")
  }
  ```

### 5.1 Batch Scheduler (serve/batch.go)

**Status: PASS** -- `MaxBatchSize` defaults to 8 with a configurable cap. Pending channel is buffered at `MaxBatchSize*4`. Cancelled requests are filtered before execution.

### 5.2 Tenant Rate Limiting (cloud/tenant.go)

**Status: PASS** -- Per-minute rate limiting on both request count and token budget with atomic operations for concurrent safety.

---

## Summary

| ID | Severity | Category | Description |
|----|----------|----------|-------------|
| SEC-01 | LOW | Crypto | API key lookup not constant-time |
| SEC-02 | HIGH | Data Exposure | No request body size limit on main endpoints |
| SEC-03 | MEDIUM | Data Exposure | Raw inference errors leaked to clients |
| SEC-04 | MEDIUM | Data Exposure | Raw errors in SSE streaming |
| SEC-05 | MEDIUM | Data Exposure | Tenant API keys exposed via Config/List |
| SEC-06 | LOW | Dependencies | genproto replace directive may block updates |
| SEC-07 | HIGH | Infrastructure | Helm template missing pod securityContext |
| SEC-08 | HIGH | Infrastructure | Cloud Run API gateway open to allUsers |
| SEC-09 | MEDIUM | Business Logic | No server-side max_tokens upper bound |
| SEC-10 | MEDIUM | Business Logic | SSRF via vision image URL fetching |

**HIGH:** 3 findings (SEC-02, SEC-07, SEC-08)
**MEDIUM:** 5 findings (SEC-03, SEC-04, SEC-05, SEC-09, SEC-10)
**LOW:** 2 findings (SEC-01, SEC-06)
