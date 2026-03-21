# Security Audit: Authentication, Authorization & Session Management

**Date:** 2026-03-21
**Scope:** `/Users/dndungu/Code/zerfoo/zerfoo` -- serve/, cloud/, security/, config/, registry/, distributed/, cmd/cli/
**Auditor:** Claude (automated)

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH     | 5 |
| MEDIUM   | 5 |
| LOW      | 3 |
| INFO     | 2 |

---

## CRITICAL Findings

### C-1: Core API Server Has Zero Authentication

**Severity:** CRITICAL
**Files:** `serve/server.go:122-132`, `cmd/cli/serve.go:108-112`
**Description:** The main `serve.Server` registers all endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `DELETE /v1/models/{id}`, `/v1/audio/transcriptions`, `/v1/classify`) with no authentication middleware whatsoever. The `cmd/cli/serve.go` serve command starts an HTTP server with `srv.Handler()` which only applies `logMiddleware` -- no auth check.

The `security` package has a `KeyStore` with proper API key hashing (SHA-256, `zf_` prefix, scoping, expiry, rotation) and the `serve/cloud/tenant.go` has a `TenantRegistry.Middleware()` with Bearer token extraction -- but **neither is wired into the core server**. They exist as standalone building blocks that are never composed into the default serving path.

**Exploitation:** Anyone with network access to the server port (default 8080) can:
- Run arbitrary inference (resource exhaustion / cost abuse)
- Delete loaded models via `DELETE /v1/models/{id}` (denial of service)
- Upload models via the repository handler (if registered)
- Exfiltrate embeddings of sensitive text

**Fix:**
```go
// In serve/server.go, add an auth option and middleware:

// WithAPIKey enables Bearer token authentication.
func WithAPIKey(key string) ServerOption {
    return func(s *Server) { s.apiKey = key }
}

// In Handler():
func (s *Server) Handler() http.Handler {
    var h http.Handler = s.mux
    if s.apiKey != "" {
        h = s.authMiddleware(h)
    }
    return s.logMiddleware(h)
}

func (s *Server) authMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        // Skip auth for health/metrics endpoints
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

// In cmd/cli/serve.go, accept --api-key flag or ZERFOO_API_KEY env var
```

---

### C-2: Tenant ID Passed Via Spoofable HTTP Header (Header Injection)

**Severity:** CRITICAL
**File:** `cloud/server.go:84,92,112`
**Description:** The `CloudServer.authMiddleware` authenticates the Bearer token, resolves the tenant, then passes the tenant ID to downstream middleware via an internal HTTP header `X-Tenant-ID`:
```go
r.Header.Set("X-Tenant-ID", tenant.ID)
```

The `rateLimitMiddleware` and `billingMiddleware` then read this header to look up the tenant. However, an attacker can set `X-Tenant-ID` in their request *before* the auth middleware runs. While the auth middleware does overwrite it for authenticated requests, there is a logic gap: if the `authMiddleware` is bypassed or reordered (e.g., `/healthz` is excluded from auth at line 52), the downstream middleware will trust the client-supplied header.

More critically, if someone registers additional routes on the mux that skip `authMiddleware` but still go through `rateLimitMiddleware`/`billingMiddleware`, the attacker controls the tenant ID -- billing fraud, rate limit bypass, or impersonation of another tenant.

**Exploitation:** Craft a request with `X-Tenant-ID: victim-tenant-id` to consume the victim's token budget or attribute billing to them.

**Fix:** Use `context.Context` to pass the tenant (like `serve/cloud/tenant.go` already does with `TenantFromContext`), not HTTP headers. The `cloud/server.go` should use the same pattern:
```go
// In authMiddleware, replace:
r.Header.Set("X-Tenant-ID", tenant.ID)
// With:
ctx := context.WithValue(r.Context(), tenantKey{}, tenant)
r = r.WithContext(ctx)

// In rateLimitMiddleware/billingMiddleware, replace:
tenantID := r.Header.Get("X-Tenant-ID")
// With:
tenant := tenantFromContext(r.Context())
```

---

## HIGH Findings

### H-1: Server Listens on Plain HTTP by Default (No TLS)

**Severity:** HIGH
**File:** `cmd/cli/serve.go:121-123`
**Description:** The serve command uses `httpServer.ListenAndServe()` (plain HTTP). There is no `--tls-cert` / `--tls-key` flag, no TLS configuration, and no way to enable HTTPS from the CLI. The `security.TLSConfig` and `security.BuildTLSConfig()` exist but are never used by the serve command.

API keys, Bearer tokens, and inference data are transmitted in plaintext.

**Exploitation:** Network eavesdropping captures API keys, prompts, and completions.

**Fix:**
```go
// Add TLS flags to ServeCommand
case "--tls-cert":
    tlsCert = args[i+1]; i++
case "--tls-key":
    tlsKey = args[i+1]; i++

// Use ListenAndServeTLS when cert/key provided
if tlsCert != "" && tlsKey != "" {
    errCh <- httpServer.ListenAndServeTLS(tlsCert, tlsKey)
} else {
    errCh <- httpServer.ListenAndServe()
}
```

---

### H-2: Model Delete Endpoint Has No Authorization Check

**Severity:** HIGH
**File:** `serve/server.go:635-657`
**Description:** `DELETE /v1/models/{id}` unloads the model (`s.model.Close()`) and sets `s.unloaded = true`, making the entire server non-functional. There is no authentication or authorization on this destructive endpoint. Any unauthenticated client can permanently disable the server.

**Exploitation:** `curl -X DELETE http://target:8080/v1/models/gemma-3-1b` -- server becomes a 404 factory.

**Fix:** Gate destructive endpoints behind admin-scope authentication. At minimum, require a Bearer token with `ScopeAdmin`.

---

### H-3: Repository Upload Handler Has No Authentication

**Severity:** HIGH
**File:** `serve/repository/handler.go:22-27`
**Description:** The `Handler.RegisterRoutes()` registers `POST /v1/models` (file upload, up to 10 GB) and `DELETE /v1/models/{id}` with no authentication middleware. Anyone can upload arbitrary files or delete models.

**Exploitation:** Upload a malicious model file; delete production models.

**Fix:** Wrap repository routes with authentication middleware before registering.

---

### H-4: SSRF via Vision Image Fetch

**Severity:** HIGH
**File:** `serve/vision.go:104-106,125-146`
**Description:** The `fetchImageData` function accepts arbitrary `http://` and `https://` URLs from user-supplied `image_url` fields in chat completion requests. It uses `http.DefaultClient` to fetch them with no URL validation, no allowlist, and no protection against internal network access.

**Exploitation:** An attacker can probe internal services:
```json
{"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}}
```
This enables cloud metadata theft, internal service scanning, and port probing.

**Fix:**
```go
func downloadImage(ctx context.Context, url string) ([]byte, error) {
    parsed, err := neturl.Parse(url)
    if err != nil {
        return nil, err
    }
    // Block private/internal IPs
    ips, err := net.DefaultResolver.LookupHost(ctx, parsed.Hostname())
    if err != nil {
        return nil, err
    }
    for _, ip := range ips {
        if isPrivateIP(net.ParseIP(ip)) {
            return nil, fmt.Errorf("blocked: URL resolves to private IP")
        }
    }
    // Use a dedicated client with timeout and redirect limits
    client := &http.Client{Timeout: 10 * time.Second, CheckRedirect: maxRedirects(3)}
    // ... rest of fetch
}
```

---

### H-5: No Rate Limiting on Core Server Endpoints

**Severity:** HIGH
**File:** `serve/server.go:122-132`
**Description:** The core `serve.Server` has no rate limiting on any endpoint. While `security.RateLimiter` (token-bucket, per-IP) exists and `serve/cloud/tenant.go` has per-tenant rate limits, neither is applied to the default server. Each inference request can be GPU-expensive (seconds of compute).

**Exploitation:** Flood the server with inference requests to exhaust GPU resources (DoS). A single client can monopolize the GPU by sending large-prompt requests in parallel.

**Fix:** Wire `security.RateLimiter` as middleware on the default server, or expose a `WithRateLimiter` option.

---

## MEDIUM Findings

### M-1: Distributed Worker gRPC Uses Insecure Credentials by Default

**Severity:** MEDIUM
**File:** `serve/disaggregated/gateway.go:18-19`, `distributed/tlsconfig.go:26-29`
**Description:** The disaggregated gateway imports `grpc/credentials/insecure` and connects to workers with plaintext gRPC. The `distributed.TLSConfig` returns `nil` when unconfigured, meaning plaintext is the default path. The worker command (`cmd/cli/worker.go`) has no TLS flags.

Gradient data, model weights, and KV cache blocks are transferred in plaintext between nodes.

**Exploitation:** Network sniffing on the cluster network captures model data and training gradients.

**Fix:** Add `--tls-cert`, `--tls-key`, `--tls-ca` flags to both serve and worker commands. Default to requiring TLS in production (allow plaintext only with explicit `--insecure` flag).

---

### M-2: OCI Registry Credentials Stored In-Memory in Plaintext

**Severity:** MEDIUM
**File:** `registry/oci.go:17-23,28-33`
**Description:** The OCI `Registry` struct stores `username` and `password` as plain strings. These are set via `WithCredentials()` and persist in memory for the lifetime of the client. They are sent over HTTP Basic Auth on every request -- if the registry URL is HTTP (not HTTPS), credentials are transmitted in cleartext.

**Exploitation:** Memory dump or core dump reveals registry credentials. Non-TLS registry connections leak credentials.

**Fix:**
- Validate that the registry URL uses HTTPS (or require explicit opt-in for HTTP).
- Consider using an OAuth2/token-based flow instead of storing raw passwords.

---

### M-3: serve/cloud/tenant.go Stores API Keys as Plaintext Map Keys

**Severity:** MEDIUM
**File:** `serve/cloud/tenant.go:98,120,127`
**Description:** The `TenantRegistry` in `serve/cloud/tenant.go` maps raw API keys to tenants (`tenants map[string]*Tenant`). API keys are stored as plaintext map keys and compared directly. A memory dump would reveal all tenant API keys.

Contrast with the well-designed `cloud/tenant.go` `TenantManager.GetByAPIKey()` which uses `subtle.ConstantTimeCompare` -- but still stores the raw key in `Tenant.APIKey`.

**Exploitation:** Memory forensics or heap inspection reveals all API keys.

**Fix:** Store SHA-256 hashes of API keys (like `security.KeyStore` does) instead of raw values. Use constant-time comparison on the hashes.

---

### M-4: Missing Security Headers on API Responses

**Severity:** MEDIUM
**File:** `serve/server.go:830-834`
**Description:** The `writeJSON` helper sets `Content-Type` but no security headers. Missing headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Strict-Transport-Security` (when TLS is enabled)
- `Cache-Control: no-store` on sensitive responses (completions contain user data)

**Exploitation:** Content sniffing attacks; cached responses on shared proxies leak user prompts/completions.

**Fix:**
```go
func (s *Server) securityHeadersMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("X-Content-Type-Options", "nosniff")
        w.Header().Set("X-Frame-Options", "DENY")
        w.Header().Set("Cache-Control", "no-store")
        next.ServeHTTP(w, r)
    })
}
```

---

### M-5: CORS Wildcard Origin Allowed

**Severity:** MEDIUM
**File:** `security/network.go:149`
**Description:** The `CORSPolicy.Middleware` supports `"*"` as an allowed origin (`origins["*"]`). If configured with `AllowedOrigins: ["*"]`, any website can make cross-origin requests to the API, potentially exfiltrating inference results via a victim's browser.

**Exploitation:** Attacker website makes fetch() calls to the victim's locally-running zerfoo server at `http://localhost:8080`, extracting model outputs.

**Fix:** Warn or reject `"*"` when authentication is enabled. Document that wildcard CORS should only be used with public, read-only endpoints.

---

## LOW Findings

### L-1: IP Spoofing via X-Forwarded-For

**Severity:** LOW
**File:** `security/network.go:167-183`
**Description:** `ClientIP()` trusts `X-Forwarded-For` and `X-Real-IP` headers from any client. Without a trusted proxy configuration, attackers can spoof their IP to bypass IP-based rate limiting and IP allowlists/denylists.

**Fix:** Add a `TrustedProxies` configuration. Only trust `X-Forwarded-For` when the direct connection comes from a known proxy IP.

---

### L-2: Streaming Error Messages Leak Internal Details

**Severity:** LOW
**File:** `serve/server.go:789,823`
**Description:** In streaming mode, errors are sent as SSE events with the raw error message:
```go
fmt.Fprintf(w, "data: {\"error\": %q}\n\n", err.Error())
```
Internal error messages (e.g., CUDA errors, file paths, memory addresses) may leak server internals.

**Fix:** Sanitize error messages before sending to clients. Map internal errors to generic messages.

---

### L-3: No Request Body Size Limits on Inference Endpoints

**Severity:** LOW
**File:** `serve/server.go:367,526`
**Description:** `handleChatCompletions` and `handleCompletions` use `json.NewDecoder(r.Body).Decode()` with no body size limit. An attacker could send a multi-GB JSON payload to exhaust server memory.

The repository handler correctly uses `http.MaxBytesReader` (line 82) but the inference endpoints do not.

**Fix:**
```go
r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10 MB limit
```

---

## INFO Findings

### I-1: Well-Designed Security Primitives Exist But Are Not Integrated

**Severity:** INFO
**Description:** The `security/` package contains production-quality building blocks:
- `KeyStore`: SHA-256 hashed API keys with scopes, expiry, rotation
- `RateLimiter`: Per-IP token-bucket rate limiter
- `IPFilter`: Allowlist/denylist
- `CORSPolicy`: CORS middleware
- `IncidentResponder`: Automatic IP lockout
- `TLSConfig`: TLS/mTLS configuration
- `Encrypt`/`Decrypt`: AES-256-GCM encryption
- `SecretConfig`: Environment variable secret loader

These are well-tested but disconnected from the actual serving layer. The gap between "security library" and "secure server" is the primary risk.

---

### I-2: Constant-Time Comparison Used Correctly in Cloud Tenant Manager

**Severity:** INFO
**File:** `cloud/tenant.go:166-176`
**Description:** `TenantManager.GetByAPIKey()` correctly uses `subtle.ConstantTimeCompare` to prevent timing side-channel attacks on API key lookup. This is a positive finding. However, the `serve/cloud/tenant.go` `TenantRegistry` does a plain map lookup by raw key, which is vulnerable to timing attacks.

---

## Recommendations (Priority Order)

1. **Wire authentication into `serve.Server`** -- Add `WithAPIKey()` or `WithKeyStore()` options that install auth middleware. Make the CLI accept `--api-key` or `ZERFOO_API_KEY`.

2. **Replace `X-Tenant-ID` header passing with context values** in `cloud/server.go` to eliminate the header injection vector.

3. **Add TLS support to `cmd/cli/serve.go`** -- Accept `--tls-cert`/`--tls-key` flags and use `ListenAndServeTLS`.

4. **Add SSRF protection to `serve/vision.go`** -- Block private IP ranges, limit redirects, add timeout.

5. **Wire rate limiting into the default server** -- Apply `security.RateLimiter` as middleware.

6. **Hash API keys in `serve/cloud/tenant.go`** -- Use the same SHA-256 pattern as `security.KeyStore`.

7. **Add security headers middleware** and request body size limits.

8. **Add `--tls-*` flags to `cmd/cli/worker.go`** for distributed training security.
