# Security Audit: serve/ Package (E106 Remediation)

**Date:** 2026-03-23
**Scope:** serve/server.go, serve/vision.go, serve/audio.go, serve/classify.go, serve/server_test.go, serve/request_id_test.go, serve/repository/, security/network.go, security/apikey.go
**Verdict:** E106 fixes are largely correct with a few findings requiring attention.

---

## 1. Auth Middleware (authMiddleware, WithAPIKey)

**Implementation:** server.go:231-252

**Assessment: PASS with one medium finding**

- **Constant-time comparison:** Uses `subtle.ConstantTimeCompare` -- correct, prevents timing attacks.
- **Bypass paths:** Exempts `/metrics`, `/healthz`, `/readyz`, `/openapi.yaml` -- appropriate for health/observability endpoints.
- **Header parsing:** Correctly requires `Bearer ` prefix and extracts token after it.

**Finding M1 (Medium): Auth bypass via path prefix manipulation**
The auth middleware uses `r.URL.Path` with exact `switch` matching, which is correct and NOT vulnerable to prefix/suffix attacks. However, if a reverse proxy normalizes paths differently (e.g., `//metrics` or `/metrics/`), the switch would fall through to the `default` case and correctly require auth. This is safe.

**Finding L1 (Low): No healthz/readyz handlers registered**
The auth middleware exempts `/healthz` and `/readyz`, but no handlers are registered for these paths in `NewServer`. Requests to these paths will get a 404 from the mux. Not a security issue, but a completeness gap.

**Tests:** Adequate -- covers no key, wrong key, correct key, and exempt paths (/metrics, /openapi.yaml). Missing: test for /healthz and /readyz exemption.

---

## 2. SSRF Protection (validateImageURL)

**Implementation:** vision.go:53-100

**Assessment: PASS with two medium findings**

- **DNS resolution:** Resolves hostname and checks each IP -- good.
- **Blocked categories:** Loopback, private (RFC 1918), link-local unicast, link-local multicast, cloud metadata IPs/hostnames -- comprehensive.
- **Redirect limit:** Set to 3 via `CheckRedirect` -- good.
- **Image size limit:** `io.LimitReader(resp.Body, maxImageSize+1)` with post-read check -- correct.

**Finding M2 (Medium): DNS rebinding not fully mitigated**
`validateImageURL` resolves the hostname and validates IPs BEFORE making the HTTP request. Between DNS validation and the actual `imageHTTPClient.Do(req)` call, the DNS record could change (TOCTOU). An attacker could set up a DNS server that returns a public IP on the first resolution (passing validation) and then a private/metadata IP on the second resolution (used by the HTTP client).

Mitigation: Use a custom `net.Dialer` in `imageHTTPClient` that validates the resolved IP at connection time, or pin the resolved IP for the request.

**Finding M3 (Medium): IPv6-mapped IPv4 addresses not explicitly blocked**
`net.IP.IsPrivate()`, `IsLoopback()`, etc. handle IPv6-mapped IPv4 (e.g., `::ffff:127.0.0.1`) correctly in Go's net package since these methods check both IPv4 and IPv6 representations. Verified safe.

**Finding L2 (Low): Scheme validation missing**
`validateImageURL` parses the URL but does not verify that the scheme is `http` or `https`. The `fetchImageData` function does check the scheme prefix before calling `downloadImage`, so non-HTTP schemes are rejected at that layer. The defense-in-depth is adequate but could be tighter.

**Finding L3 (Low): No Azure IMDS hostname blocked**
`blockedSSRFHosts` includes `metadata.google.internal` but not Azure's metadata endpoint hostname. The IP `169.254.169.254` covers both AWS and Azure, so the IP-based check is sufficient, but adding `metadata.azure.com` would be defense-in-depth.

**Tests:** Good coverage -- loopback, metadata IP, metadata hostname, private address, public URL (with SSRF bypass for local test server).

---

## 3. Path Traversal (repository.go modelDir)

**Implementation:** repository.go:66-73

**Assessment: PASS**

- Uses `filepath.Clean` + `filepath.Join` + prefix check with separator -- correct pattern.
- Rejects `joined == base` (prevents bare `.` from passing).
- Checks `strings.HasPrefix(joined, base+string(filepath.Separator))` -- prevents prefix collision attacks (e.g., `/tmp/models-evil` matching `/tmp/models`).

**Finding: No issues found**
- Encoding bypasses: `filepath.Clean` handles `%2f`, `..`, `.` correctly.
- Symlink attacks: Not mitigated at the filesystem level (a symlink inside `baseDir` could point outside), but this requires the attacker to have already written a symlink inside the repo directory, which presupposes filesystem access. Acceptable risk for the model repository use case.

**Tests:** Excellent -- covers `../../etc`, deep escape, `.`, `..`, normal ID, nested path. HTTP-level tests verify URL-encoded traversal (`..%2F..%2Fetc%2Fpasswd`). All CRUD operations tested for traversal rejection.

---

## 4. Request Body Size Limits (MaxBytesReader)

**Implementation:** server.go:526, 699, 847 (10 MB each); audio.go:38 (25 MB + 1024)

**Assessment: FAIL -- one endpoint missing**

**Finding H1 (HIGH): handleClassify missing MaxBytesReader**
`handleClassify` (classify.go:68-132) decodes the request body directly with `json.NewDecoder(r.Body).Decode(&req)` without first wrapping `r.Body` in `http.MaxBytesReader`. An attacker can send an arbitrarily large request body to `/v1/classify`, causing unbounded memory allocation.

The `maxClassifyBatch = 256` check on line 85 limits the number of array elements but not the size of each string element or the overall body size.

**Covered endpoints:**
- `/v1/chat/completions` -- 10 MB limit (line 526)
- `/v1/completions` -- 10 MB limit (line 699)
- `/v1/embeddings` -- 10 MB limit (line 847)
- `/v1/audio/transcriptions` -- ~25 MB limit (line 38)
- Repository upload -- 10 GB limit (handler.go:86)

**Missing endpoint:**
- `/v1/classify` -- **NO LIMIT**

**Tests:** Body size limit test at line 1667 covers chat/completions, completions, and embeddings but NOT classify.

---

## 5. Server.unloaded atomic.Bool

**Implementation:** server.go:37, 791, 806, 823, 835

**Assessment: PASS with one low finding**

- `unloaded` is an `atomic.Bool`, ensuring lock-free thread-safe reads/writes.
- `handleModels` (line 791), `handleModelInfo` (line 806), and `handleModelDelete` (line 820) all check `s.unloaded.Load()` before proceeding.
- `handleModelDelete` sets `s.unloaded.Store(true)` then calls `s.inflight.Wait()` then `s.model.Close()` -- correct ordering.

**Finding L4 (Low): TOCTOU in handleModelDelete**
Two concurrent DELETE requests for the same model could both pass the `s.unloaded.Load()` check on line 823 before either sets `Store(true)` on line 835. The second caller would then call `s.model.Close()` a second time. This is mitigated if `model.Close()` is idempotent, but double-close is a code smell.

A `CompareAndSwap(false, true)` pattern would be cleaner:
```go
if !s.unloaded.CompareAndSwap(false, true) {
    writeError(w, http.StatusNotFound, "model not found")
    return
}
```

**Tests:** Adequate -- TestHandleModelDelete, TestHandleModelDelete_NotFound, TestHandleModelDelete_AlreadyDeleted.

---

## 6. Security Headers Middleware

**Implementation:** server.go:186-193

**Assessment: PASS**

- Sets `X-Content-Type-Options: nosniff` -- prevents MIME sniffing.
- Sets `X-Frame-Options: DENY` -- prevents clickjacking.
- Sets `Cache-Control: no-store` -- prevents caching of sensitive responses.
- Applied as the outermost wrapper in `Handler()` (line 182), so it wraps all responses.

**Finding L5 (Low): Missing Strict-Transport-Security header**
If the server is deployed behind TLS (which the CLAUDE.md mentions as supported via TLS/mTLS), an HSTS header would be appropriate. However, omitting it is reasonable since the server may also be used over plain HTTP in development.

**Tests:** TestSecurityHeaders verifies all three headers on a `/v1/models` response.

---

## 7. Request ID Middleware

**Implementation:** server.go:207-227

**Assessment: PASS**

- Uses `crypto/rand.Read` for UUID generation -- cryptographically secure.
- Correctly sets UUID v4 version bits (0x40) and variant bits (0x80).
- Echoes client-provided `X-Request-Id` if present, generates one if absent.
- Stored in context via `requestIDKey{}` unexported type -- prevents external key collision.

**Finding L6 (Low): Client-provided request IDs not validated**
The middleware accepts any string as a request ID from the client. A malicious client could send very long strings or strings with control characters. This is low risk since the ID is only used in logging and response headers, but validation (e.g., max 256 chars, printable ASCII) would be defense-in-depth.

**Tests:** TestRequestIDEchoed and TestRequestIDGenerated cover both paths. Generated IDs are validated against a UUID v4 regex pattern.

---

## 8. Error Sanitization (sanitizeError)

**Implementation:** server.go:362-373

**Assessment: PASS with one medium finding**

- Maps OOM errors to "server temporarily overloaded, please retry".
- Maps file-not-found errors to "model not available".
- All other errors map to generic "inference failed".
- Logs the full error server-side before sanitizing.

**Finding M4 (Medium): handleClassify leaks unsanitized errors**
In classify.go:94, the error from `s.classifier.Classify()` is passed directly to `writeError`:
```go
writeError(w, inferenceErrorStatus(err), err.Error())
```
This bypasses `sanitizeError` and could leak internal details (file paths, stack traces, etc.) to the client. All other inference handlers (chat, completions, embeddings) correctly use `s.sanitizeError(err)`.

**Finding L7 (Low): recoveryMiddleware logs panic message to stderr**
Line 323 writes the panic message to stderr via `fmt.Fprintf`. The panic message could contain sensitive information. However, the HTTP response only sends "internal server error" (line 324), so no leakage to the client.

**Tests:** No dedicated tests for sanitizeError output. The error path tests (TestHandleChatCompletions_GenerateError, etc.) verify the status code but do not verify that the response body contains only sanitized messages.

---

## 9. Rate Limiting (rateLimitMiddleware)

**Implementation:** server.go:256-265, security/network.go:14-75

**Assessment: PASS with one medium finding**

- Token-bucket algorithm is correctly implemented with refill and burst.
- Per-IP keying is correct.
- Cleanup method prevents unbounded memory growth (though no background goroutine is visible to call it periodically).

**Finding M5 (Medium): ClientIP trusts X-Forwarded-For without proxy validation**
`security.ClientIP` (network.go:167-183) trusts `X-Forwarded-For` and `X-Real-IP` headers directly. An attacker without a reverse proxy can spoof these headers to:
1. **Bypass rate limiting** by rotating spoofed IPs.
2. **Attribute requests to other IPs** in logs.

This is a common pattern that requires the operator to strip/overwrite these headers at the reverse proxy layer. The code should document this assumption or provide an option to ignore forwarded headers.

**Finding L8 (Low): No rate limiter cleanup goroutine**
`RateLimiter.Cleanup()` exists but is never called automatically. Over time, the `buckets` map will grow unboundedly if clients use many different IPs. This is a memory leak, not a security vulnerability per se, but could be used for resource exhaustion.

**Tests:** TestRateLimitMiddleware verifies burst=2 with rate=0 correctly allows 2 and rejects the 3rd. Adequate for correctness but does not test IP extraction.

---

## 10. Max Tokens Cap

**Implementation:** server.go:555-559 (chat), 716-719 (completions)

**Assessment: PASS**

- Default cap is 8192, configurable via `WithMaxTokens`.
- Clamping logic: `if req.MaxTokens != nil && *req.MaxTokens > s.maxTokens` -- correct.
- Only clamps when the client provides a value; if `max_tokens` is omitted (nil), no clamping occurs, which is correct (the model's own limit applies).

**Tests:** TestMaxTokensClamp covers default clamping, custom limit, and within-limit (not clamped). Adequate.

---

## 11. In-Flight Request Draining

**Implementation:** server.go:523-524 (chat), 696-697 (completions), 835-837 (delete)

**Assessment: PASS with one observation**

- `s.inflight.Add(1)` / `defer s.inflight.Done()` brackets inference handlers.
- `handleModelDelete` calls `s.unloaded.Store(true)` then `s.inflight.Wait()` -- correct ordering.
- After `Wait()`, calls `s.model.Close()` -- correct.

**Observation:** The `handleEmbeddings` handler (line 846) does NOT call `s.inflight.Add(1)` / `s.inflight.Done()`. If a model is deleted while an embedding request is in-flight, the embedding request could race with `s.model.Close()`. Same for `handleClassify` (classify.go:68). This means the draining mechanism does not cover all inference endpoints.

---

## Summary of Findings

| ID | Severity | Component | Description |
|----|----------|-----------|-------------|
| H1 | HIGH | classify.go | Missing `MaxBytesReader` -- unbounded request body |
| M2 | MEDIUM | vision.go | DNS rebinding TOCTOU in SSRF validation |
| M4 | MEDIUM | classify.go | Unsanitized error messages leaked to client |
| M5 | MEDIUM | security/network.go | `ClientIP` trusts spoofable `X-Forwarded-For` |
| L1 | LOW | server.go | No /healthz or /readyz handlers registered |
| L4 | LOW | server.go | TOCTOU in concurrent model delete (double close) |
| L5 | LOW | server.go | Missing HSTS header |
| L6 | LOW | server.go | Client-provided request IDs not validated |
| L7 | LOW | server.go | Panic message logged to stderr (not leaked to client) |
| L8 | LOW | security/network.go | Rate limiter cleanup never auto-invoked |
| -- | INFO | server.go | handleEmbeddings and handleClassify missing inflight tracking |

## Recommended Immediate Actions

1. **H1:** Add `r.Body = http.MaxBytesReader(w, r.Body, 10<<20)` to `handleClassify`.
2. **M4:** Change classify.go:94 from `err.Error()` to `s.sanitizeError(err)`.
3. **M2:** Consider using a custom `net.Dialer` with IP validation in `imageHTTPClient.Transport` to prevent DNS rebinding.
4. **M5:** Document that `X-Forwarded-For` must be set by a trusted proxy, or add a `WithTrustedProxies` option.
5. Add `s.inflight.Add(1)` / `defer s.inflight.Done()` to `handleEmbeddings` and `handleClassify`.
6. Use `CompareAndSwap` in `handleModelDelete` to prevent double-close.
