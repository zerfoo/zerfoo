# Security Audit: Injection & Input Validation Vulnerabilities

**Date:** 2026-03-21
**Scope:** /Users/dndungu/Code/zerfoo/zerfoo (zerfoo core framework)
**Auditor:** Automated security review

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 0     |
| High     | 2     |
| Medium   | 4     |
| Low      | 3     |
| Info     | 2     |

---

## Finding 1: Path Traversal in FileSystemRepository via Model ID

- **Severity:** High
- **File:** `serve/repository/repository.go:63-73`
- **Category:** PATH TRAVERSAL

**Description:** The `FileSystemRepository` uses the user-supplied model ID directly in `filepath.Join(r.baseDir, id)` without any sanitization or containment check. The model ID originates from HTTP requests in `serve/repository/handler.go` (lines 51, 102, 152) via `r.PathValue("id")` and JSON body `req.ID`. An attacker can use path traversal sequences (e.g., `../../etc/passwd`) to read metadata, write files, or delete directories outside the repository base directory.

Note: `registry.LocalRegistry.modelDir()` (registry/registry.go:167-184) correctly validates containment, but `FileSystemRepository` does not.

**Exploitation:**
- `GET /v1/models/../../etc` -- read metadata from arbitrary directories
- `POST /v1/models` with `{"id": "../../tmp/evil"}` -- write model file outside repo
- `DELETE /v1/models/../../important-data` -- `os.RemoveAll` on arbitrary directories (line 162)

**Affected code:**

```go
// serve/repository/repository.go
func (r *FileSystemRepository) modelDir(id string) string {
    return filepath.Join(r.baseDir, id)  // No sanitization!
}
```

**Fix:**

```diff
--- a/serve/repository/repository.go
+++ b/serve/repository/repository.go
@@ -60,8 +60,17 @@ func NewFileSystemRepository(baseDir string) (*FileSystemRepository, error) {
 	return &FileSystemRepository{baseDir: baseDir}, nil
 }

-func (r *FileSystemRepository) modelDir(id string) string {
-	return filepath.Join(r.baseDir, id)
+func (r *FileSystemRepository) modelDir(id string) (string, error) {
+	joined := filepath.Join(r.baseDir, id)
+	cleaned := filepath.Clean(joined)
+	basePrefix := filepath.Clean(r.baseDir) + string(filepath.Separator)
+	if !strings.HasPrefix(cleaned+string(filepath.Separator), basePrefix) {
+		return "", fmt.Errorf("repository: model ID %q resolves outside base directory", id)
+	}
+	return cleaned, nil
 }
```

All callers of `modelDir`, `modelPath`, and `metadataPath` must be updated to handle the error return. The `Delete` method at line 162 is especially dangerous since it calls `os.RemoveAll`.

---

## Finding 2: SSRF in Vision Image Fetch (serve/vision.go)

- **Severity:** High
- **File:** `serve/vision.go:125-146`
- **Category:** SSRF

**Description:** The `downloadImage` function fetches images from user-supplied URLs in chat completion requests (via `image_url` content parts). It uses `http.DefaultClient` with no restrictions on target host, port, or protocol scheme. An attacker can probe internal services, cloud metadata endpoints (e.g., `http://169.254.169.254/latest/meta-data/`), or private network resources.

Good: The function does validate the URL scheme (http/https only) and enforces a 20 MB size limit with `io.LimitReader`. But it has no protection against internal network access.

**Exploitation:**
```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}}
    ]
  }]
}
```

**Fix:**

```diff
--- a/serve/vision.go
+++ b/serve/vision.go
@@ -124,7 +124,22 @@ func downloadImage(ctx context.Context, url string) ([]byte, error) {
 	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
 	if err != nil {
 		return nil, fmt.Errorf("create request: %w", err)
 	}
-	resp, err := http.DefaultClient.Do(req)
+	// Block requests to private/internal networks to prevent SSRF.
+	host := req.URL.Hostname()
+	ip := net.ParseIP(host)
+	if ip != nil {
+		if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() {
+			return nil, fmt.Errorf("image URL targets a private network address")
+		}
+	}
+	// Also block well-known cloud metadata hostnames.
+	if host == "metadata.google.internal" || host == "169.254.169.254" {
+		return nil, fmt.Errorf("image URL targets a cloud metadata endpoint")
+	}
+	client := &http.Client{
+		Timeout: 30 * time.Second,
+	}
+	resp, err := client.Do(req)
```

Note: A comprehensive fix requires resolving the hostname to IP before connecting and checking the resolved IP, since DNS rebinding can bypass hostname-based checks. Consider using a custom `net.Dialer` with an IP validation hook in the transport.

---

## Finding 3: Integer Overflow in GGUF Tensor Size Calculation

- **Severity:** Medium
- **File:** `model/gguf/loader.go:24-28`
- **Category:** DESERIALIZATION

**Description:** The tensor element count is computed by multiplying dimensions from the GGUF file. Since dimensions are `uint64` values cast to `int`, a crafted GGUF file can cause integer overflow when dimensions are multiplied together (e.g., two dimensions of 2^32 each). This can result in allocating a small buffer and then reading out-of-bounds, or a massive allocation causing OOM.

```go
numElements := 1
for _, d := range ti.Dimensions {
    numElements *= int(d)  // No overflow check
}
```

There is no upper bound check on `tensorCount` (line 92-93 in parser.go) either -- a crafted file could claim billions of tensors, causing memory exhaustion via `make([]TensorInfo, tensorCount)`.

**Fix:**

```diff
--- a/model/gguf/loader.go
+++ b/model/gguf/loader.go
@@ -22,9 +22,17 @@ func LoadTensors(f *File, r io.ReadSeeker) (map[string]*tensor.TensorNumeric[flo

 		// Compute number of elements.
-		numElements := 1
+		numElements := int64(1)
 		for _, d := range ti.Dimensions {
-			numElements *= int(d)
+			if d > math.MaxInt32 {
+				return nil, fmt.Errorf("tensor %q: dimension %d exceeds maximum", ti.Name, d)
+			}
+			numElements *= int64(d)
+			if numElements > 1<<34 { // ~16 billion elements max (~64 GB at float32)
+				return nil, fmt.Errorf("tensor %q: total elements %d exceeds maximum", ti.Name, numElements)
+			}
 		}
```

Also add a limit on `tensorCount` in parser.go:

```diff
--- a/model/gguf/parser.go
+++ b/model/gguf/parser.go
@@ -93,6 +93,12 @@ func Parse(r io.ReadSeeker) (*File, error) {
 	if err := binary.Read(r, binary.LittleEndian, &metadataKVCount); err != nil {
 		return nil, fmt.Errorf("read metadata kv count: %w", err)
 	}
+	if tensorCount > 100_000 {
+		return nil, fmt.Errorf("tensor count %d exceeds maximum (100000)", tensorCount)
+	}
+	if metadataKVCount > 1_000_000 {
+		return nil, fmt.Errorf("metadata kv count %d exceeds maximum (1000000)", metadataKVCount)
+	}
```

---

## Finding 4: Unbounded Read in OCI Registry Blob Download

- **Severity:** Medium
- **File:** `registry/oci.go:343`
- **Category:** DESERIALIZATION

**Description:** The `getBlob` function uses `io.ReadAll(resp.Body)` with no size limit when downloading blobs from an OCI registry. A malicious or compromised registry could return an arbitrarily large response, causing OOM. This blob data is also then passed to `os.WriteFile` (line 199).

```go
func (r *Registry) getBlob(ctx context.Context, repo, digest string) ([]byte, error) {
    // ...
    return io.ReadAll(resp.Body)  // No size limit
}
```

**Fix:**

```diff
--- a/registry/oci.go
+++ b/registry/oci.go
@@ -340,7 +340,12 @@ func (r *Registry) getBlob(ctx context.Context, repo, digest string) ([]byte, er
 		return nil, fmt.Errorf("blob GET returned status %d", resp.StatusCode)
 	}

-	return io.ReadAll(resp.Body)
+	// Limit blob size to 20 GB to prevent OOM from malicious registries.
+	const maxBlobSize = 20 << 30
+	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBlobSize+1))
+	if err != nil {
+		return nil, err
+	}
+	if int64(len(data)) > maxBlobSize {
+		return nil, fmt.Errorf("blob exceeds maximum size of %d bytes", maxBlobSize)
+	}
+	return data, nil
 }
```

---

## Finding 5: Error Message Injection in Support API

- **Severity:** Medium
- **File:** `support/api.go:165`
- **Category:** INJECTION (Response Splitting / JSON Injection)

**Description:** The `CloseTicket` handler concatenates an error message directly into a JSON string literal without escaping:

```go
http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusConflict)
```

If `err.Error()` contains double quotes or control characters (which is possible if the error message includes user-controlled data), this breaks the JSON structure. While this is JSON (not HTML), it could lead to JSON injection or confusing clients. The other handlers in this file correctly use `json.NewEncoder`.

**Fix:**

```diff
--- a/support/api.go
+++ b/support/api.go
@@ -162,7 +162,10 @@ func (a *API) CloseTicket(w http.ResponseWriter, r *http.Request) {

 	now := time.Now().UTC()
 	if err := ticket.Transition(StatusClosed, now); err != nil {
-		http.Error(w, `{"error":"`+err.Error()+`"}`, http.StatusConflict)
+		w.Header().Set("Content-Type", "application/json")
+		w.WriteHeader(http.StatusConflict)
+		resp := map[string]string{"error": err.Error()}
+		json.NewEncoder(w).Encode(resp)
 		return
 	}
```

---

## Finding 6: Missing Timeout on http.DefaultClient in Vision Fetch

- **Severity:** Medium
- **File:** `serve/vision.go:130`
- **Category:** DENIAL OF SERVICE

**Description:** The `downloadImage` function uses `http.DefaultClient` which has no timeout configured. A slow-responding server could hold the connection indefinitely, tying up a goroutine and potentially exhausting server resources.

**Fix:** Use a dedicated client with a timeout (see Finding 2 fix which also addresses this).

---

## Finding 7: No Path Traversal Check in FileSystemRepository.Delete

- **Severity:** Low (subset of Finding 1)
- **File:** `serve/repository/repository.go:154-163`
- **Category:** PATH TRAVERSAL

**Description:** `Delete(id)` calls `os.RemoveAll(dir)` where `dir` is `filepath.Join(r.baseDir, id)` with no sanitization. This is the most dangerous consequence of Finding 1, as it enables arbitrary directory deletion via `DELETE /v1/models/../../target`.

**Fix:** Addressed by Finding 1's fix.

---

## Finding 8: OCI Registry URL Injection via Repository Name

- **Severity:** Low
- **File:** `registry/oci.go:208, 248, 276, 301, 325`
- **Category:** SSRF / URL INJECTION

**Description:** The OCI registry methods construct URLs by interpolating user-provided `repo`, `tag`, and `digest` values into URL paths via `fmt.Sprintf`:

```go
url := fmt.Sprintf("%s/v2/%s/tags/list", r.url, repo)
url := fmt.Sprintf("%s/v2/%s/manifests/%s", r.url, parsed.Repository, identifier)
```

While the registry URL is developer-configured (not user input), the `repo` and `identifier` values come from parsed OCI references. Characters like `../` in the repository name could manipulate the URL path. The `parseReference` function does not validate against path traversal in repository names.

**Fix:**

```diff
--- a/registry/oci.go
+++ b/registry/oci.go
@@ -94,6 +94,9 @@ func parseReference(ref string) (Reference, error) {
 	if r.Repository == "" {
 		return Reference{}, fmt.Errorf("invalid reference %q: empty repository", ref)
 	}
+	if strings.Contains(r.Repository, "..") {
+		return Reference{}, fmt.Errorf("invalid reference %q: repository contains path traversal", ref)
+	}
 	if r.Tag == "" && r.Digest == "" {
 		r.Tag = "latest"
 	}
```

---

## Finding 9: Diagnostic Log Leaks in GGUF FP8 Quantization

- **Severity:** Low
- **File:** `model/gguf/loader.go:317-321`
- **Category:** INFORMATION DISCLOSURE

**Description:** The `QuantizeToFP8E4M3` function uses `log.Printf` (standard library logger) to output tensor names, shapes, element counts, and scale factors. In production, this leaks model architecture details to stdout/stderr.

```go
log.Printf("[FP8 diag] QuantizeToFP8E4M3: tensor=%q shape=%v elems=%d scale=%.6g ...",
    name, t.Shape(), len(f32), scale, f32Min, f32Max)
```

**Fix:** Replace with structured logging through the framework's `log.Logger` interface, or remove diagnostic logging behind a debug flag.

---

## Finding 10: No HMAC Signature on Webhook Payloads

- **Severity:** Info
- **File:** `support/webhook.go:84-117`
- **Category:** AUTHENTICATION

**Description:** The `WebhookDispatcher.Dispatch` sends JSON payloads to webhook targets without any HMAC signature or shared secret verification. Recipients cannot verify that webhook events are authentic.

**Fix:** Add an HMAC-SHA256 signature header (e.g., `X-Zerfoo-Signature`) computed from a per-target shared secret.

---

## Finding 11: No Authentication on Repository Upload/Delete Endpoints

- **Severity:** Info
- **File:** `serve/repository/handler.go:22-27`
- **Category:** ACCESS CONTROL

**Description:** The model repository endpoints (`POST /v1/models`, `DELETE /v1/models/{id}`) have no authentication middleware. Anyone with network access can upload or delete models. This is noted as informational since authentication may be handled at a higher layer (reverse proxy, middleware).

---

## Negative Findings (Areas Reviewed with No Issues)

1. **Command Injection:** All `exec.Command` usages use hardcoded command names (`git`, `nvcc`, `go`, `ollama`, `gomobile`). The `nvcc` path in `internal/codegen/compile.go` is resolved from known filesystem paths or `PATH` lookup -- not user input. Arguments are passed as separate array elements, preventing shell injection.

2. **SQL/NoSQL Injection:** No SQL or NoSQL database usage found. The `cloud/` and `support/` packages use in-memory stores with Go maps. The `.Query()` methods found are custom in-memory query methods, not database queries.

3. **Template Injection:** The only template usage is in `docsite/site/generate.go` which uses `html/template` (auto-escaping) with embedded templates and markdown content processed through goldmark. No user input reaches templates at runtime.

4. **HuggingFace Download Path Traversal:** The `registry/pull.go:downloadFile` function correctly validates that downloaded filenames do not escape the target directory (lines 195-205) using both `..` checks and prefix containment validation.

5. **LocalRegistry Path Traversal:** The `registry/registry.go:modelDir` function (lines 167-184) correctly validates containment within the cache directory.

6. **GGUF Parser String Limits:** The parser correctly limits string lengths to 1 MB (parser.go:174) and array lengths to 1M elements (parser.go:241).

7. **Audio Upload Size Limit:** `serve/audio.go` correctly limits audio file size with `io.LimitReader`.

8. **unsafe.Pointer Usage:** All `unsafe.Pointer` usage is in GPU binding code (`internal/cuda/`, `internal/cudnn/`, `internal/hip/`, etc.) for FFI calls via purego. These are necessary for the CGo-free GPU binding design and operate on GPU-allocated memory, not user-controlled buffers. No exploitable patterns found.
