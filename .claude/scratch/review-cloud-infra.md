# E106 Security Review: cloud/, registry/, infra/, deploy/, security/, marketplace/

**Date:** 2026-03-23
**Scope:** Verify E106 fixes and audit remaining security posture

---

## 1. E106 Fix Verification

### 1.1 cloud/server.go -- Tenant ID via context.Context (PASS)

The `X-Tenant-ID` header pattern has been fully removed. Tenant identity now flows via `context.Context`:

- **Line 14**: Private `tenantKey struct{}` type prevents external key collisions.
- **Line 18-21**: `tenantFromContext()` extracts tenant from context.
- **Line 95-96**: `authMiddleware` stores tenant via `context.WithValue(r.Context(), tenantKey{}, tenant)`.
- **Lines 104, 123**: `rateLimitMiddleware` and `billingMiddleware` read tenant via `tenantFromContext(r.Context())`.
- **Grep confirms**: No occurrence of `X-Tenant-ID` in any `.go` source file (only in test assertions that verify the header is absent, and in old scratch docs).

**Verdict:** Fix is complete and correct. No header-based tenant ID remains.

### 1.2 cloud/tenant.go -- API Key Redaction (PASS with one caveat)

- **Line 52-63**: `Config()` returns `redactedAPIKey` ("***redacted***") instead of the raw key.
- **Line 215-224**: `List()` delegates to `Config()`, so all listed tenants have redacted keys.
- **GetByAPIKey()** (line 170-179): Uses `subtle.ConstantTimeCompare` -- good timing-attack mitigation.

**Caveat (LOW):** `Get()` (line 157-165) and `GetByAPIKey()` return a `*Tenant` pointer with the raw `APIKey` field publicly accessible. Any code that receives a `*Tenant` can read `tenant.APIKey` directly. This is an internal API, but the `APIKey` field on the `Tenant` struct being exported is a latent leak path. Consider making `APIKey` unexported or adding a private accessor.

**Verdict:** Config()/List() redaction is correct. Raw key on the Tenant struct is exported but acceptable for internal use.

### 1.3 registry/pull.go -- SHA-256 Checksum + Atomic Writes (PASS)

- **Line 244-245**: SHA-256 computed via `io.TeeReader` during download -- correct streaming hash.
- **Line 253-260**: Checksum compared after full download; mismatch returns error. Logs warning if server provides no hash.
- **Line 218**: Atomic write: downloads to `cleaned + ".tmp"`.
- **Line 226-231**: Deferred cleanup removes temp file on any error path via `success` flag.
- **Line 263**: `os.Rename` is atomic on same filesystem.
- **Line 199-209**: Path traversal protection: rejects `..` in filename AND validates `filepath.Clean(destPath)` stays within `targetDir`.

**Verdict:** Implementation is correct. Checksum verification, atomic writes, and path traversal prevention all sound.

### 1.4 registry/oci.go -- Blob Size Limit + Path Traversal (PASS)

- **Line 17**: `maxBlobSize = 20 << 30` (20 GB).
- **Line 349**: `io.LimitReader(resp.Body, int64(maxBlobSize)+1)` -- reads at most 20GB+1 byte.
- **Line 353-354**: If data exceeds limit, returns error.
- **Line 100-101**: `parseReference` rejects `..` in repository name.
- **Line 94**: Rejects empty repository.

**Verdict:** Correct. Note that `getBlob` loads the entire blob into memory (up to 20 GB). For very large models this could OOM. Consider streaming to disk for production use, but this is a functional limitation, not a security bug.

### 1.5 deploy/helm/zerfoo/ -- securityContext (PASS)

`values.yaml` (lines 29-36):
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

`deployment.yaml` (lines 62-65): Templates the securityContext into the container spec.

Additional hardening present:
- `serviceaccount.yaml`: `automountServiceAccountToken: false` -- good.
- `ingress.yaml`: TLS configurable, disabled by default.

**Missing (LOW):** No `podSecurityContext` (e.g., `fsGroup`, `seccompProfile`) at the pod level. The container-level context is good, but adding a pod-level `seccompProfile: RuntimeDefault` would be best practice.

**Verdict:** Good security posture. Minor improvement opportunity with pod-level seccomp.

### 1.6 infra/terraform/cloud_run.tf -- allUsers Removed (PASS)

- No `allUsers` or `allAuthenticatedUsers` IAM binding exists.
- **Line 63-65**: Comment explicitly documents removal of public access (T106.25).
- **Line 66-71**: Dedicated `google_service_account` "api_invoker" created.
- **Line 73-79**: IAM binding grants `roles/run.invoker` only to that service account.

**Verdict:** Fix is complete. Cloud Run service is not publicly accessible.

---

## 2. security/*.go Audit

### 2.1 apikey.go -- KeyStore (PASS)

- Keys generated with `crypto/rand` (32 bytes = 256 bits) -- sufficient entropy.
- Raw key prefixed with `zf_` for identification.
- Only SHA-256 hash stored; raw key returned once at creation.
- `Lookup()` hashes the raw key and does map lookup -- O(1), no timing leak on hash.
- `Rotate()` properly revokes old key, copies scopes, creates new key.
- Concurrent-safe via `sync.RWMutex`.

**No issues found.**

### 2.2 encryption.go -- AES-256-GCM + TLS (PASS)

- AES-256-GCM with random nonce from `crypto/rand` -- correct.
- Nonce prepended to ciphertext; `Decrypt` extracts it -- standard pattern.
- `Decrypt` validates ciphertext length >= nonce size.
- `TLSConfig.BuildTLSConfig`: MinVersion defaults to TLS 1.2, mTLS support via ClientCAs.

**No issues found.**

### 2.3 network.go -- RateLimiter, IPFilter, CORS (PASS with notes)

- Token-bucket rate limiter keyed by IP -- correct algorithm.
- `Cleanup()` removes stale buckets -- must be called periodically by the caller.
- `IPFilter`: deny list checked before allow list -- correct precedence.
- CORS: Reflects origin if in allowed set. Handles preflight OPTIONS.

**Note (LOW):** `ClientIP()` trusts `X-Forwarded-For` and `X-Real-IP` headers unconditionally. Behind a load balancer this is fine, but if exposed directly, clients can spoof their IP. This is standard Go behavior and documented in the function name, but worth noting.

### 2.4 secrets.go -- SecretConfig (PASS)

- Reads secrets from environment variables with prefix filtering.
- No secrets hardcoded. Clean interface.

### 2.5 incident.go -- IncidentResponder (PASS)

- Automatic lockout after threshold via IPFilter integration.
- Lockout duration honored; auto-removes expired lockouts.
- Alert hooks called outside the mutex -- good.

### 2.6 vuln.go -- Vulnerability Scanning (PASS)

- Interface-only definitions for `DependencyAuditor` and `CVEChecker`.
- `AuditReport.CountBySeverity` and `HasCritical` are correct.

---

## 3. Infrastructure and Deployment Audit

### 3.1 deploy/aws/cloudformation.yaml (PASS with notes)

- ECS tasks run in Fargate with `AssignPublicIp: DISABLED` -- good, traffic goes through ALB.
- Task security group only allows ingress from ALB security group on port 8080.
- HTTP-to-HTTPS redirect when certificate is provided.
- IAM roles follow least privilege for marketplace metering and CloudWatch.
- Container Insights enabled.
- Log retention set to 30 days.

**Note (INFO):** `Resource: '*'` on marketplace metering and CloudWatch policies. This is standard for these AWS services which don't support resource-level permissions, so it's acceptable.

### 3.2 marketplace/aws/cfn/quickstart.yaml (MEDIUM -- two issues)

**Issue 1: `AssignPublicIp: ENABLED` (line 213)**
Unlike the main cloudformation.yaml which uses `DISABLED`, the quickstart template assigns public IPs directly to ECS tasks. This means tasks are directly reachable from the internet, bypassing the ALB. The ALB security group restriction is insufficient because the task's own security group (shared with the ALB) allows port 80/443 from `0.0.0.0/0`.

**Recommendation:** Use `AssignPublicIp: DISABLED` and deploy tasks in private subnets with a NAT gateway, or create a separate task security group that only allows traffic from the ALB.

**Issue 2: No HTTP-to-HTTPS redirect (line 148-155)**
The quickstart HTTP listener forwards directly to the target group even when a certificate is provided. The main cloudformation.yaml correctly redirects HTTP to HTTPS, but the quickstart does not.

### 3.3 marketplace/azure/arm/template.json (MEDIUM -- two issues)

**Issue 1: SSH open to the world (line 73-86)**
The NSG allows SSH (port 22) from `sourceAddressPrefix: "*"`. This exposes SSH to the entire internet. Should be restricted to a parameter-specified CIDR or removed entirely.

**Issue 2: Serve port open to the world (line 88-99)**
The inference API port is open to `sourceAddressPrefix: "*"` with no authentication layer in front. Unlike the AWS templates which use an ALB, this template exposes the VM directly. The API itself requires a Bearer token (via cloud/server.go), but defense in depth would restrict the source CIDR.

### 3.4 marketplace/azure/webhook.go (LOW)

**Line 82:** `io.ReadAll(r.Body)` with no size limit. A malicious (or compromised) Azure webhook caller could send an arbitrarily large body to exhaust memory. Should use `io.LimitReader` (e.g., 1 MB limit for webhook payloads).

### 3.5 infra/terraform/ -- GCP (PASS)

- GKE cluster uses Workload Identity (`GKE_METADATA` mode) -- good.
- Storage bucket has uniform bucket-level access and versioning.
- Model reader service account has minimal `roles/storage.objectViewer` permission.
- Cloud Run uses dedicated service account, no public access.
- GPU nodes use spot instances (cost optimization, not security).

### 3.6 deploy/aws/Dockerfile (PASS)

- Multi-stage build: build stage discarded.
- Runtime uses `distroless/static-debian12:nonroot` -- minimal attack surface, no shell.
- `CGO_ENABLED=0` -- fully static binary.
- Runs as nonroot user.

### 3.7 deploy/helm/ (PASS)

- ServiceAccount: `automountServiceAccountToken: false`.
- Ingress: disabled by default, TLS configurable.
- ConfigMap checksum annotation forces pod restart on config change.

---

## 4. cloud/ Additional Review

### 4.1 cloud/sso.go (LOW -- SAML signature not verified)

**Line 179-180:** Comment says "In production, this would also verify the XML signature against the IdP certificate." The `ValidateAssertion` method parses the SAML response and checks expiry but does NOT verify the cryptographic signature. An attacker who can submit a crafted SAML assertion can authenticate as any user.

This is explicitly marked as incomplete, but if SSO is deployed without completing signature verification, it is a critical vulnerability.

### 4.2 cloud/server.go -- responseCapture (LOW)

**Line 168-170:** `responseCapture.Write` appends all response bytes to `rc.body`. For streaming responses or large completions, this could accumulate significant memory. The billing middleware only needs the final usage JSON. Consider limiting capture to the first/last N bytes or using a separate billing signal.

### 4.3 cloud/tenant.go -- ConsumeTokens (INFO)

**Line 91-103:** The `ConsumeTokens` CAS loop does not check the result in `billingMiddleware` (server.go line 146: the return value of `ConsumeTokens` is ignored). If the token budget is exhausted, the response has already been sent. This means billing middleware cannot reject over-budget requests -- it only tracks. The rate-limit middleware runs before the response, so request-level limiting works, but token-level enforcement is post-hoc.

---

## 5. Summary of Findings

### E106 Fixes: All 6 verified PASS

| # | Fix | Status |
|---|-----|--------|
| 1 | Tenant ID via context.Context | PASS |
| 2 | API key redaction in Config()/List() | PASS |
| 3 | SHA-256 checksum + atomic writes | PASS |
| 4 | Blob size limit + path traversal | PASS |
| 5 | Helm securityContext | PASS |
| 6 | allUsers removed from Cloud Run | PASS |

### New Findings

| Severity | Location | Issue |
|----------|----------|-------|
| MEDIUM | `marketplace/aws/cfn/quickstart.yaml:213` | ECS tasks have public IPs; accessible bypassing ALB |
| MEDIUM | `marketplace/azure/arm/template.json:73-99` | SSH and serve port open to 0.0.0.0/0 |
| LOW | `marketplace/aws/cfn/quickstart.yaml:148` | No HTTP-to-HTTPS redirect when cert provided |
| LOW | `marketplace/azure/webhook.go:82` | `io.ReadAll` without size limit on webhook body |
| LOW | `cloud/sso.go:180` | SAML signature verification not implemented (marked TODO) |
| LOW | `cloud/server.go:168` | responseCapture unbounded memory for large responses |
| LOW | `cloud/tenant.go:37-38` | Exported `APIKey` field on `Tenant` struct (latent leak path) |
| LOW | `deploy/helm/values.yaml` | Missing pod-level seccomp profile |
| INFO | `security/network.go:167` | `ClientIP()` trusts X-Forwarded-For unconditionally |
| INFO | `cloud/server.go:146` | `ConsumeTokens` return value ignored; over-budget not enforced pre-response |

### No Hardcoded Secrets Found

Grep for hardcoded passwords, secrets, and credentials found no issues in production code. Test files use obvious test values (`"test-secret-key"`, `"s3cret"`) which is acceptable.
