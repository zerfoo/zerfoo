# ADR 065: Security Middleware Integration

## Status

Accepted

## Date

2026-03-21

## Context

A deep security review of zerfoo v1.10.0 (1321 Go files, 10 audit agents) revealed that
the `security/` package contains production-quality authentication, rate limiting, TLS,
IP filtering, and CORS primitives -- but none of them are wired into the default
`serve.Server` HTTP handler chain. All 10+ API endpoints (including destructive
operations like `DELETE /v1/models/{id}`) are unauthenticated by default.

Additionally, the `cloud/server.go` passes tenant identity via the spoofable
`X-Tenant-ID` HTTP header instead of `context.Context`, enabling billing fraud
and tenant impersonation if any route bypasses the auth middleware.

The review found 2 Critical, 11 High, 24 Medium, 9 Low, and 4 Info findings.

## Decision

1. Wire security middleware into `serve.Server` using the existing functional options
   pattern: `WithAPIKey(key)`, `WithRateLimiter(rl)`, `WithTLS(cert, key)`,
   `WithSecurityHeaders()`. The middleware stack order is:
   recovery > security headers > rate limiting > authentication > logging > handler.

2. Replace all `X-Tenant-ID` header passing in `cloud/server.go` with
   `context.WithValue` and `tenantFromContext()` pattern already used in
   `serve/cloud/tenant.go`.

3. Add `--api-key` / `ZERFOO_API_KEY`, `--tls-cert` / `--tls-key`, and
   `--rate-limit` flags to `cmd/cli/serve.go`.

4. Add `http.MaxBytesReader` to all inference endpoints (10 MB default).

5. Add SSRF protection to `serve/vision.go` with private IP blocking and
   DNS resolution validation.

6. Convert all `panic()` calls in `layers/core/` and `layers/attention/` to
   error returns, using the existing `Forward() error` and `Backward() error`
   signatures.

7. Add pod `securityContext` to the Helm deployment template matching the
   enterprise docs recommendations.

8. Restrict Cloud Run IAM from `allUsers` to authenticated service accounts.

## Consequences

**Positive:**
- All API endpoints are authenticated by default when an API key is configured.
- Existing security primitives are reused, not rewritten.
- The functional options pattern remains consistent with the rest of the codebase.
- Breaking change risk is minimal: authentication is opt-in via flag/env var.

**Negative:**
- Users who rely on unauthenticated access must set `ZERFOO_API_KEY` or pass `--api-key`.
- The `cloud/server.go` refactor requires updating all middleware that reads tenant identity.
- Converting 45+ panics to errors requires updating callers throughout the inference pipeline.
