# ADR-001: Enterprise Production Readiness

**Status:** Accepted
**Phase:** 4 + 7
**Date:** 2026-03-01

## Context

Zerfoo had strong foundations (clean interfaces, modular architecture, type-safe
generics, 95%+ test coverage) but lacked operational hardening for enterprise
production deployment. Gaps existed in observability, security, reliability,
configuration management, and CI/CD enforcement.

Additionally, a Phase 7 architecture review identified structural issues: dead
code (pkg/prelude, tests/helpers nil stubs), inverted layer registration
dependency (layers/core -> model), and thread-unsafe graph memo map.

## Decision

### Observability (E21-E22)

- **Structured logging** via `log/` package: Logger interface with Debug/Info/
  Warn/Error levels, JSON output mode, NopLogger for zero-overhead disabling.
  All packages instrumented (compute, distributed, training, model, cmd).
- **Metrics** via `metrics/runtime/` package: Collector interface with Counter,
  Gauge, Histogram. InMemoryCollector for testing; NopCollector for production
  disabling. CPUEngine/GPUEngine and distributed ops instrumented.

### Security (E23)

- TLS/mTLS for all gRPC via `distributed.TLSConfig`. Plaintext fallback for
  local dev (nil config). Input validation on all RPC handlers (completed in
  Phase 5 via E32).

### Configuration (E24)

- `config/` package: `Load[T](path)` and `LoadWithEnv[T](path, prefix)` for
  JSON config with env var overrides via struct tags. Validation via
  `validate:"required"` tag. Standard structs: EngineConfig, TrainingConfig,
  DistributedConfig.

### Reliability (E25-E26, E28)

- **Graceful shutdown** via `shutdown/` package: Coordinator with reverse-order
  Closer execution, per-closer timeout, signal handling (SIGINT/SIGTERM) in CLI.
- **Health checks** via `health/` package: HTTP /healthz (liveness), /readyz
  (readiness with configurable checks), /debug/pprof/. Engine health check
  verifies compute is operational.
- **Resource limits**: MemoryTracker with CAS-based enforcement at Engine level.
  Per-operation timeout via context.Context deadline checks.

### CI/CD Hardening (E27)

- Parity and numerics tests blocking in CI (removed `|| true`).
- Coverage gate via `cmd/coverage-gate/`: fails if any testable package drops
  below 93%.
- Benchmark regression via `cmd/bench-compare/`: fails on >10% regression.
- Race detector on all unit tests. Go 1.25 on Ubuntu + macOS runners.

### Architecture Cleanup (Phase 7: E44-E46)

- **Dead code removal**: Deleted pkg/prelude (empty), tests/helpers/wire.go
  (4 nil stubs), 7 dead test files (17 always-skipping tests).
- **Registration consolidation**: Removed init() from layers/core/registry.go.
  Single entry point: `layers/registry.RegisterAll()`. Exported BuildFFN.
- **Graph thread safety**: Added sync.Mutex to graph.Graph protecting memo map
  in Forward/Backward. Coarse-grained lock (correct for graphs < 1000 nodes).

## Consequences

- All packages use leveled structured logging; no raw fmt.Printf in production.
- Runtime metrics available for Prometheus scraping or in-memory snapshot.
- gRPC is TLS-capable; plaintext is opt-in (nil TLSConfig).
- CI enforces coverage >= 93%, benchmark regression < 10%, zero races.
- Graph.Forward is safe for concurrent use from multiple goroutines.
- No init()-based registration; single wiring point reduces coupling.

### Blocked Item

- **E29 GPU hardware validation**: Blocked on GCP GPU quota = 0. Quota request
  pending (preference ID: zerfoo-gpu-test, project: numerai-488804). Unblock by
  checking quota status or trying a different cloud provider.

### Key Files

- `log/logger.go` -- Logger interface, StdLogger, NopLogger
- `metrics/runtime/metrics.go` -- Collector, InMemoryCollector, NopCollector
- `config/loader.go` -- Load[T], LoadWithEnv[T]
- `shutdown/coordinator.go` -- Closer, Coordinator
- `health/server.go` -- Health HTTP server
- `cmd/coverage-gate/main.go` -- CI coverage enforcement
- `cmd/bench-compare/main.go` -- CI benchmark regression detection
