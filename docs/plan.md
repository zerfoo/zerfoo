# Zerfoo Enterprise Production Readiness Plan

## 1. Context

### Problem Statement

Zerfoo is a Go-based ML framework with 40+ packages, a 34-method compute
Engine[T] interface, CPU and CUDA GPU backends, gRPC-based distributed
training, and comprehensive test coverage (95%+ across testable packages).

The framework has strong foundations -- clean interfaces, modular architecture,
type-safe generics, and high test coverage -- but lacks the operational
hardening required for enterprise production deployment. This plan addresses
the gaps in observability, security, reliability, configuration management,
and CI/CD enforcement needed to reach production grade.

Architecture and design details are documented in docs/design.md.
GPU-specific documentation is in docs/gpu.md.

### Objectives

- O1: Add structured logging with configurable log levels across all packages.
- O2: Export runtime metrics (throughput, latency, memory, errors) via a
  metrics interface suitable for Prometheus or similar backends.
- O3: Harden gRPC distributed services with TLS and mutual authentication.
- O4: Add file-based configuration loading with validation and env var overrides.
- O5: Implement graceful shutdown with resource cleanup across all components.
- O6: Add health check endpoints for readiness and liveness probes.
- O7: Make parity and numerics tests blocking in CI; add coverage gates.
- O8: Add benchmark regression detection to prevent performance degradation.
- O9: Add resource limits (memory caps, timeouts) to prevent unbounded allocation.
- O10: Validate GPU implementation on real NVIDIA hardware (blocked on GCP quota).
- O11: Create production deployment runbook and troubleshooting guide.

### Non-Goals

- Multi-GPU or distributed GPU support.
- cuDNN, TensorRT, or other NVIDIA library integration.
- AMD ROCm or OpenCL backends.
- Mixed precision training.
- Breaking changes to the Engine[T] or Node[T] interfaces.
- Replacing gRPC with a different RPC framework.
- Adding third-party test frameworks (testify, etc.).

### Constraints and Assumptions

- Use Go standard library only where possible. Minimize new dependencies.
- All CUDA code behind `//go:build cuda` build tags.
- Pre-commit hook rejects commits spanning multiple directories.
- All changes must pass golangci-lint, go vet, and gofmt.
- Tests must pass with -race flag.
- No Docker Compose. Prefer DevSpace if orchestration is needed.
- Table-driven tests using the standard testing package.

### Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Structured logging | All packages use leveled logger | Grep for raw fmt.Print/log.Print in non-test code = 0 |
| Metrics export | Runtime metrics available via interface | Metrics interface has >= 10 counters/gauges |
| TLS coverage | All gRPC endpoints use TLS | No plaintext gRPC listeners in production config |
| Config loading | YAML/JSON config from file + env vars | Config loads from file, env overrides work |
| Graceful shutdown | All components clean up on SIGTERM | Integration test verifies orderly shutdown |
| Health checks | Readiness + liveness probes | HTTP endpoint returns status within 100ms |
| CI blocking tests | Parity + numerics tests block merges | CI fails on parity/numerics test failure |
| Benchmark gates | CI fails on > 10% regression | Benchmark comparison in CI workflow |
| Resource limits | Memory caps enforced | Allocation above limit returns error |
| Coverage gate | >= 95% enforced in CI | CI fails if coverage drops below threshold |

---

## 2. Scope and Deliverables

### In Scope

- Structured logging library (log levels, JSON output, context propagation).
- Metrics collection interface and default implementation.
- TLS/mTLS configuration for gRPC services.
- File-based configuration with validation and environment variable overrides.
- Graceful shutdown coordination across Engine, distributed workers, gRPC server.
- Health check HTTP endpoints.
- CI hardening: blocking parity tests, coverage gates, benchmark regression detection.
- Resource limit enforcement (memory, timeouts).
- Production documentation (deployment runbook, troubleshooting guide).
- GPU hardware validation (when quota available).

### Out of Scope

- Web UI or dashboard.
- Model serving HTTP API (inference server).
- Automatic device placement or tensor migration.
- Database or persistent storage integration.
- Container image building or Kubernetes manifests.

### Deliverables

| ID | Description | Acceptance Criteria |
|----|-------------|---------------------|
| D1 | Structured logging | Logger interface with Debug/Info/Warn/Error levels; JSON output mode; all packages instrumented |
| D2 | Metrics interface | Counters, gauges, histograms; default in-memory impl; export-ready |
| D3 | gRPC TLS | TLS config struct; mTLS support; integration test with TLS |
| D4 | Config management | YAML/JSON loader; env var overrides; validation errors |
| D5 | Graceful shutdown | Context-based cancellation; cleanup ordering; integration test |
| D6 | Health checks | HTTP /healthz and /readyz endpoints; configurable checks |
| D7 | CI hardening | Blocking parity/numerics; coverage gate; benchmark gate |
| D8 | Resource limits | Memory cap on Engine; per-operation timeout; GPU memory limit |
| D9 | Production docs | Deployment runbook; troubleshooting guide; performance tuning |
| D10 | GPU validation | Tests pass on real T4; benchmark results documented |

---

## 3. Checkable Work Breakdown

### Completed Work (Phases 1-3)

Phase 1 (Test Coverage), Phase 2 (GPU Engine), and Phase 3 (GPU Production
Readiness) are complete. Details in docs/design.md Section 7.

Remaining blocked items from Phase 3:
- T15.1 GPU hardware validation -- BLOCKED on GCP GPU quota
- T20.1 Production benchmarks on T4 -- BLOCKED on T15.1

---

### Phase 4: Enterprise Production Readiness

#### E21: Structured Logging

Add a logging abstraction that supports leveled output, structured fields,
and JSON format. Instrument all packages that currently use raw fmt.Printf
or the distributed Logger interface.

- [x] T21.1 Define Logger interface in a new `log` package  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: Interface has Debug, Info, Warn, Error methods. Each accepts a message string and key-value fields. A NopLogger and a StdLogger (writing to io.Writer) are provided. JSON output mode is available via a constructor option.
  - [x] S21.1.1 Create log/logger.go with Logger interface and Level type  Est: 20m
  - [x] S21.1.2 Implement StdLogger with level filtering and text/JSON output  Est: 25m
  - [x] S21.1.3 Implement NopLogger (zero-allocation no-op)  Est: 5m
  - [x] S21.1.4 Write unit tests for StdLogger (level filtering, JSON format, field rendering)  Est: 20m
  - [x] S21.1.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T21.2 Integrate Logger into compute package  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: T21.1
  - Acceptance: CPUEngine and GPUEngine accept a Logger at construction. OOM fallback, stream errors, and pool operations log at appropriate levels. No raw fmt.Printf calls remain in compute/.
  - [x] S21.2.1 Add Logger field to CPUEngine; log parallelFor errors at Warn  Est: 15m
  - [x] S21.2.2 Add Logger field to GPUEngine; log OOM fallback, pool stats, stream errors  Est: 20m
  - [x] S21.2.3 Update tests to verify log output in error scenarios  Est: 15m
  - [x] S21.2.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T21.3 Integrate Logger into distributed package  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: T21.1
  - Acceptance: Replace existing distributed.Logger interface with log.Logger. All coordinator and worker components use leveled logging. Connection events logged at Info, errors at Error.
  - [x] S21.3.1 Update distributed.ServerManager, coordinator to accept log.Logger  Est: 15m
  - [x] S21.3.2 Replace all fmt.Printf calls in distributed/ with logger calls  Est: 15m
  - [x] S21.3.3 Update tests to use StdLogger or NopLogger  Est: 10m
  - [x] S21.3.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T21.4 Integrate Logger into remaining packages  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T21.1
  - Acceptance: training/, model/, cmd/cli/ use log.Logger. No raw fmt.Printf in non-test production code.
  - [x] S21.4.1 Add Logger to training.WorkflowConfig and optimizer constructors  Est: 10m
  - [x] S21.4.2 Add Logger to model package and cmd/cli framework  Est: 10m
  - [x] S21.4.3 Audit all packages for remaining fmt.Printf; replace with logger  Est: 10m
  - [x] S21.4.4 Run golangci-lint and go test -cover  Est: 5m

#### E22: Metrics Interface

Add a metrics collection abstraction for runtime observability. The interface
must be backend-agnostic (usable with Prometheus, StatsD, or in-memory).

- [x] T22.1 Define Metrics interface in a new `metrics/runtime` package  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: Interface has Counter(name), Gauge(name), Histogram(name, buckets) methods. Each returns a typed metric with Inc/Set/Observe methods. A default in-memory implementation is provided for testing and local use. A NopMetrics implementation is provided for zero overhead when metrics are disabled.
  - [x] S22.1.1 Create metrics/runtime/metrics.go with Collector interface  Est: 20m
  - [x] S22.1.2 Implement InMemoryCollector with thread-safe counters/gauges  Est: 25m
  - [x] S22.1.3 Implement NopCollector (zero-allocation no-op)  Est: 5m
  - [x] S22.1.4 Write unit tests for InMemoryCollector (concurrent access, snapshot)  Est: 15m
  - [x] S22.1.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T22.2 Instrument compute.Engine with metrics  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: T22.1
  - Acceptance: CPUEngine and GPUEngine report: op_count (counter per operation type), op_duration_seconds (histogram), oom_fallback_total (counter), pool_hit_total / pool_miss_total (counters for GPU pool).
  - [x] S22.2.1 Add Collector field to CPUEngine; instrument Add/MatMul/etc. with counters and timers  Est: 20m
  - [x] S22.2.2 Add Collector field to GPUEngine; instrument kernel dispatch, OOM, pool  Est: 20m
  - [x] S22.2.3 Write tests verifying metric increments after operations  Est: 15m
  - [x] S22.2.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T22.3 Instrument distributed package with metrics  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T22.1
  - Acceptance: Distributed workers report: allreduce_count (counter), allreduce_duration_seconds (histogram), barrier_count, broadcast_count, connection_errors_total.
  - [x] S22.3.1 Add Collector to Strategy and coordinator  Est: 15m
  - [x] S22.3.2 Instrument AllReduceGradients, Barrier, BroadcastTensor  Est: 10m
  - [x] S22.3.3 Write tests verifying metrics after distributed operations  Est: 10m
  - [x] S22.3.4 Run golangci-lint and go test -cover  Est: 5m

#### E23: gRPC Security Hardening

Add TLS and mutual authentication to all gRPC communication channels.

- [x] T23.1 Add TLS configuration to gRPC server and client  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: A TLSConfig struct supports: CA cert path, server cert/key paths, client cert/key paths for mTLS. ServerManager.Start() uses TLS credentials when TLSConfig is provided. Worker connections use TLS. Plaintext is still supported (for local development) when TLSConfig is nil.
  - [x] S23.1.1 Create distributed/tlsconfig.go with TLSConfig struct and credential helpers  Est: 20m
  - [x] S23.1.2 Update ServerManager to accept TLSConfig and create TLS listener  Est: 15m
  - [x] S23.1.3 Update NetworkManager.ConnectToPeers to use TLS dial options  Est: 15m
  - [x] S23.1.4 Write integration test: server + client with self-signed TLS certs  Est: 20m
  - [x] S23.1.5 Write integration test: mTLS with client cert verification  Est: 15m
  - [x] S23.1.6 Run golangci-lint and go test -cover  Est: 5m

- [x] T23.2 Add input validation to distributed RPC handlers  Owner: TBD  Est: 30m  Completed: 2026 03 01 via T32.5  Note: Implemented as part of Phase 5 E32 workerService.
  - Dependencies: None
  - Acceptance: All RPC handlers validate request fields (non-empty rank, valid tensor shapes, non-nil data). Invalid requests return gRPC InvalidArgument status. Tests verify each validation path.
  - [x] S23.2.1 Add validation to AllReduce, Barrier, Broadcast RPC handlers  Est: 15m
  - [x] S23.2.2 Write tests for each validation error case  Est: 10m
  - [x] S23.2.3 Run golangci-lint and go test -cover  Est: 5m

#### E24: Configuration Management

Add file-based configuration loading with validation and environment
variable overrides. Use encoding/json and os.Getenv from the standard library.

- [x] T24.1 Create config package with file loader  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: A config.Load[T](path string) function reads a JSON file into a struct. A config.LoadWithEnv[T](path, prefix string) function additionally applies environment variable overrides using the `env` struct tag. Validation errors list all invalid fields. Missing required fields produce clear error messages.
  - [x] S24.1.1 Create config/loader.go with Load[T] function (JSON decoder)  Est: 15m
  - [x] S24.1.2 Implement env var override via struct tag reflection  Est: 20m
  - [x] S24.1.3 Implement validation via `validate:"required"` struct tag  Est: 15m
  - [x] S24.1.4 Write unit tests: valid config, missing file, invalid JSON, missing required, env override  Est: 20m
  - [x] S24.1.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T24.2 Define standard config structs for Engine and Training  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T24.1
  - Acceptance: EngineConfig (device type, memory limit, log level), TrainingConfig (batch size, learning rate, optimizer, epochs, checkpoint interval), DistributedConfig (coordinator address, TLS config, timeout). Each struct has JSON tags and validation tags.
  - [x] S24.2.1 Define EngineConfig, TrainingConfig, DistributedConfig structs  Est: 15m
  - [x] S24.2.2 Write tests loading each config from JSON with env overrides  Est: 10m
  - [x] S24.2.3 Run golangci-lint and go test -cover  Est: 5m

#### E25: Graceful Shutdown

Implement orderly shutdown coordination using context cancellation
and cleanup callbacks.

- [x] T25.1 Add Closer interface and shutdown coordinator  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: A shutdown.Coordinator registers Closer instances in order. On Shutdown(ctx), it calls Close() on each in reverse registration order. If a Closer does not complete within the context deadline, it is skipped and logged. Integration test demonstrates orderly cleanup.
  - [x] S25.1.1 Create shutdown/coordinator.go with Closer interface and Coordinator  Est: 20m
  - [x] S25.1.2 Implement reverse-order shutdown with timeout per closer  Est: 15m
  - [x] S25.1.3 Write tests: orderly shutdown, timeout on slow closer, empty coordinator  Est: 15m
  - [x] S25.1.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T25.2 Implement Closer for Engine and distributed components  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T25.1
  - Acceptance: GPUEngine.Close() drains memory pool and destroys CUDA handles. CPUEngine.Close() is a no-op (satisfies interface). Distributed Strategy.Shutdown() deregisters from coordinator and closes connections. All Close methods are idempotent.
  - [x] S25.2.1 Make CPUEngine implement Closer (no-op Close)  Est: 5m
  - [x] S25.2.2 Verify GPUEngine.Close() is idempotent  Est: 10m
  - [x] S25.2.3 Make distributed Strategy implement Closer  Est: 10m
  - [x] S25.2.4 Write integration test: register Engine + Strategy, trigger shutdown  Est: 15m
  - [x] S25.2.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T25.3 Add signal handling to CLI commands  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T25.1, T25.2
  - Acceptance: cmd/zerfoo and cmd/zerfoo-predict catch SIGINT/SIGTERM, trigger shutdown coordinator, and exit cleanly. Integration test verifies signal handling.
  - [x] S25.3.1 Add signal listener in cmd framework that cancels root context  Est: 15m
  - [x] S25.3.2 Wire shutdown coordinator into CLI lifecycle  Est: 10m
  - [x] S25.3.3 Write test verifying clean exit on SIGTERM  Est: 10m
  - [x] S25.3.4 Run golangci-lint and go test -cover  Est: 5m

#### E26: Health Checks

Add health check endpoints for deployment probes (Kubernetes liveness
and readiness).

- [x] T26.1 Create health check HTTP server  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: T21.1
  - Acceptance: A health.Server exposes /healthz (liveness) and /readyz (readiness) HTTP endpoints. Each returns 200 OK with JSON body when healthy, 503 when unhealthy. Readiness checks are configurable (register check functions). Server starts on a configurable port. Logger is used for startup/error messages.
  - [x] S26.1.1 Create health/server.go with Server struct and HTTP handlers  Est: 15m
  - [x] S26.1.2 Implement configurable readiness checks (func() error callbacks)  Est: 10m
  - [x] S26.1.3 Write tests: healthy response, unhealthy readiness, concurrent access  Est: 15m
  - [x] S26.1.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T26.2 Add engine health check  Owner: TBD  Est: 20m  Completed: 2026 03 01
  - Dependencies: T26.1
  - Acceptance: A check function verifies Engine is operational (e.g., small tensor add succeeds). For GPU, additionally verify CUDA context is valid. Register as readiness check.
  - [x] S26.2.1 Implement engine health check function  Est: 10m
  - [x] S26.2.2 Write test for healthy and unhealthy engine  Est: 10m
  - [x] S26.2.3 Run golangci-lint and go test -cover  Est: 5m

#### E27: CI/CD Hardening

Make CI pipeline enforce quality gates strictly.

- [x] T27.1 Make parity and numerics tests blocking  Owner: TBD  Est: 15m  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: Remove `|| true` from parity and numerics test steps in .github/workflows/ci.yml. CI fails if any parity or numerics test fails.
  - [x] S27.1.1 Update ci.yml: remove `|| true` from parity test step  Est: 5m
  - [x] S27.1.2 Update ci.yml: remove `|| true` from numerics test step  Est: 5m
  - [x] S27.1.3 Verify CI passes with current test suite  Est: 5m

- [x] T27.2 Add coverage gate to CI  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: CI step runs `go test -coverprofile=coverage.out ./...`, parses output, and fails if any testable package (excluding documented exceptions) drops below 93%. Coverage summary is posted as a CI artifact.
  - [x] S27.2.1 Add coverage step to ci.yml that generates coverage.out  Est: 10m
  - [x] S27.2.2 Write a Go script (cmd/coverage-gate/main.go) that parses coverage.out and exits non-zero if below threshold  Est: 20m
  - [x] S27.2.3 Add tests for coverage-gate script  Est: 10m
  - [x] S27.2.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T27.3 Add benchmark regression detection  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: CI runs benchmarks on each PR. A Go script compares benchmark results against a baseline (stored in repo). CI fails if any benchmark regresses by more than 10%. Baseline is updated via a manual workflow dispatch.
  - [x] S27.3.1 Add benchmark step to ci.yml (go test -bench=. -benchmem -count=3)  Est: 10m
  - [x] S27.3.2 Write cmd/bench-compare/main.go to parse benchstat output and enforce threshold  Est: 25m
  - [x] S27.3.3 Add baseline benchmark results file (benchmarks/baseline.txt)  Est: 5m
  - [x] S27.3.4 Add tests for bench-compare script  Est: 10m
  - [x] S27.3.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T27.4 Update CI Go version and add race detector  Owner: TBD  Est: 15m  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: CI uses Go 1.25 (matching go.mod). Race detector runs on unit tests. Both Ubuntu and macOS runners are used.
  - [x] S27.4.1 Update ci.yml go-version to match go.mod  Est: 5m
  - [x] S27.4.2 Add -race flag to unit test step  Est: 5m
  - [x] S27.4.3 Add macOS runner to test matrix  Est: 5m

#### E28: Resource Limits

Add configurable resource limits to prevent unbounded allocation and
runaway operations.

- [x] T28.1 Add memory limit to Engine  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: Engine accepts a MaxMemoryBytes option. Tensor allocation that would exceed the limit returns an error instead of allocating. GPU engine tracks device memory usage. The limit is enforced at the Engine level, not the allocator level (so it applies to both CPU and GPU).
  - [x] S28.1.1 Add MemoryTracker to compute package (atomic int64 tracking allocated bytes)  Est: 15m
  - [x] S28.1.2 Integrate MemoryTracker into tensor allocation (New, NewWithStorage)  Est: 15m
  - [x] S28.1.3 Add MaxMemoryBytes option to Engine constructors  Est: 10m
  - [x] S28.1.4 Write tests: allocation within limit succeeds, over limit returns error  Est: 15m
  - [x] S28.1.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T28.2 Add per-operation timeout enforcement  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: Engine respects context.Context deadlines. Long-running operations (MatMul, Softmax) check ctx.Done() periodically and return context.DeadlineExceeded if expired. GPU operations use CUDA stream synchronization with timeout.
  - [x] S28.2.1 Add ctx.Done() checks in CPUEngine parallelFor loops  Est: 15m
  - [x] S28.2.2 Add stream sync timeout in GPUEngine operations  Est: 10m
  - [x] S28.2.3 Write tests: operation completes within deadline, times out correctly  Est: 15m
  - [x] S28.2.4 Run golangci-lint and go test -cover  Est: 5m

#### E29: GPU Hardware Validation (Blocked)

Validate all GPU code on real NVIDIA hardware.

- [ ] T29.1 Create GCP T4 spot VM and validate GPU tests  Owner: TBD  Est: 1h  **BLOCKED:** GCP GPU quota = 0. Quota increase request pending (preference ID: zerfoo-gpu-test).
  - Dependencies: None
  - Acceptance: `go test -tags cuda ./...` passes on real T4 hardware. Benchmark results captured and documented in docs/gpu.md.
  - Unblock action: Check quota status via `gcloud beta quotas preferences describe zerfoo-gpu-test --project=numerai-488804`. If still denied, try a different GCP project or cloud provider.
  - [ ] S29.1.1 Create n1-standard-4 spot VM with T4 GPU  Est: 5m
  - [ ] S29.1.2 Install CUDA Toolkit 12.x and Go 1.25, clone repo  Est: 15m
  - [ ] S29.1.3 Build with `go build -tags cuda ./...` and fix any build issues  Est: 10m
  - [ ] S29.1.4 Run `go test -tags cuda ./...` and capture output  Est: 10m
  - [ ] S29.1.5 Run benchmarks and save results  Est: 5m
  - [ ] S29.1.6 Delete VM immediately  Est: 2m
  - [ ] S29.1.7 Document results in docs/gpu.md  Est: 10m

- [ ] T29.2 Run optimized benchmarks on T4  Owner: TBD  Est: 30m  **BLOCKED:** Depends on T29.1.
  - Dependencies: T29.1
  - Acceptance: Benchmark results for MatMul (128/512/1024), Softmax, and chained attention ops documented with Phase 3 device-resident pipeline.
  - [ ] S29.2.1 Run benchmarks with -benchmem and capture results  Est: 10m
  - [ ] S29.2.2 Update docs/gpu.md with benchmark table  Est: 15m
  - [ ] S29.2.3 Delete VM  Est: 2m

#### E30: Production Documentation

Create operational documentation for production deployment.

- [x] T30.1 Write deployment runbook  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: E21, E23, E24, E25, E26
  - Acceptance: docs/runbook.md covers: system requirements, installation steps, configuration reference (all config fields documented), startup sequence, health check verification, log interpretation, common operational tasks (scale workers, update model, restart), shutdown procedure.
  - [x] S30.1.1 Write system requirements and installation section  Est: 15m
  - [x] S30.1.2 Write configuration reference (all config structs documented)  Est: 15m
  - [x] S30.1.3 Write startup, health check, and shutdown sections  Est: 15m
  - [x] S30.1.4 Write common operational tasks  Est: 15m

- [x] T30.2 Write troubleshooting guide  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: E21, E22
  - Acceptance: docs/troubleshooting.md covers: common error messages with root causes and fixes, GPU-specific issues (CUDA not found, OOM, driver mismatch), distributed training issues (connection refused, timeout, split brain), performance diagnosis (how to identify bottlenecks, pprof usage).
  - [x] S30.2.1 Document common error messages and fixes  Est: 15m
  - [x] S30.2.2 Document GPU troubleshooting  Est: 10m
  - [x] S30.2.3 Document distributed training troubleshooting  Est: 10m
  - [x] S30.2.4 Document performance diagnosis with pprof  Est: 10m

- [x] T30.3 Add pprof endpoints to health server  Owner: TBD  Est: 20m  Completed: 2026 03 01
  - Dependencies: T26.1
  - Acceptance: Health server registers net/http/pprof handlers. CPU profile, heap profile, goroutine dump available at /debug/pprof/*.
  - [x] S30.3.1 Register pprof handlers in health.Server  Est: 10m
  - [x] S30.3.2 Write test verifying pprof endpoints respond  Est: 10m
  - [x] S30.3.3 Run golangci-lint and go test -cover  Est: 5m

#### E31: Final Verification

Run the full quality gate suite after all enterprise features are implemented.

- [x] T31.1 Run full test suite with coverage  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: E21, E22, E23, E24, E25, E26, E27, E28
  - Acceptance: `go test ./... -cover` shows all packages at target coverage. `go test ./... -race` shows zero races. New packages (log, config, health, shutdown, metrics/runtime) are all at >= 95%.
  - [x] S31.1.1 Run go test ./... -cover  Est: 10m
  - [x] S31.1.2 Run go test ./... -race  Est: 10m
  - [x] S31.1.3 Verify new packages meet 95% coverage  Est: 10m

- [x] T31.2 Run linters and formatters  Owner: TBD  Est: 15m  Completed: 2026 03 01
  - Dependencies: T31.1
  - Acceptance: golangci-lint 0 issues, go vet clean, gofmt clean.
  - [x] S31.2.1 Run golangci-lint run ./...  Est: 5m
  - [x] S31.2.2 Run go vet ./...  Est: 5m
  - [x] S31.2.3 Run gofmt -l . and verify no files  Est: 5m

- [x] T31.3 Run integration smoke test  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T31.1
  - Acceptance: End-to-end test: load config from file, create Engine, run forward pass, verify health check, trigger graceful shutdown. All within a single test binary.
  - [x] S31.3.1 Write integration test covering config -> engine -> health -> shutdown  Est: 20m
  - [x] S31.3.2 Run integration test  Est: 5m
  - [x] S31.3.3 Run golangci-lint  Est: 5m

---

### Phase 5: Concrete Distributed Service Server

#### Phase 5 Context

Phase 4 enterprise production readiness is complete except for E29 (GPU
validation, blocked on GCP quota) and T23.2 (RPC input validation, skipped
because no concrete DistributedServiceServer implementation existed).

The distributed package currently has:
- Auto-generated protobuf stubs for DistributedService (AllReduce bidi stream,
  Barrier unary, Broadcast unary) in distributed/pb/dist.proto.
- Auto-generated protobuf stubs for Coordinator (RegisterWorker,
  UnregisterWorker, Heartbeat, StartCheckpoint, EndCheckpoint) in
  distributed/pb/coordinator.proto.
- A fully implemented Coordinator gRPC server
  (distributed/coordinator/coordinator.go) with worker management, heartbeat
  reaper, and checkpoint coordination.
- InternalStrategy[T] interface (distributed/interfaces.go) defining Init,
  AllReduceGradients, Barrier, BroadcastTensor, Rank, Size, Shutdown.
- AllReduceStrategy[T] (distributed/all_reduce.go) implementing hierarchical
  all-reduce using local + cross-node InternalStrategy instances.
- NetworkManager (distributed/network_manager.go) for establishing peer gRPC
  client connections.
- ServerManager (distributed/network_manager.go) for gRPC server lifecycle
  management (start, stop, graceful stop).
- TLS/mTLS configuration (distributed/tlsconfig.go).
- GrpcServer, ListenerFactory, Dialer, ServiceClientFactory type aliases
  (distributed/interfaces.go).
- CoordinatorClient interface (distributed/interfaces.go).
- Comprehensive custom mock implementations for testing
  (distributed/custom_mocks_test.go).

What is missing:
1. A concrete DistributedServiceServer implementation -- the actual gRPC
   handler that runs on each worker node and processes incoming AllReduce,
   Barrier, and Broadcast RPCs from peers.
2. A GrpcStrategy[T] that implements InternalStrategy[T] using gRPC transport,
   connecting the high-level AllReduceStrategy to the network layer.
3. A WorkerNode struct that ties together the server, strategy, coordinator
   registration, health checks, and shutdown coordination.
4. Input validation on RPC handlers (the previously skipped T23.2).
5. Multi-worker integration tests proving distributed operations work
   end-to-end over real gRPC connections.

#### Phase 5 Objectives

- P5-O1: Implement a concrete DistributedServiceServer with AllReduce,
  Barrier, and Broadcast handlers including input validation.
- P5-O2: Implement GrpcStrategy[T] connecting InternalStrategy[T] to gRPC
  transport.
- P5-O3: Create multi-worker integration tests proving correctness over
  real gRPC connections (using bufconn for in-process testing).
- P5-O4: Implement worker lifecycle management (init, run, shutdown)
  integrated with existing CLI, health checks, and shutdown coordinator.

#### Phase 5 Non-Goals

- Ring all-reduce optimization. Use star topology (reduce to root, broadcast
  from root) for correctness first. Ring optimization is a future Phase 6
  task.
- Gradient compression or sparsification.
- Fault-tolerant training with automatic recovery from worker failures.
- Dynamic worker join or leave during a training step.
- Multi-GPU per worker.

#### Phase 5 Design Decisions

**AllReduce Protocol (Star Topology):**
The AllReduce bidi stream implements a star-topology reduce. Root (rank 0)
runs the server that collects gradients from all peers. Each non-root worker
opens a bidi stream to root, sends its gradients as AllReduceRequest messages
(one per named tensor), then waits for root to send back AllReduceResponse
messages with the averaged result. Root accumulates all peer gradients plus
its own local gradients, computes the element-wise average (sum / world_size),
and streams the result back to each peer.

The server uses a reduceSession struct to coordinate across concurrent
AllReduce stream handlers. The session collects tensors by name from each
peer, waits for all peers via a sync barrier, computes the reduction, and
distributes the result.

**Barrier Protocol:**
Barrier uses a simple counter-based approach. Each worker calls Barrier RPC
on the root. Root counts arrivals and blocks each caller until all workers
have arrived. Uses sync.Cond for efficient waiting. Each barrier has an
epoch number to prevent stale barrier responses.

**Broadcast Protocol:**
Root sends a BroadcastRequest with the tensor. Non-root workers call
Broadcast RPC on root. Root returns the tensor in the BroadcastResponse.
Root stores the broadcast tensor in a thread-safe map keyed by name so
concurrent callers all receive the same data.

**Tensor Serialization:**
The pb.Tensor message uses repeated float for data (float32 only). The
GrpcStrategy[T] converts tensor.TensorNumeric[T] to/from pb.Tensor. For
T=float32, this is a direct copy. For T=float64, values are narrowed to
float32 for transport (acceptable for gradient averaging where precision
loss is tolerable).

---

#### E32: Worker Service (DistributedServiceServer)

Implement the concrete gRPC service handler that runs on each worker node,
processing AllReduce, Barrier, and Broadcast RPCs from peers.

- [x] T32.1 Create workerService struct with reduce session coordinator  Owner: TBD  Est: 1.5h  Completed: 2026 03 01
  - Dependencies: None
  - Acceptance: A workerService struct in distributed/worker_service.go implements pb.DistributedServiceServer. It embeds pb.UnimplementedDistributedServiceServer. Fields include rank (int32), worldSize (int32), logger (log.Logger), collector (metrics/runtime.Collector). A reduceSession struct coordinates all-reduce across concurrent streams: it collects tensors by name from each peer, uses a sync barrier (sync.Cond or channels) to wait for all peers, computes the element-wise sum, and distributes the result. Static interface assertion var _ pb.DistributedServiceServer = (*workerService)(nil) compiles.
  - [x] S32.1.1 Create distributed/worker_service.go with workerService struct, constructor NewWorkerService(rank, worldSize int32, logger log.Logger) *workerService  Est: 15m
  - [x] S32.1.2 Implement reduceSession struct with Submit(peerRank int32, tensors map[string]*pb.Tensor) and WaitForResult() map[string]*pb.Tensor methods  Est: 30m
  - [x] S32.1.3 Implement NewReduceSession(worldSize int32) *reduceSession constructor  Est: 10m
  - [x] S32.1.4 Write unit tests for reduceSession: two peers submit, both get averaged result; timeout when one peer missing; concurrent submission safety  Est: 30m
  - [x] S32.1.5 Run golangci-lint and go test -cover on distributed/  Est: 5m

- [x] T32.2 Implement AllReduce bidi stream handler  Owner: TBD  Est: 1.5h  Completed: 2026 03 01
  - Dependencies: T32.1
  - Acceptance: workerService.AllReduce(stream) receives all AllReduceRequest messages from a peer until EOF, submits them to the active reduceSession, waits for the global result, and sends AllReduceResponse messages back on the stream. Root (rank 0) also contributes its own local tensors via a SetLocalTensors method. Multiple concurrent streams (one per non-root peer) are handled correctly. Metrics are recorded: allreduce_server_count (counter), allreduce_server_duration_seconds (histogram).
  - [x] S32.2.1 Implement AllReduce method on workerService: recv loop, submit to session, wait, send loop  Est: 30m
  - [x] S32.2.2 Add SetLocalTensors(tensors map[string]*pb.Tensor) method for root to inject its own gradients  Est: 15m
  - [x] S32.2.3 Add NewSession() method to reset the reduce session for each training step  Est: 10m
  - [x] S32.2.4 Write unit tests using mock bidi streams: single peer, two peers, stream error mid-recv  Est: 30m
  - [x] S32.2.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T32.3 Implement Barrier handler  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: T32.1
  - Acceptance: workerService.Barrier(ctx, req) increments an arrival counter for the current barrier epoch. When arrivals equal worldSize, all blocked callers are released and BarrierResponse is returned. If the context deadline expires before all peers arrive, the handler returns a DeadlineExceeded gRPC status. Barrier epoch increments after each completed barrier to prevent stale responses.
  - [x] S32.3.1 Add barrierState struct to workerService with epoch int64, arrived int32, mu sync.Mutex, cond *sync.Cond  Est: 15m
  - [x] S32.3.2 Implement Barrier method: increment arrived, wait on cond, broadcast when all arrived  Est: 15m
  - [x] S32.3.3 Write unit tests: 3 concurrent callers all released, timeout when one missing, sequential barriers with epoch increment  Est: 20m
  - [x] S32.3.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T32.4 Implement Broadcast handler  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: T32.1
  - Acceptance: workerService.Broadcast(ctx, req) stores the broadcast tensor in a thread-safe map keyed by name. Non-root workers call this RPC on root to retrieve the broadcast tensor. Root sets the tensor via a SetBroadcastTensor(name string, tensor *pb.Tensor) method before non-root workers call. If the tensor is not yet available, the handler waits (with context deadline) for it to be set.
  - [x] S32.4.1 Add broadcastStore (sync.Map or mutex-guarded map) to workerService with wait channels  Est: 15m
  - [x] S32.4.2 Implement Broadcast method and SetBroadcastTensor method  Est: 15m
  - [x] S32.4.3 Write unit tests: set then retrieve, wait then set (concurrent), timeout  Est: 15m
  - [x] S32.4.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T32.5 Add input validation to all RPC handlers  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T32.2, T32.3, T32.4
  - Acceptance: AllReduce validates non-nil tensor, non-empty name, valid shape (all dimensions > 0, product matches data length). Barrier validates rank is in range [0, worldSize). Broadcast validates non-nil tensor, non-empty name, valid shape. Invalid requests return gRPC InvalidArgument status with descriptive message. This task completes the previously skipped T23.2.
  - Risk: Must not break existing Coordinator RPC validation (already has validation in coordinator.go).
  - [x] S32.5.1 Add validateTensor(t *pb.Tensor, fieldName string) error helper  Est: 10m
  - [x] S32.5.2 Add validation calls at the top of AllReduce, Barrier, Broadcast  Est: 10m
  - [x] S32.5.3 Write tests for each validation error case (nil tensor, empty name, shape mismatch, rank out of range)  Est: 15m
  - [x] S32.5.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T32.6 Run linters and verify coverage for E32  Owner: TBD  Est: 15m  Completed: 2026 03 01
  - Dependencies: T32.5
  - Acceptance: golangci-lint reports 0 issues on distributed/. go test -cover -race ./distributed/ shows >= 95% coverage on worker_service.go. go vet ./distributed/ clean.
  - [x] S32.6.1 Run golangci-lint run ./distributed/  Est: 5m
  - [x] S32.6.2 Run go test -cover -race ./distributed/  Est: 5m
  - [x] S32.6.3 Fix any remaining lint or coverage gaps  Est: 5m

#### E33: gRPC Strategy (InternalStrategy[T] over gRPC)

Implement GrpcStrategy[T] that connects the InternalStrategy[T] interface
to the gRPC transport layer, bridging the high-level AllReduceStrategy with
the concrete WorkerService.

- [x] T33.1 Create GrpcStrategy[T] struct  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: E32
  - Acceptance: A GrpcStrategy[T] struct in distributed/grpc_strategy.go implements InternalStrategy[T]. Fields: rank int, size int, workerService *workerService, serverManager ServerManager, networkManager NetworkManager, peerClients []pb.DistributedServiceClient, peerConns []*grpc.ClientConn, coordinatorClient CoordinatorClient, coordinatorConn *grpc.ClientConn, logger log.Logger, collector metrics/runtime.Collector. Static interface assertion var _ InternalStrategy[float32] = (*GrpcStrategy[float32])(nil) compiles.
  - [x] S33.1.1 Create distributed/grpc_strategy.go with struct and NewGrpcStrategy constructor  Est: 20m
  - [x] S33.1.2 Add tensor conversion helpers: tensorToProto(t *tensor.TensorNumeric[T]) *pb.Tensor and protoToTensor(p *pb.Tensor) (*tensor.TensorNumeric[T], error)  Est: 20m
  - [x] S33.1.3 Write unit tests for tensor conversion round-trip (float32, various shapes)  Est: 15m
  - [x] S33.1.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T33.2 Implement Init (register, connect, start server)  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: T33.1
  - Acceptance: GrpcStrategy.Init(rank, size, coordinatorAddress) registers the worker with the coordinator via RegisterWorker RPC, receives the assigned rank and peer addresses, starts the local gRPC server (workerService) via ServerManager, and connects to all peer workers via NetworkManager.ConnectToPeers. After Init, the strategy is ready for AllReduceGradients calls.
  - [x] S33.2.1 Implement Init method: register with coordinator, start server, connect to peers  Est: 30m
  - [x] S33.2.2 Write unit tests with mock coordinator, ServerManager, and NetworkManager  Est: 25m
  - [x] S33.2.3 Run golangci-lint and go test -cover  Est: 5m

- [x] T33.3 Implement AllReduceGradients  Owner: TBD  Est: 1.5h  Completed: 2026 03 01
  - Dependencies: T33.2
  - Acceptance: GrpcStrategy.AllReduceGradients(gradients) converts each gradient tensor to pb.Tensor, opens an AllReduce bidi stream to root (rank 0), sends all gradients, receives the averaged result, and converts back to tensor.TensorNumeric[T], updating the gradient map in place. If this worker IS root (rank 0): sets local tensors on workerService, creates a new reduce session, and waits for peers to complete the all-reduce. Metrics: allreduce_client_count, allreduce_client_duration_seconds.
  - [x] S33.3.1 Implement AllReduceGradients for non-root workers: open stream to root, send gradients, recv result  Est: 30m
  - [x] S33.3.2 Implement AllReduceGradients for root worker: set local tensors, new session, wait for completion  Est: 30m
  - [x] S33.3.3 Write unit tests: non-root sends and receives (mock stream), root processes (mock peers)  Est: 25m
  - [x] S33.3.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T33.4 Implement Barrier and BroadcastTensor  Owner: TBD  Est: 45m  Completed: 2026 03 01
  - Dependencies: T33.2
  - Acceptance: GrpcStrategy.Barrier() calls Barrier RPC on root (rank 0). Root calls its own workerService.Barrier locally. Non-root workers send BarrierRequest with their rank. GrpcStrategy.BroadcastTensor(t, rootRank) root converts tensor to proto and sets it on workerService via SetBroadcastTensor, then non-root workers call Broadcast RPC on root to retrieve it. After receiving, non-root workers update the tensor in place.
  - [x] S33.4.1 Implement Barrier: non-root calls RPC on root, root calls local service  Est: 15m
  - [x] S33.4.2 Implement BroadcastTensor: root sets, non-root retrieves via RPC  Est: 15m
  - [x] S33.4.3 Write unit tests with mock clients  Est: 15m
  - [x] S33.4.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T33.5 Implement Shutdown  Owner: TBD  Est: 30m  Completed: 2026 03 01
  - Dependencies: T33.2
  - Acceptance: GrpcStrategy.Shutdown() unregisters the worker from the coordinator (UnregisterWorker RPC), closes all peer connections (NetworkManager.CloseConnections), stops the gRPC server (ServerManager.GracefulStop), and closes the coordinator connection. All operations are idempotent via sync.Once. No panic on double-call.
  - [x] S33.5.1 Implement Shutdown with sync.Once and ordered cleanup  Est: 15m
  - [x] S33.5.2 Write unit tests: single shutdown, double shutdown (idempotent), shutdown with failed unregister  Est: 15m
  - [x] S33.5.3 Run golangci-lint and go test -cover  Est: 5m

- [x] T33.6 Run linters and verify coverage for E33  Owner: TBD  Est: 15m  Completed: 2026 03 01
  - Dependencies: T33.5
  - Acceptance: golangci-lint reports 0 issues on distributed/. go test -cover -race ./distributed/ shows >= 95% coverage on grpc_strategy.go. go vet clean.
  - [x] S33.6.1 Run golangci-lint, go vet, go test -cover -race  Est: 10m
  - [x] S33.6.2 Fix any remaining issues  Est: 5m

#### E34: Multi-Worker Integration Tests

Prove distributed operations work correctly over real gRPC connections
using in-process bufconn listeners (same pattern as coordinator tests).

- [x] T34.1 In-process multi-worker AllReduce integration test  Owner: TBD  Est: 1.5h  Completed: 2026 03 01
  - Dependencies: E33
  - Acceptance: A test starts a coordinator and 3 GrpcStrategy workers in the same process using bufconn. Each worker has different gradient tensors. After AllReduceGradients, all workers have identical averaged gradients. Mathematical correctness: if worker 0 has [1,2,3], worker 1 has [4,5,6], worker 2 has [7,8,9], all should get [4,5,6] after averaging. Test runs with -race flag.
  - [x] S34.1.1 Create distributed/integration_test.go with bufconn test harness (start coordinator, create workers)  Est: 30m
  - [x] S34.1.2 Write TestMultiWorkerAllReduce with 3 workers and verify averaged gradients  Est: 30m
  - [x] S34.1.3 Write TestMultiWorkerAllReduce_SingleWorker edge case (world size = 1)  Est: 15m
  - [x] S34.1.4 Run with -race flag  Est: 5m

- [x] T34.2 In-process Barrier and Broadcast integration tests  Owner: TBD  Est: 1h  Completed: 2026 03 01
  - Dependencies: T34.1
  - Acceptance: Barrier test: 3 workers call Barrier concurrently; all are released after the last worker arrives; timing proves no worker proceeds early. Broadcast test: root broadcasts tensor [10,20,30] to all workers; all non-root workers receive exact copy.
  - [x] S34.2.1 Write TestMultiWorkerBarrier with 3 workers and timing verification  Est: 20m
  - [x] S34.2.2 Write TestMultiWorkerBroadcast from root to 2 non-root workers  Est: 20m
  - [x] S34.2.3 Run with -race flag  Est: 5m

- [x] T34.3 Error and edge case integration tests  Owner: TBD  Est: 45m  Completed: 2026 03 01  Note: TestAllReduce_ContextCancellation implemented; S34.3.2 and S34.3.3 covered by existing tests
  - Dependencies: T34.1
  - Acceptance: Test context cancellation during AllReduce (one worker cancels mid-stream, others get error). Test invalid inputs rejected over the wire (gRPC InvalidArgument status). Test single-worker mode (world size = 1, all ops are no-ops or self-reduces).
  - [x] S34.3.1 Write TestAllReduce_ContextCancellation  Est: 15m
  - [x] S34.3.2 Write TestAllReduce_InvalidInput over gRPC  Est: 15m
  - [x] S34.3.3 Write TestSingleWorker (world size 1)  Est: 10m
  - [x] S34.3.4 Run with -race flag  Est: 5m

- [x] T34.4 TLS multi-worker integration test  Owner: TBD  Est: 30m  Completed: 2026 03 02
  - Dependencies: T34.1
  - Acceptance: Same as T34.1 but with TLS enabled using self-signed certificates (generated at test time). Verifies TLS handshake works for both coordinator and peer connections. Uses the existing TLSConfig from distributed/tlsconfig.go.
  - [x] S34.4.1 Add TLS cert generation helper to test (reuse pattern from tlsconfig_test.go)  Est: 10m
  - [x] S34.4.2 Write TestMultiWorkerAllReduce_TLS with TLS-enabled coordinator and workers  Est: 15m
  - [x] S34.4.3 Run with -race flag  Est: 5m

- [x] T34.5 Run linters and verify coverage for E34  Owner: TBD  Est: 15m  Completed: 2026 03 02
  - Dependencies: T34.4
  - Acceptance: golangci-lint 0 issues. go test -cover -race ./distributed/... shows integration tests pass. go vet clean.
  - [x] S34.5.1 Run golangci-lint, go vet, go test -cover -race  Est: 10m
  - [x] S34.5.2 Fix any remaining issues  Est: 5m

#### E35: Worker Lifecycle and CLI Integration

Create a WorkerNode struct that ties together the distributed components
and integrate with the CLI, health checks, and shutdown coordinator.

- [x] T35.1 Create WorkerNode struct  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: E33
  - Acceptance: A WorkerNode struct in distributed/worker_node.go encapsulates: GrpcStrategy (or AllReduceStrategy wrapping two GrpcStrategies), coordinator connection, health check registration, and shutdown.Closer implementation. WorkerNode.Start(ctx, cfg) initializes the strategy, registers with the coordinator, starts the gRPC server, connects to peers, and registers an engine health check. WorkerNode.Close(ctx) triggers orderly shutdown. WorkerNode can be registered with the shutdown.Coordinator from the shutdown package.
  - [x] S35.1.1 Create distributed/worker_node.go with WorkerNode struct and constructor  Est: 20m
  - [x] S35.1.2 Implement Start method: init strategy, register health check  Est: 20m
  - [x] S35.1.3 Implement Close method satisfying shutdown.Closer  Est: 10m
  - [x] S35.1.4 Write unit tests: start/stop lifecycle, double close is safe  Est: 15m
  - [x] S35.1.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T35.2 Add worker CLI command  Owner: TBD  Est: 45m  Completed: 2026 03 02  Note: Created in cmd/cli/worker.go and registered in cmd/zerfoo/main.go
  - Dependencies: T35.1, T25.3
  - Acceptance: A `worker` subcommand in cmd/zerfoo starts a distributed training worker. Flags: --coordinator-address (required), --worker-address (required), --worker-id (defaults to hostname), --config (optional JSON config path). The command creates a WorkerNode, registers it with the shutdown coordinator, connects signal handling via cli.SignalContext, and blocks until SIGTERM/SIGINT. On signal, graceful shutdown is triggered.
  - [x] S35.2.1 Create cmd/zerfoo/worker.go with worker command registration  Est: 15m
  - [x] S35.2.2 Implement worker command: parse flags, create WorkerNode, start, wait for signal  Est: 20m
  - [x] S35.2.3 Write test verifying command parses flags and creates worker (mock coordinator)  Est: 15m
  - [x] S35.2.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T35.3 End-to-end worker lifecycle integration test  Owner: TBD  Est: 45m  Completed: 2026 03 02
  - Dependencies: T35.1, T35.2
  - Acceptance: Test starts a coordinator, starts 2 WorkerNodes, verifies both workers register successfully (coordinator reports 2 workers), runs a health check on each worker, then triggers shutdown. After shutdown, both workers have deregistered from the coordinator (coordinator reports 0 workers). Test runs with -race.
  - [x] S35.3.1 Write TestWorkerNodeLifecycle in distributed/integration_test.go  Est: 25m
  - [x] S35.3.2 Verify health check integration (readiness check passes during run, fails after stop)  Est: 15m
  - [x] S35.3.3 Run with -race flag  Est: 5m

- [x] T35.4 Run linters and verify coverage for E35  Owner: TBD  Est: 15m  Completed: 2026 03 02  Note: distributed/ 96.0%, cmd/cli/ 91.4%
  - Dependencies: T35.3
  - Acceptance: golangci-lint 0 issues. go test -cover -race ./distributed/... and ./cmd/zerfoo/... pass. go vet clean.
  - [x] S35.4.1 Run golangci-lint, go vet, go test -cover -race  Est: 10m
  - [x] S35.4.2 Fix any remaining issues  Est: 5m

- [x] T35.5 Update plan and documentation  Owner: TBD  Est: 30m  Completed: 2026 03 02
  - Dependencies: T35.4
  - Acceptance: docs/plan.md has all Phase 5 tasks marked complete. docs/runbook.md has a new "Distributed Worker Setup" section. docs/troubleshooting.md updated if new error patterns were discovered. T23.2 marked as completed via T32.5.
  - [x] S35.5.1 Update docs/plan.md: mark all Phase 5 tasks [x], update progress log  Est: 10m
  - [x] S35.5.2 Add "Distributed Worker Setup" section to docs/runbook.md  Est: 10m
  - [x] S35.5.3 Review and update docs/troubleshooting.md  Est: 10m

#### E36: Phase 5 Final Verification

Run the full quality gate suite after all Phase 5 work is complete.

- [x] T36.1 Run full test suite with coverage and race detector  Owner: TBD  Est: 30m  Completed: 2026 03 02  Note: distributed/ 96.0% coverage, all tests pass with -race
  - Dependencies: E32, E33, E34, E35
  - Acceptance: go test ./... -cover -race passes. distributed/ package coverage >= 95%. No new data races. All existing tests still pass (no regressions).
  - [x] S36.1.1 Run go test ./... -cover -race  Est: 15m
  - [x] S36.1.2 Verify distributed/ package coverage >= 95%  Est: 10m
  - [x] S36.1.3 Fix any regressions  Est: 5m

- [x] T36.2 Run linters and verify CI compatibility  Owner: TBD  Est: 15m  Completed: 2026 03 02  Note: golangci-lint 0 issues, go vet clean on all packages
  - Dependencies: T36.1
  - Acceptance: golangci-lint run ./... reports 0 issues. go vet ./... clean. CI workflow (ci.yml) does not need changes (existing test commands cover new code).
  - [x] S36.2.1 Run golangci-lint run ./...  Est: 5m
  - [x] S36.2.2 Run go vet ./...  Est: 5m
  - [x] S36.2.3 Verify ci.yml covers new code without changes  Est: 5m

---

### Phase 6: Open Weights Model Import Support

#### Phase 6 Context

Zerfoo can train and run inference on models built directly with its layer API.
Importing pre-trained open-weights models (Gemma 3, Kimi-VL) requires closing
gaps in the ONNX import pipeline (zonnx repo) and in the zerfoo layer registry.

Gap analysis conducted on 2026 03 02 identified the following blockers:

**Gemma 3 (4-bit quantized transformer, 18 layers, ONNX opset 21):**
- zonnx converter: AttributeProto_TENSOR case missing in convertAttribute()
  blocks 7 Constant nodes from converting.
- zonnx converter: UINT8 dtype missing in convertTensorWithPath() blocks
  MatMulNBits (126 instances) quantized weight tensors.
- MatMulNBits and Constant layers exist in zerfoo (layers/core/) but lack
  registry builder functions and are not registered in layers/registry/.
- model/builder.go has no dispatch for "MatMulNBits" or "Constant" ZMF node types.

**Kimi-VL-A3B (MoonLight language model + SigLIP vision encoder):**
- Vision encoder uses Conv2d, Pad, Slice, Resize, BatchNormalization,
  GlobalAveragePool -- none implemented in zerfoo.
- Softmax exists in the compute engine but is not registered as a graph layer node.
- Standard LayerNormalization (with bias) is not registered (only Simplified and
  Skip variants are).
- Slice, Pad, TopK, Erf are missing entirely.
- MoE (Mixture of Experts) gate routing and expert dispatch are not implemented.

#### Phase 6 Objectives

- P6-O1: Fix zonnx converter to handle TENSOR attributes and UINT8 dtype.
- P6-O2: Register MatMulNBits and Constant in zerfoo layer registry.
- P6-O3: Implement Softmax, Sigmoid, LayerNormalization, Slice, Pad, TopK, Erf.
- P6-O4: Implement Conv2d, GlobalAveragePool, BatchNormalization, Resize.
- P6-O5: Implement MixtureOfExperts layer for Kimi-VL language model.
- P6-O6: Validate Gemma 3 end-to-end with a forward pass integration test.
- P6-O7: Validate Kimi-VL vision encoder end-to-end with a forward pass test.

#### Phase 6 Non-Goals

- KV cache and autoregressive decoding (future phase; requires graph execution changes).
- Beam search or nucleus sampling strategies.
- Operator fusion or CUDA acceleration for new operators (correctness first).
- Model quantization at import time (only loading pre-quantized 4-bit weights).
- ZMF sub-graph support (MoE will hold expert tensors directly as a workaround).

#### Phase 6 Design Decisions

**4-bit weight packing:** MatMulNBits stores 4-bit weights packed two-per-byte
in UINT8 tensors. ZMF uses DataType=UINT8 for these. Dequantization happens in
MatMulNBits.Forward() which already uses numeric.Unpack4BitSlice internally.

**Conv2d strategy:** Use im2col + MatMul for correctness. im2col reshapes input
patches into a 2D matrix that is multiplied by the flattened kernel matrix.
This reuses the existing MatMul implementation without a specialized kernel.

**Multi-repo discipline:** zonnx and zerfoo are separate repos. Pre-commit hooks
reject multi-directory commits. All zonnx changes are committed in the zonnx
repo; all zerfoo layer/model changes are committed in the zerfoo repo.

---

#### E37: Complete Gemma 3 ONNX Import

Fix the zonnx converter and zerfoo layer registry to support all operators
used by Gemma 3 4B-IT quantized (ONNX opset 21).

- [ ] T37.1 Fix TENSOR attribute and UINT8 dtype in zonnx converter  Owner: TBD  Est: 1h
  - Dependencies: None
  - Files: zonnx/pkg/converter/converter.go (convertAttribute, convertTensorWithPath)
  - Acceptance: Add case AttributeProto_TENSOR in convertAttribute(): read embedded
    TensorProto via attr.T, call convertTensorWithPath(), store as ZMF TENSOR attribute.
    Add UINT8 and INT8 cases in convertTensorWithPath() dtype switch: read raw bytes
    from RawData field, store as []byte in ZMF Tensor.Data with DataType=UINT8/INT8.
    Test: convert a minimal ONNX graph with a Constant node (value = float32 [[1,2],[3,4]]);
    assert ZMF attribute type=TENSOR and values match. Test: UINT8 tensor with 4 bytes
    [0x12,0x34,0x56,0x78] is preserved in ZMF.
  - Risk: ONNX TensorProto has multiple storage formats (float_data, raw_data). Prioritize
    RawData for UINT8; fall through to typed fields for other types.
  - [ ] S37.1.1 Add AttributeProto_TENSOR case in convertAttribute()  Est: 20m
  - [ ] S37.1.2 Add UINT8 and INT8 dtype cases in convertTensorWithPath()  Est: 20m
  - [ ] S37.1.3 Write unit tests for TENSOR attribute and UINT8 dtype conversion  Est: 20m
  - [ ] S37.1.4 Run golangci-lint and go test -cover in zonnx/pkg/converter/  Est: 5m

- [ ] T37.2 Add BuildConstant[T] to zerfoo and register  Owner: TBD  Est: 45m
  - Dependencies: T37.1
  - Files: layers/core/constant.go, layers/registry/registry.go, model/builder.go
  - Acceptance: Add func BuildConstant[T generic.TensorNumeric](node *zmf.Node,
    inputs []graph.Node[T]) (graph.Node[T], error) to layers/core/constant.go. Read
    "value" attribute (ZMF Tensor) from node.Attributes. Decode into tensor.TensorNumeric[T].
    Construct and return a Constant[T] holding the decoded tensor. Register "Constant":
    BuildConstant[T] in layers/registry/RegisterAll[T](). Add case "Constant" in
    model/builder.go dispatch. Unit test: ZMF graph with single Constant node; forward
    pass returns tensor [1.0, 2.0, 3.0].
  - [ ] S37.2.1 Add BuildConstant[T] function to layers/core/constant.go  Est: 15m
  - [ ] S37.2.2 Register "Constant" in layers/registry/registry.go  Est: 5m
  - [ ] S37.2.3 Add "Constant" case in model/builder.go  Est: 5m
  - [ ] S37.2.4 Write unit tests for Constant layer build and forward  Est: 15m
  - [ ] S37.2.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T37.3 Add BuildMatMulNBits[T] to zerfoo and register  Owner: TBD  Est: 1.5h
  - Dependencies: T37.1
  - Files: layers/core/matmul_nbits.go, layers/registry/registry.go, model/builder.go
  - Acceptance: Add func BuildMatMulNBits[T generic.TensorNumeric](node *zmf.Node,
    inputs []graph.Node[T]) (graph.Node[T], error). Read node attributes: K (int),
    N (int), bits (int, default 4), block_size (int). Read weight tensor from node's
    UINT8 initializer (packed 4-bit bytes). Read scale tensor (float32). Optionally read
    zero_point tensor (UINT8). Construct MatMulNBits[T] with pre-loaded fields. Register
    "MatMulNBits" in layers/registry/RegisterAll[T](). Add "MatMulNBits" case in
    model/builder.go. Unit test: BuildMatMulNBits K=4, N=2, bits=4, block_size=4; run
    forward with float32 activations [1,4]; verify output shape [1,2] and values match
    manual dequantization reference (tolerance 1e-4).
  - [ ] S37.3.1 Add BuildMatMulNBits[T] function to layers/core/matmul_nbits.go  Est: 30m
  - [ ] S37.3.2 Register "MatMulNBits" in layers/registry/registry.go  Est: 5m
  - [ ] S37.3.3 Add "MatMulNBits" case in model/builder.go  Est: 5m
  - [ ] S37.3.4 Write unit tests: build, forward pass, dequantization correctness  Est: 30m
  - [ ] S37.3.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T37.4 Add zonnx converter handler for MatMulNBits  Owner: TBD  Est: 45m
  - Dependencies: T37.1
  - File: zonnx/pkg/converter/converter.go (convertNode function)
  - Acceptance: Add case "MatMulNBits" in convertNode(). Extract ONNX attrs: K (int),
    N (int), bits (int, default=4), block_size (int). Map inputs: [0]=A (float activations),
    [1]=B (uint8 quantized weights), [2]=scales (float32), [3]=zero_points (uint8 optional).
    Emit a ZMF Node with type="MatMulNBits" and attributes K, N, bits, block_size. Store B
    as a node initializer weight in ZMF. Test: convert a minimal ONNX graph with one
    MatMulNBits node; verify ZMF node type="MatMulNBits" and all attributes are present.
  - [ ] S37.4.1 Add "MatMulNBits" case in convertNode()  Est: 25m
  - [ ] S37.4.2 Write unit test for MatMulNBits node conversion  Est: 15m
  - [ ] S37.4.3 Run golangci-lint and go test -cover in zonnx/pkg/converter/  Est: 5m

- [ ] T37.5 Add zonnx importer builders for Constant and MatMulNBits  Owner: TBD  Est: 1h
  - Dependencies: T37.1, T37.4
  - Files: zonnx/pkg/importer/layers/constant.go (new), matmul_nbits.go (new)
  - Acceptance: BuildConstant reads ONNX Constant node "value" attribute (TensorProto),
    calls convertTensorWithPath(), emits a ZMF Constant node. BuildMatMulNBits reads ONNX
    attrs (K, N, bits, block_size), reads UINT8 weight initializer, emits ZMF MatMulNBits
    node. Both registered in zonnx importer layer registry. Test: ONNX graph with Constant
    and MatMulNBits nodes imports to ZMF; ZMF graph has nodes of correct types.
  - [ ] S37.5.1 Create zonnx/pkg/importer/layers/constant.go with BuildConstant  Est: 20m
  - [ ] S37.5.2 Create zonnx/pkg/importer/layers/matmul_nbits.go with BuildMatMulNBits  Est: 25m
  - [ ] S37.5.3 Register both builders in zonnx importer registry  Est: 5m
  - [ ] S37.5.4 Write unit tests for both builders  Est: 15m
  - [ ] S37.5.5 Run golangci-lint and go test -cover in zonnx/pkg/importer/  Est: 5m

- [ ] T37.6 Gemma 3 ONNX import smoke test  Owner: TBD  Est: 1.5h
  - Dependencies: T37.1, T37.2, T37.3, T37.4, T37.5
  - File: tests/parity/gemma3_import_test.go (new, in zerfoo repo)
  - Acceptance: Test is skipped if GEMMA3_ONNX_PATH env var is not set. Load Gemma 3
    4B-IT quantized ONNX model. Run zonnx convert. Assert: conversion completes without
    error. Assert: ZMF graph contains MatMulNBits (126 nodes), SimplifiedLayerNormalization,
    GroupQueryAttention, Constant, Reshape, Transpose nodes. Load ZMF into zerfoo. Run one
    forward pass with input shape [1, 8] int64 token IDs. Assert: output shape [1, 8, V]
    where V >= 256000. Assert: no NaN or Inf in output.
  - [ ] S37.6.1 Write TestGemma3Import (skip if env var not set)  Est: 30m
  - [ ] S37.6.2 Run test with a real Gemma 3 model file; fix any import errors  Est: 45m
  - [ ] S37.6.3 Run golangci-lint and go test -cover  Est: 5m

- [ ] T37.7 Run linters and verify coverage for E37  Owner: TBD  Est: 15m
  - Dependencies: T37.6
  - Acceptance: golangci-lint 0 issues in all modified directories. go test -cover on
    layers/core/ >= 85%. go test -cover on zonnx/pkg/converter/ >= 80%.
  - [ ] S37.7.1 Run golangci-lint in zerfoo layers/core/, layers/registry/, model/  Est: 5m
  - [ ] S37.7.2 Run golangci-lint in zonnx/pkg/converter/ and zonnx/pkg/importer/  Est: 5m
  - [ ] S37.7.3 Verify coverage thresholds; fix any gaps  Est: 5m

#### E38: Core Missing Operators

Implement graph-level layer nodes for operators missing from the zerfoo registry.
These are needed for general transformer inference and as building blocks for VLMs.

- [x] T38.1 Implement Softmax layer and register  Owner: TBD  Est: 45m  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/activations/softmax.go (new), layers/registry/registry.go, model/builder.go
  - Acceptance: Softmax[T] struct with axis int attribute. Forward: for each slice along
    axis subtract max (numerical stability), exponentiate, divide by sum. BuildSoftmax[T]
    reads "axis" from node attributes (default -1). Register "Softmax" in RegisterAll[T].
    Add "Softmax" case in model/builder.go. Test: Softmax([[1,2,3],[4,5,6]], axis=1) matches
    scipy.special.softmax reference (tolerance 1e-6).
  - [x] S38.1.1 Create layers/activations/softmax.go with Softmax[T] and BuildSoftmax[T]  Est: 20m
  - [x] S38.1.2 Register "Softmax" in RegisterAll  Est: 5m
  - [x] S38.1.3 Write unit tests with numerical reference  Est: 15m
  - [x] S38.1.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T38.2 Implement Sigmoid layer and register  Owner: TBD  Est: 30m  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/activations/registry.go (BuildSigmoid added), layers/registry/registry.go
  - Acceptance: BuildSigmoid[T] wrapping existing NewSigmoid. Register "Sigmoid". Tests pass.
  - [x] S38.2.1 Add BuildSigmoid to layers/activations/registry.go  Est: 10m
  - [x] S38.2.2 Register "Sigmoid" in RegisterAll  Est: 5m
  - [x] S38.2.3 Write unit tests  Est: 10m
  - [x] S38.2.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T38.3 Implement standard LayerNormalization layer and register  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/normalization/registry.go (BuildLayerNormalization + resolveParam added)
  - Acceptance: BuildLayerNormalization[T] reads epsilon, resolves scale/bias params via
    multiple naming patterns, creates LayerNormalization with featureDim from param shape.
    Register "LayerNormalization". Tests pass including forward pass verification.
  - [x] S38.3.1 Add BuildLayerNormalization to layers/normalization/registry.go  Est: 25m
  - [x] S38.3.2 Register "LayerNormalization" in RegisterAll  Est: 5m
  - [x] S38.3.3 Write unit tests vs reference  Est: 20m
  - [x] S38.3.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T38.4 Implement Slice layer and register  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/core/slice.go (new), layers/registry/registry.go
  - Acceptance: Slice[T] with starts/ends/axes/steps. Returns dense copy. BuildSlice[T].
    Register "Slice". Tests cover 1D/2D/negative indices/clamped end.
  - [x] S38.4.1 Create layers/core/slice.go  Est: 25m
  - [x] S38.4.2 Register "Slice" in RegisterAll  Est: 5m
  - [x] S38.4.3 Write unit tests  Est: 20m
  - [x] S38.4.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T38.5 Implement Pad layer and register  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/core/pad.go (new), layers/registry/registry.go
  - Acceptance: Pad[T] with pads []int64 and constantValue. BuildPad[T]. Register "Pad".
    Tests cover 1D/2D/constant value/mismatch errors.
  - [x] S38.5.1 Create layers/core/pad.go  Est: 25m
  - [x] S38.5.2 Register "Pad" in RegisterAll  Est: 5m
  - [x] S38.5.3 Write unit tests  Est: 20m
  - [x] S38.5.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T38.6 Implement TopK layer and register  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/core/topk.go (new), layers/registry/registry.go
  - Acceptance: TopK[T] with k/axis/largest/sorted. Returns values only (not indices).
    BuildTopK[T]. Register "TopK". Tests cover largest/smallest/large-k/builder paths.
  - [x] S38.6.1 Create layers/core/topk.go  Est: 30m
  - [x] S38.6.2 Register "TopK" in RegisterAll  Est: 5m
  - [x] S38.6.3 Write unit tests  Est: 20m
  - [x] S38.6.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T38.7 Implement Erf layer and register  Owner: TBD  Est: 30m  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/activations/erf.go (new), layers/registry/registry.go
  - Acceptance: NewErf[T tensor.Float] using math.Erf via BaseActivation. BuildErf[T].
    Register "Erf". Tests verify erf(0)=0, erf(1)~0.8427, erf(-1)~-0.8427.
  - [x] S38.7.1 Create layers/activations/erf.go  Est: 10m
  - [x] S38.7.2 Register "Erf" in RegisterAll  Est: 5m
  - [x] S38.7.3 Write unit tests  Est: 10m
  - [x] S38.7.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T38.8 Add zonnx importer builders for E38 operators  Owner: TBD  Est: 1.5h  Completed: 2026 03 02
  - Dependencies: T38.1 through T38.7
  - Files: zonnx/pkg/converter/converter.go (Slice/Pad/TopK cases),
    zonnx/pkg/importer/layers/{softmax,sigmoid,erf,layer_norm,slice,pad,topk}.go (new)
  - Note: Slice/Pad/TopK needed converter special cases to promote input tensors to ZMF
    attributes. Softmax/Sigmoid/Erf/LayerNorm work via the generic convertNode path.
  - [x] S38.8.1 Create zonnx importer builders for Softmax, Sigmoid, LayerNorm  Est: 30m
  - [x] S38.8.2 Create zonnx importer builders for Slice, Pad  Est: 20m
  - [x] S38.8.3 Create zonnx importer builders for TopK, Erf  Est: 20m
  - [x] S38.8.4 Register all builders in zonnx importer registry via init()  Est: 5m
  - [x] S38.8.5 Write round-trip tests for each operator in converter_test.go  Est: 20m
  - [x] S38.8.6 Run golangci-lint and go test -cover in zonnx/  Est: 5m

- [x] T38.9 Run linters and verify coverage for E38  Owner: TBD  Est: 15m  Completed: 2026 03 02
  - Dependencies: T38.1 through T38.7 (T38.8 pending)
  - Acceptance: golangci-lint 0 issues in layers/activations/, layers/core/,
    layers/normalization/, layers/registry/. go test -race ./... all pass.
  - [x] S38.9.1 Run golangci-lint and go test -cover -race in all modified dirs  Est: 10m
  - [x] S38.9.2 Fix any remaining issues  Est: 5m  Note: fixed copyloopvar and SA9003 in pad/topk

#### E39: Vision Encoder Operators

Implement operators for the SigLIP vision encoder used in MoondreamV2 and
Kimi-VL. All operators use NCHW tensor format [N, C, H, W].

- [x] T39.1 Implement Conv2d layer and register  Owner: TBD  Est: 2h  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/core/conv2d.go (new), layers/registry/registry.go, model/builder.go
  - Acceptance: Conv2d[T] struct. Attributes: strides [2]int, pads [4]int
    (top,left,bottom,right), dilations [2]int, groups int. Fields: kernel
    [out_C, in_C/groups, kH, kW], bias [out_C] optional. Forward: im2col reshapes input
    patches to [N*H_out*W_out, in_C*kH*kW]; multiply by flattened kernel
    [in_C*kH*kW, out_C]; reshape to [N, out_C, H_out, W_out]. BuildConv2d[T] reads kernel
    and bias from node initializers. Register "Conv". Test 1: [1,1,5,5] all-ones input,
    [1,1,3,3] all-ones kernel, stride=1, pad=0 returns [1,1,3,3] where each value = 9.0.
    Test 2: stride=2 halves spatial dims. Test 3: padding preserves spatial dims.
  - Deviation: Used direct nested-loop convolution instead of im2col+MatMul to avoid
    allocating a large intermediate matrix. Simpler and correct for inference workloads.
  - [x] S39.1.1 Implement Conv2d Forward using nested loops with ops.Mul/ops.Add  Est: 30m
  - [x] S39.1.2 Implement BuildConv2d[T] reading strides/pads/dilations/group attributes  Est: 20m
  - [x] S39.1.3 Register "Conv"  Est: 5m
  - [x] S39.1.4 Write unit tests (table-driven: stride=1, stride=2, with-bias)  Est: 25m
  - [x] S39.1.5 Run golangci-lint and go test -cover  Est: 5m

- [x] T39.2 Implement GlobalAveragePool layer and register  Owner: TBD  Est: 30m  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/core/global_avg_pool.go (new), layers/registry/registry.go
  - [x] S39.2.1 Create layers/core/global_avg_pool.go  Est: 10m
  - [x] S39.2.2 Register "GlobalAveragePool"  Est: 5m
  - [x] S39.2.3 Write unit tests  Est: 10m
  - [x] S39.2.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T39.3 Implement BatchNormalization layer (inference mode) and register  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/normalization/batch_norm.go (new), layers/registry/registry.go
  - [x] S39.3.1 Create layers/normalization/batch_norm.go  Est: 25m
  - [x] S39.3.2 Register "BatchNormalization"  Est: 5m
  - [x] S39.3.3 Write unit tests (zero-mean, scale+bias, spatial dims)  Est: 20m
  - [x] S39.3.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T39.4 Implement Resize layer and register  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: None
  - Files: layers/core/resize.go (new), layers/registry/registry.go
  - [x] S39.4.1 Create layers/core/resize.go (nearest neighbor)  Est: 25m
  - [x] S39.4.2 Register "Resize"  Est: 5m
  - [x] S39.4.3 Write unit tests (scales and sizes modes)  Est: 15m
  - [x] S39.4.4 Run golangci-lint and go test -cover  Est: 5m

- [x] T39.5 Add zonnx importer builders for E39 operators  Owner: TBD  Est: 1.5h  Completed: 2026 03 02
  - Dependencies: T39.1 through T39.4
  - Files: zonnx/pkg/importer/layers/conv.go (new), global_avg_pool.go (new),
    batch_norm.go (new), resize.go (new); zonnx/pkg/converter/converter.go
  - [x] S39.5.1 Create importer stubs for Conv, GlobalAveragePool  Est: 30m
  - [x] S39.5.2 Create importer stubs for BatchNormalization, Resize  Est: 30m
  - [x] S39.5.3 Add Resize special case in converter (promote scales/sizes inputs)  Est: 10m
  - [x] S39.5.4 Fix converter to skip empty optional ONNX inputs  Est: 5m
  - [x] S39.5.5 Write round-trip tests for Resize (scales + sizes variants)  Est: 20m
  - [x] S39.5.6 Run golangci-lint and go test ./...  Est: 5m

- [x] T39.6 Run linters and verify coverage for E39  Owner: TBD  Est: 15m  Completed: 2026 03 02
  - Dependencies: T39.5
  - [x] S39.6.1 Run golangci-lint and go test -cover -race: 0 issues, all pass  Est: 10m

#### E40: Mixture of Experts

Implement MixtureOfExperts[T] for Kimi-VL-A3B (MoonLight uses sparse MoE
with top-2 expert routing per token).

- [x] T40.1 Implement MoE gate routing layer  Owner: TBD  Est: 1.5h  Completed: 2026 03 02
  - Dependencies: T38.1 (Softmax), T38.6 (TopK)
  - Files: layers/core/moe.go (new)
  - Deviation: gateWeight is passed as a runtime Forward input (not from params) to match
    the ONNX/ZMF pattern used by Conv2d and BatchNorm. MoEGate.route() is a private method
    called by both Forward and MixtureOfExperts. Returns [seqLen, topK] weight tensor.
  - [x] S40.1.1 Implement MoEGate struct and Forward  Completed: 2026 03 02
  - [x] S40.1.2 Add BuildMoEGate[T] and register "MoEGate"  Completed: 2026 03 02
  - [x] S40.1.3 Write unit tests  Completed: 2026 03 02
  - [x] S40.1.4 Run golangci-lint and go test -cover  Completed: 2026 03 02

- [x] T40.2 Implement MoE expert dispatch and aggregate  Owner: TBD  Est: 2h  Completed: 2026 03 02
  - Dependencies: T40.1
  - Files: layers/core/moe.go (extended), layers/registry/registry.go
  - Deviation: Experts are graph.Node[T] instances set at construction time. ZMF sub-graph
    loading not yet supported; BuildMixtureOfExperts leaves experts=nil (documented as tech
    debt). Test uses identityExpert and scale2Expert helper types.
  - [x] S40.2.1 Implement MixtureOfExperts struct and Forward  Completed: 2026 03 02
  - [x] S40.2.2 Add BuildMixtureOfExperts[T] with expert loading strategy  Completed: 2026 03 02
  - [x] S40.2.3 Register "MixtureOfExperts"  Completed: 2026 03 02
  - [x] S40.2.4 Write unit tests  Completed: 2026 03 02
  - [x] S40.2.5 Run golangci-lint and go test -cover  Completed: 2026 03 02

- [x] T40.3 Add zonnx importer builders for MoE operators  Owner: TBD  Est: 1h  Completed: 2026 03 02
  - Dependencies: T40.1, T40.2
  - File: zonnx/pkg/importer/layers/moe.go (new)
  - [x] S40.3.1 Create zonnx/pkg/importer/layers/moe.go  Completed: 2026 03 02
  - [x] S40.3.3 Run golangci-lint and go test -cover  Completed: 2026 03 02

- [x] T40.4 Run linters and verify coverage for E40  Owner: TBD  Est: 15m  Completed: 2026 03 02
  - Dependencies: T40.3
  - Result: golangci-lint 0 issues. moe.go package-level coverage 93.8%; all functions >= 87%.
  - [x] S40.4.1 Run golangci-lint and go test -cover -race  Completed: 2026 03 02
  - [x] S40.4.2 Fix any remaining issues  Completed: 2026 03 02

#### E41: Gemma 3 End-to-End Validation

- [ ] T41.1 Gemma 3 forward pass parity test  Owner: TBD  Est: 2h
  - Dependencies: E37, E38
  - File: tests/parity/gemma3_test.go (new, in zerfoo repo)
  - Acceptance: Skipped if GEMMA3_ZMF_PATH env var not set. Load ZMF model. Create
    float32 CPUEngine. Build graph. Run forward pass with input [1, 8] int64 token IDs
    [1,2,3,4,5,6,7,8]. Assert: output shape [1, 8, V] where V >= 256000. Assert: no NaN
    or Inf in output. Optionally compare top-5 logits for first token against golden file
    tests/parity/gemma3_golden.json (tolerance 0.1 for 4-bit quantized model).
  - [ ] S41.1.1 Create tests/parity/gemma3_test.go  Est: 45m
  - [ ] S41.1.2 Run test with real model; capture golden output  Est: 45m
  - [ ] S41.1.3 Save golden file and verify test passes  Est: 20m
  - [ ] S41.1.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T41.2 Gemma 3 greedy decode smoke test  Owner: TBD  Est: 1h
  - Dependencies: T41.1
  - File: tests/parity/gemma3_test.go (extend)
  - Acceptance: Skipped if GEMMA3_ZMF_PATH not set. Run 5 greedy decode steps: argmax
    over vocab of last-position logits, append token, re-run. Assert: 5 output tokens in
    valid range [0, V). Assert: no panic or error. NOTE: without KV cache each step runs
    the full sequence (slow but correct for smoke testing).
  - [ ] S41.2.1 Implement greedy decode loop in test  Est: 30m
  - [ ] S41.2.2 Run and verify  Est: 20m
  - [ ] S41.2.3 Run golangci-lint and go test  Est: 5m

#### E42: Kimi-VL Vision Encoder Validation

- [ ] T42.1 SigLIP vision encoder forward pass test  Owner: TBD  Est: 2h
  - Dependencies: E39
  - File: tests/parity/siglip_test.go (new, in zerfoo repo)
  - Acceptance: Skipped if SIGLIP_ZMF_PATH env var not set. Load ZMF model. Create
    float32 CPUEngine. Run forward pass with input [1, 3, 224, 224] float32 (normalized
    image). Assert: output shape [1, 196, embed_dim] (patch_size=16, 224/16=14, 14*14=196).
    Assert: no NaN or Inf. Optionally compare CLS token embedding vs HuggingFace reference
    (tolerance 1e-2).
  - [ ] S42.1.1 Create tests/parity/siglip_test.go  Est: 45m
  - [ ] S42.1.2 Run test with real SigLIP model; fix any import or runtime errors  Est: 45m
  - [ ] S42.1.3 Verify output and run golangci-lint  Est: 20m

- [ ] T42.2 Kimi-VL connector forward pass test  Owner: TBD  Est: 1h
  - Dependencies: T42.1
  - File: tests/parity/siglip_test.go (extend)
  - Acceptance: Skipped if env vars not set. Load connector ZMF model. Input: vision
    embeddings [1, 196, embed_dim] (output of T42.1). Assert: connector output shape
    [1, 196, lm_dim]. Assert: no NaN or Inf.
  - [ ] S42.2.1 Implement connector test  Est: 30m
  - [ ] S42.2.2 Run and verify  Est: 20m
  - [ ] S42.2.3 Run golangci-lint and go test  Est: 5m

#### E43: Phase 6 Final Verification

- [ ] T43.1 Run full test suite with coverage and race detector  Owner: TBD  Est: 30m
  - Dependencies: E37, E38, E39, E40, E41, E42
  - Acceptance: go test ./... -cover -race passes in zerfoo. go test ./... -cover passes
    in zonnx. All existing tests still pass (no regressions). New files meet >= 85%
    coverage (Conv2d accepted at 85%; others at >= 90%).
  - [ ] S43.1.1 Run go test ./... -cover -race in zerfoo  Est: 10m
  - [ ] S43.1.2 Run go test ./... -cover in zonnx  Est: 10m
  - [ ] S43.1.3 Fix any regressions  Est: 10m

- [ ] T43.2 Run linters across all modified directories  Owner: TBD  Est: 15m
  - Dependencies: T43.1
  - Acceptance: golangci-lint 0 issues. go vet clean. gofmt clean.
  - [ ] S43.2.1 Run golangci-lint run ./... in zerfoo  Est: 5m
  - [ ] S43.2.2 Run golangci-lint run ./... in zonnx  Est: 5m
  - [ ] S43.2.3 Fix any remaining lint issues  Est: 5m

- [ ] T43.3 Update documentation  Owner: TBD  Est: 30m
  - Dependencies: T43.2
  - Acceptance: docs/plan.md Phase 6 tasks marked [x]. Hand-off notes updated with
    supported models and new operator list. zonnx/missing_operators_analysis.md updated.
  - [ ] S43.3.1 Update docs/plan.md Phase 6 tasks to [x]  Est: 10m
  - [ ] S43.3.2 Update Hand-off Notes with new operators and supported models  Est: 10m
  - [ ] S43.3.3 Update zonnx/missing_operators_analysis.md  Est: 10m

---

## 4. Timeline and Milestones

| ID | Milestone | Dependencies | Exit Criteria |
|----|-----------|--------------|---------------|
| M15 | Logging and metrics | E21, E22 | All packages instrumented; metrics exported |
| M16 | Security and config | E23, E24 | TLS on gRPC; config loads from file with env overrides |
| M17 | Reliability | E25, E26, E28 | Graceful shutdown; health checks; resource limits |
| M18 | CI hardening | E27 | Parity tests blocking; coverage + benchmark gates |
| M19 | Documentation | E30 | Runbook, troubleshooting guide, pprof endpoints |
| M20 | GPU validation | E29 | Tests pass on real T4 hardware (when quota available) |
| M21 | Enterprise ready | E31 | Full suite green, all quality gates pass |
| M22 | Worker service | E32, E33 | Concrete DistributedServiceServer + GrpcStrategy implemented |
| M23 | Distributed integration | E34 | Multi-worker tests prove AllReduce/Barrier/Broadcast correctness |
| M24 | Worker lifecycle | E35 | WorkerNode + CLI command; health + shutdown integrated |
| M25 | Phase 5 complete | E36 | Full suite green, distributed coverage >= 95% |
| M26 | Gemma 3 converter fixed | E37 | TENSOR attr handled; 126 MatMulNBits + 7 Constant nodes convert; smoke test passes |
| M27 | Core operators complete | E38 | Softmax, Sigmoid, LayerNorm, Slice, Pad, TopK, Erf registered and tested |
| M28 | Vision encoder ready | E39 | Conv2d, GlobalAveragePool, BatchNorm, Resize registered and tested |
| M29 | MoE complete | E40 | MoEGate and MixtureOfExperts registered and tested |
| M30 | VLM parity validated | E41, E42 | Gemma 3 forward pass test passes; SigLIP encoder test passes |
| M31 | Phase 6 complete | E43 | Full suite green; all quality gates pass |

### Recommended Sequence

**Phase 4 (Complete):**
1. **E21** (Logging) -- Foundation for all other observability work
2. **E22** (Metrics) -- Can start after T21.1; depends on Logger
3. **E27** (CI Hardening) -- Independent; can run in parallel with E21/E22
4. **E23** (gRPC Security) -- Independent
5. **E24** (Config Management) -- Independent
6. **E25** (Graceful Shutdown) -- Independent; benefits from Logger
7. **E26** (Health Checks) -- Depends on Logger
8. **E28** (Resource Limits) -- Independent
9. **E29** (GPU Validation) -- Blocked on external quota; do when unblocked
10. **E30** (Documentation) -- After E21-E26 are complete
11. **E31** (Final Verification) -- After all other epics

**Phase 5 (Concrete Server):**
12. **E32** (Worker Service) -- No new dependencies; uses existing log, metrics, pb stubs
13. **E33** (gRPC Strategy) -- Depends on E32
14. **E34** (Integration Tests) -- Depends on E33
15. **E35** (Worker Lifecycle + CLI) -- Depends on E33; can partially parallel E34
16. **E36** (Final Verification) -- After E32-E35

**Phase 6 (Open Weights Model Import):**
17. **E37** (Gemma 3 ONNX Import) -- No new zerfoo deps; zonnx converter fix first
18. **E38** (Core Missing Operators) -- Parallel with E37; independent of E39/E40
19. **E39** (Vision Encoder Operators) -- Parallel with E38; independent
20. **E40** (MoE) -- Depends on E38 (Softmax + TopK needed by MoEGate)
21. **E41** (Gemma 3 Validation) -- Depends on E37, E38
22. **E42** (Kimi-VL Validation) -- Depends on E39, E40
23. **E43** (Final Verification) -- After E37-E42

Parallelism opportunities:
- E21 + E27 can run in parallel (independent)
- E23 + E24 + E25 can run in parallel (independent)
- E22 starts after T21.1 (needs Logger interface)
- E26 starts after T21.1 (needs Logger interface)
- E34 + E35 can partially overlap (E34 tests E33 output; E35 builds on E33 independently)

---

## 5. Operating Procedure

### Definition of Done

A task is done when:
1. Implementation matches the acceptance criteria.
2. All existing tests pass (`go test ./... -count=1`).
3. New code has unit tests with >= 95% coverage.
4. `golangci-lint run ./package/` reports 0 issues.
5. `go vet ./package/` reports no issues.
6. Tests pass with `-race` flag.
7. Non-CUDA build (`go build ./...` without cuda tag) compiles.
8. Changes are committed in a small commit touching one directory only.

### Review and QA Steps

1. Read existing implementation before writing code.
2. Write tests first or alongside implementation. Use table-driven tests.
3. After implementation, run `go test -cover ./package/` to verify coverage.
4. Run `golangci-lint run --fix ./package/` to fix lint issues.
5. Run `gofmt -w .` to ensure formatting.
6. Run `go test ./... -count=1` to verify no regressions.
7. Run `go build ./...` (without cuda tag) to verify non-CUDA build.

### Commit Discipline

- Never commit files from different directories in the same commit.
- Make small, logical commits: one task or subtask per commit.
- Use Conventional Commits: `feat(log): add structured logger`, `fix(distributed): add TLS config`.
- Never allow changes to pile up. Commit after each completed subtask.
- Always run linters and formatters before committing.

---

## 6. Progress Log

- **2026 03 02 (update 14):** Change Summary: Completed T38.8 (zonnx importer builders). Added converter special cases for Slice/Pad/TopK to promote positional ONNX input tensors to named ZMF attributes (starts/ends/axes/steps, pads/constant_value, k). Added 7 layer builder stubs in zonnx/pkg/importer/layers/ (softmax, sigmoid, erf, layer_norm, slice, pad, topk), each registered via init(). 10 new round-trip tests added to converter_test.go covering all E38 operators. All zonnx tests pass; golangci-lint 0 issues. Commits (zonnx): 2a7bd4f, 04726bb.

- **2026 03 02 (update 13):** Change Summary: Completed E38 core missing operators (T38.1-T38.7, T38.9). Implemented and registered: Softmax (layers/activations/softmax.go), Erf (layers/activations/erf.go), BuildSigmoid builder (layers/activations/registry.go), BuildLayerNormalization with resolveParam helper (layers/normalization/registry.go), Slice (layers/core/slice.go), Pad (layers/core/pad.go), TopK (layers/core/topk.go). All seven operators registered in layers/registry/registry.go. All 50 packages pass go test -race ./...; golangci-lint 0 issues. Commits: 5c15cab, cf93bf7, d1ad6fa, 3370f25.

- **2026 03 02 (update 12):** Change Summary: Added Phase 6 -- Open Weights Model Import Support. Gap analysis identified blockers for Gemma 3 (TENSOR attribute missing in zonnx converter, UINT8 dtype missing, MatMulNBits and Constant not registered in zerfoo) and Kimi-VL (Conv2d, Pad, Slice, Resize, BatchNorm, GlobalAveragePool all missing; Softmax/Sigmoid/TopK/Erf not registered as layer nodes; MoE not implemented). New epics: E37 (Gemma 3 ONNX import fixes: 7 tasks), E38 (core missing operators: Softmax, Sigmoid, LayerNorm, Slice, Pad, TopK, Erf: 9 tasks), E39 (vision encoder operators: Conv2d, GlobalAveragePool, BatchNorm, Resize: 6 tasks), E40 (MoE: 4 tasks), E41 (Gemma 3 end-to-end validation: 2 tasks), E42 (Kimi-VL vision encoder validation: 2 tasks), E43 (Phase 6 final verification: 3 tasks). Added milestones M26-M31. Phase 6 is unblocked and can begin immediately.

- **2026 03 02 (update 11):** Change Summary: Completed Phase 5 -- Concrete Distributed Service Server. E32: workerService implementing pb.DistributedServiceServer with AllReduce (bidi stream), Barrier (unary), Broadcast (unary) handlers, reduceSession, barrierState, input validation (validateTensor). E33: GrpcStrategy[T] implementing InternalStrategy[T] with Init, AllReduceGradients (star-topology), Barrier, BroadcastTensor, Shutdown (idempotent). Fixed Init to accept explicit world size parameter for sequential registration. E34: Multi-worker integration tests (AllReduce 3-worker, single-worker, Barrier, Broadcast, context cancellation). T34.4 (TLS integration) deferred. E35: WorkerNode struct (worker_node.go), WorkerCommand (cmd/cli/worker.go), registered in cmd/zerfoo/main.go, lifecycle integration test. E36: Full test suite pass, distributed/ 96.0% coverage, golangci-lint 0 issues, go vet clean. Commits: a20fe4c, ab72e98, 34a784e, 9922af5, ddbea47, c3f8fcf, b668d28, afdea4a, 3574de4.

- **2026 03 01 (update 10):** Change Summary: Added Phase 5 -- Concrete Distributed Service Server. New epics E32 (WorkerService implementing pb.DistributedServiceServer with AllReduce/Barrier/Broadcast handlers and input validation), E33 (GrpcStrategy[T] implementing InternalStrategy[T] over gRPC transport), E34 (multi-worker integration tests using bufconn), E35 (WorkerNode lifecycle + CLI worker command + health/shutdown integration), E36 (Phase 5 final verification). Added milestones M22-M25. Star-topology AllReduce protocol (reduce to root, broadcast back). T32.5 completes previously skipped T23.2 (RPC input validation). 20 new tasks, estimated ~15 hours total.

- **2026 03 01 (update 9):** Change Summary: Completed remaining Phase 4 tasks. T25.3 signal handling (cmd/cli, cmd/zerfoo, cmd/zerfoo-predict). T28.1 memory limit (MemoryTracker with CAS-based enforcement). T28.2 per-operation timeout (parallelForCtx, context checks in UnaryOp/binaryOp/MatMul). T30.1 deployment runbook (docs/runbook.md). T30.2 troubleshooting guide (docs/troubleshooting.md). T31.1 full test suite with race detector (0 data races, 1 pre-existing flaky test in distributed/coordinator). T31.2 golangci-lint 0 issues, go vet clean. T31.3 integration smoke test (config->engine->health->shutdown). CI regex fixed (Go 1.25 does not support Perl negative lookahead). T23.2 skipped (no concrete RPC server implementation). E29 remains BLOCKED on GCP GPU quota.

- **2026 03 01 (update 8):** Change Summary: Completed T22.1-T22.3 metrics interface/instrumentation, T23.1 TLS config, T25.2 Closer implementations, T26.2 engine health check, T27.2 coverage gate, T27.3 benchmark regression detection, T30.3 pprof endpoints. All with tests, lint clean, coverage above thresholds.

- **2026 03 01 (update 7):** Change Summary: Created enterprise production readiness plan (Phase 4, E21-E31). Extracted architecture and design knowledge to docs/design.md. Trimmed plan.md to remove completed Phase 1-3 task details (preserved as summary in design.md Section 7). New epics: E21 structured logging, E22 metrics interface, E23 gRPC TLS, E24 config management, E25 graceful shutdown, E26 health checks, E27 CI hardening, E28 resource limits, E29 GPU validation (re-numbered from E15/E20), E30 production docs, E31 final verification. Added milestones M15-M21.

- **2026 03 01 (update 6):** Completed E6 T6.1 (testutil tests, 98.5%), E6 T6.2 (testutils tests, 94.5%), E7 T7.1 (full suite green, zero races, regularization 92.9% -> 97.6%), E7 T7.2 (0 lint issues, gofmt clean). All Phase 1 remaining tasks done.

- **2026 03 01 (updates 1-5):** Completed Phase 2 (GPU Engine, E8-E14) and Phase 3 (GPU Production Readiness, E16-E19). Details in docs/design.md Section 7.

- **2026 02 25:** Completed Phase 1 test coverage (E1-E5). 30 of 33 packages at >= 95%.

- **2026 02 24:** Initial plan created for Phase 1 test coverage improvement.

---

## 7. Hand-off Notes

### For a New Contributor

- **Architecture:** Read docs/design.md for interface contracts, package layout, and GPU architecture.
- **GPU details:** Read docs/gpu.md for build requirements, kernel inventory, and memory model.
- **Phase 1-3 status:** Complete. See docs/design.md Section 7 for summary.
- **Phase 4 status:** Complete (except E29 GPU validation, blocked on GCP quota).
- **Phase 5 status:** Complete. Concrete DistributedServiceServer, GrpcStrategy, WorkerNode, CLI worker command. 96% coverage.
- **Phase 6 status:** Planned, not started. Open weights model import (Gemma 3 + Kimi-VL).
- **GPU hardware validation:** Blocked on GCP GPU quota (E29).
- **Key files to read first:**
  - compute/engine.go -- Engine[T] interface (34 methods)
  - graph/node.go -- Node[T] interface
  - tensor/storage.go -- Storage[T] interface
  - distributed/interfaces.go -- Distributed training interfaces (InternalStrategy[T], NetworkManager, ServerManager, CoordinatorClient)
  - distributed/pb/dist.proto -- DistributedService proto (AllReduce, Barrier, Broadcast)
  - distributed/pb/coordinator.proto -- Coordinator proto (RegisterWorker, Heartbeat, Checkpoint)
  - distributed/coordinator/coordinator.go -- Fully implemented Coordinator gRPC server
  - distributed/all_reduce.go -- AllReduceStrategy[T] hierarchical implementation
  - distributed/network_manager.go -- NetworkManager and ServerManager implementations
- **How to run tests:** `go test ./... -cover` for full suite. `go test -tags cuda ./...` for GPU.
- **How to build:** `go build ./...` (CPU). `go build -tags cuda ./...` (GPU).
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.
- **No credentials required.** All work is local. CUDA Toolkit needed for GPU work.
- **Testing pattern for gRPC:** Use google.golang.org/grpc/test/bufconn for in-process gRPC tests. See distributed/coordinator/coordinator_test.go for the established pattern.

### External Dependencies

- GCP GPU quota increase for hardware validation (preference ID: zerfoo-gpu-test, project: numerai-488804).

---

## 8. Appendix

### Production Readiness Scorecard (Current State)

| Category | Score | Notes |
|----------|-------|-------|
| Architecture | 9/10 | Clean interfaces, modular, type-safe |
| Core Functionality | 8/10 | Engine complete, GPU in progress |
| Testing | 8/10 | 95%+ coverage, missing hardware validation |
| Error Handling | 6/10 | Basic validation, no structured errors |
| Security | 3/10 | No TLS, no auth, minimal validation |
| Observability | 3/10 | Minimal logging, no metrics export |
| Configuration | 4/10 | Programmatic only, no file support |
| Operations | 3/10 | No health checks, no shutdown coordination |
| Documentation | 5/10 | Design docs good, missing runbooks |
| CI/CD | 7/10 | Comprehensive tests, non-blocking parity |

### Target Scorecard (After Phase 4)

| Category | Target | How Achieved |
|----------|--------|-------------|
| Architecture | 9/10 | No changes needed |
| Core Functionality | 9/10 | GPU hardware validation (E29) |
| Testing | 9/10 | Blocking parity tests, benchmark gates (E27) |
| Error Handling | 8/10 | Structured logging with context (E21) |
| Security | 7/10 | TLS, input validation (E23) |
| Observability | 8/10 | Logging, metrics, pprof (E21, E22, E30) |
| Configuration | 8/10 | File loading, env overrides, validation (E24) |
| Operations | 8/10 | Health checks, graceful shutdown, limits (E25, E26, E28) |
| Documentation | 8/10 | Runbook, troubleshooting, pprof (E30) |
| CI/CD | 9/10 | Blocking tests, coverage gate, benchmark gate (E27) |

### Target Scorecard (After Phase 6)

| Category | Target | How Achieved |
|----------|--------|-------------|
| Architecture | 10/10 | No changes from Phase 5 |
| Core Functionality | 10/10 | Gemma 3 + Kimi-VL inference via ONNX import (E37-E42) |
| Testing | 10/10 | Parity tests for real open-weights models (E41, E42) |
| Error Handling | 9/10 | No changes from Phase 5 |
| Security | 8/10 | No changes from Phase 5 |
| Observability | 8/10 | No changes from Phase 5 |
| Configuration | 8/10 | No changes from Phase 5 |
| Operations | 9/10 | No changes from Phase 5 |
| Documentation | 9/10 | Gap analysis resolved; operator coverage documented (T43.3) |
| CI/CD | 9/10 | No changes from Phase 5 |

### Target Scorecard (After Phase 5)

| Category | Target | How Achieved |
|----------|--------|-------------|
| Architecture | 10/10 | Concrete server completes distributed architecture (E32, E33) |
| Core Functionality | 9/10 | GPU validation still pending (E29) |
| Testing | 10/10 | Multi-worker integration tests over real gRPC (E34) |
| Error Handling | 9/10 | RPC input validation on all handlers (T32.5) |
| Security | 8/10 | TLS integration tests with distributed workers (T34.4) |
| Observability | 8/10 | No changes from Phase 4 |
| Configuration | 8/10 | No changes from Phase 4 |
| Operations | 9/10 | Worker lifecycle + CLI command + health integration (E35) |
| Documentation | 9/10 | Distributed worker setup in runbook (T35.5) |
| CI/CD | 9/10 | No changes from Phase 4 |

### New Packages and Files Created

| Package / File | Purpose | Epic |
|---------|---------|------|
| log/ | Structured logging with levels | E21 |
| metrics/runtime/ | Runtime metrics collection | E22 |
| config/ | File-based configuration loading | E24 |
| shutdown/ | Graceful shutdown coordinator | E25 |
| health/ | HTTP health check server | E26 |
| cmd/coverage-gate/ | CI coverage enforcement script | E27 |
| cmd/bench-compare/ | CI benchmark regression detection | E27 |
| distributed/worker_service.go | Concrete DistributedServiceServer (AllReduce, Barrier, Broadcast) | E32 |
| distributed/grpc_strategy.go | GrpcStrategy[T] implementing InternalStrategy[T] over gRPC | E33 |
| distributed/integration_test.go | Multi-worker integration tests using bufconn | E34 |
| distributed/worker_node.go | WorkerNode lifecycle management | E35 |
| cmd/zerfoo/worker.go | Worker CLI subcommand | E35 |
| layers/activations/softmax.go | Softmax graph layer node | E38 |
| layers/activations/sigmoid.go | Sigmoid graph layer node | E38 |
| layers/activations/erf.go | Erf (error function) graph layer node | E38 |
| layers/normalization/layer_norm.go | Standard LayerNormalization (with gamma+beta) | E38 |
| layers/normalization/batch_norm.go | BatchNormalization inference mode | E39 |
| layers/core/slice.go | Slice operator for tensor cropping | E38 |
| layers/core/pad.go | Pad operator (constant mode) | E38 |
| layers/core/topk.go | TopK selection operator | E38 |
| layers/core/conv2d.go | 2D convolution via im2col + MatMul | E39 |
| layers/core/global_avg_pool.go | GlobalAveragePool [N,C,H,W] -> [N,C,1,1] | E39 |
| layers/core/resize.go | Resize (nearest + bilinear) | E39 |
| layers/core/moe.go | MoEGate and MixtureOfExperts layers | E40 |
| tests/parity/gemma3_test.go | Gemma 3 forward pass parity test | E41 |
| tests/parity/siglip_test.go | SigLIP + Kimi-VL connector parity tests | E42 |
| zonnx/pkg/importer/layers/constant.go | zonnx builder for Constant nodes | E37 |
| zonnx/pkg/importer/layers/matmul_nbits.go | zonnx builder for MatMulNBits nodes | E37 |
| zonnx/pkg/importer/layers/softmax.go | zonnx builder for Softmax | E38 |
| zonnx/pkg/importer/layers/sigmoid.go | zonnx builder for Sigmoid | E38 |
| zonnx/pkg/importer/layers/layer_norm.go | zonnx builder for LayerNormalization | E38 |
| zonnx/pkg/importer/layers/slice.go | zonnx builder for Slice | E38 |
| zonnx/pkg/importer/layers/pad.go | zonnx builder for Pad | E38 |
| zonnx/pkg/importer/layers/topk.go | zonnx builder for TopK | E38 |
| zonnx/pkg/importer/layers/erf.go | zonnx builder for Erf | E38 |
| zonnx/pkg/importer/layers/conv.go | zonnx builder for Conv (Conv2d) | E39 |
| zonnx/pkg/importer/layers/global_avg_pool.go | zonnx builder for GlobalAveragePool | E39 |
| zonnx/pkg/importer/layers/batch_norm.go | zonnx builder for BatchNormalization | E39 |
| zonnx/pkg/importer/layers/resize.go | zonnx builder for Resize | E39 |
| zonnx/pkg/importer/layers/moe.go | zonnx builders for MoEGate and MixtureOfExperts | E40 |
