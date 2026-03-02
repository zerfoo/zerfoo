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

- [ ] T21.1 Define Logger interface in a new `log` package  Owner: TBD  Est: 1h
  - Dependencies: None
  - Acceptance: Interface has Debug, Info, Warn, Error methods. Each accepts a message string and key-value fields. A NopLogger and a StdLogger (writing to io.Writer) are provided. JSON output mode is available via a constructor option.
  - [ ] S21.1.1 Create log/logger.go with Logger interface and Level type  Est: 20m
  - [ ] S21.1.2 Implement StdLogger with level filtering and text/JSON output  Est: 25m
  - [ ] S21.1.3 Implement NopLogger (zero-allocation no-op)  Est: 5m
  - [ ] S21.1.4 Write unit tests for StdLogger (level filtering, JSON format, field rendering)  Est: 20m
  - [ ] S21.1.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T21.2 Integrate Logger into compute package  Owner: TBD  Est: 45m
  - Dependencies: T21.1
  - Acceptance: CPUEngine and GPUEngine accept a Logger at construction. OOM fallback, stream errors, and pool operations log at appropriate levels. No raw fmt.Printf calls remain in compute/.
  - [ ] S21.2.1 Add Logger field to CPUEngine; log parallelFor errors at Warn  Est: 15m
  - [ ] S21.2.2 Add Logger field to GPUEngine; log OOM fallback, pool stats, stream errors  Est: 20m
  - [ ] S21.2.3 Update tests to verify log output in error scenarios  Est: 15m
  - [ ] S21.2.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T21.3 Integrate Logger into distributed package  Owner: TBD  Est: 45m
  - Dependencies: T21.1
  - Acceptance: Replace existing distributed.Logger interface with log.Logger. All coordinator and worker components use leveled logging. Connection events logged at Info, errors at Error.
  - [ ] S21.3.1 Update distributed.ServerManager, coordinator to accept log.Logger  Est: 15m
  - [ ] S21.3.2 Replace all fmt.Printf calls in distributed/ with logger calls  Est: 15m
  - [ ] S21.3.3 Update tests to use StdLogger or NopLogger  Est: 10m
  - [ ] S21.3.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T21.4 Integrate Logger into remaining packages  Owner: TBD  Est: 30m
  - Dependencies: T21.1
  - Acceptance: training/, model/, cmd/cli/ use log.Logger. No raw fmt.Printf in non-test production code.
  - [ ] S21.4.1 Add Logger to training.WorkflowConfig and optimizer constructors  Est: 10m
  - [ ] S21.4.2 Add Logger to model package and cmd/cli framework  Est: 10m
  - [ ] S21.4.3 Audit all packages for remaining fmt.Printf; replace with logger  Est: 10m
  - [ ] S21.4.4 Run golangci-lint and go test -cover  Est: 5m

#### E22: Metrics Interface

Add a metrics collection abstraction for runtime observability. The interface
must be backend-agnostic (usable with Prometheus, StatsD, or in-memory).

- [ ] T22.1 Define Metrics interface in a new `metrics/runtime` package  Owner: TBD  Est: 1h
  - Dependencies: None
  - Acceptance: Interface has Counter(name), Gauge(name), Histogram(name, buckets) methods. Each returns a typed metric with Inc/Set/Observe methods. A default in-memory implementation is provided for testing and local use. A NopMetrics implementation is provided for zero overhead when metrics are disabled.
  - [ ] S22.1.1 Create metrics/runtime/metrics.go with Collector interface  Est: 20m
  - [ ] S22.1.2 Implement InMemoryCollector with thread-safe counters/gauges  Est: 25m
  - [ ] S22.1.3 Implement NopCollector (zero-allocation no-op)  Est: 5m
  - [ ] S22.1.4 Write unit tests for InMemoryCollector (concurrent access, snapshot)  Est: 15m
  - [ ] S22.1.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T22.2 Instrument compute.Engine with metrics  Owner: TBD  Est: 45m
  - Dependencies: T22.1
  - Acceptance: CPUEngine and GPUEngine report: op_count (counter per operation type), op_duration_seconds (histogram), oom_fallback_total (counter), pool_hit_total / pool_miss_total (counters for GPU pool).
  - [ ] S22.2.1 Add Collector field to CPUEngine; instrument Add/MatMul/etc. with counters and timers  Est: 20m
  - [ ] S22.2.2 Add Collector field to GPUEngine; instrument kernel dispatch, OOM, pool  Est: 20m
  - [ ] S22.2.3 Write tests verifying metric increments after operations  Est: 15m
  - [ ] S22.2.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T22.3 Instrument distributed package with metrics  Owner: TBD  Est: 30m
  - Dependencies: T22.1
  - Acceptance: Distributed workers report: allreduce_count (counter), allreduce_duration_seconds (histogram), barrier_count, broadcast_count, connection_errors_total.
  - [ ] S22.3.1 Add Collector to Strategy and coordinator  Est: 15m
  - [ ] S22.3.2 Instrument AllReduceGradients, Barrier, BroadcastTensor  Est: 10m
  - [ ] S22.3.3 Write tests verifying metrics after distributed operations  Est: 10m
  - [ ] S22.3.4 Run golangci-lint and go test -cover  Est: 5m

#### E23: gRPC Security Hardening

Add TLS and mutual authentication to all gRPC communication channels.

- [ ] T23.1 Add TLS configuration to gRPC server and client  Owner: TBD  Est: 1h
  - Dependencies: None
  - Acceptance: A TLSConfig struct supports: CA cert path, server cert/key paths, client cert/key paths for mTLS. ServerManager.Start() uses TLS credentials when TLSConfig is provided. Worker connections use TLS. Plaintext is still supported (for local development) when TLSConfig is nil.
  - [ ] S23.1.1 Create distributed/tlsconfig.go with TLSConfig struct and credential helpers  Est: 20m
  - [ ] S23.1.2 Update ServerManager to accept TLSConfig and create TLS listener  Est: 15m
  - [ ] S23.1.3 Update NetworkManager.ConnectToPeers to use TLS dial options  Est: 15m
  - [ ] S23.1.4 Write integration test: server + client with self-signed TLS certs  Est: 20m
  - [ ] S23.1.5 Write integration test: mTLS with client cert verification  Est: 15m
  - [ ] S23.1.6 Run golangci-lint and go test -cover  Est: 5m

- [ ] T23.2 Add input validation to distributed RPC handlers  Owner: TBD  Est: 30m
  - Dependencies: None
  - Acceptance: All RPC handlers validate request fields (non-empty rank, valid tensor shapes, non-nil data). Invalid requests return gRPC InvalidArgument status. Tests verify each validation path.
  - [ ] S23.2.1 Add validation to AllReduce, Barrier, Broadcast RPC handlers  Est: 15m
  - [ ] S23.2.2 Write tests for each validation error case  Est: 10m
  - [ ] S23.2.3 Run golangci-lint and go test -cover  Est: 5m

#### E24: Configuration Management

Add file-based configuration loading with validation and environment
variable overrides. Use encoding/json and os.Getenv from the standard library.

- [ ] T24.1 Create config package with file loader  Owner: TBD  Est: 1h
  - Dependencies: None
  - Acceptance: A config.Load[T](path string) function reads a JSON file into a struct. A config.LoadWithEnv[T](path, prefix string) function additionally applies environment variable overrides using the `env` struct tag. Validation errors list all invalid fields. Missing required fields produce clear error messages.
  - [ ] S24.1.1 Create config/loader.go with Load[T] function (JSON decoder)  Est: 15m
  - [ ] S24.1.2 Implement env var override via struct tag reflection  Est: 20m
  - [ ] S24.1.3 Implement validation via `validate:"required"` struct tag  Est: 15m
  - [ ] S24.1.4 Write unit tests: valid config, missing file, invalid JSON, missing required, env override  Est: 20m
  - [ ] S24.1.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T24.2 Define standard config structs for Engine and Training  Owner: TBD  Est: 30m
  - Dependencies: T24.1
  - Acceptance: EngineConfig (device type, memory limit, log level), TrainingConfig (batch size, learning rate, optimizer, epochs, checkpoint interval), DistributedConfig (coordinator address, TLS config, timeout). Each struct has JSON tags and validation tags.
  - [ ] S24.2.1 Define EngineConfig, TrainingConfig, DistributedConfig structs  Est: 15m
  - [ ] S24.2.2 Write tests loading each config from JSON with env overrides  Est: 10m
  - [ ] S24.2.3 Run golangci-lint and go test -cover  Est: 5m

#### E25: Graceful Shutdown

Implement orderly shutdown coordination using context cancellation
and cleanup callbacks.

- [ ] T25.1 Add Closer interface and shutdown coordinator  Owner: TBD  Est: 45m
  - Dependencies: None
  - Acceptance: A shutdown.Coordinator registers Closer instances in order. On Shutdown(ctx), it calls Close() on each in reverse registration order. If a Closer does not complete within the context deadline, it is skipped and logged. Integration test demonstrates orderly cleanup.
  - [ ] S25.1.1 Create shutdown/coordinator.go with Closer interface and Coordinator  Est: 20m
  - [ ] S25.1.2 Implement reverse-order shutdown with timeout per closer  Est: 15m
  - [ ] S25.1.3 Write tests: orderly shutdown, timeout on slow closer, empty coordinator  Est: 15m
  - [ ] S25.1.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T25.2 Implement Closer for Engine and distributed components  Owner: TBD  Est: 30m
  - Dependencies: T25.1
  - Acceptance: GPUEngine.Close() drains memory pool and destroys CUDA handles. CPUEngine.Close() is a no-op (satisfies interface). Distributed Strategy.Shutdown() deregisters from coordinator and closes connections. All Close methods are idempotent.
  - [ ] S25.2.1 Make CPUEngine implement Closer (no-op Close)  Est: 5m
  - [ ] S25.2.2 Verify GPUEngine.Close() is idempotent  Est: 10m
  - [ ] S25.2.3 Make distributed Strategy implement Closer  Est: 10m
  - [ ] S25.2.4 Write integration test: register Engine + Strategy, trigger shutdown  Est: 15m
  - [ ] S25.2.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T25.3 Add signal handling to CLI commands  Owner: TBD  Est: 30m
  - Dependencies: T25.1, T25.2
  - Acceptance: cmd/zerfoo and cmd/zerfoo-predict catch SIGINT/SIGTERM, trigger shutdown coordinator, and exit cleanly. Integration test verifies signal handling.
  - [ ] S25.3.1 Add signal listener in cmd framework that cancels root context  Est: 15m
  - [ ] S25.3.2 Wire shutdown coordinator into CLI lifecycle  Est: 10m
  - [ ] S25.3.3 Write test verifying clean exit on SIGTERM  Est: 10m
  - [ ] S25.3.4 Run golangci-lint and go test -cover  Est: 5m

#### E26: Health Checks

Add health check endpoints for deployment probes (Kubernetes liveness
and readiness).

- [ ] T26.1 Create health check HTTP server  Owner: TBD  Est: 45m
  - Dependencies: T21.1
  - Acceptance: A health.Server exposes /healthz (liveness) and /readyz (readiness) HTTP endpoints. Each returns 200 OK with JSON body when healthy, 503 when unhealthy. Readiness checks are configurable (register check functions). Server starts on a configurable port. Logger is used for startup/error messages.
  - [ ] S26.1.1 Create health/server.go with Server struct and HTTP handlers  Est: 15m
  - [ ] S26.1.2 Implement configurable readiness checks (func() error callbacks)  Est: 10m
  - [ ] S26.1.3 Write tests: healthy response, unhealthy readiness, concurrent access  Est: 15m
  - [ ] S26.1.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T26.2 Add engine health check  Owner: TBD  Est: 20m
  - Dependencies: T26.1
  - Acceptance: A check function verifies Engine is operational (e.g., small tensor add succeeds). For GPU, additionally verify CUDA context is valid. Register as readiness check.
  - [ ] S26.2.1 Implement engine health check function  Est: 10m
  - [ ] S26.2.2 Write test for healthy and unhealthy engine  Est: 10m
  - [ ] S26.2.3 Run golangci-lint and go test -cover  Est: 5m

#### E27: CI/CD Hardening

Make CI pipeline enforce quality gates strictly.

- [ ] T27.1 Make parity and numerics tests blocking  Owner: TBD  Est: 15m
  - Dependencies: None
  - Acceptance: Remove `|| true` from parity and numerics test steps in .github/workflows/ci.yml. CI fails if any parity or numerics test fails.
  - [ ] S27.1.1 Update ci.yml: remove `|| true` from parity test step  Est: 5m
  - [ ] S27.1.2 Update ci.yml: remove `|| true` from numerics test step  Est: 5m
  - [ ] S27.1.3 Verify CI passes with current test suite  Est: 5m

- [ ] T27.2 Add coverage gate to CI  Owner: TBD  Est: 30m
  - Dependencies: None
  - Acceptance: CI step runs `go test -coverprofile=coverage.out ./...`, parses output, and fails if any testable package (excluding documented exceptions) drops below 93%. Coverage summary is posted as a CI artifact.
  - [ ] S27.2.1 Add coverage step to ci.yml that generates coverage.out  Est: 10m
  - [ ] S27.2.2 Write a Go script (cmd/coverage-gate/main.go) that parses coverage.out and exits non-zero if below threshold  Est: 20m
  - [ ] S27.2.3 Add tests for coverage-gate script  Est: 10m
  - [ ] S27.2.4 Run golangci-lint and go test -cover  Est: 5m

- [ ] T27.3 Add benchmark regression detection  Owner: TBD  Est: 45m
  - Dependencies: None
  - Acceptance: CI runs benchmarks on each PR. A Go script compares benchmark results against a baseline (stored in repo). CI fails if any benchmark regresses by more than 10%. Baseline is updated via a manual workflow dispatch.
  - [ ] S27.3.1 Add benchmark step to ci.yml (go test -bench=. -benchmem -count=3)  Est: 10m
  - [ ] S27.3.2 Write cmd/bench-compare/main.go to parse benchstat output and enforce threshold  Est: 25m
  - [ ] S27.3.3 Add baseline benchmark results file (benchmarks/baseline.txt)  Est: 5m
  - [ ] S27.3.4 Add tests for bench-compare script  Est: 10m
  - [ ] S27.3.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T27.4 Update CI Go version and add race detector  Owner: TBD  Est: 15m
  - Dependencies: None
  - Acceptance: CI uses Go 1.25 (matching go.mod). Race detector runs on unit tests. Both Ubuntu and macOS runners are used.
  - [ ] S27.4.1 Update ci.yml go-version to match go.mod  Est: 5m
  - [ ] S27.4.2 Add -race flag to unit test step  Est: 5m
  - [ ] S27.4.3 Add macOS runner to test matrix  Est: 5m

#### E28: Resource Limits

Add configurable resource limits to prevent unbounded allocation and
runaway operations.

- [ ] T28.1 Add memory limit to Engine  Owner: TBD  Est: 45m
  - Dependencies: None
  - Acceptance: Engine accepts a MaxMemoryBytes option. Tensor allocation that would exceed the limit returns an error instead of allocating. GPU engine tracks device memory usage. The limit is enforced at the Engine level, not the allocator level (so it applies to both CPU and GPU).
  - [ ] S28.1.1 Add MemoryTracker to compute package (atomic int64 tracking allocated bytes)  Est: 15m
  - [ ] S28.1.2 Integrate MemoryTracker into tensor allocation (New, NewWithStorage)  Est: 15m
  - [ ] S28.1.3 Add MaxMemoryBytes option to Engine constructors  Est: 10m
  - [ ] S28.1.4 Write tests: allocation within limit succeeds, over limit returns error  Est: 15m
  - [ ] S28.1.5 Run golangci-lint and go test -cover  Est: 5m

- [ ] T28.2 Add per-operation timeout enforcement  Owner: TBD  Est: 30m
  - Dependencies: None
  - Acceptance: Engine respects context.Context deadlines. Long-running operations (MatMul, Softmax) check ctx.Done() periodically and return context.DeadlineExceeded if expired. GPU operations use CUDA stream synchronization with timeout.
  - [ ] S28.2.1 Add ctx.Done() checks in CPUEngine parallelFor loops  Est: 15m
  - [ ] S28.2.2 Add stream sync timeout in GPUEngine operations  Est: 10m
  - [ ] S28.2.3 Write tests: operation completes within deadline, times out correctly  Est: 15m
  - [ ] S28.2.4 Run golangci-lint and go test -cover  Est: 5m

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

- [ ] T30.1 Write deployment runbook  Owner: TBD  Est: 1h
  - Dependencies: E21, E23, E24, E25, E26
  - Acceptance: docs/runbook.md covers: system requirements, installation steps, configuration reference (all config fields documented), startup sequence, health check verification, log interpretation, common operational tasks (scale workers, update model, restart), shutdown procedure.
  - [ ] S30.1.1 Write system requirements and installation section  Est: 15m
  - [ ] S30.1.2 Write configuration reference (all config structs documented)  Est: 15m
  - [ ] S30.1.3 Write startup, health check, and shutdown sections  Est: 15m
  - [ ] S30.1.4 Write common operational tasks  Est: 15m

- [ ] T30.2 Write troubleshooting guide  Owner: TBD  Est: 45m
  - Dependencies: E21, E22
  - Acceptance: docs/troubleshooting.md covers: common error messages with root causes and fixes, GPU-specific issues (CUDA not found, OOM, driver mismatch), distributed training issues (connection refused, timeout, split brain), performance diagnosis (how to identify bottlenecks, pprof usage).
  - [ ] S30.2.1 Document common error messages and fixes  Est: 15m
  - [ ] S30.2.2 Document GPU troubleshooting  Est: 10m
  - [ ] S30.2.3 Document distributed training troubleshooting  Est: 10m
  - [ ] S30.2.4 Document performance diagnosis with pprof  Est: 10m

- [ ] T30.3 Add pprof endpoints to health server  Owner: TBD  Est: 20m
  - Dependencies: T26.1
  - Acceptance: Health server registers net/http/pprof handlers. CPU profile, heap profile, goroutine dump available at /debug/pprof/*.
  - [ ] S30.3.1 Register pprof handlers in health.Server  Est: 10m
  - [ ] S30.3.2 Write test verifying pprof endpoints respond  Est: 10m
  - [ ] S30.3.3 Run golangci-lint and go test -cover  Est: 5m

#### E31: Final Verification

Run the full quality gate suite after all enterprise features are implemented.

- [ ] T31.1 Run full test suite with coverage  Owner: TBD  Est: 30m
  - Dependencies: E21, E22, E23, E24, E25, E26, E27, E28
  - Acceptance: `go test ./... -cover` shows all packages at target coverage. `go test ./... -race` shows zero races. New packages (log, config, health, shutdown, metrics/runtime) are all at >= 95%.
  - [ ] S31.1.1 Run go test ./... -cover  Est: 10m
  - [ ] S31.1.2 Run go test ./... -race  Est: 10m
  - [ ] S31.1.3 Verify new packages meet 95% coverage  Est: 10m

- [ ] T31.2 Run linters and formatters  Owner: TBD  Est: 15m
  - Dependencies: T31.1
  - Acceptance: golangci-lint 0 issues, go vet clean, gofmt clean.
  - [ ] S31.2.1 Run golangci-lint run ./...  Est: 5m
  - [ ] S31.2.2 Run go vet ./...  Est: 5m
  - [ ] S31.2.3 Run gofmt -l . and verify no files  Est: 5m

- [ ] T31.3 Run integration smoke test  Owner: TBD  Est: 30m
  - Dependencies: T31.1
  - Acceptance: End-to-end test: load config from file, create Engine, run forward pass, verify health check, trigger graceful shutdown. All within a single test binary.
  - [ ] S31.3.1 Write integration test covering config -> engine -> health -> shutdown  Est: 20m
  - [ ] S31.3.2 Run integration test  Est: 5m
  - [ ] S31.3.3 Run golangci-lint  Est: 5m

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

### Recommended Sequence

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

Parallelism opportunities:
- E21 + E27 can run in parallel (independent)
- E23 + E24 + E25 can run in parallel (independent)
- E22 starts after T21.1 (needs Logger interface)
- E26 starts after T21.1 (needs Logger interface)

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
- **Phase 4 status:** Not started. This plan covers enterprise production readiness.
- **GPU hardware validation:** Blocked on GCP GPU quota (E29).
- **Key files to read first:**
  - compute/engine.go -- Engine[T] interface (34 methods)
  - graph/node.go -- Node[T] interface
  - tensor/storage.go -- Storage[T] interface
  - distributed/interfaces.go -- Distributed training interfaces
- **How to run tests:** `go test ./... -cover` for full suite. `go test -tags cuda ./...` for GPU.
- **How to build:** `go build ./...` (CPU). `go build -tags cuda ./...` (GPU).
- **Pre-commit hook:** Runs golangci-lint and tests. Rejects multi-directory commits.
- **No credentials required.** All work is local. CUDA Toolkit needed for GPU work.

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

### New Packages Created by This Plan

| Package | Purpose | Epic |
|---------|---------|------|
| log/ | Structured logging with levels | E21 |
| metrics/runtime/ | Runtime metrics collection | E22 |
| config/ | File-based configuration loading | E24 |
| shutdown/ | Graceful shutdown coordinator | E25 |
| health/ | HTTP health check server | E26 |
| cmd/coverage-gate/ | CI coverage enforcement script | E27 |
| cmd/bench-compare/ | CI benchmark regression detection | E27 |
