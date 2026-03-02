# Zerfoo Troubleshooting Guide

## Common Error Messages

### "memory limit exceeded"

**Cause:** The total tensor allocation exceeds the configured `memory_limit_mb`.

**Fix:**
- Increase the memory limit in configuration or set to 0 (unlimited).
- Reduce batch size or model dimensions.
- Check for tensor leaks (tensors allocated but never freed).

### "input tensor cannot be nil"

**Cause:** An operation received a nil tensor pointer.

**Fix:**
- Verify that all tensors are properly initialized with `tensor.New`.
- Check the output of prior operations for errors that may have returned nil.

### "invalid shapes for matrix multiplication"

**Cause:** Inner dimensions of tensors are incompatible for MatMul.

**Fix:**
- Verify tensor shapes. For `A @ B`, A must be `[..., m, k]` and B must be `[..., k, n]`.
- Use `Reshape` to fix shape mismatches before MatMul.

### "context canceled" / "context deadline exceeded"

**Cause:** The operation took too long or the parent context was canceled.

**Fix:**
- Increase the timeout for long-running operations.
- For SIGTERM during inference, this is expected -- the shutdown coordinator
  is canceling in-flight work.

### "coverage gate failed"

**Cause:** CI coverage dropped below the 93% threshold.

**Fix:**
- Add tests for new or modified code.
- Run `go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out`
  to find uncovered lines.

### "benchmark regression detected"

**Cause:** A benchmark regressed by more than 10% compared to the baseline.

**Fix:**
- Profile the regression using `go test -bench=BenchmarkName -cpuprofile=cpu.out`.
- If the regression is expected (e.g. added safety checks), update the baseline:
  `go test -bench=. -benchmem -count=3 ./compute/ > benchmarks/baseline.txt`.

## GPU-Specific Issues

### "CUDA not found" / build fails with cuda tag

**Cause:** CUDA Toolkit is not installed or not in PATH.

**Fix:**
- Install CUDA Toolkit 12.x from NVIDIA.
- Ensure `nvcc` is in PATH: `which nvcc`.
- Set `CUDA_HOME` if installed in a non-standard location.

### GPU Out of Memory (OOM)

**Cause:** GPU device memory is exhausted.

**Symptoms:**
- Operations fail with CUDA allocation errors.
- Log shows "GPU OOM fallback to CPU" at WARN level.

**Fix:**
- Reduce batch size.
- Set a lower `memory_limit_mb`.
- Monitor GPU memory: `nvidia-smi -l 1`.
- Verify the memory pool is releasing tensors (check pool_hit/pool_miss metrics).

### CUDA Driver Version Mismatch

**Cause:** The installed CUDA runtime is incompatible with the GPU driver.

**Fix:**
- Check driver version: `nvidia-smi`.
- Check CUDA version: `nvcc --version`.
- Upgrade the driver or install a compatible CUDA Toolkit.
- Refer to the NVIDIA CUDA compatibility matrix.

### Device Not Available

**Cause:** No GPU detected or GPU index out of range.

**Fix:**
- Verify GPU is visible: `nvidia-smi`.
- Check `CUDA_VISIBLE_DEVICES` environment variable.
- Ensure the GPU is not occupied by another process.

## Distributed Training Issues

### "connection refused"

**Cause:** The coordinator or peer node is not reachable.

**Fix:**
- Verify the coordinator is running and listening on the configured address.
- Check firewall rules for port 50051/TCP (default gRPC port).
- Ensure `coordinator_address` matches the actual host:port.
- For TLS: verify certificates are valid and not expired.

### "transport: authentication handshake failed"

**Cause:** TLS certificate mismatch between server and client.

**Fix:**
- Verify both sides use the same CA certificate.
- Check certificate expiry: `openssl x509 -in cert.pem -noout -dates`.
- For mTLS: ensure client cert is signed by the server's trusted CA.

### Timeout During Barrier / AllReduce

**Cause:** One or more workers are slow or unreachable.

**Fix:**
- Check all worker processes are running.
- Increase `timeout_seconds` in the distributed config.
- Look for network latency: `ping <peer-host>`.
- Check for stragglers (uneven data distribution causing load imbalance).

### Split Brain / Inconsistent State

**Cause:** Network partition caused workers to diverge.

**Fix:**
- Stop all workers.
- Restart from the last consistent checkpoint.
- If no checkpoint, restart training from the beginning.

## Performance Diagnosis

### Identifying Bottlenecks

1. Enable pprof endpoints (automatically registered at `/debug/pprof/`).
2. Capture a CPU profile during a representative workload.
3. Analyze with `go tool pprof`.

### CPU Profiling

```
# Capture 30-second profile
go tool pprof http://localhost:8081/debug/pprof/profile?seconds=30

# In the pprof shell:
(pprof) top 20
(pprof) web        # Opens flame graph in browser
(pprof) list MatMul  # Show annotated source
```

### Memory Profiling

```
# Heap profile
go tool pprof http://localhost:8081/debug/pprof/heap

(pprof) top 20 -cum
(pprof) list getOrCreateDest
```

### Goroutine Analysis

```
# Dump all goroutines
curl http://localhost:8081/debug/pprof/goroutine?debug=2 > goroutines.txt

# Look for stuck goroutines (blocked on channels, mutexes)
grep -A5 "semacquire\|chanrecv\|select" goroutines.txt
```

### Using Metrics for Diagnosis

Access runtime metrics programmatically:

```go
snap := collector.Snapshot()
for name, count := range snap.Counters {
    fmt.Printf("%s: %d\n", name, count)
}
```

Key metrics to monitor:

- `op_count_MatMul` -- If disproportionately high, investigate batching.
- `op_duration_seconds` -- Histogram; check p99 for tail latency.
- `allreduce_duration_seconds` -- If high, check network or straggler workers.

### Benchmark Comparison

Compare current performance against the baseline:

```
go test -bench=. -benchmem -count=3 -run='^$' ./compute/ > bench-current.txt
go run ./cmd/bench-compare -baseline benchmarks/baseline.txt -current bench-current.txt -threshold 10
```
