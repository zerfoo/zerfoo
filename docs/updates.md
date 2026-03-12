# S100.1.1 DGX Spark Integration Test Results

Date: 2026-03-11

## Environment

- **Host**: ndungu@192.168.86.250 (DGX Spark, NVIDIA GB10)
- **OS**: Linux 6.17.0-1008-nvidia aarch64
- **Go**: 1.25.0 linux/arm64
- **CUDA**: 13.0 (/usr/local/cuda)
- **Code**: upstream/main at commit 765108e (Merge PR #45 feat/neon-softmax)
- **Build**: `go build -tags cuda` with CGO_CFLAGS/CGO_LDFLAGS pointing to
  /usr/local/cuda

## Performance Results

| Model | Device | Tokens | tok/s | Megakernel Log? |
|-------|--------|--------|-------|-----------------|
| gemma3 (F32) | cuda | 64 | 12.84 | NO |
| gemma3 (F32) | cuda | 16 | 12.19 | NO |
| gemma3-q4 | cuda | 64 | 8.61 | NO |
| gemma3-q4 | cpu | 16 | 5.82 | NO |

### Baselines (from plan.md)

| Config | tok/s |
|--------|-------|
| CPU ARM64 (post Track D) | 8.15 median |
| GPU cuda (previous) | 10.32 peak / 7.78 median |

## Findings

### 1. Megakernel Did Not Fire

The "megakernel: compiled and loaded" log message never appeared.
`tryCompileMegakernel` (generate/megakernel.go:21) is called at
generate/generator.go:152 but silently fails. All error paths in
`tryCompileMegakernel` return without logging, making it impossible to
determine from output alone which step failed:

- `codegen.CheckSupport` (unsupported ops)
- `codegen.EmitMegakernel` (source generation)
- `codegen.CachedCompile` (nvcc compilation)
- `codegen.LoadMegakernel` (dlopen)

The most likely failure point is `codegen.CheckSupport`, which probably finds
unsupported ops in the Gemma 3 execution plan (KV cache ops, rotary
embeddings, or attention ops). This aligns with T100.2 (GPU KV cache wiring)
being listed as a prerequisite.

### 2. GPU Throughput Improved

The F32 model at 12.84 tok/s exceeds the previous baseline of 10.32 peak.
This improvement comes from the regular (non-megakernel) GPU execution path.

### 3. Output Quality Issues

Both models produce gibberish/repetitive output on CPU and GPU. The F32 model
repeats "land" indefinitely. The Q4 model outputs random tokens. This may
indicate model or quantization issues unrelated to the megakernel path.

### 4. Q4 vs F32 Performance Gap

Q4 on GPU (8.61 tok/s) is slower than F32 on GPU (12.84 tok/s). This is
unexpected and may indicate the Q4 kernel path is not GPU-optimized.

## Recommendation

Add diagnostic logging to `tryCompileMegakernel` at each failure point so the
exact failure cause can be identified. Example:

```go
unsupported := codegen.CheckSupport(instructions)
if len(unsupported) > 0 {
    log.Printf("megakernel: %d unsupported ops: %v", len(unsupported), unsupported)
    return
}
```

T100.2 (GPU KV cache wiring) is likely required before the megakernel can
fire on a real model.

## Quality Gate Assessment

| Gate | Status |
|------|--------|
| "megakernel: compiled and loaded" appears | FAIL |
| bench_tps runs on DGX Spark | PASS |
| Performance baseline recorded | PASS |
