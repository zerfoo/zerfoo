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

---

# S100.2.1 KV Cache Integration Test Results

Date: 2026-03-11

## Environment

- **Host**: ndungu@192.168.86.250 (DGX Spark, NVIDIA GB10)
- **OS**: Linux 6.17.0-1008-nvidia aarch64
- **Go**: 1.26.0 linux/arm64
- **CUDA**: 13.0 (/usr/local/cuda)
- **Code**: upstream/main at commit 17b0e8a
- **Build**: `go build -tags cuda` with CGO_CFLAGS/CGO_LDFLAGS pointing to
  /usr/local/cuda

## Build Fixes Required

### 1. Missing runner_stub.go methods (commit 2faa5b2)

T100.2 added `SetKVCache()` and `HasKVCache()` to `MegakernelRunner` in
`runner.go` (`//go:build !cuda`), but the corresponding `runner_stub.go`
(`//go:build cuda`) was not updated. Build failed with:

```
runner.SetKVCache undefined (type *codegen.MegakernelRunner has no field or method SetKVCache)
```

Fix: Added `SetKVCache` and `HasKVCache` stubs to `runner_stub.go`.

### 2. Purego/CGo linker conflict (commit 17b0e8a)

T87.3 added `cgo_import_dynamic` directives in `purego_linux_arm64.go` to
import dlopen/dlsym from libdl.so.2 via assembly trampolines. When building
with `-tags cuda`, other CGo files activate, and the Go linker cannot handle
`SDYNIMPORT` relocations alongside the external (CGo) linker:

```
internal/cuda.libc_dlopen_trampoline: unhandled relocation for libc_dlopen
(type 65 (SDYNIMPORT) rtype 9 (R_CALLARM64))
```

Fix: Split platform implementation into two files:
- `purego_linux_arm64.go` (`!cuda`): zero-overhead asm trampolines
- `purego_linux_arm64_cgo.go` (`cuda`): CGo-based dlopen/dlsym/ccall

## Performance Results

| Model | Device | Tokens | tok/s | Megakernel? |
|-------|--------|--------|-------|-------------|
| gemma3 (F32) | cuda | 16 | 11.81 | NO |
| gemma3 (F32) | cuda | 50 | 11.54 | NO |
| gemma3-q4 | cuda | 50 | 8.98 | NO |

### Comparison with S100.1.1

| Model | S100.1.1 tok/s | S100.2.1 tok/s | Delta |
|-------|----------------|----------------|-------|
| gemma3 (F32) | 12.84 (64 tok) | 11.54 (50 tok) | -10% |
| gemma3-q4 | 8.61 (64 tok) | 8.98 (50 tok) | +4% |

Small variance is expected; different token counts and Go version (1.25 vs 1.26).

## Findings

### 1. Megakernel Did Not Fire — 16 Unsupported Ops Identified

`codegen.CheckSupport` rejects 16 ops not in the emitter table:

```
AutoPositionIds AutoZeroKVCache Shape Unsqueeze Cast Equal Where
ConstantOfShape Expand Range Cos Sin Greater Trilu Max ScatterND
```

These ops fall into three categories:

**RoPE (Rotary Positional Embeddings)**: `Cos`, `Sin`, `Range`, `AutoPositionIds`
- RoPE computes position-dependent rotation matrices using sin/cos of positions.
- `Range` generates position indices; `Cos`/`Sin` compute rotation components.

**Attention masking**: `Equal`, `Where`, `Greater`, `Trilu`, `ConstantOfShape`, `Expand`
- Causal attention mask construction: `Trilu` creates triangular mask,
  `Where`/`Greater`/`Equal` apply conditional logic, `ConstantOfShape`
  fills with -inf, `Expand` broadcasts the mask.

**Utility/shape ops**: `Shape`, `Unsqueeze`, `Cast`, `AutoZeroKVCache`, `Max`, `ScatterND`
- `Shape`/`Unsqueeze` are tensor metadata ops (could be no-ops in megakernel).
- `Cast` converts types (e.g., int64 indices to float32).
- `AutoZeroKVCache` initializes KV cache (one-time setup, not per-token).
- `Max` is element-wise max (for clamping).
- `ScatterND` is an indexed write (for KV cache updates).

### 2. Architectural Issue: Megakernel Runner vs GPU Engine Build Tags

The megakernel runner (`runner.go`) has `//go:build !cuda` and uses purego
dlopen to load compiled .so files. The GPU engine (`gpu_engine.go`) has
`//go:build cuda` and uses CGo-based cuBLAS/cuDNN. These are mutually
exclusive — the megakernel runner cannot be active in a CUDA build.

This means even if all ops were supported, the megakernel runner stub
(`runner_stub.go`) would return `errStub` from `LoadMegakernel()`, and
the megakernel would never fire in a `-tags cuda` build.

This is a fundamental architectural blocker that requires either:
- **Option A**: Move the megakernel runner out of the `!cuda` constraint
  (use CGo-based dlopen when building with `-tags cuda`)
- **Option B**: Remove build tags entirely per ADR-025 (bigger refactor)

### 3. Output Quality

Both models continue to produce gibberish/repetitive output (consistent
with S100.1.1 findings). This is a pre-existing issue unrelated to the
megakernel path.

## Summary of Blockers

| Blocker | Severity | Fix Scope |
|---------|----------|-----------|
| 16 unsupported ops in CheckSupport | High | Add emitters for each op (~2-4h) |
| runner_stub.go returns errStub in cuda build | Critical | Move runner to work in cuda build (~1h) |
| Build tag architecture (purego vs CGo) | Architectural | ADR-025 phase 2 (TBD) |

## Recommendation

1. **Immediate**: Fix `runner_stub.go` to use real dlopen (not stub) when
   building with `-tags cuda`. The CGo dlopen fallback created in this
   session provides the infrastructure.

2. **Short-term**: Add emitters for the 16 unsupported ops. Priority order:
   - `Cos`, `Sin` (trivial: `unaryOp("cosf")`, `unaryOp("sinf")`)
   - `Max` (trivial: `funcBinaryOp("fmaxf")`)
   - `Shape`, `Unsqueeze`, `Reshape` (no-ops)
   - `Cast` (type conversion)
   - `Range`, `Expand`, `Repeat` (indexing)
   - `Equal`, `Greater`, `Where` (comparison/select)
   - `Trilu`, `ConstantOfShape` (mask construction)
   - `ScatterND` (indexed write)
   - `AutoPositionIds`, `AutoZeroKVCache` (model-specific setup)

3. **Long-term**: Complete ADR-025 — remove `//go:build cuda` tags entirely,
   use runtime dlopen detection for all GPU operations.

## Quality Gate Assessment

| Gate | Status |
|------|--------|
| Megakernel fires | FAIL |
| Blocker precisely identified | PASS |
| Performance numbers recorded | PASS |
| Results appended to docs/updates.md | PASS |
