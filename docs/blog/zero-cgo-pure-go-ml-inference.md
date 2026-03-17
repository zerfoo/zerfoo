# Zero CGo: Why We Chose Pure Go for ML Inference

*Architecture deep-dive: how Zerfoo runs GPU-accelerated ML inference without a single CGo call.*

## The Problem

If you want to run ML inference in a Go application today, your options are limited. You can shell out to a Python process, wrap a C++ runtime like ONNX Runtime or TensorRT via CGo, or send HTTP requests to an external inference server. All of these add operational complexity that Go developers have learned to avoid.

CGo is the most tempting option because it looks like a first-class Go feature. It is not. The moment you add `import "C"` to your codebase, you lose most of what makes Go productive.

## What CGo Costs You

**Build complexity.** CGo requires a C compiler toolchain on every machine that builds your code. For CUDA, that means the CUDA toolkit headers and `nvcc`. Your `go build ./...` is now a multi-step build script with platform-specific flags, `CGO_CFLAGS`, `CGO_LDFLAGS`, and library paths. CI pipelines need CUDA-dev Docker images. New contributors need a setup guide instead of `git clone && go build`.

**Broken cross-compilation.** Go's cross-compilation story is one of its best features: `GOOS=linux GOARCH=arm64 go build` just works. CGo breaks this completely. Cross-compiling CGo code requires a cross-compilation toolchain for the target platform, and for CUDA specifically, you need the target architecture's CUDA libraries. In practice, you build on the target platform or not at all.

**No static binaries.** CGo forces dynamic linking. Your single, statically-linked Go binary becomes a binary that depends on `libcudart.so`, `libcublas.so`, and whatever else you linked. Deploying means shipping shared libraries or building on the target machine.

**Debugging across language boundaries.** When a CGo-linked CUDA call segfaults, you get a crash in C land with no Go stack trace. CUDA memory errors surface as opaque error codes through the CGo barrier. GDB and Delve do not play well together across the boundary.

**Runtime overhead.** Every CGo call goes through `runtime.cgocall`, which increments the CGo counter, switches to the system stack, and coordinates with the garbage collector. This overhead is small per call but measurable at the thousands-of-calls-per-second rate that inference demands.

## The purego Alternative

Zerfoo takes a different approach. Instead of linking to CUDA at build time, we load it at runtime using `dlopen`/`dlsym` -- the same mechanism that every dynamically-loaded library on Linux uses, but called from pure Go without CGo.

### Loading the CUDA Runtime

At startup, Zerfoo attempts to open `libcudart.so` and resolve the function pointers it needs:

```go
// CUDALib holds dlopen handles and resolved function pointers for
// CUDA runtime functions.
type CUDALib struct {
    handle uintptr // dlopen handle for libcudart

    cudaMalloc            uintptr
    cudaFree              uintptr
    cudaMemcpy            uintptr
    cudaMemcpyAsync       uintptr
    cudaStreamCreate      uintptr
    cudaStreamSynchronize uintptr
    // ... 14 runtime functions total
}

func Open() (*CUDALib, error) {
    lib := &CUDALib{}
    for _, path := range []string{"libcudart.so.12", "libcudart.so"} {
        h := dlopenImpl(path, rtldLazy|rtldGlobal)
        if h != 0 {
            lib.handle = h
            break
        }
    }
    if lib.handle == 0 {
        return nil, fmt.Errorf("cuda: dlopen libcudart failed")
    }
    // Resolve each function pointer via dlsym...
    syms := []sym{
        {"cudaMalloc", &lib.cudaMalloc},
        {"cudaFree", &lib.cudaFree},
        // ...
    }
    for _, s := range syms {
        addr := dlsymImpl(lib.handle, s.name)
        if addr == 0 {
            return nil, fmt.Errorf("cuda: dlsym %s failed", s.name)
        }
        *s.ptr = addr
    }
    return lib, nil
}
```

If CUDA is not installed, `Open()` returns an error and inference falls back to the CPU engine. The binary is the same either way -- no build tags, no conditional compilation.

### Calling C Functions Without CGo

The resolved function pointers are raw addresses. To call them, we need to invoke C calling convention functions from Go. The platform-specific implementations use Go's internal runtime machinery directly, bypassing CGo entirely.

On Linux arm64 (where CUDA GPUs live), we use `runtime.asmcgocall` to run on the system stack with an assembly trampoline that sets up the AAPCS64 calling convention:

```go
//go:linkname asmcgocall runtime.asmcgocall
func asmcgocall(fn, arg unsafe.Pointer) int32

type ccallArgs struct {
    fn   uintptr
    args [20]uintptr
    ret  uintptr
}

func ccall(fn uintptr, a ...uintptr) uintptr {
    var args ccallArgs
    args.fn = fn
    copy(args.args[:], a)
    runTrampoline(&args)
    return args.ret
}
```

The assembly trampoline (`purego_linux_arm64.s`) loads arguments into registers X0-X7 per AAPCS64, calls the function pointer, and stores the return value. This is exactly what CGo does internally -- but without the `runtime.cgocall` bookkeeping overhead.

### Custom CUDA Kernels

CUDA kernels are written in CUDA C, compiled separately into `libkernels.so` via `nvcc`, and loaded the same way:

```go
func openKernelLib() (*KernelLib, error) {
    lib, err := cuda.DlopenKernels() // dlopen("libkernels.so")
    if err != nil {
        return nil, err
    }
    k := &KernelLib{handle: lib}
    // Resolve 60+ kernel launch functions
    syms := []struct {
        name string
        dest *uintptr
    }{
        {"launch_rmsnorm", &k.launchRMSNorm},
        {"flash_attention_forward_f32", &k.launchFlashAttentionF32},
        {"fused_rope_f32", &k.launchFusedRoPEF32},
        {"gemm_q4_f32", &k.launchGemmQ4F32},
        // ...
    }
    for _, s := range syms {
        ptr, _ := cuda.Dlsym(lib, s.name)
        *s.dest = ptr
    }
    return k, nil
}
```

Launching a kernel is then a direct function call through the resolved pointer:

```go
func RMSNorm(input, weight, output, scales unsafe.Pointer,
    eps float32, rows, D int, stream unsafe.Pointer) error {
    k := klib()
    ret := cuda.Ccall(k.launchRMSNorm,
        uintptr(input), uintptr(weight), uintptr(output), uintptr(scales),
        floatBits(eps), uintptr(rows), uintptr(D), uintptr(stream))
    return checkKernel(ret, "rmsnorm")
}
```

The kernel itself is standard CUDA C. The `launch_rmsnorm` function is a thin C wrapper that calls `<<<blocks, threads, 0, stream>>>` and returns the CUDA error code. From Go's perspective, it is just a function pointer that takes integers and returns an integer.

### Optional Symbol Resolution

Not all kernels need to be present. Flash attention decode, FP8 operations, and specialized memory copy kernels are marked optional:

```go
optionalSyms := map[string]bool{
    "flash_attention_decode_f32": true,
    "launch_fp8_add":            true,
    "launch_fp8_rmsnorm":        true,
    // ...
}
for _, s := range syms {
    ptr, dlErr := cuda.Dlsym(lib, s.name)
    if dlErr != nil {
        if optionalSyms[s.name] {
            continue // leave pointer as 0; callers check
        }
        return nil, fmt.Errorf("kernels: dlsym %s: %w", s.name, dlErr)
    }
    *s.dest = ptr
}
```

Callers check for a zero function pointer before invoking. This means you can compile a partial `libkernels.so` during development -- the framework gracefully degrades to CPU paths for missing kernels.

## What This Means for Deployment

**Single binary.** `go build -o zerfoo ./cmd/zerfoo` produces one static binary. Copy it to your server. If CUDA is installed, it uses the GPU. If not, it runs on CPU. Same binary.

**Cross-compilation works.** `GOOS=linux GOARCH=arm64 go build` produces an arm64 Linux binary on your Mac. It will load CUDA at runtime on a machine that has it.

**No build infrastructure changes.** Embedding ML inference into an existing Go service is `go get github.com/zerfoo/zerfoo` and a few lines of code. Your CI pipeline, your Dockerfile, your Makefile -- none of them need to know about CUDA.

**Graceful degradation.** The `cuda.Available()` function is a one-liner that tries `dlopen` and caches the result:

```go
func Available() bool {
    globalOnce.Do(func() {
        globalLib, errGlobal = Open()
    })
    return errGlobal == nil
}
```

Application code never needs to check for CUDA. The compute engine dispatches to GPU or CPU transparently.

## Performance Without Compromise

The obvious concern: does avoiding CGo make it slower?

No. The bottleneck in ML inference is GPU compute, not the binding layer. Whether you call `cudaLaunchKernel` through CGo or through a dlsym'd function pointer, the GPU executes the same kernel. The call overhead difference is on the order of nanoseconds; a single transformer layer takes milliseconds.

Here are the numbers. On a DGX Spark (GB10 Grace Blackwell), running Gemma 3 1B with Q4_K_M quantization:

| Runtime | Decode throughput | Notes |
|---------|------------------|-------|
| **Zerfoo** | **245 tok/s** | Pure Go, zero CGo, custom CUDA kernels via dlopen |
| Ollama | 204 tok/s | Go wrapper around llama.cpp (C++) |

Zerfoo is 20% faster than Ollama on the same hardware, despite Ollama being a thin wrapper around C++. The performance comes from the kernels, not the binding mechanism:

- **25+ custom CUDA kernels** including fused RoPE, fused SwiGLU, fused Add+RMSNorm, fused QK-Norm+RoPE, flash attention (prefill and decode), quantized GEMM/GEMV (Q4_0, Q4_K_M, Q8_0)
- **CUDA graph capture** replays the entire decode step as a single graph launch, eliminating per-kernel launch overhead. 99.5% of decode instructions are captured.
- **ARM NEON SIMD assembly** for CPU-bound operations (GEMM, RMSNorm, RoPE, SiLU, softmax) written in hand-tuned `.s` files

The purego binding layer is not on the critical path. Kernel execution time dominates.

*Methodology note: benchmarks run on NVIDIA GB10 (DGX Spark), CUDA 13.0, Ubuntu 24.04 arm64. Gemma 3 1B Q4_K_M with 512-token context. Ollama v0.6.x with default settings. Throughput measured over 100 decode iterations after warmup.*

## Trade-offs

This approach is not free. There are real costs.

**Assembly per platform.** The `ccall` trampoline requires platform-specific assembly. We have implementations for `linux/arm64` (production) and `darwin/arm64` (development). Adding a new platform means writing a new `.s` file that implements the target's calling convention. In practice, CUDA only runs on Linux x86_64 and arm64, so two assembly files cover the GPU-relevant platforms.

**No inline C.** CGo lets you write C code directly in Go source files. With purego, all C code lives in separately compiled shared libraries. For the custom CUDA kernels this is natural (they are `.cu` files compiled by `nvcc` anyway). For one-off C calls, it means you cannot just drop a `// #include` into a Go file.

**Debugging GPU crashes is harder.** When a CUDA kernel segfaults through a dlsym'd function pointer, the crash dump does not show a CGo frame. You get a SIGSEGV in the assembly trampoline. In practice, CUDA kernel bugs manifest as CUDA error codes (not segfaults), and those propagate cleanly through the `checkKernel` error path.

**`unsafe.Pointer` everywhere.** The GPU binding code is inherently unsafe -- it passes raw device pointers across language boundaries. This is equally true of CGo. The unsafety is confined to the `internal/cuda` package and does not leak into the public API.

**`go:linkname` dependency.** The Linux arm64 path uses `//go:linkname asmcgocall runtime.asmcgocall` to access an internal Go runtime function. This is not a stable API and could break with a Go release. In practice, `asmcgocall` has been stable for 10+ years and is unlikely to change, but it is a risk worth acknowledging.

## Conclusion

Zero CGo is not a limitation. It is a deliberate design choice that preserves everything Go developers expect: `go build` works everywhere, cross-compilation works, binaries are self-contained, and GPU acceleration is a runtime capability, not a build-time dependency.

The alternative -- wrapping a C++ inference runtime via CGo -- would make Zerfoo easier to build initially but harder to use in practice. Every Go developer who wanted to add ML inference to their application would need to set up a C++ toolchain, manage shared library dependencies, and debug across language boundaries.

With purego, adding ML inference to a Go application is:

```bash
go get github.com/zerfoo/zerfoo
```

That is the entire build change. The rest is just Go code.
