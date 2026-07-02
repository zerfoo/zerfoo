# ADR 025: Replace CGo with purego for CUDA Bindings

## Status
Accepted

## Date
2026-03-06

## Context
Zerfoo's GPU backend uses CGo (`import "C"`) to call CUDA runtime functions and
custom kernel launchers. This has several downsides:

1. **Build complexity**: All 8 files in internal/cuda/ require `//go:build cuda`
   tags. The binary must be compiled with `-tags cuda` on a machine with CUDA
   headers and nvcc. Cross-compilation is impossible without a CUDA toolkit.

2. **CGo overhead**: Each CGo call costs ~100-200ns due to goroutine stack
   switching and C ABI setup. With 650+ calls per token, this adds ~65-130us
   per token (~0.5-1% of decode time). Small but non-zero.

3. **Build time**: CGo compilation is slow. Each `go build` recompiles C stubs.
   The CUDA kernel shared library (libkernels.so) is already pre-compiled via
   make, so the CGo layer is just a function-pointer bridge.

4. **Runtime detection impossible**: With CGo, GPU support is a compile-time
   decision. A single binary cannot detect GPU availability at runtime.

The Go standard library provides `syscall.NewLazyDLL` (Windows) and Go 1.21+
provides `purego` patterns using `plugin.Open` or `dlopen` via unsafe. The
ebitengine/purego library provides a mature, well-tested dlopen wrapper that
works on Linux, macOS, and Windows.

However, purego is a third-party dependency, which conflicts with the project's
"Go standard library only" convention. An alternative is to use `plugin.Open`
(stdlib) or raw `syscall` dlopen calls. The `plugin` package only loads Go
plugins, not C shared libraries. Raw dlopen via `unix.Dlopen` + `unix.Dlsym`
from `golang.org/x/sys/unix` is viable and avoids third-party deps.

## Decision
Replace CGo bindings in internal/cuda/ with dlopen-based function loading using
`golang.org/x/sys/unix` (Dlopen, Dlsym, Dlclose). This is a semi-standard
dependency (golang.org/x) already used by many Go projects.

The approach:
1. Compile CUDA kernels into a shared library (libkernels.so) -- already done.
2. Compile CUDA runtime wrappers (malloc, memcpy, stream) into the same .so or
   load libcudart.so directly via dlopen.
3. Replace each `import "C"` file with a pure Go file that uses dlopen/dlsym to
   load function pointers and `purego`-style register+call to invoke them.
4. Remove `//go:build cuda` tags. Instead, attempt dlopen at runtime; if the .so
   is not found, fall back to CPU-only mode.
5. The .cu source files and Makefile are unchanged. Only the Go calling layer
   changes.

This applies to ALL GPU operations regardless of model architecture (transformers,
RNNs, CNNs, etc.) since it replaces the transport layer, not the compute layer.

## Consequences
Positive:
- Single binary works on both GPU and CPU machines (runtime detection)
- No CGo build tags; `go build ./...` always compiles
- Eliminates CGo goroutine stack overhead (~100-200ns per call)
- Faster build times (no CGo compilation step)
- Cross-compilation becomes possible
- Architecture-agnostic: benefits every layer type

Negative:
- Adds golang.org/x/sys dependency (semi-standard, widely accepted)
- Manual function signature management (must match C ABI exactly)
- Slightly more complex calling convention than CGo for pointer types
- Must handle dlopen errors gracefully for runtime detection
- Kernel shared library (.so) must be distributed alongside the binary
