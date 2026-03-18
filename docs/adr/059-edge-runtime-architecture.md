# ADR 059: Zerfoo Runtime -- Edge Inference Architecture

## Status
Accepted

## Date
2026-03-18

## Context
Edge deployment (Raspberry Pi, Jetson, mobile, embedded) requires a minimal inference
binary. Go cross-compiles natively (GOOS/GOARCH), produces static binaries (zero CGo),
and already has ARM NEON SIMD. However, full Zerfoo binary includes training,
distributed, serve, and other packages unnecessary for edge inference.

Alternatives considered:
1. TinyGo compilation -- smaller binaries but limited stdlib and no generics support.
2. Build tags to exclude packages -- standard Go approach, no toolchain change.
3. Separate minimal repo -- duplicates code, maintenance burden.

## Decision
Zerfoo Runtime is a build-tag-gated minimal binary:

1. `go build -tags edge ./cmd/zerfoo-runtime` excludes training/, distributed/,
   serve/, and unnecessary layer packages. Target: under 10MB static ARM64 binary.

2. Pre-optimized model format: `zerfoo optimize --target arm64` pre-computes graph
   fusion decisions and stores them in GGUF metadata. Runtime skips optimization pass.

3. Accelerator delegates via Engine[T]: CoreML delegate (macOS/iOS), NNAPI delegate
   (Android) as optional Engine[T] implementations behind build tags.

4. Cross-compilation targets (priority order): linux/arm64 (Pi, Jetson),
   darwin/arm64 (macOS, iOS via gomobile), linux/amd64 (containers),
   android/arm64 (gomobile).

## Consequences
Positive:
- Static binary with zero external dependencies -- unmatched deployment simplicity.
- Same code path as full Zerfoo -- no correctness divergence.
- Go cross-compilation is free (no toolchain setup).

Negative:
- Go binary baseline is 5-10MB even stripped -- larger than TFLite (~1MB).
- gomobile for iOS/Android adds build complexity.
- Platform-specific delegates (CoreML, NNAPI) require CGo or purego bindings.
