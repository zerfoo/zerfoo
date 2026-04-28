# PJRT CPU Parity Tests (T126.1.1, E126)

This directory contains a build-tagged test scaffold for validating that the
PJRT CPU plugin produces results that match the native CPU compute engine on
the GGUF inference path.

## Status

**Blocked.** The scaffold compiles and skips cleanly, but the parity
assertion itself cannot run yet. Two prerequisites are missing:

1. **PJRT CPU plugin .so** is not yet vendored or built in this repo. The
   plan calls for acquiring or building `pjrt_c_api_cpu_plugin.so` from
   the OpenXLA source tree and exposing it via `PJRT_CPU_PLUGIN`.
2. **First-token logits accessor.** `inference.Model` exposes
   `Generate`, `GenerateStream`, `Chat`, and `ChatStream`, all of which
   sample tokens internally. There is no public hook returning the raw
   `[1, seqLen, vocabSize]` logits tensor (or its last-position slice) for
   a given prompt. Without that hook the parity test cannot perform the
   numerical comparison the plan requires.

Once both prerequisites land, replace the `t.Skip(...)` line in
`pjrt_parity_test.go` with the actual two-load comparison and lift the
build tag if appropriate.

## Running

```
go test -tags pjrt_test -run TestPJRTCPUParity -count=1 ./tests/parity/...
```

Required environment:

| Variable           | Meaning                                          |
| ------------------ | ------------------------------------------------ |
| `PJRT_CPU_PLUGIN`  | Absolute path to the PJRT CPU plugin shared lib. |
| `GEMMA3_MODEL_DIR` | Directory containing the Gemma 3 1B GGUF.        |

Without `PJRT_CPU_PLUGIN`, the test skips. Without
`GEMMA3_MODEL_DIR`, individual sub-tests skip. Default
`go test ./...` (no tag) skips this file entirely.

## Acceptance criterion

Per `docs/plan.md` E126 / T126.1.1: native vs PJRT first-token logits
match within absolute tolerance `1e-4`. The constant `pjrtTolerance` in
`pjrt_parity_test.go` codifies the threshold.

## Follow-ups

- Add a `(*inference.Model).FirstTokenLogits(ctx, prompt) ([]float32, error)`
  method (or equivalent debug/inspection hook) so parity tests can compare
  pre-sampling outputs deterministically.
- Wire the PJRT CPU plugin acquisition into `tools/` or the CI runner so
  `PJRT_CPU_PLUGIN` is set automatically once a host has it cached.
- After the CPU path is green, T126.1.2 extends the same harness to a
  CUDA PJRT plugin on DGX Spark.
