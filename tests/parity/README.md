# Model parity tests

## How a parity suite finds its model (T136.6)

The flagship GGUFs are staged **flat** in one directory on the GB10
(`/var/lib/zerfoo/models`), with no per-model subdirectories. `inference.Load`
resolves a *directory* through `findGGUF` (`inference/inference.go:317`), which
returns the **first** `.gguf` it sees -- so pointing every suite's
`*_MODEL_DIR` at that directory made every suite load the same file and the
verified-model matrix go green while proving nothing.

Model resolution therefore works like this, and must keep working like this:

1. `tests/parity/modelset/model-matrix.json` is the checked-in
   **matrix row -> exact GGUF filename** table. It is the only source of
   filenames.
2. Each `testutil.ModelParityConfig` names its row via `MatrixRow`. A suite
   without a row fails; an unknown row fails; a row with no pinned file skips.
   Nothing ever falls back to scanning.
3. `ZERFOO_MODELS_DIR` (or the row's own `*_MODEL_DIR`) names the **directory**
   only. The filename is joined from the table.
4. Before loading, the runner asserts the resolved absolute path is the
   declared file, checks the GGUF header's `general.architecture` (plus
   `general.name` and size when the row declares them), and records the claim
   so two rows can never resolve to one file.
5. Generation tests load via `inference.LoadFile(exactPath)`, never
   `inference.Load(dir)`.

Guard tests, all host-side and GPU-free:

| Test | Guards |
| ---- | ------ |
| `TestDirectoryScanCollapsesDistinctRowsOntoOneFile` | Red-proof witness: directory resolution collapses two rows onto one file. |
| `TestMatrixResolverKeepsFlatDirRowsDistinct` | Every staged row resolves to its own file in a flat directory. |
| `TestParityConfigsPinKnownDistinctMatrixRows` | No suite lacks a row; no two suites share one. |
| `TestStagedModelIdentity` | Per-row identity of the real staged GGUFs. |
| `tests/parity/modelset` unit tests | Unknown row, unstaged row, missing file, duplicate resolution, identity mismatch. |

Run them with:

```
go test ./tests/parity/ ./tests/parity/modelset/ -run 'Matrix|Identity|DirectoryScan|Parity' -count=1
```

Adding a model to the matrix means adding a row to `model-matrix.json` (exact
filename, architecture, and ideally `gguf_name` + `size_bytes`) and pointing a
`ModelParityConfig.MatrixRow` at it.

# PJRT CPU Parity Tests (T126.1.1, E126)

This section covers a build-tagged test scaffold for validating that the
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
