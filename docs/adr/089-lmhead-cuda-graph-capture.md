# ADR-089: CUDA Graph Capture Compatibility for LMHead

**Status:** Proposed (awaiting ztensor ship)
**Date:** 2026-04-16
**Epic:** E99 (gemma4e CUDA graph capture)
**Task:** T99.1.4

## Context

After ADR-088 (T99.1.2) made `Gemma4PLECombinedProducer` capture-safe,
the CUDA graph capture region for gemma4e expanded from a single layer
to `[2, 569)` — the full transformer body. However, `cudaStreamEndCapture`
still fails with:

```
cuda graph: end capture failed: cudaStreamEndCapture failed: operation
failed due to a previous error during capture
CUDA GRAPH: capture failed: instruction 568 (LMHead): number of axes 3
must match tensor dimensions 2
```

Instruction 568 is the last instruction in the plan (LMHead is the
terminal op of the transformer graph: `hidden -> logits`). The
`number of axes 3 must match tensor dimensions 2` error originates
from `CPUEngine.Transpose` in
`ztensor/compute/cpu_engine.go:1306`, reached via a fallback inside
`GPUEngine.MatMulTransposeB`
(`ztensor/compute/gpu_engine.go:1248,1258,1307,1317,1338`). The
fallback calls `e.Transpose(ctx, b, []int{0, 2, 1})` — 3 axes — on the
LMHead weight `b` which is always 2D `[vocabSize, hiddenDim]`.

The fallback fires when any of these is true:
- BLAS does not implement `BLASTransposeB` (SgemmNT).
- `T` is not `float32` (e.g. FP16 path on gemma4e could end up here).
- `getDevicePtr(e, a)` or `getDevicePtr(e, b)` fails (common during
  graph capture because CUDA rejects synchronous H2D copies on a
  capturing stream when a tensor is CPU-resident).
- `e.pool.Alloc` fails (VRAM pressure).

On DGX with gemma4e Q4_K_M, at least one of these triggers during
capture, and the 3-axis Transpose on a 2D weight explodes.

A previous incarnation of this same bug was observed at an earlier
commit: 2026-03-26 devlog entry reports
`instruction 184 (LMHead): number of axes 3 must match tensor
dimensions 2` on the mmap loading path — same root cause, different
trigger density.

## Options Considered

### A. Register `LMHead` in ztensor's `nonCapturableOps`

**What it means.** Add `"LMHead": true` to the `nonCapturableOps` map
in `ztensor/graph/cuda_graph.go`. The capture-region selector
(`NewCUDAGraphExecutor` in `cuda_graph.go:264`) picks the longest
contiguous capturable run; with LMHead excluded, the region becomes
`[2, 568)` and LMHead runs post-capture on the host stream (same path
used for `EmbeddingLookup` at pre-capture).

**Capture region impact.** We lose exactly one instruction at the
tail. The 35-layer transformer body (~566 instructions) stays captured.
No layer-body fragmentation, unlike the pre-T99.1.2 state of the PLE
slicers. This is the ideal outcome for a last-in-plan op.

**Cost.** One-line change in ztensor. Requires release-please cut and a
go.mod bump in zerfoo. No behavior change on CPU, no throughput
regression on GPU uncaptured path, no correctness impact.

### B. Fix the axes/dims path in `GPUEngine.MatMulTransposeB` fallback (ztensor)

**What it means.** Change every `e.Transpose(ctx, b, []int{0, 2, 1})`
in `gpu_engine.go` `MatMulTransposeB` to branch on `len(b.Shape())`:
use `[]int{1, 0}` when `b` is 2D and `[]int{0, 2, 1}` when 3D.

**Why more risky.** Touches five fallback sites, each with subtly
different context (unsupported BLAS, wrong dtype, H2D blocked, OOM).
Also the failure mode inside capture is a CUDA error that may be
partially synchronous — even with the correct Transpose axes, the
resulting `e.MatMul(ctx, a, kT, dst...)` would execute on the host
stream and could still break capture by issuing a synchronous memcpy.
We would need to confirm all four preconditions that drive into the
fallback are capture-safe; option A sidesteps the question entirely.

**Why deferred.** This is a general-purpose bugfix in ztensor worth
doing independently of T99.1.4. It does not need to block the capture
fix. File as a separate ztensor issue.

### C. Avoid `MatMulTransposeB` in `lmHeadNode.Forward` (zerfoo-only)

**What it means.** In `inference/arch_llama.go:lmHeadNode.Forward`,
skip the `TransposeBMatMuler` branch for the non-quantized case and
always do an explicit `Transpose(weight, [1,0])` + `MatMul`.

**Why rejected.**

1. Regresses the original `MatMulTransposeB` motivation. The code's
   existing comment (line 98-102) explains that explicit Transpose
   allocates a 1GB+ temporary on large-vocab models and caused a prior
   use-after-free. While we could avoid caching, the allocation cost
   alone is a real per-step hit during decode.
2. Even if the explicit Transpose runs fine uncaptured, it would still
   hit the same `getDevicePtr`-during-capture failure mode during
   graph capture on CPU-resident weights.
3. Does not fix the analogous problem for any other op that routes
   through `MatMulTransposeB` with a 2D B (e.g., future attention
   variants that reuse the helper).

### D. Fail-open: catch the CPU Transpose error in LMHead and recover

Not viable. The error happens mid-capture, at which point the CUDA
stream is already invalidated. Recovering requires tearing down the
whole capture and starting over — which is exactly what the current
fall-through-to-uncaptured-path does.

## Decision

**Adopt Option A.** Register `LMHead` in ztensor's `nonCapturableOps`.

Rationale:
- Matches the T99.1.1 / ADR-088 playbook for an op that does
  host-visible work during capture.
- LMHead is the last instruction; post-capture placement costs zero
  capture-region coverage.
- Minimal cross-repo surface (one line in ztensor, one go.mod bump in
  zerfoo).
- Option B is a ztensor-internal hardening that should land separately
  as a robustness improvement, not as a blocker for gemma4e capture.

## Consequences

### Positive

- gemma4e on CUDA can drop `ZERFOO_DISABLE_CUDA_GRAPH=1` once the
  ztensor change ships and the go.mod bump lands. Capture region is
  35-layer transformer body, equivalent to other GGUF architectures.
- Pattern generalizes to any terminal op that triggers host-visible
  fallbacks during capture (e.g., a future sampled-softmax head).

### Negative / cost

- One extra instruction (LMHead) runs post-capture on the host stream
  instead of replayed inside the graph. Measurable impact: a single
  MatMul + Reshape + optional softcap at the very end of the pass.
  Based on the 184/185 instruction capture ratio seen on Gemma 3, the
  throughput delta from excluding one terminal op is small enough to
  be inside bench noise.
- Cross-repo dependency: zerfoo's gemma4e capture fix requires a
  ztensor release. Not new (ADR-088 had the same pattern).

### Follow-ups

- **Ship the one-line ztensor change** (T99.1.4a, this ADR's
  implementation). Coordinator ships `zerfoo/ztensor` first, then a
  bump PR in `zerfoo/zerfoo`.
- **T99.1.3** (currently blocked): verify on DGX that
  `cudaStreamEndCapture` succeeds without the env var.
- **Optional: fix `MatMulTransposeB` fallback axes** (file as new
  ztensor task). Low-risk hardening that prevents this class of bug
  recurring for other 2D-B callers.
- **T99.2.1** (throughput regression) and **T99.2.2** (decode
  correctness) remain separate and out of scope here.

## Handoff diff (ztensor)

The exact change required in `github.com/zerfoo/ztensor` at
`graph/cuda_graph.go:57-66`:

```diff
 var nonCapturableOps = map[string]bool{
 	"EmbeddingLookup":            true,
 	"Gather":                     true,
 	"AutoAttentionMask":          true,
 	"AutoPositionIds":            true,
 	"Slice":                      true,
 	"ConstantOfShape":            true,
 	"Shape":                      true,
 	"Gemma4PLECombinedProducer":  true,
+	"LMHead":                     true,
 }
```

Accompanying comment block (`graph/cuda_graph.go`, near the
`Gemma4PLECombinedProducer` justification):

```go
// LMHead is the terminal op of transformer graphs (hidden -> logits).
// Its internal MatMulTransposeB fallback calls Transpose(weight,
// [0,2,1]) on a 2D weight tensor, raising "number of axes 3 must
// match tensor dimensions 2" from CPUEngine.Transpose when any of
// BLAS-NT absence, non-float32 T, getDevicePtr failure, or
// Alloc-during-capture drives into the fallback path. Since LMHead
// is the last instruction, placing it post-capture costs zero
// capture-region coverage. See zerfoo/docs/adr/089.
```

After the ztensor PR merges and release-please cuts a version X:

- Update `zerfoo/go.mod`: bump `github.com/zerfoo/ztensor` to version X.
- Run `go mod tidy` to update `go.sum`.
- Drop `ZERFOO_DISABLE_CUDA_GRAPH: "1"` from
  `docs/bench/manifests/gemma4-e2e.yaml`.
- Verify on DGX via `scripts/gemma4-spark.sh -mode generate -device cuda
  -steps 32 -cleanup`. Success = no
  `cudaStreamEndCapture failed` or `capture failed: instruction ... (LMHead)`
  lines in the log. Unblocks T99.1.3.
