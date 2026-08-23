# Vacuous-assertion audit — 2026-08 (T152.3)

**Scope:** `tests/parity/` (priority), `layers/**/*_test.go`. CPU-only, no GPU pods.
**Method:** every finding below was **red-proofed** — the production code the test
covers was deliberately broken in a scratch worktree and the test re-run. A test
that stays green against broken code is CONFIRMED vacuous. Nothing in this
document is an unverified suspicion; candidates that survived red-proofing are
listed as HEALTHY, with the break that they caught.

**Why this audit exists:** three vacuous gates surfaced on 2026-08-20 — the
directory-scan parity harness (PR #987, lore L-0018), `ztensor#182`'s
thousandfold-loose Q5_K bound, and `TestGQA_CachedForward`'s saturated softmax
(lore L-0009 / L-0010, PR #993). Three in one day is a pattern, not coincidence.

## Headline

| | |
| --- | --- |
| Tests/assertions audited | **107** (79 golden-driven parity comparisons instrumented + 28 individually red-proofed) |
| CONFIRMED vacuous | **11 tests** across 6 packages |
| Confirmed HEALTHY under a real break | **13 tests** (plus 78 of 79 golden tolerances measured as adequately tight) |
| Fixed in this PR | 6 tests + 1 shared helper + 1 golden tolerance |
| Filed, not fixed (needs a behaviour ruling) | 5 |

## Method note: how the tolerance sweep was done

`tests/parity/testutil.AssertClose` backs ~100 golden-driven parity tests, so it
was temporarily instrumented to log, for every comparison, the declared
tolerance, the actual `maxAbsDiff`, and the largest expected magnitude. One run
of `go test ./tests/parity/` yielded 79 distinct measurements. The instrumentation
was reverted; the permanent fix it motivated is described under F1.

The vacuity condition for an absolute-tolerance comparison is exact and cheap to
state: **if `tolerance >= max|expected|`, an all-zero output passes.** Exactly one
golden in the suite violated it.

| comparison | tol | measured maxAbsDiff | max\|expected\| | tol as % of signal |
| --- | --- | --- | --- | --- |
| `mamba_block_forward` | 1e-03 | 9.09e-12 | 1.13e-04 | **886.32%** |
| `ssm_layer_forward` | 1e-04 | 3.73e-09 | 0.038 | 0.26% |
| `gqa_forward` | 1e-04 | 3.73e-09 | 0.0504 | 0.20% |
| `nbeats_forecast` | 1e-05 | 7.45e-09 | 0.0414 | 0.02% |
| `moe_forward` | 1e-04 | 2.98e-08 | 0.463 | 0.02% |
| `block_attn_res_forward` | 1e-04 | 8.94e-08 | 0.752 | 0.01% |
| ... 73 more | | | | all ≤ 0.01% |

The tolerances are round numbers (1e-4/1e-5/1e-6) rather than sized to measured
error — typically 10²–10⁵ looser than the observed worst case. That is loose, but
for 78 of 79 it is still far below the signal, so a real regression of any
meaningful size still trips them. Only `mamba_block_forward` crossed into
vacuity. **No zero-length comparisons were found** — `AssertClose`'s
length-mismatch `Fatalf` correctly prevents the nil-expected/empty-loop trap.

## CONFIRMED VACUOUS

Ranked by blast radius — what product claim currently rests on the test.

| # | Test | File:line | Shape | Red-proof (break applied) | Result | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | `TestGQABackward_MHA` (+ `gradClose` helper) | `layers/attention/gqa_backward_test.go:117`, helper `:321` | Tolerance: "relative" bound whose denominator is floored at 1.0, making `tol=1e-2` **absolute** | Dropped the entire dK path from GQA's input gradient (`dInputTotal -= dInputK`) | **stays green** — and so did every other test in `layers/attention`, `tests/parity`, `layers/transformer` | **FILED** — needs a ruling on the intended gradient-check contract |
| V2 | `TestParity_MambaBlock` | `tests/parity/layer_parity_test.go:1178` | Tolerance far looser than measured error: `tol=1e-3` vs max expected 1.13e-4 (886%) | `MambaBlock.Forward` returns all zeros | **stays green** | **FIXED** (F1, F2) |
| V3 | `TestMultiHeadLatentAttention_PartialRoPE_PositionIndependence` | `layers/attention/multi_head_latent_attention_test.go:309` | Self-passing gate: asserts shape, NaN-freeness, and a constructor config echo | Forced full RoPE in `applyPartialRoPE` while leaving `m.ropeHeadDim` intact (behaviour-only break) | **stays green** — whole `layers/attention` package green | **FILED** |
| V4 | `TestComplexSSMState_RoPEEffect` | `layers/ssm/complex_state_test.go:115` | Self-passing gate: `out1` and `out2` are the same call on the same object; asserts they are *equal* | Removed RoPE on B and C entirely | **stays green** — whole `layers/ssm` and `tests/parity` green | **FILED** |
| V5 | `TestArchitectureBuilderNonNil` | `tests/parity/arch_registry_test.go:79` | Self-passing gate: loop draws its cases from the registry under test; zero rows = pass | `inference.ListArchitectures` returns nil | **stays green** | **FIXED** (F3) |
| V6 | `TestGroupedQueryAttention_ScaleRope` | `layers/attention/grouped_query_attention_extended_test.go:79` | Self-passing gate: only asserts `err == nil` | `ScaleRope` becomes `return nil` | **stays green** — nothing in `layers/attention` caught it | **FIXED** (F4) |
| V7 | `TestGlobalAttention_ScaleRope` | `layers/attention/global_attention_extended_test.go:99` | as V6 | as V6 | **stays green** | **FIXED** (F4) |
| V8 | `TestScaledDotProductAttention_ForwardWithMask` | `layers/attention/scaled_dot_product_attention_extended_test.go:13` | Self-passing gate: builds a real causal mask, then asserts only shape + NaN | `Forward` silently discards the caller's `mask` | **stays green** (4 unrelated error-injection tests in the package did trip, incidentally) | **FIXED** (F5) |
| V9 | `TestParity_TransformerBlock` / `TestParity_TransformerBlock_Structural` | `tests/parity/layer_parity_test.go:1246`, `:2671` | Parity-named test delegating to a shape+NaN structural body ("golden file not yet implemented") | Removed the FFN sub-layer and its residual from `transformer.Block.Forward` | **stays green** | **FILED** |
| V10 | `TestTSMixerBlock_ChannelMixing_DiffersFromIndependent` | `layers/timeseries/tsmixer_test.go:75` | Saturation/degeneracy: compares two *independently random-initialised* blocks, so they differ regardless | Disabled the channel-mixing MLP even when requested | **stays green** | **FILED** |
| V11 | `TestMatMulNBits_Asymmetric`, `TestMatMulNBits_GlobalScale` | `layers/core/remaining_coverage_test.go:636`, `:667` | Self-passing gate: asserts `out != nil` and a config echo, never the dequantized values | Zero point ignored in dequant arithmetic (config left intact) | **stays green** — but `TestMatMulNBitsDequantization` **did** fail, so the capability is still covered | **FILED** (low priority) |

### Worst one: V1 — the GQA backward gradient check

`gradClose` (`layers/attention/gqa_backward_test.go:321`) computes

```go
denom := float32(math.Max(1.0, math.Max(math.Abs(float64(a)), math.Abs(float64(b)))))
return diff/denom < tol
```

Flooring the denominator at `1.0` turns the stated *relative* tolerance into an
*absolute* one for every gradient element smaller than 1.0 — which, given the
fixture's input scaling, is all of them. With `tol = 1e-2`, any analytic gradient
under 0.01 passes against a numerical value of exactly zero.

Red-proof: dropping the **entire K path** from GQA's input gradient — a severe,
real backward-pass bug — was caught by **nothing**: not `TestGQABackward_MHA`,
not `TestGQABackward_WeightGradients`, not `tests/parity`, not
`layers/transformer`. `TestGQABackward` itself, the only test covering the
grouped case (`numQ != numKV`), is `t.Skip`ped at `:16` with a known
analytical-vs-numerical mismatch, so the head-replication path — the one actually
suspected of being wrong — has **no live gradient coverage at all**.

**Claim that rests on it:** correctness of the backward pass for every
GQA-based architecture, i.e. all training and fine-tuning on Llama/Mistral/Qwen/
Gemma-class models. The same floored-denominator construction recurs in
`layers/attention/sdpa_backward_finitediff_test.go:163` (`tol=5e-2`),
`layers/attention/mla_backward_test.go:140`, and `layers/ssm/verify_learn_test.go:230`,
where the constant is literally named `relTol` while behaving absolutely.

Filed rather than fixed because choosing the right bound requires a ruling on the
intended gradient-check contract (per-element relative with a small absolute
floor, and what floor) — exactly the kind of judgement this audit was told not to
guess at.

## CONFIRMED HEALTHY

Verified by breaking the code they cover and observing a real failure. Recording
these is half the value: it says what has actually been checked.

| Test | Break applied | Result |
| --- | --- | --- |
| `TestParity_SDPA_Causal` | Causal masking disabled in `SetCausal` | **FAILS** — 24/32 values exceed tol, maxDiff 2.27 |
| `TestParity_RotaryEmbedding` | RoPE angle-table position order reversed | **FAILS** — maxAbsDiff 3.93 vs tol 1e-5 |
| `TestRotaryPositionalEmbedding_Forward` | same | **FAILS** |
| `TestRotaryPositionalEmbedding_Backward` | same | **FAILS** |
| `TestRotaryPositionalEmbedding_NewRotaryPositionalEmbedding` | same | **FAILS** |
| `TestDocumentWiseRoPE_PositionReset` | same | **FAILS** |
| `TestDocumentWiseRoPE_PositionResetAtBoundaries` | same | **FAILS** |
| `TestDocumentWiseRoPE_SingleDocumentMatchesStandard` | same | **FAILS** |
| `TestMatMulNBitsDequantization` | Zero point ignored in dequantization | **FAILS** |
| `TestSDPA_FusedScaledSoftmax_SkippedWithMask` | Caller mask discarded | **FAILS** |
| `TestSDPA_Forward_MaskReshapeError` / `_MaskAddError` / `_MaskReshapeBackError` | same | **FAIL** |
| `TestBlock_Forward_Errors` | FFN sub-layer removed | **FAILS** |
| `TestMultiHeadLatentAttention_PartialRoPE_*` (config assertion) | `ropeHeadDim` forced to `headDim` | **FAILS** — catches a *config* regression, but not the behavioural one (see V3) |
| 78 of 79 golden parity comparisons | tolerance measured against signal magnitude | tolerance ≤ 0.26% of signal in every case |

`layers/attention/gqa_rope_position_parity_test.go` (`TestGQA_PrefillDecodeRoPEPositionParity`,
the L-0010 replacement) deserves explicit mention as the **positive model**: it
pins its weights and asserts its own sensitivity before asserting parity, citing
L-0009. It is the template the fixes below copy.

### One probe that did NOT prove anything, and why

An early sweep reversed the global RoPE angle table and found that only three
tests in `layers/embeddings` failed — every test in `layers/attention` stayed
green, including the L-0010 parity test. **This is not evidence of vacuity.** A
consistent global relabeling of positions is invisible to a *relative*-position
test by construction, and RoPE is relative — L-0010 says so directly ("that shift
is UNIFORM within one pass and RoPE is relative, so a pure prefill and a pure
token-at-a-time decode are each internally consistent"). Recorded here so the
next auditor does not re-derive it and file a false positive.

## Fixes in this PR

**F1 — `AssertClose` gains a sensitivity control** (`tests/parity/testutil/assertions.go`).
New `AssertSensitive` fails when `tolerance >= max|expected|`, i.e. when an
all-zero output would satisfy the comparison, and when the expected slice is
empty. Called from `AssertClose`, so **every** golden-driven parity test now
carries the L-0009 sensitivity control automatically. Run against the suite as
written, it flagged exactly one comparison (`mamba_block_forward`) and no others —
the fix validates itself.

**F2 — `tests/golden/layers/ssm_mamba.json` tolerance 1e-3 → 1e-8.** Sized to the
measured error: `maxAbsDiff` is 9.095e-12, so 1e-8 leaves ~1100x headroom while
sitting 200x below the *smallest* expected magnitude (8.3e-7). Re-red-proofed:
with `MambaBlock.Forward` zeroed, the test now fails 32/32 values.

**F3 — `TestArchitectureBuilderNonNil` guards against an empty registry.**
Re-red-proofed: now fails when `ListArchitectures` returns nil.

**F4 — `TestGroupedQueryAttention_ScaleRope` / `TestGlobalAttention_ScaleRope`
assert the angle table actually scaled.** Both capture `rope.GetAngles` before and
after and check every element against `prev * factor`. Re-red-proofed: both now
fail against a no-op `ScaleRope`. (`TestGroupedQueryAttention_NoRoPE_ScaleRope`
correctly still passes — a no-op *is* the right behaviour when `rope == nil`.)

**F5 — `TestScaledDotProductAttention_ForwardWithMask` compares masked against
unmasked output.** Re-red-proofed: now fails when `Forward` discards the mask.

All fixes are RED-on-break and GREEN-on-correct-code. Full CPU suite
(`./layers/... ./tests/parity/... ./inference/...`, 36 packages) passes.

## Filed, not fixed

Each needs a ruling on intended behaviour rather than a guess:

1. **V1 `gradClose` and its three siblings** — what is the intended gradient-check
   contract? Per-element relative with an absolute floor is the standard shape,
   but the floor value is a judgement call. Also: `TestGQABackward` is skipped
   with a known mismatch, leaving the grouped-head backward path uncovered. That
   skip is a live correctness gap, not just a test-quality one.
2. **V3 MLA partial RoPE** — the test needs to assert that dims past
   `ropeHeadDim` are genuinely position-invariant (same input at two positions →
   identical values in the passthrough half). Requires confirming that is the
   intended contract.
3. **V4 `TestComplexSSMState_RoPEEffect`** — the doc comment promises a
   RoPE-on vs RoPE-off comparison; the body compares a call to itself. Needs a
   way to construct the block without B/C rotation, which does not exist today.
4. **V9 `TestParity_TransformerBlock`** — a structural test wearing a parity
   name. Either generate the golden file or rename it so a green run does not
   read as numeric parity. Naming is the maintainers' call. Note the broader
   pattern: ~20 `TestParity_*_Structural` tests assert only shape + finiteness.
   They are honest at the `_Structural` suffix, but `TestParity_TransformerBlock`
   and `TestParity_AttnRes` alias them under parity names.
5. **V10 TSMixer / V11 MatMulNBits** — both need fixtures that share weights and
   vary only the feature under test. Mechanical, but the correct fixture depends
   on what the test was meant to prove.

## Reachability caveat (not vacuity, but adjacent)

Of 218 tests in `tests/parity`, **53 SKIP** without `ZERFOO_MODELS_DIR` — every
per-architecture generation and forward-pass row (Gemma, Llama, Mistral, Phi,
Qwen, DeepSeek, Falcon, Mixtral, RWKV, Command-R, LLaVA, Qwen-VL, SigLIP). A
default `go test ./tests/parity/` reports `ok` having exercised no model at all.
This is by design post-#987 and the identity gate in `scripts/dgx-validate-inpod.sh`
guards the CI path, but it means the package-level `ok` is not evidence for any
model-support claim. Flagged for UC-H2-013 ("public claims never exceed verified
evidence").

## Not reached this session

- `generate/` and `inference/` were swept only incidentally (both were included
  in the break-blast-radius runs, and `generate/` caught none of the RoPE or GQA
  breaks — but no test in either package was individually red-proofed).
- The ~20 `TestParity_*_Structural` tests were characterised but only
  `TransformerBlock` and `AttentionHead` were red-proofed individually.
- GPU-gated parity (`gpu_parity_ops_test.go`, `gqa_gpu_parity_test.go`,
  `mamba3_parity_test.go` CUDA rows) — deliberately untouched, CPU-only session.
- The 16 `t.Skipf("Forward failed: %v", err)` sites in `layers/*/comprehensive_test.go`
  turn a broken Forward into a skip rather than a failure. Mechanical to convert
  in bulk; not done here to keep this PR reviewable.
