# Verified-Model Matrix

**This file is the single source of truth for Zerfoo's public model-support
claims. Marketing may not exceed this file: no public claim about model
support may go beyond what this matrix evidences** (ADR-093 rule 1 —
verification is the gate for claims).

A model earns a row only when its support is backed by a link to a real repo
artifact — a parity test in `tests/parity/`, a benchmark row in
`docs/benchmarks.md`, and/or a reproduction manifest in
`docs/bench/manifests/`. A row without evidence links is **not permitted** as a
support claim; such rows are listed below explicitly as candidates that are
**not yet verified**, with empty evidence cells, so the gap is visible rather
than hidden.

Honesty over optics: several candidate rows below are pending. Do not cite them
as supported until their evidence cells are populated. T136.3 (GB10 parity
re-runs, blocked on the T136.2 model provisioning) and T136.4 (benchmark
re-runs with reproduction manifests) will populate and refresh the evidence
here; until then the "Status" column is authoritative.

## Status legend

| Status | Meaning |
|--------|---------|
| `verified` | Has at least one existing, checked evidence link (parity test that exists and/or a `benchmarks.md` row). GPU-verified only if the GPU-Verified column says yes. |
| `pending` | Architecture is registered in Zerfoo (`inference.ListArchitectures()`), but no parity/benchmark evidence has been recorded yet. Evidence cells intentionally empty until T136.3/T136.4. |
| `arch-unconfirmed` | Zerfoo has no GGUF-registered architecture builder for this model (it is absent from `inference.ListArchitectures()`), so gate check 1 fails. Kept in the candidate list per the plan, but must not be claimed as supported. |

## Matrix

| Model | Architecture (Zerfoo builder) | GGUF source (public) | Quant | Parity evidence | Benchmark evidence | GPU-verified | Status |
|-------|------------------------------|----------------------|-------|-----------------|--------------------|--------------|--------|
| Gemma 3 1B | `gemma3` (`gemma`/`gemma3` alias) | `ggml-org/gemma-3-1b-it-GGUF` (file `gemma-3-1b-it-Q4_K_M.gguf`, per `docs/bench/manifests/gemma3-tps.yaml`) | Q4_K_M | `TestGemma3ForwardPass` / `TestGemma3GreedyDecode` / `TestGemma3Generation` (`tests/parity/gemma3_test.go`) — require `GEMMA3_MODEL_DIR`; GB10 re-run pending T136.3 | 241 tok/s (1.28x Ollama 188), `docs/benchmarks.md` "Throughput vs Ollama" row, 2026-03-31, v1.38.4; reproduction manifest `docs/bench/manifests/gemma3-tps.yaml` | Yes (2026-03-31) | verified |
| Gemma 3 4B | `gemma3` (same builder as 1B) | `ggml-org/gemma-3-4b-it-GGUF` (exact file pinned at provisioning, T136.2) | Q4_K_M (intended) | Same `TestGemma3*` harness, parameterized by `GEMMA3_MODEL_DIR`; not yet run against a 4B GGUF | — (pending T136.4) | No | pending |
| Llama 3.2 3B | `llama` | `bartowski/Llama-3.2-3B-Instruct-GGUF` (exact file pinned at provisioning, T136.2) | Q4_K_M | `TestLlama3ForwardPass` / `TestLlama3GreedyDecode` / `TestLlama3Generation` (`tests/parity/llama3_test.go`) — require `LLAMA3_MODEL_DIR`; GB10 re-run pending T136.3 | 92 tok/s (0.99x Ollama 93), `docs/benchmarks.md` "Throughput vs Ollama" row, 2026-03-30, v1.38.4; per-model reproduction manifest pending T136.4 | Yes (2026-03-30) | verified |
| Llama 4 (Scout, smallest runnable) | `llama4` | `unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF` (exact file pinned at provisioning, T136.2) | Q4_K_M (intended) | `TestLlama4ForwardPass` / `TestLlama4GreedyDecode` / `TestLlama4Generation` (`tests/parity/llama4_test.go`) — require `LLAMA4_MODEL_DIR`; not yet run | — (pending T136.4) | No | pending |
| Mistral 7B | `mistral` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (exact file pinned at provisioning, T136.2) | Q4_K_M | `TestMistralForwardPass` / `TestMistralGreedyDecode` / `TestMistralGeneration` (`tests/parity/mistral_test.go`) — require `MISTRAL_MODEL_DIR`; GB10 re-run pending T136.3 | 44 tok/s (1.00x Ollama 44), `docs/benchmarks.md` "Throughput vs Ollama" row, 2026-03-30, v1.38.4; per-model reproduction manifest pending T136.4 | Yes (2026-03-30) | verified |
| Qwen 2 | `qwen2` | `Qwen/Qwen2-7B-Instruct-GGUF` (exact file pinned at provisioning, T136.2) | Q4_K_M (intended) | `TestQwen25ForwardPass` / `TestQwen25GreedyDecode` / `TestQwen25Generation` (`tests/parity/qwen_test.go`, `qwen2` builder; test fixtures target Qwen2.5) — require `QWEN25_MODEL_DIR`; not yet run against a Qwen 2 GGUF | — (pending T136.4) | No | pending |
| Phi-4 | `phi` / `phi3` alias | `microsoft/phi-4-gguf` (exact file pinned at provisioning, T136.2) | Q4_K_M (intended) | `TestPhi4ForwardPass` / `TestPhi4GreedyDecode` / `TestPhi4Generation` (`tests/parity/phi4_test.go`) — require `PHI4_MODEL_DIR`; not yet run | — (pending T136.4) | No | pending |
| DeepSeek-R1-Distill 1.5B | `qwen2` (the 1.5B distill is a Qwen2-architecture model; see note below) | `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` (exact file pinned at provisioning, T136.2) | Q4_K_M | No dedicated parity test; `tests/parity/deepseek_test.go` (`TestDeepSeekV3*`) targets the `deepseek_v3` architecture, **not** this Qwen2 distill. Parity via the `qwen2` path pending T136.3 | 186 tok/s (1.11x Ollama 168), `docs/benchmarks.md` "Throughput vs Ollama" row ("DeepSeek-R1 1.5B"), 2026-03-30, v1.38.4; per-model reproduction manifest pending T136.4 | Yes (2026-03-30) | verified |
| MiniMax-M2 229B (CPU / over-RAM) | MoE; loaded via the GGUF path on CPU (over-RAM mmap). Builder not in the `arch_registry_test` expected list — see note below | MiniMaxAI MiniMax-M2 GGUF (229B MoE, 3 shards, 128.8 GB Q4_K_M; exact repo/file pinned at provisioning, T136.2) | Q4_K_M | No GPU parity test (CPU-only over-RAM path); load + decode demonstrated in `docs/devlog.md` 2026-03-29 entry ("MiniMax-M2 229B inference verified on 128 GB DGX Spark") | 0.06 tok/s CPU over-RAM, `docs/benchmarks.md` "Over-RAM Inference" row, 2026-03-29 (NVMe-bound; Ollama failed to load with a 500 error on same hardware) | No (CPU-only; no GPU over-RAM path yet) | verified |
| Chronos-2 (timeseries, non-LLM) | `BuildChronos` T5 graph builder exists (`inference/timeseries/arch_chronos.go`, with a weight converter in `convert_chronos.go`) but is not registered in the GGUF architecture registry (`inference.ListArchitectures()`) — gate check 1 fails | Amazon `amazon/chronos-2` (not currently distributed as GGUF; conversion path unconfirmed) | — | Structural/shape tests only (`timeseries/chronos_test.go`, `inference/timeseries/arch_chronos_test.go`); no numeric parity test in `tests/parity/` | — | No | arch-unconfirmed |

### Notes

- **DeepSeek-R1-Distill 1.5B architecture.** DeepSeek-R1-Distill-Qwen-1.5B is a
  Qwen2-architecture distillation, so it loads through Zerfoo's `qwen2` builder,
  not the `deepseek_v3` builder that `tests/parity/deepseek_test.go` exercises.
  The `benchmarks.md` generate/throughput row for "DeepSeek-R1 1.5B" is genuine
  and cited above; a dedicated parity assertion via the `qwen2` path is a T136.3
  follow-up.
- **MiniMax-M2 architecture.** MiniMax-M2 loads and decodes on the CPU over-RAM
  path (evidence: `benchmarks.md` Over-RAM row + `devlog.md` 2026-03-29). Its
  builder is not among the architectures asserted in
  `tests/parity/arch_registry_test.go`; the row is evidenced by the recorded
  load+generate run, not by the architecture-registry test. GPU acceleration for
  the over-RAM path does not exist yet, so GPU-verified is `No`.
- **Chronos-2 (the required non-LLM entry).** Zerfoo carries a Chronos-2 T5
  graph builder (`inference/timeseries.BuildChronos`) and a weight converter,
  with structural/shape tests — but the architecture is not registered in the
  GGUF registry (`inference.ListArchitectures()` does not include it), Chronos-2
  is not distributed as GGUF, and no numeric parity test exists in
  `tests/parity/`, so no GGUF-path evidence can exist today. Per the plan's
  deviation rule this candidate is kept but marked `arch-unconfirmed`; it must
  not be claimed as supported. Registering the builder + a GGUF conversion path
  would be the route to a claimable row. **The framework does
  have real non-LLM (timeseries) parity coverage** — e.g. `TestParity_PatchTST`,
  `TestParity_DLinear`, `TestParity_NBEATS`, `TestParity_ITransformer`,
  `TestParity_TFT_Structural`, `TestParity_CfC`, `TestParity_FreTS`,
  `TestParity_TimeMixer` in `tests/parity/model_parity_test.go`. If a benchmarked,
  GGUF-loadable non-LLM row is required for the general-purpose support claim, one
  of these architectures (with an existing parity test) is the honest candidate to
  promote, not Chronos-2.

## Gate criteria — what a model must pass to earn a `verified` row

A model row may claim support only when **every** applicable check below is
satisfied and linked from its evidence cells. This mirrors the universal quality
gates in `CLAUDE.md` ("Quality gates are universal") and ADR-093 rule 1
(verification gates claims).

1. **Architecture is registered.** The model's GGUF `general.architecture` maps
   to a builder returned by `inference.ListArchitectures()` (see
   `tests/parity/arch_registry_test.go`). If not, the row is `arch-unconfirmed`
   and cannot be claimed.
2. **GGUF source is public and exact.** The GGUF-source cell names a public
   repository and the specific file (and, at provisioning, the pinned revision).
   The file is staged on the DGX host under `/var/lib/zerfoo/models` by T136.2
   before any GB10 run.
3. **Parity evidence exists and is named.** A parity test in `tests/parity/`
   (e.g. the `Test<Model>ForwardPass` / `GreedyDecode` / `Generation` trio)
   names the model and, when run against the staged GGUF, passes. The evidence
   cell records the exact test name(s), the run date, and a run reference.
   Structural-only parity (e.g. the `*_Structural` timeseries tests) must be
   labeled as such — it is weaker than a full numeric parity assertion.
4. **GPU verification via the standing gate.** For a `GPU-verified: Yes` claim,
   the model's parity subset must pass under the standing GPU gate
   `scripts/dgx-validate.sh` on the GB10 (which builds natively in a Spark pod,
   runs the cuda-tagged tests + the mounted parity subset via
   `docs/bench/manifests/validate-arm64.yaml`, and exits on a JSON report). The
   parity stage only runs for models whose GGUF is mounted at the manifest's
   `MODELS_DIR` (`/var/lib/zerfoo/models`); the mount's `hostPath` must
   pre-exist (Spark/Podman fails on a missing `hostPath`). This is T136.3.
5. **Benchmark evidence is reproducible.** Any throughput figure cites a
   `docs/benchmarks.md` row (with tok/s, quant, and date) **and** a reproduction
   manifest under `docs/bench/manifests/` (e.g. `gemma3-tps.yaml` recertifies the
   241 tok/s Gemma 3 1B figure). Rows whose benchmark predates the manifest
   convention are marked "per-model reproduction manifest pending T136.4" until
   T136.4 adds one.
6. **CPU-only rows are labeled.** Over-RAM / CPU-only support (e.g. MiniMax-M2)
   is a valid `verified` row but must carry `GPU-verified: No` and identify the
   path, since no GPU claim is evidenced.

A row that cannot satisfy checks 1–3 for a full (non-structural) assertion, or
whose benchmark number lacks a `benchmarks.md` source, is not a support claim —
it stays `pending` or `arch-unconfirmed` with empty evidence cells.

## Candidate summary (10 seeded rows)

| Candidate | Current status | Basis |
|-----------|----------------|-------|
| Gemma 3 1B | verified (GPU) | `benchmarks.md` 241 tok/s row (2026-03-31) + `gemma3-tps.yaml` + `TestGemma3*` |
| Gemma 3 4B | pending | `gemma3` builder registered; no evidence yet |
| Llama 3.2 3B | verified (GPU) | `benchmarks.md` 92 tok/s row (2026-03-30) + `TestLlama3*` |
| Llama 4 (Scout) | pending | `llama4` builder registered; `TestLlama4*` present but not run |
| Mistral 7B | verified (GPU) | `benchmarks.md` 44 tok/s row (2026-03-30) + `TestMistral*` |
| Qwen 2 | pending | `qwen2` builder registered; `TestQwen25*` fixtures target Qwen2.5 |
| Phi-4 | pending | `phi`/`phi3` builder registered; `TestPhi4*` present but not run |
| DeepSeek-R1-Distill 1.5B | verified (GPU) | `benchmarks.md` 186 tok/s row (2026-03-30); loads via `qwen2` builder |
| MiniMax-M2 229B (CPU) | verified (CPU) | `benchmarks.md` Over-RAM row + `devlog.md` 2026-03-29 |
| Chronos-2 (timeseries) | arch-unconfirmed | `BuildChronos` exists but is not GGUF-registered; kept per plan, must not be claimed |

## Known documentation discrepancy

`docs/benchmarks.md` cites "Full results: `results/benchmark-2026-03-31.json`"
for the Throughput-vs-Ollama table, but that file does not exist in `results/`
(only `benchmark-2026-03-25.json` and `benchmark-2026-03-27.json` are present).
The tok/s figures above are therefore cited to the `benchmarks.md` rows
themselves and, for Gemma 3 1B, to the `gemma3-tps.yaml` reproduction manifest —
**not** to the missing JSON. T136.4 should either restore the referenced results
file or correct the `benchmarks.md` reference.
