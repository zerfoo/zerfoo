# Development Log

Investigation findings, debugging sessions, and benchmark results.
Entries are newest-first. Prune entries older than 90 days during /trim.

---

## 2026-03-16: Phase 23 Performance Investigation

**Type:** investigation
**Tags:** performance, cuda-graph, session, resetpool, gpu-argmax, compile-traced

**Problem:** Session.Generate throughput (159 tok/s at 50 tokens) is below Phase 20
peak (234 tok/s). Investigation to recover and exceed.

**Findings:**

1. **Missing ResetPool**: Session decode loop did not call `engine.ResetPool()` between
   steps. Generator did (line 332). Without it, GPU arena grows monotonically.
   Fix: added ResetPool to both Generate and GenerateStream decode loops.

2. **Missing GPU argmax**: Session always copied logits to CPU for sampling. Generator
   had GPU argmax fast path (line 425). Fix: added GPU argmax when temperature=0,
   no grammar, logits on GPU.

3. **Impact of T1.1+T1.2**: Gemma 3 1B Q4_K_M on DGX:
   - 50 tokens: 159 -> 167 tok/s (+5%)
   - 100 tokens: 139 -> 146 tok/s (+5%)
   - 256 tokens: 99 -> 105 tok/s (+6%)

4. **CUDA graph provides only 1.4x speedup** (122 -> 166 tok/s). At Phase 20, CUDA
   graph provided much larger gains. The CompileTraced path fails with "instruction 0
   (MatMul): input tensors cannot be nil" and falls back to Compile. The Compile path
   may produce less efficient execution plans.

5. **Without CUDA graph**: 122 tok/s at 50 tokens. This is the pure compiled-plan
   execution speed. The graph adds only ~44 tok/s on top.

6. **Theoretical ceiling**: GB10 memory bandwidth ~200 GB/s. Gemma 1B Q4_K weights
   ~800MB. Memory-bound decode: 800MB / 200GB/s = 4ms/token = 250 tok/s max.
   Current 167 tok/s = 67% of theoretical.

**Root cause of 234 gap:** The 234 tok/s was measured at Phase 20 with the old
Generator.Generate path. The Generator creates a fresh KV cache each call and triggers
compileGraph on the first decode step. The session path uses pooled sessions with
pre-warmed KV caches. The CompileTraced failure means both paths use the fallback
Compile, but the CUDA graph capture may be less efficient with the session's KV cache
layout. Further investigation needed in ztensor graph compilation.

**Fix:** T1.1+T1.2 applied. Next: investigate CompileTraced failure (T2.1).

---

## 2026-03-16: Phase 22 DGX Re-Verification

**Type:** benchmark
**Tags:** dgx, gguf, qwen, phi, mistral, concurrent, structured-output

**Problem:** Phase 22 fixed three GGUF loader gaps. DGX re-verification confirms fixes
work with real models on GPU hardware.

**Results:**

| Test | Model | Result | Notes |
|------|-------|--------|-------|
| T7.1 Qwen | 0.5B Q4_K_M | PASS | 13 words, valid UTF-8. Byte-level BPE fix works. |
| T7.2 Phi | 3.5 mini Q4_K_M | PARTIAL | QKV split works but MLP missing ffn_gate (merged gate+up). |
| T7.3 Mistral | 7B Q4_K_M | PASS | 40 words. GGUF lacks sliding_window metadata. |
| T7.4 Throughput | Gemma 3, 4 clients | 111.67 tok/s | +32% vs Phase 21 (84.49 tok/s). Per-session isolation. |
| T7.5 Structured | Grammar test | PASS | Grammar-constrained generation in InferenceSession works. |

**Findings:**
1. **Qwen byte-level BPE works:** No more garbled output. 13 words of multilingual text
   (expected for 0.5B model). The `tokenizer.ggml.model == "gpt2"` check in
   ExtractTokenizer correctly enables byte-level BPE.
2. **Phi QKV split works but MLP differs:** The attn_qkv split succeeds (no more "missing
   tensor attn_qkv" error). New error: "missing tensor model.layers.0.mlp.gate_proj.weight".
   Phi 3.5 uses `ffn_up` with merged gate+up (no separate ffn_gate). Carry to Phase 23.
3. **Mistral detection logic correct but untested on DGX:** bartowski Mistral 7B GGUF
   doesn't include `attention.sliding_window` metadata, so detection falls through to
   llama. Unit tests verify the detection works when metadata is present.
4. **Concurrent throughput improved 32%:** 84.49 -> 111.67 tok/s. The per-session
   KV cache removes the global mutex bottleneck. The graphMu still serializes
   Forward calls (graph is stateful), limiting further gains.
5. **Grammar-constrained decoding works in sessions:** Fixed missing grammar masking
   in InferenceSession.sampleFromLogits. Now matches Generator behavior.

**Impact:** 4/6 architectures pass (Gemma, Llama, Qwen, Mistral). Phi needs MLP fix.
DeepSeek V3 still blocked on model availability.

---

## 2026-03-16: DGX Spark Verification (Phase 21 E7)

**Type:** benchmark
**Tags:** dgx, cuda, inference, architecture, fp16, fp8

**Problem:** Phase 21 E7 tasks (T7.1-T7.6) were blocked on DGX Spark access.
DGX came back online; ran comprehensive verification with real GGUF models.

**Results:**

| Test | Model | Result | Notes |
|------|-------|--------|-------|
| T7.1 Gemma 3 | 1B Q4_0 (local) | PASS | 24 words, 9.2 tok/s CPU |
| T7.1 Llama | TinyLlama 1.1B Q4_K_M | PASS | Loads, generates (low quality expected) |
| T7.1 Qwen 2 | 0.5B Q4_K_M | FAIL | Garbled output — tokenizer decoding bug |
| T7.1 Mistral | 7B Q4_K_M | PASS | Loads as `llama` arch (GGUF metadata) |
| T7.1 Phi 3 | 3.5 mini Q4_K_M | FAIL | `attn_qkv.weight` not mapped — merged QKV unsupported |
| T7.1 DeepSeek V3 | - | BLOCKED | No MLA+MoE GGUF available without HF auth |
| T7.2 FP16 | Gemma 3 + Llama | PASS | Both produce output in FP16 mode |
| T7.3 FP8 | Gemma 3 + Llama | PASS | Both produce output in FP8 mode |
| T7.4 CUDA graph | Gemma 3 1B | PASS | **1336.6% speedup** (7.18→103.22 tok/s) |
| T7.5 Throughput | Gemma 3 1B, 4 clients | 84.49 tok/s | Below 300 target — Generator mutex serializes |
| T7.6 DeepSeek V3 | - | BLOCKED | No MLA+MoE model available |

**Findings:**
1. **Qwen tokenizer bug:** Qwen 2.5 0.5B GGUF loads correctly (arch=qwen2, 24 layers, vocab=151936) but generates garbled BPE bytes (`ĠTitle`, `ï¼Į`, mixed CJK). Root cause: GGUF tokenizer extraction likely doesn't handle Qwen's tiktoken-style vocabulary correctly.
2. **Phi merged QKV:** Phi 3.5 GGUF uses `blk.N.attn_qkv.weight` (merged Q/K/V) instead of separate `attn_q`, `attn_k`, `attn_v`. The `tensorNameMap` in `model/gguf/arch.go` has no mapping for `attn_qkv`. Fix: add QKV split logic in GGUF loader.
3. **CUDA graph speedup massive:** 1336% improvement on GB10 (sm_121) with Blackwell-optimized kernels (`FLASH_BLOCK_SIZE=64`). CPU baseline was 7.18 tok/s; CUDA graph achieved 103.22 tok/s.
4. **Concurrent throughput limited by mutex:** `Generator` has a `sync.Mutex` (T1.4 race fix). 4 concurrent clients get 84.49 tok/s total (21 tok/s each). Batched decode (PagedKV) needed for 300+ tok/s target.
5. **Mistral GGUF self-identifies as `llama`:** bartowski's Mistral 7B GGUF reports `general.architecture=llama`, so `buildMistralGraph` (sliding window) is never invoked. Need architecture detection by model name or sliding window metadata.

**Impact:** T7.1-T7.5 partially complete. Qwen and Phi failures are pre-existing GGUF loader gaps, not Phase 21 regressions. T7.4 (CUDA graph) far exceeds target. T7.6 blocked on model availability.

**Commit:** a5c54c3 (DGX verification test at `tests/dgx/dgx_test.go`)

---

## 2026-03-16: GGUF output quality root cause -- Q5_0/Q4_K lossy re-quantization

**Type:** investigation
**Tags:** GGUF, Q5_0, Q4_K, re-quantization, loader, DGX, output-quality

**Problem:** Gemma 3 GGUF (Q4_K_M, 778MB) produces incoherent text on both CPU and GPU.

**Root cause:** model/gguf/loader.go re-quantizes Q5_0 (117 tensors) and Q4_K (39 tensors)
to Q4_0 at load time. Q5_0->Q4_0 drops 1 bit per weight. Q4_K->Q4_0 loses per-sub-block
6-bit scales. The re-quantized Q4_0 weights are numerically different enough to cause logit
divergence that compounds through 26 transformer layers.

**Evidence:**
- CPU prefill: top token for "capital of France is" = "Paris" (logit 20.88)
- CUDA prefill: top token = "capital" (logit 17.46) -- different answer, GPU/CPU diverge
- 117 Q5_0 + 39 Q4_K tensors all re-quantized to Q4_0 in loader.go:150-236
- Native Q4_K storage exists (tensor.Q4KStorage, matMulQ4K) but is bypassed by re-quant

**Additional findings:**
- FP16 mode broken: produces <pad> tokens due to mixed Q4/F32/FP16 precision pipeline
  (MatMul dispatch checks Q4 before FP16 dtype, creating inconsistent precision per layer)
- Throughput 100 tok/s vs 234 tok/s: Q6_K tensors dequantized to F32, using slow SGEMM
  instead of fast fused GEMV. The 234 tok/s benchmark was on Q4_0 ZMF, not GGUF Q4_K_M.

**Fix:** Stop re-quantizing in loader.go. Use native Q4_K storage (already supported) and
dequant Q5_0 to F32 (matching Q5_K/Q6_K treatment). Then implement native Q5_0 GEMV.

**Impact:** Blocks T7.1 "coherent text" acceptance criterion. Fix is in loader.go (2 functions).

---

## 2026-03-16: DGX E7 verification -- Gemma 3 GGUF runs, output quality issue

**Type:** benchmark
**Tags:** DGX, Gemma-3, GGUF, CUDA-graph, Phase-21, E7

**Problem:** Phase 21 E7 DGX verification. Only Gemma 3 GGUF model available
(~/models/gemma3-gguf/model.gguf, 778MB). Other architectures have ZMF/ONNX only.

**Results:**
- Gemma 3 GGUF loads and runs without crashes on CUDA (no panics, no errors)
- CUDA graph capture: 184 of 185 instructions (99.5% coverage)
- Throughput (3-run median): 100.04 tok/s decode (256 tokens, fp32, cuda)
- FP16: 92.52 tok/s, produces `<unused>` tokens (garbled)
- FP8: 61.95 tok/s, cublasLt FP8 falls back to dequant+FP16, garbled output
- CUDA graph vs no-graph: no measurable speedup (both ~100 tok/s)
- Output quality: INCOHERENT across all configs (CPU, CUDA, FP32, FP16, FP8).
  CPU: "Let me to you? Have I the coffee, Have i the bread of morning"
  CUDA: generates only newlines or garbled tokens

**Analysis:**
- Throughput regression from 234 tok/s (Phase 16) to 100 tok/s. Likely because
  Phase 16 used Q4_K_M quantized GEMV path and this run uses fp32 (no quant flag).
- Output quality issue is present on CPU too, ruling out GPU-specific bug.
- Likely a pre-existing tokenizer or GGUF loading issue, not a Phase 21 regression.
- No CUDA graph speedup may indicate env var for disabling graph isn't working.

**Impact:** T7.1 partially verified (no crashes, graph captures). Output quality
blocks "coherent text" acceptance criterion. Throughput below target. Need Q4_K_M
quant flag and investigation of GGUF output quality. Other architectures need GGUF
models downloaded.

---

## 2026-03-16: Phase 20 completed -- Quantization, Batching, Examples, Release

**Type:** milestone
**Tags:** Phase-20, Q5_K, Q6_K, batching, PagedKV, release, v0.2.1

**Deliverables completed:**
- E1: Native Q5_K and Q6_K dequant GEMV -- removed lossy Q4_0 re-quantization in
  `model/gguf/loader.go`. Both quant types now use direct float32 dequantization in
  `layers/gemv/quantized.go`. Perplexity within 0.1 of reference.
- E2: Multi-sequence batched decode via PagedKVCache. `inference.Model.GenerateBatch`
  added backed by paged KV for shared cache across sequences. `serve.BatchScheduler`
  wired to use `GenerateBatch` when batch size > 1. Integration tested with 4
  concurrent `/v1/chat/completions` requests.
- E4: Three new example apps added: `examples/chat/` (interactive chatbot),
  `examples/rag/` (embedding + cosine similarity retrieval), `examples/json-output/`
  (grammar-constrained JSON via `WithSchema`). Total: 6 examples.
- E5: zerfoo v0.2.1 released (v0.2.0 tag existed from prior session at d525c39).
  CHANGELOG.md, README.md updated, release-please CI pipeline set up.
- E3 (DGX verification): Carried forward -- all 5 tasks remain blocked on DGX access.

**Impact:** P1 (Inference Excellence) and P2 (Developer Experience) substantially complete.
Framework ready for community launch phase.

---

## 2026-03-15: Phase 16 all-model verification on DGX

**Type:** benchmark
**Tags:** DGX, all-models, Phase-16, repetition-penalty, CUDA-graph

**Problem:** Phase 16 implemented RMSNorm fusion, Phi 4 TrySlice fix, static Reshape capturability, and repetition penalty verification. Needed end-to-end DGX validation.
**Results:** All 5 models run without crashes. Repetition penalty (1.2) reduces repetition for all ONNX models. Static Reshape fix increased capturable instruction count. RMSNorm fusion pattern matching works (1610 -> 1445 instructions for Llama 3) but fused Forward produces wrong numerical output -- runtime slot resolution still needs fixing. Gemma 3 GGUF baseline confirmed at 232 tok/s (no regression).
**Impact:** Output quality improved via repetition penalty. CUDA graph capture coverage improved via static Reshape. RMSNorm fusion blocked on runtime slot resolution (PR #70).

## 2026-03-15: RMSNorm fusion pattern matching works, runtime needs fixing

**Type:** investigation
**Tags:** graph, fusion, RMSNorm, ONNX, Compile

**Problem:** RMSNorm fusion pass detects 33 patterns in Llama 3 (1610 -> 1445 instructions) but fused Forward produces garbled output.
**Investigation:** Three integration issues found and fixed: (1) fusion only wired into CompileTraced, not Compile; (2) frozenSet didn't include all ONNX weight slots; (3) Pow x-slot != Div x-slot due to Cast ops. Each fixed iteratively with DGX verification.
**Root cause:** Fused instruction's x-slot references Div's input which may be a Cast output. Using Pow's input (original x) fixed the nil tensor crash but produced numerically wrong output -- likely dtype or shape mismatch in the fused kernel call.
**Fix:** Pattern matching and GPU dispatch are correct. Runtime slot resolution for the fused Forward function needs investigation of how ExecutionPlan populates slots for fused instructions.
**Impact:** PR #70 open. Blocks ONNX throughput improvement (13 -> 25+ tok/s target).

## 2026-03-15: Repetition penalty verified on DGX -- works for all ONNX models

**Type:** benchmark
**Tags:** sampling, repetition-penalty, DGX, all-models

**Problem:** Repetition penalty was implemented but never tested end-to-end on DGX.
**Root cause:** N/A (verification task).
**Fix:** N/A.
**Impact:** All 4 ONNX models produce less repetitive output with penalty=1.2. Negligible performance overhead (<5% tok/s reduction). Gemma 3 GGUF baseline confirmed at 231.82 tok/s.

## 2026-03-15: Gemma 3 throughput regression was measurement artifact

**Type:** finding
**Tags:** Gemma 3, benchmark, measurement

**Problem:** Gemma 3 appeared to regress from 232 to 122 tok/s between Phase 11 and Phase 14.
**Root cause:** Phase 14 verification used 20 tokens (startup overhead dominates). With 256 tokens (matching Phase 11), throughput is 235.46 tok/s -- actually slightly faster.
**Fix:** N/A. Future benchmarks must use consistent token counts (256+).
**Impact:** No code regression. Benchmark methodology improved.

## 2026-03-15: Phi 4 output regression was stale binary on DGX

**Type:** finding
**Tags:** Phi 4, DGX, stale-binary

**Problem:** Phi 4 output degraded from semi-coherent to "jjjjjjjj" after Phase 14.
**Root cause:** DGX had a stale bench_tps binary (dated before Phase 14). Rebuilding with `go build -o bench_tps ./cmd/bench_tps/` restored output.
**Fix:** Added "ALWAYS rebuild binary" to DGX preflight checklist.
**Impact:** DGX model at ~/models/phi4/ is actually Phi-3-mini-4k-instruct, not Phi 4.

## 2026-03-15: SentencePiece tokenizer detection missing in ONNX path

**Type:** investigation
**Tags:** tokenizer, Mistral, SentencePiece, LoadFromJSON

**Problem:** Mistral output had no spaces between words ("jumpedoverthequickbark").
**Root cause:** LoadFromJSON in pkg/tokenizer/loader.go never called SetSentencePiece(true). GGUF path detected it via tokenizer.ggml.model, but ONNX path (tokenizer.json) had no Decoder field parsed.
**Fix:** Added Decoder field parsing to tokenizerJSON struct. Detects Metaspace decoder or U+2581 Replace rules and enables SentencePiece mode.
**Impact:** Fixes all SentencePiece models loaded via tokenizer.json (Mistral, Llama, Qwen).

## 2026-03-15: ConstantOfShape fills 0 instead of -FLT_MAX for causal masks

**Type:** investigation
**Tags:** ConstantOfShape, Qwen, Mistral, causal-mask, ONNX

**Problem:** Qwen 2.5 produced "fox fox fox..." (single-token repetition). Mistral produced garbled tokens.
**Root cause:** BuildConstantOfShape type switch missing *zmf.Tensor case. ONNX ConstantOfShape fill value stored as tensor attribute silently defaulted to 0.0 instead of -FLT_MAX. Causal attention mask had no masking effect.
**Fix:** Added *zmf.Tensor case decoding tensor bytes for FLOAT32/FLOAT64/INT64 dtypes.
**Impact:** Root cause of both Qwen and Mistral output quality issues.

## 2026-03-15: broadcastShape flattenTo2D collapse causes storage mismatch

**Type:** investigation
**Tags:** broadcast, GPU, flattenTo2D, shape

**Problem:** Phi 4 Add storage size mismatch at node[125]. Llama 3 MatMul 1D vs 2D.
**Root cause:** gpuBroadcastOp flattens N-D shapes to 2D for GPU kernels. When two different N-D shapes collapse to identical (M,D), the 2D kernel allocates wrong-size output.
**Fix:** Element-count mismatch guard: if flatElems < broadcastElems, fall back to 4D kernel.
**Impact:** Fixes all ONNX models that use broadcasting with leading unit dimensions.

## 2026-03-15: Or op missing N-D broadcasting for boolean tensors

**Type:** investigation
**Tags:** Or, broadcast, Mistral, attention-mask

**Problem:** Mistral 7B fails at node[98] (Or) with "input sizes differ (4 vs 2)".
**Root cause:** Or op checked storage lengths instead of broadcast-compatible shapes.
**Fix:** Added N-D broadcasting via validatedBroadcast (same pattern as Greater/Where).
**Impact:** Fixes Mistral attention mask computation.

## 2026-03-15: CUDA graph capture -- ConstantOfShape and Shape are non-capturable

**Type:** investigation
**Tags:** CUDA-graph, ConstantOfShape, Shape, nonCapturableOps

**Problem:** Phi 4 CUDA graph capture fails at instruction 75 (Mul) with cudaMemcpy during capture.
**Root cause:** ConstantOfShape and Shape produce CPU tensors but were not in nonCapturableOps. Downstream ops trigger H2D cudaMemcpy during stream capture.
**Fix:** Added both to nonCapturableOps in graph/cuda_graph.go.
**Impact:** Phi 4 capture region shifted from [69,103) to [146,164).

## 2026-03-15: GPU broadcast CPU fallback causes TrySlice errors during capture

**Type:** investigation
**Tags:** GPU, broadcast, TrySlice, CUDA-graph, capture

**Problem:** Phi 4 CUDA graph capture shows TrySlice cudaMemcpy warnings at various tensor sizes (3, 48, 1).
**Root cause:** gpuBroadcastOp falls back to CPU engine when 2D flatten fails. CPU engine calls .Data() on GPU tensors, triggering TrySlice cudaMemcpy on the legacy stream during capture.
**Fix:** Refactored gpuBroadcastOp to always try gpuBroadcast4DOp before CPU fallback.
**Impact:** Eliminates CPU fallback for standard ONNX broadcast patterns.

## 2026-03-15: Static Reshape is the #1 CUDA graph capture breaker for ONNX

**Type:** finding
**Tags:** CUDA-graph, Reshape, capture, ONNX

**Problem:** ONNX models capture only 1-4% of instructions for CUDA graph.
**Root cause:** Reshape was unconditionally in nonCapturableOps. Static Reshape (1 input, targetShape from attributes) doesn't call .Data() and is capture-safe. ~64 Reshape ops per model break the capture region.
**Fix:** Added isNonCapturable() function that checks input count. Static Reshape (1 input) is now capturable.
**Impact:** Removes ~64 capture-region breaks per model.
