# Development Log

Investigation findings, debugging sessions, and benchmark results.
Entries are newest-first. Prune entries older than 90 days during /trim.

---

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
