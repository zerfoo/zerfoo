# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.1](https://github.com/zerfoo/zerfoo/compare/v1.4.0...v1.4.1) (2026-03-17)


### Bug Fixes

* **normalization:** add nil guard in RMSNorm.Backward before cached tensor access ([f956329](https://github.com/zerfoo/zerfoo/commit/f9563296969614e1376a156b7f262478cecb15f6))

## [1.4.0](https://github.com/zerfoo/zerfoo/compare/v1.3.0...v1.4.0) (2026-03-17)


### Features

* **api:** add zerfoo.Load high-level inference API ([b3f804b](https://github.com/zerfoo/zerfoo/commit/b3f804b6db651d81a10ba36bb9f3e9b3a8ef8ee4))
* **api:** implement Model.ChatStream for streaming generation ([c6d6fcd](https://github.com/zerfoo/zerfoo/commit/c6d6fcd356e45efaf0a8042de242ac751d970bdf))
* **api:** implement Model.Embed with mean-pooling ([9d4cce7](https://github.com/zerfoo/zerfoo/commit/9d4cce7b1fbc9012a59dcea118024ae306f30ae4))
* **cli:** add --help flag support to CLI framework ([8fab982](https://github.com/zerfoo/zerfoo/commit/8fab9824107cd7048244cb26f607df7eed4ee216))
* **cli:** add --json-schema flag for structured output in run command ([2a4ed26](https://github.com/zerfoo/zerfoo/commit/2a4ed261c05313dcf904917357ee00cb69ebec80))
* **cli:** add download progress indicator for pull command ([9db7cbd](https://github.com/zerfoo/zerfoo/commit/9db7cbdee706d8e7d1f2a37d6edb739971e9c426))
* **cli:** add model loading progress indicator ([118af89](https://github.com/zerfoo/zerfoo/commit/118af89122b3e9766fb12bdebe9d15c4ba929782))
* **cli:** add version command ([f357b57](https://github.com/zerfoo/zerfoo/commit/f357b573beced2bb66c171275aa0e1b9a00ecaa7))
* **cli:** support --flag=value syntax in argument parser ([8e3caf9](https://github.com/zerfoo/zerfoo/commit/8e3caf9b68bedd2b535b5c2303a644bcd34789d4))
* **cmd:** add --quant flag to zerfoo pull command ([3cfd62d](https://github.com/zerfoo/zerfoo/commit/3cfd62d1c610c237c63901c89a461f70ec946fb3))
* **cmd:** add zerfoo list command ([03239f7](https://github.com/zerfoo/zerfoo/commit/03239f7018f9bde1c756e6ff9757333a9486e857))
* **cmd:** add zerfoo rm command ([4693513](https://github.com/zerfoo/zerfoo/commit/469351392bb902799352c1ab162a7329b96d22a7))
* **examples:** add streaming chat example ([6ed52f1](https://github.com/zerfoo/zerfoo/commit/6ed52f15d09eec3aefa3828822a3d22777287223))
* **examples:** chatbot CLI example ([edf9151](https://github.com/zerfoo/zerfoo/commit/edf9151d297970aefae72f9016155cfb6d9e039b))
* **examples:** RAG retrieval-augmented generation demo ([d1131c7](https://github.com/zerfoo/zerfoo/commit/d1131c71072065e909c857891db976f792279e76))
* **examples:** structured JSON output with grammar-guided decoding ([4cfe91e](https://github.com/zerfoo/zerfoo/commit/4cfe91e8abde6cf45ff3d5b38793e912f56a6707))
* **generate/grammar:** implement JSON Schema to CFG state machine ([32f5b52](https://github.com/zerfoo/zerfoo/commit/32f5b52506d4e3907da18b347e57890e5e9483cd))
* **generate/grammar:** implement token mask from CFG state ([32a3ed5](https://github.com/zerfoo/zerfoo/commit/32a3ed5a179866bf4c5883728d8836f4a16a9205))
* **generate:** add InferenceSession struct for per-request inference isolation ([a85434e](https://github.com/zerfoo/zerfoo/commit/a85434e33e70c37575848ba5771af8df555b3f4a))
* **generate:** add shared graph mutex and GenerateStream to InferenceSession ([8989dc6](https://github.com/zerfoo/zerfoo/commit/8989dc67e0b97e1e22f00d97d746f366d8f14d71))
* **generate:** add token mask and grammar advancement for constrained decoding ([85c56af](https://github.com/zerfoo/zerfoo/commit/85c56afd473f7839083011d08954061f71375e56))
* **generate:** cache PoolResetter at session creation ([43d8d9a](https://github.com/zerfoo/zerfoo/commit/43d8d9a89b76ddd10fca01dcdbca74955122ed31))
* **generate:** cache PoolResetter at session creation (T1.3) ([e02266e](https://github.com/zerfoo/zerfoo/commit/e02266eaa91ee39275ae596c471218d2b869b873))
* **generate:** cache stopSet and pre-allocate generatedIDs ([2873fac](https://github.com/zerfoo/zerfoo/commit/2873facbe973e2693c4ea546b1abb764b7bfb1ab))
* **generate:** cache stopSet and pre-allocate generatedIDs (T1.4) ([1bec8fa](https://github.com/zerfoo/zerfoo/commit/1bec8fa5195d5a782a12b386198a5f224c2a5297))
* **generate:** implement per-session KV cache and Generate method ([b4bd691](https://github.com/zerfoo/zerfoo/commit/b4bd691957a97c6fc8adac8e3e7236dde6845334))
* **generate:** thread grammar state through sampling config and generation loop ([945208b](https://github.com/zerfoo/zerfoo/commit/945208bf7e5c552a36cdf11909f8a99188fc8503))
* **gguf:** add attn_qkv tensor name mapping for Phi merged QKV ([6a8be73](https://github.com/zerfoo/zerfoo/commit/6a8be73d072e731432bc8967dd1decb9db84cf88))
* **gguf:** implement merged gate+up MLP tensor split for Phi ([02250c8](https://github.com/zerfoo/zerfoo/commit/02250c84fbbd6d28d14f2ed4365ed0095b2dc1d2))
* **gguf:** implement QKV tensor split for Phi merged attn_qkv ([73a6eba](https://github.com/zerfoo/zerfoo/commit/73a6eba84394c9d6f9c4cb0444b9578773e9c843))
* **inference:** add buildMistralGraph with sliding window attention ([1f4e5dd](https://github.com/zerfoo/zerfoo/commit/1f4e5dd4ecbe32577d37eefd69cdad93f4295866))
* **inference:** add buildPhiGraph with partial rotary embedding ([192b9b6](https://github.com/zerfoo/zerfoo/commit/192b9b6c6264a7716fe818358916cad685aa2a0f))
* **inference:** add buildQwenGraph with attention bias support ([d437816](https://github.com/zerfoo/zerfoo/commit/d437816be5603feaef20d5045ad4fa6afe902d90))
* **inference:** add WithGrammar generation option ([1ce5600](https://github.com/zerfoo/zerfoo/commit/1ce560012ece98283981658b77e73a4936014c24))
* **inference:** GenerateBatch for multi-sequence generation ([81df874](https://github.com/zerfoo/zerfoo/commit/81df874035d8d2bf53408093da61c291af105e16))
* **inference:** wire per-request sessions into Model.Generate for concurrent inference ([2587480](https://github.com/zerfoo/zerfoo/commit/25874806418b5c0b611f35c2fd01029e1ac5f42f))
* **model/huggingface:** add HuggingFace API client ([db7631f](https://github.com/zerfoo/zerfoo/commit/db7631fb38a8886d5484e0b9daf978038fa0ee1c))
* **model/huggingface:** implement cache manifest and management ([d164e0d](https://github.com/zerfoo/zerfoo/commit/d164e0d629039b8c9dedb5a047b6c0c37d1f6729))
* **model/huggingface:** implement download with resume and progress ([466c22a](https://github.com/zerfoo/zerfoo/commit/466c22a231ff3478b61df3e8741bae1ffad63b68))
* **moe:** implement batched expert execution for DeepSeek V3 ([017fd20](https://github.com/zerfoo/zerfoo/commit/017fd20e78a3c9b246bbd87d1718c8c542cba884))
* **serve:** add response_format json_schema support ([9756aa2](https://github.com/zerfoo/zerfoo/commit/9756aa23d0891bb6f82107e045e12b0e28d683a9))
* **serve:** add tool definition parsing to chat completions API ([6e51788](https://github.com/zerfoo/zerfoo/commit/6e51788cac11ab62222cbfadb40e0ff42280e8f4))
* **serve:** implement forced tool_choice for function calling ([59c28e7](https://github.com/zerfoo/zerfoo/commit/59c28e7df0575cb22be0ee876ebf500bc2789eab))
* **serve:** implement tool call detection and response formatting ([c55f693](https://github.com/zerfoo/zerfoo/commit/c55f693f1e0b2d907b756c35a9f704b3e99f3d3d))
* **serve:** wire BatchScheduler to GenerateBatch for concurrent requests ([b084ea7](https://github.com/zerfoo/zerfoo/commit/b084ea739f798491d346e4ab1bc2d6382234ee98))


### Bug Fixes

* **attention:** apply partial RoPE to rope_head_dim subset in MLA ([a216259](https://github.com/zerfoo/zerfoo/commit/a216259f54564415ad331e8051d5fe6ea6ff43b5))
* **attention:** pre-allocate FlashAttentionDecode buffer, remove debug logging ([9918041](https://github.com/zerfoo/zerfoo/commit/99180414093113ad6e1aa1625b46d34f2e1c4b51))
* **attention:** pre-allocate FlashAttentionDecode output buffer for CUDA graph capture ([7d6657b](https://github.com/zerfoo/zerfoo/commit/7d6657b8ccd3ebe40f60d8a0aba2be5a0f7ff78d))
* **attention:** revert FlashAttentionDecode buffer pre-allocation ([6d2052c](https://github.com/zerfoo/zerfoo/commit/6d2052cdedfc3266849633e887a450f1aaf22b74))
* **cli:** sort commands alphabetically in help output ([727d1de](https://github.com/zerfoo/zerfoo/commit/727d1de7857782a120afc599ca2362fbc5e23494))
* **cli:** wire --system flag to Model.Chat in run command ([1062caf](https://github.com/zerfoo/zerfoo/commit/1062caf23b98720b77461a6d9aafea9c0dd3d617))
* **deps:** use correct ztensor with PreUploadFrozenWeights restored ([97cbd60](https://github.com/zerfoo/zerfoo/commit/97cbd607e68c7d83e210d350424d90599c7eabf6))
* **deps:** ztensor with correct cuBLAS SGEMM path ([eb9a4d2](https://github.com/zerfoo/zerfoo/commit/eb9a4d20a17571791418cc8fc7a1ab5b369fd8d9))
* **generate:** add grammar-constrained decoding to InferenceSession ([00ce9e7](https://github.com/zerfoo/zerfoo/commit/00ce9e7e4bbe20368a113ccc174ff3b8e6189b43))
* **generate:** eliminate data race in Generator by serializing Generate calls ([59e16a3](https://github.com/zerfoo/zerfoo/commit/59e16a3a910e0e053f7112c8ed6b26f4701b8ec3))
* **generate:** only reset stateful nodes on prefill, not every decode step ([0d90ee1](https://github.com/zerfoo/zerfoo/commit/0d90ee16f329614193334c0f5133b91c272aaf46))
* **gguf:** dequantize Q4_K and Q5_0 directly to float32 (no lossy Q4_0 step) ([a53fa08](https://github.com/zerfoo/zerfoo/commit/a53fa08c0a8418d90791345f13ce8655492c434c))
* **gguf:** enable byte-level BPE for gpt2 tokenizer model type ([696fa84](https://github.com/zerfoo/zerfoo/commit/696fa84fb48f4e6a9825b2a23706597e0ed4663a))
* **gguf:** Q5_K native float32 dequant, remove lossy Q4_0 re-quantization ([a3147fd](https://github.com/zerfoo/zerfoo/commit/a3147fd5b401ee12402a6b0be03f7ca0494f89f8))
* **gguf:** Q5_K native float32 dequant, remove lossy Q4_0 re-quantization ([343bad2](https://github.com/zerfoo/zerfoo/commit/343bad217745fafa0de0374fad11a8376c933e2d))
* **gguf:** Q6_K native float32 dequant, remove lossy Q4_0 re-quantization ([420b157](https://github.com/zerfoo/zerfoo/commit/420b1573aa085ee600840e960168b11a99f2b670))
* **gguf:** remove duplicate attn_qkv.weight key in tensorNameMap ([ccf2a58](https://github.com/zerfoo/zerfoo/commit/ccf2a58715c73d710852c5b887c1568e9bed6f57))
* **gguf:** revert Q5_0 to Q4_0 re-quantization (perf), keep Q4_K native ([5a2fb87](https://github.com/zerfoo/zerfoo/commit/5a2fb8795ddbf2220c3806ec05e22a4c9069967e))
* **gguf:** stop lossy re-quantization of Q4_K and Q5_0 tensors ([7ca225d](https://github.com/zerfoo/zerfoo/commit/7ca225df8435557e5bdf3873d36105c70bebef15))
* **grammar:** reject trailing comma in object when all properties emitted ([07a2ade](https://github.com/zerfoo/zerfoo/commit/07a2ade6e637c5e66c09383d88ceef1a8b0fe29a))
* **grammar:** vet/lint clean ([fd57670](https://github.com/zerfoo/zerfoo/commit/fd57670cc5e95f94c76bb179a949bde5a54fd7bc))
* **inference:** add FP8 E4M3 transpose handling to CPU path ([72c9df4](https://github.com/zerfoo/zerfoo/commit/72c9df49058ad735c7eb604abd9f5e57ee259b9b))
* **inference:** correct FP16 GQA tensor storage size mismatch ([78c9a5f](https://github.com/zerfoo/zerfoo/commit/78c9a5fc10a276b5f17692aab3860f5f3a1822bb))
* **inference:** detect Mistral architecture from sliding window metadata ([e9ea11b](https://github.com/zerfoo/zerfoo/commit/e9ea11b16e508bda0723b3d1f72c92fc2fcb621a))
* **inference:** eliminate D2H transfer in GQA to enable CUDA graph capture ([c39ca9f](https://github.com/zerfoo/zerfoo/commit/c39ca9faa45fda3f071a8365f2f98b404c571b57))
* **inference:** update Embed test assertions to match new error messages ([b4081ad](https://github.com/zerfoo/zerfoo/commit/b4081adf216d80be05f01d2f25dc85e8150f82b5))
* **lint:** update test model loader registration from zmf to gguf ([99a7b72](https://github.com/zerfoo/zerfoo/commit/99a7b72ca3ed3658de3b47d579e8016882d70aa0))
* remove debug env vars, document model file issue ([70364be](https://github.com/zerfoo/zerfoo/commit/70364bedc1c6cd965bf4b5c46c1bafc9a9de52b5))
* restore stable 188 tok/s baseline ([7b674e7](https://github.com/zerfoo/zerfoo/commit/7b674e7e58a56fade52aa2ca89edf0c267e47d26))


### Performance Improvements

* **attention:** disable FlashAttentionDecode, use SDPA for decode path ([d0fe532](https://github.com/zerfoo/zerfoo/commit/d0fe532ce4a4f206a8df3f0a2f1f6715bdcfdcbe))
* **deps:** ztensor Phase 6-compatible paths ([e30d56f](https://github.com/zerfoo/zerfoo/commit/e30d56f26b2cfb134923665fb24e36322afb108f))
* **deps:** ztensor with clean Q4 GEMV path ([cc3c27b](https://github.com/zerfoo/zerfoo/commit/cc3c27bfd478f67b40d29385cd17fc1413b19c62))
* **deps:** ztensor with EnsureCaptureInputsGPU Q4 fix ([476bf53](https://github.com/zerfoo/zerfoo/commit/476bf53fabda3acf78d076a7b3db4d04491910a0))
* **deps:** ztensor with Q4 GEMV path restored ([2ec6c59](https://github.com/zerfoo/zerfoo/commit/2ec6c595eef474c444048e5510f9ae1d9a96139d))
* **generate:** add ResetPool and GPU argmax to InferenceSession decode ([6fb4da5](https://github.com/zerfoo/zerfoo/commit/6fb4da55ad55b70241280389406f52266d13c0e3))
* **generate:** hold graph mutex for entire generation, not per-step ([9ef4789](https://github.com/zerfoo/zerfoo/commit/9ef47892c973adc28223c5f2757c51eb8e40ae0e))
* **generate:** wire CUDA graph execution plan into InferenceSession ([3237c15](https://github.com/zerfoo/zerfoo/commit/3237c15cc524038565c2a69fa48b3b5a81aa5b30))
* **gguf:** dequantize Q5_0/Q4_K/Q5_K/Q6_K to FP16 for tensor core GEMM ([ec014a0](https://github.com/zerfoo/zerfoo/commit/ec014a0724036e2b20bdf44c298684261b2fc3d1))
* **gguf:** re-quantize Q5_0 and Q4_K to Q8_0 instead of Q4_0 ([e206d9e](https://github.com/zerfoo/zerfoo/commit/e206d9e3500a40766aee1337e3835ea6871096f0))
* **gguf:** restore Q4_0 re-quantization for Q5_K and Q6_K weights ([8717a12](https://github.com/zerfoo/zerfoo/commit/8717a125a1c248c0d47f04ef94189dca0055b3f4))
* **gguf:** revert Q4_K to Q4_0 re-quantization (20% faster) ([d26a3bf](https://github.com/zerfoo/zerfoo/commit/d26a3bf5ed94378eee2181d6989c7338b3128d03))
* **gguf:** use BFloat16 dequant for Q5_0/Q4_K/Q5_K/Q6_K weights ([84851f2](https://github.com/zerfoo/zerfoo/commit/84851f2640bb60b00cdf66fa71c51cd930708142))
* **gguf:** use native Q5_K and Q5_0 storage (117+0 tensors) ([b14b22a](https://github.com/zerfoo/zerfoo/commit/b14b22a64f751d166374db63258432fd5a3980ff))
* **gguf:** use native Q6_K storage (13 tensors, incremental) ([9fbdf6e](https://github.com/zerfoo/zerfoo/commit/9fbdf6e91dc8c43c8941c36d2e8e336a89b8b86f))
* **gguf:** use native quantized storage for Q4_K, Q5_K, Q6_K, Q5_0 ([a84a855](https://github.com/zerfoo/zerfoo/commit/a84a8557da2dfceedb8c78e53d3bffbf29f90920))
* **gguf:** use native quantized storage for Q4_K, Q5_K, Q6_K, Q5_0 ([ce4a761](https://github.com/zerfoo/zerfoo/commit/ce4a7613ae9f9cd37c0cb3567e39080db197a742))
* **inference:** add session pool to preserve GPU addresses for CUDA graph replay ([1524a5f](https://github.com/zerfoo/zerfoo/commit/1524a5fca45a1d573195c74596f8593e2f090db3))
* **inference:** use Generator directly for maximum throughput ([5a09695](https://github.com/zerfoo/zerfoo/commit/5a0969534cd0e3b591f9d718c3aa68c8edef8241))
* Q4_0 CPU + BF16 GPU upload for 2x bandwidth reduction ([65fd4b0](https://github.com/zerfoo/zerfoo/commit/65fd4b00735068ee1aeead9d036b75a78f338bc2))


### Reverts

* **gguf:** restore Q4_0 re-quant baseline (170 tok/s) ([4dda2b8](https://github.com/zerfoo/zerfoo/commit/4dda2b8bed9142841d10b803220c94fc7318f08b))
* **gguf:** restore Q4_0 re-quantization pending kernel optimization ([8ad132a](https://github.com/zerfoo/zerfoo/commit/8ad132a53c74985e5c5f4e6fb1e02c1525c9f10f))
* restore Q4_0 + F32 GPU baseline (170 tok/s) ([cad04e3](https://github.com/zerfoo/zerfoo/commit/cad04e358c6015ff6985ab2f0508f8a11b09f4c5))

## [Unreleased]

## [0.2.1] - 2026-03-16

### Added

- `inference.GenerateBatch` for multi-sequence parallel generation
- `serve.BatchScheduler` wired to `GenerateBatch` for concurrent request handling
- Q5_K and Q6_K native float32 dequantization (replaces lossy Q4_0 re-quantization)
- Release-please CI pipeline (`.github/workflows/release-please.yml`)
- README Quick Start guide, badges, and feature examples

### Fixed

- `go vet` and `golangci-lint` clean across all packages (ptrFromUintptr, errcheck, gosec)

## [0.2.0] - 2026-03-16

### Added

- `zerfoo.Load(pathOrID)` one-line model loader supporting local GGUF paths and HuggingFace model IDs (e.g. `"google/gemma-3-4b"`)
- `Model.Chat(prompt)` single-turn text generation
- `Model.Generate(ctx, prompt, opts...)` generation with options (max tokens, temperature, top-p)
- `Model.Embed(texts)` text embedding extraction with L2 normalization
- `Model.ChatStream(ctx, prompt)` streaming token generation via channel
- `Model.Close()` resource cleanup
- `WithSchema(schema)` grammar-guided JSON output constrained to a JSON Schema
- `WithTools(tools...)` / `WithToolChoice(choice)` tool/function calling via chat completions API
- `zerfoo pull` CLI command to download GGUF models from HuggingFace with resume, SHA256 verification, and progress display
- `zerfoo list` CLI command to list cached models
- `zerfoo rm` CLI command to remove a cached model
- `response_format: json_schema` support in OpenAI-compatible chat completions API
- GGUF graph builders for Mistral (sliding window attention), Qwen 2 (attention bias, RoPE theta=1M), Phi 3/4 (partial rotary), and DeepSeek V3/V2-Lite (MLA + MoE) bringing total supported architectures to six
- Example applications: `examples/chat/` (chatbot), `examples/rag/` (retrieval-augmented generation), `examples/json-output/` (structured output)

### Fixed

- FP16 inference: GQA tensor storage mismatch causing incorrect outputs
- FP8 inference: same GQA root cause in E4M3FN path
- CUDA graph capture: eliminated device-to-host transfer in GQA decode that prevented graph closure
- Concurrent generation race condition in `Generator` serialized with mutex
- Q5_K_M and Q6_K weights: removed lossy re-quantization to Q4_0; now dequantize accurately to float32

### Changed

- `zerfoo.Load` now accepts HuggingFace model IDs in addition to local file paths
- All exported symbols in the top-level `zerfoo` package now have stability markers (`Stable.` / `Experimental.`)
