# Changelog

## [1.0.1](https://github.com/zerfoo/zerfoo/compare/v1.0.0...v1.0.1) (2026-03-14)


### Bug Fixes

* **ci:** add zmf checkout to release workflow for GoReleaser ([c01cd0c](https://github.com/zerfoo/zerfoo/commit/c01cd0c8d1ad2a09a9b2873ed8784decc0f9f810))

## 1.0.0 (2026-03-14)


### Features

* **attention:** add decode fast path using GPU-resident KV length ([7b63abd](https://github.com/zerfoo/zerfoo/commit/7b63abd42447a37b004d947502b5e6e2318b5cb2))
* **attention:** add Q/K normalization support for Gemma 3 ([b3a1e8e](https://github.com/zerfoo/zerfoo/commit/b3a1e8e220dbff71630107f58a458de1fd7e2686))
* **attention:** fuse QK RMSNorm + RoPE into single kernel launch ([42f4008](https://github.com/zerfoo/zerfoo/commit/42f4008d1807db3f40cf59e1157be545756bfcf5))
* **attention:** pass GQA head counts to flash decode kernel ([1512408](https://github.com/zerfoo/zerfoo/commit/1512408e306a394c199b4a871024fdd3a450f6f1))
* **bench:** add --kv-dtype flag for FP16 KV cache benchmarking ([fa3fc85](https://github.com/zerfoo/zerfoo/commit/fa3fc85cbb588b247c3f5b6bbdaa8e4f913058a7))
* **bench:** report arena pool stats in bench_tps ([144f44f](https://github.com/zerfoo/zerfoo/commit/144f44f03270b8aca720ceb749c01defbb18b54f))
* **cmd/bench_tps:** add fp8 to dtype flag help text ([cda27e1](https://github.com/zerfoo/zerfoo/commit/cda27e192f60c60fe3e3c2f77a3d236f6205afff))
* **cmd:** add debug-infer tool for testing GGUF inference ([05ee6fd](https://github.com/zerfoo/zerfoo/commit/05ee6fd3bdefd4634fff0993c6ba295aa2938a30))
* **compute:** accept Float16Storage in element-wise ops ([6492b16](https://github.com/zerfoo/zerfoo/commit/6492b168922dc79a57b20e72b8e167df74e6bfda))
* **compute:** accept Float16Storage in MatMul ([7c23aca](https://github.com/zerfoo/zerfoo/commit/7c23aca28cfc61fd8ea2f15b9cd4cd45be238e88))
* **compute:** add &gt;4D CPU fallback for GPU Transpose ([da4357e](https://github.com/zerfoo/zerfoo/commit/da4357e1e9cb717bd6d42ba2d6a72805780b623c))
* **compute:** add batch broadcasting to CPU MatMul ([78d38cc](https://github.com/zerfoo/zerfoo/commit/78d38cc1af58dffaddef7fe485332b280b814aa2))
* **compute:** add DTypeFP8 constant and wire FP16 paths for FP8 ([402fe51](https://github.com/zerfoo/zerfoo/commit/402fe5165d9dc65df39d544ca8b05e11dad7606f))
* **compute:** add FP16 native path to FusedRMSNormGPU ([ccde777](https://github.com/zerfoo/zerfoo/commit/ccde777fe852b983423d35f0e1354b54df9e1c0d))
* **compute:** add FP16ToF32Converter interface and GPUEngine implementation ([cdc8264](https://github.com/zerfoo/zerfoo/commit/cdc8264ce36058879a24c67a5cb82b80c7333ab6))
* **compute:** add FP8 scale factor diagnostic logging ([10417cd](https://github.com/zerfoo/zerfoo/commit/10417cd3e157d02ef4939c25bb8060b860052ae9))
* **compute:** add full FP16 inference path ([dedef6c](https://github.com/zerfoo/zerfoo/commit/dedef6c42c89a6ded46638e81a9fbcbfd61a55b3))
* **compute:** add fused GPU operations ([45c1c72](https://github.com/zerfoo/zerfoo/commit/45c1c72a3d4835c67004ccfa9dff3abf7a502b8c))
* **compute:** add fused SwiGLU GPU operation ([c88d5f8](https://github.com/zerfoo/zerfoo/commit/c88d5f87a9fa44000d6b9d8d6cd8af09101222cd))
* **compute:** add GPUArgmaxer interface and GPUEngine implementation ([9bea19a](https://github.com/zerfoo/zerfoo/commit/9bea19a27240844348ffb720962119afceb5d75c))
* **compute:** add PoolResetter support to EngineProxy ([3d2f15e](https://github.com/zerfoo/zerfoo/commit/3d2f15ef78572522a78c0f2269698e3fa6bdd1dd))
* **compute:** add Q4 B-side weight GEMV for GPU MatMul ([2b63ec1](https://github.com/zerfoo/zerfoo/commit/2b63ec106bbf68ba0c964867e52b8de4c15a13e1))
* **compute:** add StreamProvider interface for CUDA graph integration ([b3eebc9](https://github.com/zerfoo/zerfoo/commit/b3eebc98e483d129453c92738a8d2511f5be5b04))
* **compute:** add tests for Float16Storage MatMul ([1f5c91c](https://github.com/zerfoo/zerfoo/commit/1f5c91cd3fe42a686d4b0ad79acd1f6bcddbe054))
* **compute:** add ZERFOO_DISABLE_MANAGED_MEM env var for benchmarking ([cc0a1e8](https://github.com/zerfoo/zerfoo/commit/cc0a1e83e46852e643b29bd832f7a318f872b719))
* **compute:** convert embedding output to FP16 at inference start ([282b7b8](https://github.com/zerfoo/zerfoo/commit/282b7b8ab025d7d0ff39d0545697dc56d9293158))
* **compute:** eliminate D2H copy in GPUEngine.Gather ([3aa0010](https://github.com/zerfoo/zerfoo/commit/3aa001013d084317b534d3245000bef157f39286))
* **compute:** eliminate D2H copy in GPUEngine.Gather ([0750c4e](https://github.com/zerfoo/zerfoo/commit/0750c4e8a0e3885a82ed4dd647fb11f0460b8cf4))
* **compute:** implement GPU-native Repeat to eliminate D2H roundtrips ([b797073](https://github.com/zerfoo/zerfoo/commit/b797073b2ec2f31e13482e4841a54065c8302a15))
* **compute:** pre-allocate FP8 MatMul scratch buffers ([90ed2fd](https://github.com/zerfoo/zerfoo/commit/90ed2fdff8e144ee0a625c28efad21b6c7606bec))
* **compute:** pre-convert weights to FP16 at upload time ([be2c478](https://github.com/zerfoo/zerfoo/commit/be2c4784da4e491d96f1bae94413cbadb26f5c41))
* **compute:** upload Q8 weights to GPU by dequantizing to F32 ([48230b2](https://github.com/zerfoo/zerfoo/commit/48230b2bbbb5ff95c4fd8b4d557a79f9482af917))
* **compute:** use managed memory for weight uploads on GB10 ([10573eb](https://github.com/zerfoo/zerfoo/commit/10573ebfa07691a8ba31c77cac1bc6ebeb8160d3))
* **compute:** use managed memory for weight uploads on GB10 ([764aa6e](https://github.com/zerfoo/zerfoo/commit/764aa6e6f280e50973e0811391296d813e1cbb48))
* **compute:** wire 4D broadcast kernels into GPUEngine binary ops ([5fe5166](https://github.com/zerfoo/zerfoo/commit/5fe5166755fc29a26ae8026d2c59ed5ebd5a0a8c))
* **compute:** wire BFloat16 MatMul through cublasGemmEx ([471d7e0](https://github.com/zerfoo/zerfoo/commit/471d7e0d660d6f0534f9f91c7ad85071c69a00e2))
* **compute:** wire FP8 MatMul through cublasLtMatmul ([3a303d0](https://github.com/zerfoo/zerfoo/commit/3a303d0329ff457165e2fb1eb50d5d641eed447f))
* **compute:** wire fused dequant+GEMV Q4_K into GPUEngine MatMul ([79c6884](https://github.com/zerfoo/zerfoo/commit/79c688449f023fe615be2b40ba1e3189d296f9c8))
* **cublas:** add CudaR8F_E4M3 data type constant ([a707200](https://github.com/zerfoo/zerfoo/commit/a70720033567fdb475f7f09e784457b1117292f5))
* **cublas:** add purego wrappers for cublasLt API ([6158507](https://github.com/zerfoo/zerfoo/commit/6158507a3940e2ca8fdabc4f08106563be382a8c))
* **cublas:** add SgemmNT to purego path for MatMulTransposeB on linux arm64 ([bb5e5fd](https://github.com/zerfoo/zerfoo/commit/bb5e5fdda14b567659f05c12fe9ecdb3f32365b7))
* **cublas:** add SgemmNTStridedBatched for single-call batched A*B^T ([3512821](https://github.com/zerfoo/zerfoo/commit/3512821aca4d34acc136ca36705d367014276d03))
* **cublas:** add SgemmNTStridedBatched to purego path ([2682450](https://github.com/zerfoo/zerfoo/commit/268245035b9a24197498274fd161a1002f66d8aa))
* **cublas:** implement cublasGemmEx purego wrapper ([8e8ba8f](https://github.com/zerfoo/zerfoo/commit/8e8ba8f41a9db039abb4ee3e386464c6660dd04d))
* **cublas:** replace CGo bindings with purego implementation ([058c252](https://github.com/zerfoo/zerfoo/commit/058c252761184adf01e903c94f25e30bacfcb58d))
* **cuda:** add CUDA graph capture and replay API wrappers ([ac6b72d](https://github.com/zerfoo/zerfoo/commit/ac6b72d2ea7e0dc3fd0b1317ca4c8da2869cf657))
* **cuda:** add custom sgemv_m1 kernel for M=1 decode GEMV ([c2d6835](https://github.com/zerfoo/zerfoo/commit/c2d6835fa8661a86a151c0712f3036b926a20b6f))
* **cuda:** add decode-specific flash attention kernel with GPU-resident kv_len ([67ea195](https://github.com/zerfoo/zerfoo/commit/67ea195424bdcd6f396ee3d210c8c54d2f9a4f1e))
* **cuda:** add FP16 kernel variants for element-wise ops ([ebfeb9d](https://github.com/zerfoo/zerfoo/commit/ebfeb9d779bcb98036f67cda666262f53ef9a917))
* **cuda:** add FP8 E4M3 dequantize-on-load element-wise kernels ([2b7c87f](https://github.com/zerfoo/zerfoo/commit/2b7c87f9eda03ba792e2ec82f7557925b17c9832))
* **cuda:** add fused dequant+GEMV kernel for Q4_K_M ([2fb1921](https://github.com/zerfoo/zerfoo/commit/2fb1921ad33f21e3fdee80fe81b5c44ea2969653))
* **cuda:** add fused dequant+GEMV kernel for Q4_K_M to build ([30ef54b](https://github.com/zerfoo/zerfoo/commit/30ef54b792e84a58567679561cfe0aed7f98410a))
* **cuda:** add GPU repeat kernel to replace D2D memcpy loop ([69bdfa4](https://github.com/zerfoo/zerfoo/commit/69bdfa41b817ea01e93591739c583c34ae187687))
* **cuda:** add GQA support to flash_attention_decode kernel ([7ced794](https://github.com/zerfoo/zerfoo/commit/7ced794a6a627514074e1215696e2f2886036b8c))
* **cuda:** add increment_counter kernel for GPU-resident position tracking ([7ec23c0](https://github.com/zerfoo/zerfoo/commit/7ec23c0be667da6413fbcfa3b19341d5221ffe1f))
* **cuda:** add managed memory detection and arena support for GB10 ([836e1bf](https://github.com/zerfoo/zerfoo/commit/836e1bfa1646ba9366c1bfea020e87ec53d28bfe))
* **cuda:** add managed memory detection and arena support for GB10 ([c93f9b8](https://github.com/zerfoo/zerfoo/commit/c93f9b8034d5c8c0ca827d78b4db4ac8072c6318))
* **cuda:** add offset_memcpy kernel for GPU-driven KV cache append ([9a09960](https://github.com/zerfoo/zerfoo/commit/9a09960a6427b7bfa2c1631d235d1f09c318f20f))
* **cuda:** add Q4_K dequantization kernel for non-GEMV MatMul ([af6bede](https://github.com/zerfoo/zerfoo/commit/af6bede3b5231270a9d22f1bd35833f28cddb0cd))
* **cuda:** convert flash attention dispatch to purego ([0dc4ef2](https://github.com/zerfoo/zerfoo/commit/0dc4ef2a92229e419289d068f88fafa4a7864965))
* **cuda:** expose ArenaPool.Capacity() for profiling ([b123463](https://github.com/zerfoo/zerfoo/commit/b1234639bd472e7bbbeee250d1b9dbe74d77252c))
* **cuda:** optimize N-D transpose kernel with precomputed output strides ([e0a2921](https://github.com/zerfoo/zerfoo/commit/e0a292134ac625e02d1fa66e60fe3e72a4cdec75))
* **cuda:** optimize N-D transpose kernel with precomputed output strides ([82c8aea](https://github.com/zerfoo/zerfoo/commit/82c8aea37c10f79b0b2176c78fc1a36a7f6a9ca8))
* **cuda:** update TransposeND dispatch to pass precomputed output strides ([24c17b8](https://github.com/zerfoo/zerfoo/commit/24c17b825ba2e4c3099ba9c02fca898ad32a0e9d))
* **cuda:** update TransposeND dispatch to pass precomputed output strides ([b77fe8a](https://github.com/zerfoo/zerfoo/commit/b77fe8a405204bf73f2ac78f62d27f7daf0891c0))
* **cudnn:** add purego wrappers for cuDNN API ([6da6756](https://github.com/zerfoo/zerfoo/commit/6da6756011758d5d03449b6132aa595cdaa8bc33))
* **debug:** add layer-by-layer comparison to debug-infer ([1846ba1](https://github.com/zerfoo/zerfoo/commit/1846ba12b42304c952d25a200ded1920a0d8bb60))
* **generate:** add FP16 storage mode to TensorCache ([ae508a2](https://github.com/zerfoo/zerfoo/commit/ae508a23b252fa825927884fa48814d496326421))
* **generate:** add GPU-resident KV sequence length counter to TensorCache ([3b77a7a](https://github.com/zerfoo/zerfoo/commit/3b77a7ac1fecf045e84ac87fb1494db86260fe8d))
* **generate:** add GPU-resident position counter to GPUKVCache ([889685d](https://github.com/zerfoo/zerfoo/commit/889685d91b0fb7c9184625692eeb56b8a8baaa3a))
* **generate:** add GPU-resident TensorCache to eliminate KV cache D2H copies ([9fb2b38](https://github.com/zerfoo/zerfoo/commit/9fb2b381d908ff9a63da0846b67fd44bf4f198ac))
* **generate:** enable CUDA graph capture in compileGraph ([84b933e](https://github.com/zerfoo/zerfoo/commit/84b933e7c34b2c114272dc82cadb3fee1feba748))
* **generate:** enable CUDA graph capture in compileGraph ([9db1236](https://github.com/zerfoo/zerfoo/commit/9db12366a70789fed521ad7351581deef10687f9))
* **generate:** use offset_memcpy kernel for GPU-driven KV cache append ([b28bfc9](https://github.com/zerfoo/zerfoo/commit/b28bfc9a3deb75a23c2f6503064b08c7ea7aabb0))
* **generate:** wire CUDA graph executor into decode loop ([a7e1efc](https://github.com/zerfoo/zerfoo/commit/a7e1efca966a80a15a9d63ae77c12228b88d3db1))
* **generate:** wire FP16 KV dtype through Generator to TensorCache ([10f2458](https://github.com/zerfoo/zerfoo/commit/10f2458b744711e08b07e3bab2d29149b622efb6))
* **gguf:** add BFloat16 weight loading support ([1db38fa](https://github.com/zerfoo/zerfoo/commit/1db38fa25ad9ad78635e17cac1445a83f8c8645d))
* **gguf:** add FP8 E4M3 weight quantization support ([8357cdc](https://github.com/zerfoo/zerfoo/commit/8357cdc41e2b74aa436f27e4cb6368157d4c4f41))
* **gguf:** add Q5_0 dequantization support ([28028f7](https://github.com/zerfoo/zerfoo/commit/28028f7d7c9cb05ffed107103583a96d1a78073b))
* **gguf:** preserve Q4_K storage instead of re-quantizing to Q4_0 ([64633c5](https://github.com/zerfoo/zerfoo/commit/64633c56cef2164c26cc87f739ca20bf1a7ca44a))
* **gpuapi:** add arena allocation profiling logging ([f3f546d](https://github.com/zerfoo/zerfoo/commit/f3f546d5b3303ca3ccf084f1b425c79213960c73))
* **gpuapi:** add Argmax to KernelRunner interface ([ae985e7](https://github.com/zerfoo/zerfoo/commit/ae985e7d1a9fcd454aa07c31e5e241c2fe2e52ae))
* **gpuapi:** add counter and memcpy stubs to OpenCL, ROCm, and test ([eb41279](https://github.com/zerfoo/zerfoo/commit/eb41279b66de4b0848f04d14a95ec43353710a09))
* **gpuapi:** add DequantFP8E4M3ToFP16 to KernelRunner interface ([70dbbbe](https://github.com/zerfoo/zerfoo/commit/70dbbbe100dddb32b1b247b05c1d70941676dc51))
* **gpuapi:** add DequantQ4KF32 to KernelRunner interface ([6f39798](https://github.com/zerfoo/zerfoo/commit/6f39798b212264c2161473ff24514dd38dd76bb7))
* **gpuapi:** add FusedSwiGLU to KernelRunner interface ([921b5f3](https://github.com/zerfoo/zerfoo/commit/921b5f323131259ae8cd7c07edc8520cfb34a4be))
* **gpuapi:** add GemmQ8F32 to KernelRunner interface ([ad3c734](https://github.com/zerfoo/zerfoo/commit/ad3c734671b2bfcbc8b99f03df5649c27579b7c7))
* **gpuapi:** add GemvQ4KF32 to KernelRunner interface ([1685514](https://github.com/zerfoo/zerfoo/commit/1685514a478b4f5ddb4fbab7c6f9837622613a0c))
* **gpuapi:** add OffsetMemcpy to KernelRunner interface ([7b561f5](https://github.com/zerfoo/zerfoo/commit/7b561f53f93609b722ee51c90da8cadea27164bf))
* **gpuapi:** add Repeat method to KernelRunner interface ([d5a1533](https://github.com/zerfoo/zerfoo/commit/d5a15337964fc44b64a7052342403681ad5437e4))
* **gpuapi:** remove cuda build tag from cuda_blas, add runtime guard ([89a66c1](https://github.com/zerfoo/zerfoo/commit/89a66c103fbe96a356fe944f3a73cbb39dee8d9e))
* **graph:** add MatMulTransposeB to traced execution plan ([6df83f4](https://github.com/zerfoo/zerfoo/commit/6df83f4d2bd0ef400d65e13ea3b9843045271ad0))
* **graph:** implement CUDA graph capture and replay for execution plans ([566e5ea](https://github.com/zerfoo/zerfoo/commit/566e5ea21f0adba3e42ed58d4df6e82247775002))
* **graph:** implement partial CUDA graph capture ([9fa7ded](https://github.com/zerfoo/zerfoo/commit/9fa7dedbdb5d0009d55e243ea34dc4c9ec9ecb9d))
* **graph:** pre-allocate fixed buffer layout in ExecutionPlan ([bf967d1](https://github.com/zerfoo/zerfoo/commit/bf967d1155a9e6dfb7da48606232ccecd1527fc7))
* **hip:** add purego wrappers for HIP runtime API ([fa0aafb](https://github.com/zerfoo/zerfoo/commit/fa0aafb500d4e4c14b0f5b5e4ac936891aaf0f6f))
* **inference:** add fp8 dtype support with weight quantization ([88b763a](https://github.com/zerfoo/zerfoo/commit/88b763a6b578fbf27289c41100adb95243e17306))
* **inference:** add Gemma 3 architecture support for GGUF loading ([9b03cb1](https://github.com/zerfoo/zerfoo/commit/9b03cb1a09c59820128d5233411087cb73cebe5b))
* **inference:** add Gemma 3 post-norm support and fix tensor name mapping ([2bb8cff](https://github.com/zerfoo/zerfoo/commit/2bb8cffd7449d87e5a8a164083d6caac748bba70))
* **inference:** add prompt/completion token counting and Tokenizer accessor ([cda1f29](https://github.com/zerfoo/zerfoo/commit/cda1f29afd0e5b8eb0ea068e7cbf8dfe4bd42b9d))
* **inference:** add prompt/completion token counting and Tokenizer accessor ([da539d3](https://github.com/zerfoo/zerfoo/commit/da539d387dc559a27aa9cf63546d5e0f1539c456))
* **inference:** add WithKVDtype option for FP16 KV cache ([1ee43cc](https://github.com/zerfoo/zerfoo/commit/1ee43cc75dcc8acec093402232aae406f3c795fe))
* **inference:** convert LMHead FP16 logits to F32 for sampling ([a3e868b](https://github.com/zerfoo/zerfoo/commit/a3e868bf5fece470c54e2f1a4ca8b97297148dde))
* **inference:** fuse post-FFN RMSNorm + residual Add into single kernel ([6b22b47](https://github.com/zerfoo/zerfoo/commit/6b22b4762251c8587e97a8143222c34050af827e))
* **inference:** GPU-aware embedding lookup and softcapping ([71d04ae](https://github.com/zerfoo/zerfoo/commit/71d04aee4a9e6caafab2938b7f9262bdd1ee7ca2))
* **inference:** implement alternating global/local attention for Gemma 3 ([817f48a](https://github.com/zerfoo/zerfoo/commit/817f48a5d76b387961ad9c42639bd7749aa1b3d9))
* **inference:** merge QKV and Gate+Up projections for decode optimization ([c3835ad](https://github.com/zerfoo/zerfoo/commit/c3835ad63302683c84a78f756b38684a6191e410))
* **kernels:** accept int64 indices in gather kernel ([46d435e](https://github.com/zerfoo/zerfoo/commit/46d435e323441cff2ca648047444f22739a4fa6c))
* **kernels:** accept int64 indices in gather kernel ([f698a29](https://github.com/zerfoo/zerfoo/commit/f698a2969de0fc6cbcb713e8f3b5e7217169adae))
* **kernels:** add 4D broadcast element-wise CUDA kernels ([0d64322](https://github.com/zerfoo/zerfoo/commit/0d643220cfd08f1c1a54108e460c74a85edc6f35))
* **kernels:** add data_offset parameter to GemmQ4F32 ([6a9dae1](https://github.com/zerfoo/zerfoo/commit/6a9dae12ea89da79de5e4ab18450aa142ec4bfec))
* **kernels:** add fused SwiGLU CUDA kernel ([930d074](https://github.com/zerfoo/zerfoo/commit/930d0743030a3863799f370cf7c0f480e018d52f))
* **kernels:** add fused_add_rmsnorm and scaled_softmax kernels ([062d8fc](https://github.com/zerfoo/zerfoo/commit/062d8fcb81caf9874fbd741a2f7c487d16bb5399))
* **kernels:** add GPU argmax kernel for greedy decoding ([a1f1980](https://github.com/zerfoo/zerfoo/commit/a1f1980ec089cce01d6212f450b1beb49dcf694c))
* **kernels:** add int32 index support to gather kernel and parity tests ([ddd14d9](https://github.com/zerfoo/zerfoo/commit/ddd14d9426f3ee267aca6eece71ea1199bdfc4d7))
* **kernels:** add Q8_0 dequant-GEMM CUDA kernel ([ab06c61](https://github.com/zerfoo/zerfoo/commit/ab06c613937af344432fc046835c10d4e4c6b0c7))
* **kernels:** add rope_select CUDA kernel for GPU-indexed RoPE table selection ([f3d2da6](https://github.com/zerfoo/zerfoo/commit/f3d2da61533bc67781bc71483817be383d139186))
* **layers/gather:** implement EmbeddedFrozenProvider interface ([8a0706b](https://github.com/zerfoo/zerfoo/commit/8a0706b68f662ca583bf12c9557ed092f2b79c11))
* **layers:** add GetAnglesGPU to RotaryPositionalEmbedding ([60c251f](https://github.com/zerfoo/zerfoo/commit/60c251fc919f790adee2796495d376d846eae8a6))
* **model/gguf:** skip 1D tensors in FP8 quantization ([83e5897](https://github.com/zerfoo/zerfoo/commit/83e5897d4502217f9580a64d9eb67fd1e523895c))
* **model:** add FP8 scale factor diagnostic logging in QuantizeToFP8E4M3 ([fa50993](https://github.com/zerfoo/zerfoo/commit/fa509939a5f434910b1cee767ef15b912767d59f))
* **opencl:** add purego wrappers for OpenCL runtime API ([63ca987](https://github.com/zerfoo/zerfoo/commit/63ca9874f7ae3f830d9bbe9790065e2fd949248e))
* **q4k:** re-enable Q4_K preservation and add dispatch diagnostics ([ef567c6](https://github.com/zerfoo/zerfoo/commit/ef567c62def69491a187e07b52baeac421018ed3))
* **rocblas:** add purego wrappers for rocBLAS API ([4aed62d](https://github.com/zerfoo/zerfoo/commit/4aed62db13f5e6f11bbb94c2f5f7153d592f9ac4))
* **serve:** add embeddings, model info, model delete endpoints and usage token counting ([36f176b](https://github.com/zerfoo/zerfoo/commit/36f176b6566f10e7f06e442d06aad7bce47ce09b))
* **serve:** add embeddings, model info, model delete endpoints and usage token counting ([1b17557](https://github.com/zerfoo/zerfoo/commit/1b175576554799134cd8a1ffbab6f7f9c7e49a8c))
* **serve:** add GET /openapi.yaml endpoint for client discovery ([728a966](https://github.com/zerfoo/zerfoo/commit/728a96619674a890cd9139678f93f380bfc9cf31))
* **tensor:** add BFloat16Storage for half-precision weight storage ([b86820c](https://github.com/zerfoo/zerfoo/commit/b86820cbe2186e7a04cf810955f24e100a7d36dc))
* **tensor:** add DequantizeBlock method to Q8Storage ([84dd932](https://github.com/zerfoo/zerfoo/commit/84dd932a4949454209ecacbb34e8a7945f34eb7b))
* **tensor:** add DequantizeRange method to Q8Storage ([4d5a77c](https://github.com/zerfoo/zerfoo/commit/4d5a77cdc72faa464d0b0b8599055ae49d6e891f))
* **tensor:** add Float16Storage type ([078a415](https://github.com/zerfoo/zerfoo/commit/078a4150112bbda2b82f3c79386a5f991776a6a4))
* **tensor:** add FP8 E4M3 and E5M2 storage types ([c509f77](https://github.com/zerfoo/zerfoo/commit/c509f7759d104e7802d801bfb83f8593728e8c31))
* **tensor:** add GPU pointer support to FP8E4M3Storage ([3834080](https://github.com/zerfoo/zerfoo/commit/3834080abbf2c22c1b72cd8da0f68a328b82b610))
* **tensor:** add GPU support to Q4KStorage ([9801661](https://github.com/zerfoo/zerfoo/commit/980166171df79cd7291dac9d0f53098e94c8f116))
* **tensor:** add GPU-optimized separated Q4 layout ([8cccf9e](https://github.com/zerfoo/zerfoo/commit/8cccf9eb9ade9e457399335e95f418a0d64e4926))
* **tensor:** add GPUStorage.SubSlice for GPU-side slicing without D2H copy ([e63f7d3](https://github.com/zerfoo/zerfoo/commit/e63f7d3dce9f5a2f7a0ab7ce3aa3cb37b27a8ef5))
* **tensor:** add NewFloat16StorageGPU constructor ([ccdcc44](https://github.com/zerfoo/zerfoo/commit/ccdcc444b1503a3cc18a58a96ea280865c2dcf08))
* **tensor:** add RawBytes, GPUPtr, SetGPUPtr to Q8Storage ([3b48c18](https://github.com/zerfoo/zerfoo/commit/3b48c180a59bbab6856314e369abfbe3e18d705f))
* **tensorrt:** add purego wrappers for TensorRT C API ([a3177cf](https://github.com/zerfoo/zerfoo/commit/a3177cf1b4b951138c46a1531b990a54b311a3fa))
* **tokenizer:** add special token handling for GGUF control tokens ([055d5ac](https://github.com/zerfoo/zerfoo/commit/055d5ac49c44784196439005c0e7bac0c0b9d8d3))
* **tokenizer:** implement SentencePiece pre-tokenization for GGUF models ([c9d0877](https://github.com/zerfoo/zerfoo/commit/c9d08778062c912e1492c47e771f7c510a0e6cf9))
* **xblas:** add GemmF32Q8NT for Q8 weight matrix-vector multiply ([a754756](https://github.com/zerfoo/zerfoo/commit/a754756def0dabc3a25c0248d19548f59212ced9))


### Bug Fixes

* **activations:** correct SwiGLU to use silu(x1)*x2 instead of x1*sigmoid(x2) ([a2b37d2](https://github.com/zerfoo/zerfoo/commit/a2b37d2813d806422cfa42c0e411d134fba8f363))
* **attention:** add automatic causal masking during prefill ([87c422e](https://github.com/zerfoo/zerfoo/commit/87c422ef19c30aea18dba73fc1c4eb651b5cc75f))
* **attention:** add D2H copy warnings to GQA CPU fallback paths ([c303128](https://github.com/zerfoo/zerfoo/commit/c3031280a79f601104f4facc403de6fecc59225f))
* **attention:** add Float16Storage zero-copy views in GQA split paths ([c9f006b](https://github.com/zerfoo/zerfoo/commit/c9f006ba1bb6d5f7ad3933ff4f695d76c4764a88))
* **attention:** apply causal masking directly in SDPA scores ([d712806](https://github.com/zerfoo/zerfoo/commit/d712806350db712f34e6c45dc57cc85a7104e3b2))
* **attention:** eliminate D2H fallback in fused QK norm+RoPE path ([267a85d](https://github.com/zerfoo/zerfoo/commit/267a85d360b1b819f72519ec0c75238c13082c9e))
* **attention:** eliminate D2H fallback in splitMergedQKV ([87f2735](https://github.com/zerfoo/zerfoo/commit/87f27351338154f711db9d34ca4f5bf33f7572b5))
* **attention:** skip decode fast path for GQA to avoid maxSeqLen Repeat regression ([3f4fb8e](https://github.com/zerfoo/zerfoo/commit/3f4fb8eeb1fb296cc087e2153d0cafe26ffd85c9))
* **attention:** skip decode fast path for GQA to avoid Repeat regression ([dcda448](https://github.com/zerfoo/zerfoo/commit/dcda448706c46c6dc27b23eac42620c07043bb54))
* **attention:** use GPUStorage.SubSlice instead of NewGPUStorageView in GQA ([0e3ebc2](https://github.com/zerfoo/zerfoo/commit/0e3ebc2893a1a875880e98be06c89c8232a028c7))
* **ci:** checkout zmf repo as sibling for go.mod replace directive ([ea1f649](https://github.com/zerfoo/zerfoo/commit/ea1f64971e719b51efaadf8a847ea53d25805c44))
* **ci:** lint only new code to avoid failing on 276 pre-existing issues ([09a5131](https://github.com/zerfoo/zerfoo/commit/09a513114cf79b299bb4d067faf253dbdc6a153e))
* **ci:** make benchmark regression check non-blocking ([ae6bdf6](https://github.com/zerfoo/zerfoo/commit/ae6bdf6eded3375d1a036509471320f42746f93b))
* **compute:** add Float16Storage support to GPUEngine.Reshape ([4c5e996](https://github.com/zerfoo/zerfoo/commit/4c5e996412bec7025effcfc2b1bc587c314f09d5))
* **compute:** add Float16Storage support to Reshape, Transpose, Split, Concat, Repeat ([a369bd2](https://github.com/zerfoo/zerfoo/commit/a369bd2f20eb9a223302471e89badef4c8c9af4e))
* **compute:** add FP16 dequant fallback for FP8 MatMul on SM &lt; 8.9 ([59da4fb](https://github.com/zerfoo/zerfoo/commit/59da4fbfa25635ad65170392ba60827397d9046f))
* **compute:** handle batch dimensions in FP16 MatMul ([f261aa1](https://github.com/zerfoo/zerfoo/commit/f261aa1ef8754cc61b5fdd4bafcdf7906426e9cf))
* **compute:** handle Float16Storage in getDevicePtr ([afefd6a](https://github.com/zerfoo/zerfoo/commit/afefd6a653d8bb6eabcc5b25a1ee5b5d7b792613))
* **compute:** handle Float16Storage weights in F32 RMSNorm GPU path ([c61f2b0](https://github.com/zerfoo/zerfoo/commit/c61f2b03a215fbff2d8d3a448193870694175cdc))
* **compute:** keep F32 weights as GPUStorage, fix FP16 garbage output ([efdd87b](https://github.com/zerfoo/zerfoo/commit/efdd87b964b605158ce2508aff5762d066d70c04))
* **compute:** merge FP8 scratchpad reset and output buffer methods ([ad7b9f9](https://github.com/zerfoo/zerfoo/commit/ad7b9f9eb08cc9e92c9f70b3b7a983bbbeb9a7bd))
* **compute:** remove Q4 GEMM debug logging ([07bb6c2](https://github.com/zerfoo/zerfoo/commit/07bb6c22e25c65d79d1157a5b28aea28ef429459))
* **compute:** skip Float16Storage tensors in UploadWeights re-entry ([f0b1bc5](https://github.com/zerfoo/zerfoo/commit/f0b1bc553c2b9cba8422aff372d0af19d4984c84))
* **compute:** skip Q8 tensors in F32 upload loop ([db8ecbf](https://github.com/zerfoo/zerfoo/commit/db8ecbfea204362c15027bcb21aaf68093faed32))
* **compute:** use dual scratchpad buffers in FP8 dequant fallback ([06afb2c](https://github.com/zerfoo/zerfoo/commit/06afb2c20d62ed86ae056251f0d029962c2fc6ef))
* **compute:** use GPU dequant+cuBLAS for non-GEMV Q4_K MatMul ([d78f7dd](https://github.com/zerfoo/zerfoo/commit/d78f7dd4970aa2b43afb3a7d56c09d7843b8341b))
* **cuda-graph:** mark GQA non-capturable, add arena reset floor and captured slot restore ([ea1fddd](https://github.com/zerfoo/zerfoo/commit/ea1fddd0d04ea6aec700edf2e1574166c4dec90e))
* **cuda-graph:** upload pre-capture CPU slots to GPU before capture ([aab529c](https://github.com/zerfoo/zerfoo/commit/aab529c130846aec8fc9ea18448cb3b4df730d76))
* **cuda:** add !cuda build constraint to purego kernel stubs ([1e8d4cd](https://github.com/zerfoo/zerfoo/commit/1e8d4cd2a9d9fad8e15a74d4161ed09e444a145d))
* **cuda:** add CGo helpers and !cuda constraint for purego loader ([312e85d](https://github.com/zerfoo/zerfoo/commit/312e85d021035f3fbaff613f508493c5b8eea18c))
* **cuda:** arena allocator must respect ZERFOO_ENABLE_MANAGED_MEM ([0ab1755](https://github.com/zerfoo/zerfoo/commit/0ab1755815b286be52a3670c1a411872b7fd6a15))
* **cuda:** correct CGo declaration for fused RMSNorm kernel ([4e82605](https://github.com/zerfoo/zerfoo/commit/4e826057fdb4eb3cfcddba04c9d7384635e8ed7e))
* **cuda:** disable arena reset and increase to 2GB for correctness ([6e55bf0](https://github.com/zerfoo/zerfoo/commit/6e55bf0ca79f3fe2b300f9a395fe2a1632c620c8))
* **cuda:** revert !cuda constraint from purego files without CGo counterparts ([07b9bbe](https://github.com/zerfoo/zerfoo/commit/07b9bbe2815ca729f487c695bd16a3733eee49a0))
* **cuda:** sync CGo kernel wrappers with purego API signatures ([fa2b336](https://github.com/zerfoo/zerfoo/commit/fa2b33613a11fd1c9a4a52be07723639b98cc4d7))
* **cuda:** update CUDA Q4 GEMM kernel to GGML split nibble ordering ([36398ab](https://github.com/zerfoo/zerfoo/commit/36398abdedb6a514bde8c6e63ff19937f4839b14))
* **cuda:** use relaxed capture mode to allow synchronous memcpy during graph capture ([6dbd439](https://github.com/zerfoo/zerfoo/commit/6dbd439f5e9a031f9340057dfde037abe2bafef5))
* **cuda:** use strided KV access in decode kernel for packed head layout ([5130532](https://github.com/zerfoo/zerfoo/commit/51305324ce754409a270658c66203c3bd08bd11f))
* **fp8:** clear stale scratchpad pointers after arena reset, skip embed/lm_head from FP8 ([2168358](https://github.com/zerfoo/zerfoo/commit/2168358a93177541bac16582a8cf47423fa25c78))
* **gemm_q4:** update header to match separated layout signature ([8ac45f4](https://github.com/zerfoo/zerfoo/commit/8ac45f4c261b5591fcfafcd3e98f959dc73051b3))
* **generate:** add ZERFOO_DISABLE_CUDA_GRAPH env var to skip graph capture ([987f63e](https://github.com/zerfoo/zerfoo/commit/987f63e6c3170968bd8d1687b736573e8f467116))
* **generate:** disable CUDA graph and managed memory by default ([08476ef](https://github.com/zerfoo/zerfoo/commit/08476ef1a7b894858c46106ed5eff1fbe5640db5))
* **generate:** disable CUDA graph capture until D2H copies are eliminated ([587c6cd](https://github.com/zerfoo/zerfoo/commit/587c6cd8d5e14aafc299fc31db2cb8e4aa1b655e))
* **generate:** eliminate D2H copies in KV cache CPU fallback path ([1b08d9e](https://github.com/zerfoo/zerfoo/commit/1b08d9e221c099e7db76818208de1eca3acd1f56))
* **generate:** eliminate data race on logitsBuf in BatchGenerate ([8980cbb](https://github.com/zerfoo/zerfoo/commit/8980cbb87cd58223bcd32e1fda47f0c2687bac8b))
* **generate:** eliminate temp buffer race in FP16 KV cache append ([4648d24](https://github.com/zerfoo/zerfoo/commit/4648d24f51f6e2904be92addd2a62b47468fc4a0))
* **generate:** free old GPU buffers in TensorCache to prevent OOM ([6fab037](https://github.com/zerfoo/zerfoo/commit/6fab03768f9a3497b400b77457f2814e461fed35))
* **generate:** prepend BOS token in GenerateStream ([510257e](https://github.com/zerfoo/zerfoo/commit/510257e6d30f9fdce4783b20f75069d4af008adc))
* **generate:** serialize BatchGenerate to eliminate ExecutionPlan data race ([690202b](https://github.com/zerfoo/zerfoo/commit/690202b10965577f237cb99e7edb9dcf788d1396))
* **generate:** sync GPU counter during prefill, not during decode ([3284e35](https://github.com/zerfoo/zerfoo/commit/3284e359389515f832ad44f4cd661af9b9819af0))
* **generate:** sync GPU counter to CPU seqLen after prefill ([768c336](https://github.com/zerfoo/zerfoo/commit/768c336de0c502130983a1e4486a0146c3be8c8b))
* **gguf:** correct GGUF magic constant byte order ([df7b5b5](https://github.com/zerfoo/zerfoo/commit/df7b5b572e0d491b8b2f21f1fbce16fc007bc4c9))
* **gguf:** preserve Q4_K storage instead of re-quantizing to Q4_0 ([7a1e9cf](https://github.com/zerfoo/zerfoo/commit/7a1e9cfb30978f4b6a4133247a479552f9da56cf))
* **gguf:** replace nonexistent ByteSize() with len(RawBytes()) in BF16 test ([c9e91c3](https://github.com/zerfoo/zerfoo/commit/c9e91c3dce43f7a8cbf4c0accbea1c06fb24ccd5))
* **gguf:** reverse GGML dimension order to match PyTorch convention ([9b170d3](https://github.com/zerfoo/zerfoo/commit/9b170d318761d0601d06001e23c911ce7dedcbb3))
* **gguf:** revert Q4_K preservation, use Q4_0 re-quantization for GPU path ([5a40225](https://github.com/zerfoo/zerfoo/commit/5a402257c16258462374f653882eec4dd16840e1))
* **gguf:** revert Q4_K to Q4_0 — GPU dequant path 29% slower than Q4_0 ([60097fb](https://github.com/zerfoo/zerfoo/commit/60097fb8ee074cfed32d532ba7a9b42fa5c65ea2))
* **gguf:** use name-based skip for FP8 quantization instead of shape ([f8bb573](https://github.com/zerfoo/zerfoo/commit/f8bb57384283e579bdb271987922e32be0f85fd7))
* **graph:** gracefully fall back when CUDA graph capture fails ([f9d4f90](https://github.com/zerfoo/zerfoo/commit/f9d4f90ca3ebd3af049dd01ed650f6fe14ead7a0))
* **graph:** include EmbeddedFrozen tensors in ConstantTensors for GPU upload ([154a5cc](https://github.com/zerfoo/zerfoo/commit/154a5cc8cb44172fc8930a12ba67d328fbefc17a))
* **graph:** return error instead of panic on nil tensor in ExecutionPlan.Run ([04c4d34](https://github.com/zerfoo/zerfoo/commit/04c4d3411327ad476bd82cac1a805a44b4f9ad39))
* **inference:** add BOS prepending and logit softcapping for Gemma 3 ([9f5c538](https://github.com/zerfoo/zerfoo/commit/9f5c538be6d2f26b7c615ef6582492c3ff97e3af))
* **inference:** dequantize Q8 weights before transpose ([2588630](https://github.com/zerfoo/zerfoo/commit/25886300e870b5a8cc96f76256864f9fa71d5c66))
* **inference:** implement EmbeddedFrozenProvider on lmHeadNode and embeddingLookupNode ([cafd902](https://github.com/zerfoo/zerfoo/commit/cafd902879cee46bab9482d9278a710d05b1bd82))
* **inference:** preserve FP8E4M3Storage through weight transpose ([6bc41a6](https://github.com/zerfoo/zerfoo/commit/6bc41a6e551eb72a9351a5db6c66c96740e8882d))
* **inference:** real transpose for quantized weights on GPU ([a7bb500](https://github.com/zerfoo/zerfoo/commit/a7bb500d289751e40a8ab00ca426b03e925df5e8))
* **inference:** restore GPU F32 transpose for quantized weights ([8e02446](https://github.com/zerfoo/zerfoo/commit/8e02446e3f2bf3003ee875190b3f0d7f6e3e9952))
* **inference:** use model-specified RMSNorm epsilon instead of hardcoded 1e-5 ([bdc2e98](https://github.com/zerfoo/zerfoo/commit/bdc2e98be0bdfabf9613b7458fdb61e9618e5035))
* **inference:** use virtual transpose for all quantized weights ([f020d15](https://github.com/zerfoo/zerfoo/commit/f020d15367b65fcf6cca66245643af2c682d0253))
* **kernels:** make FP8 and FP16 conversion symbols optional in dlopen ([7c36a43](https://github.com/zerfoo/zerfoo/commit/7c36a43f50da81a257eb890b85d8b1f58e547e8c))
* **kernels:** update FusedAddRMSNorm to write separate normed/sum outputs ([2f690a8](https://github.com/zerfoo/zerfoo/commit/2f690a81dd7f4d10f3390bbd2be35d29e5451fa6))
* **kv-cache:** use async memcpy for D2D copies during CUDA graph capture ([20a5f51](https://github.com/zerfoo/zerfoo/commit/20a5f5156350264eb098ecca18abd18d5cd0176c))
* **layers:** handle nil scales from FP16 FusedRMSNormGPU ([9767a75](https://github.com/zerfoo/zerfoo/commit/9767a758ee16293a96575bbc2fd3420dd3f97880))
* **layers:** unwrap EngineProxy for CPU RoPE and use Split for GPU path ([87e5d0e](https://github.com/zerfoo/zerfoo/commit/87e5d0eada66a8fed2466ef9dbd3c937dc82a46f))
* **matmul:** eliminate D2H copy in weight pointer caching ([12aac09](https://github.com/zerfoo/zerfoo/commit/12aac097cfb5852f2fcd28edf4a52e494b1eef24))
* **model:** update Q5_0 loader and Q4_0 tests for split nibble format ([a0b9781](https://github.com/zerfoo/zerfoo/commit/a0b978175cee28bf2a7989b4ead7b84215393e50))
* restore FusedAddRMSNorm 5-param interface after fused attention revert ([c684a92](https://github.com/zerfoo/zerfoo/commit/c684a927cde34dcfe14aaa4936cbb4a06862e865))
* **rope:** skip backward cache on GPU fused path to avoid D2H copies ([014e0f9](https://github.com/zerfoo/zerfoo/commit/014e0f90040ac3069a63f7018d1c8f1f007effc1))
* **rope:** unwrap EngineProxy for WeightUploader check in lazy GPU upload ([ff54bca](https://github.com/zerfoo/zerfoo/commit/ff54bca205e723e629d69c9171b0202796589ca4))
* **tensor:** add non-owning view mode to GPUStorage for safe reshape ([631a29d](https://github.com/zerfoo/zerfoo/commit/631a29d86526fa10edce045b4ba5f4418c6399fc))
* **tensor:** correct Q4_K/Q5_K/Q6_K dequantization to match llama.cpp ([8b150fc](https://github.com/zerfoo/zerfoo/commit/8b150fce8efabefc66888c9113ae8cad2267c61a))
* **tensor:** make Float16Storage.Slice safe for GPU-only storage ([5f1f9c0](https://github.com/zerfoo/zerfoo/commit/5f1f9c047faaea9a5823e1d055c20cc6bd83231b))
* **tensor:** update all dequantization to GGML split nibble ordering ([1ef239e](https://github.com/zerfoo/zerfoo/commit/1ef239ee4283d5d4d805a3513c198a02b60310bc))
* **tokenizer:** do not prepend leading space after special tokens ([6aa56a2](https://github.com/zerfoo/zerfoo/commit/6aa56a2337443d2ae778c9b89112d49bc025a8f5))
* **tokenizer:** split on newlines in SentencePiece pre-tokenization ([84bdf99](https://github.com/zerfoo/zerfoo/commit/84bdf99b5cd0e09d6ea27519492f9a23bfd559bc))
* **xblas:** correct FMAXV register encoding in NEON softmax ([c172b5e](https://github.com/zerfoo/zerfoo/commit/c172b5e4bc5996db6a7788eb404d14d4fc4d301e))
* **xblas:** update fused Q4 GEMM kernels to GGML split nibble ordering ([8bb827a](https://github.com/zerfoo/zerfoo/commit/8bb827a0c651a4edbcff50213e31eb932c78c8c2))


### Performance Improvements

* **attention:** add fused decode attention kernel for single-token generation ([f7d2498](https://github.com/zerfoo/zerfoo/commit/f7d24984f64f4bddd9f778e898e73c9b1379ec69))
* **attention:** avoid Concat in fused QK norm+RoPE decode path ([27bf4d3](https://github.com/zerfoo/zerfoo/commit/27bf4d34c81a01c9c4b6f028808f3ee3ea04a8f9))
* **attention:** pass KV buffer directly to decode kernel without reshape ([5ce6112](https://github.com/zerfoo/zerfoo/commit/5ce6112761249f5d3e9b8b5ceff7842d6786d979))
* **attention:** re-enable decode fast path for GQA models ([6fc52b0](https://github.com/zerfoo/zerfoo/commit/6fc52b0696673454453fccd83394a49510cb6a3e))
* **attention:** skip KV head replication when numKVHeads=1 ([e92a04a](https://github.com/zerfoo/zerfoo/commit/e92a04ac57175934a662cc295b2dca59c97e2c51))
* **attention:** use cuBLAS SgemmNT to avoid explicit K transpose in SDPA ([74cac33](https://github.com/zerfoo/zerfoo/commit/74cac333ac2ec8cf42c12c68c55eef7e30673467))
* **bench:** add PGO profile for optimized builds ([44e1d9e](https://github.com/zerfoo/zerfoo/commit/44e1d9e2f169b0348c4d989859e7b2011a1f6b30))
* **bench:** add PGO profile for optimized builds ([922177f](https://github.com/zerfoo/zerfoo/commit/922177f45d1f73b72876aa8af46138b0c7ccb0d9))
* **bench:** enrich PGO profile with merged CPU+CUDA samples ([bb741c3](https://github.com/zerfoo/zerfoo/commit/bb741c37bf3bc207c67bfab88dc8f2adb44835c7))
* **bench:** print cuBLAS profile summary when ZERFOO_PROFILE_CUBLAS=1 ([7821c64](https://github.com/zerfoo/zerfoo/commit/7821c64ad8e1cf9c797abc8f660dde8a9eef9406))
* **compute:** add 3D [0,2,1] fast path to Transpose ([c07d128](https://github.com/zerfoo/zerfoo/commit/c07d128c62ecf6093bf2371d1788e43c88eb5848))
* **compute:** add output buffer to fp8Scratchpad ([b3cb11a](https://github.com/zerfoo/zerfoo/commit/b3cb11a0556180d23dadb89ed0d4a79bc5f290fa))
* **compute:** cache RMSNorm weight on GPU to eliminate per-pass H2D upload ([8f64c8f](https://github.com/zerfoo/zerfoo/commit/8f64c8f3bd30cf51542e2d99500259f9c3fb13f9))
* **compute:** delegate FusedRMSNormer through EngineProxy ([b502ac1](https://github.com/zerfoo/zerfoo/commit/b502ac1cee56fd6ec5881951f311e28709a6b096))
* **compute:** enable Q4 B-weight GEMV dispatch in MatMul ([585538b](https://github.com/zerfoo/zerfoo/commit/585538b5c7284be5ae995818c215831ceb52599c))
* **compute:** free GPU tensors immediately in TensorPool.Release ([e7e0820](https://github.com/zerfoo/zerfoo/commit/e7e0820b2367c9c9291963e06a7fab91a18df28d))
* **compute:** handle Q8 storage on B-side in MatMul dispatch ([76c661f](https://github.com/zerfoo/zerfoo/commit/76c661f8715de427ddf9c85270e42e6f5ffe13a3))
* **compute:** skip FP16/FP8 type dispatch on F32 compute path ([837b210](https://github.com/zerfoo/zerfoo/commit/837b2102c7429e020ffb0a16a3339efdec2a891d))
* **compute:** skip transpose kernel when it reduces to a reshape ([cea6ff4](https://github.com/zerfoo/zerfoo/commit/cea6ff4d0d0f5cd0a46c1640b2c0b448bf20784e))
* **compute:** support N-D leading-dimension broadcast in GPU kernels ([45abf02](https://github.com/zerfoo/zerfoo/commit/45abf02860aad2717f039853ef2b40793d2e621f))
* **compute:** upload Q8 weights as raw bytes and use Q8 dequant-GEMM ([cd135e5](https://github.com/zerfoo/zerfoo/commit/cd135e5b3e64f3ae4e1e4998f460e4cbc2cd451d))
* **compute:** use batched NT GEMM in MatMulTransposeB for multi-head attention ([d72be2f](https://github.com/zerfoo/zerfoo/commit/d72be2fb46d34aa4c8acf1fb6cd27baee98bfb37))
* **compute:** use direct cudaMalloc for permanent weight storage ([33b0dee](https://github.com/zerfoo/zerfoo/commit/33b0deeff2979c87039181c5104be29ca85d1cca))
* **compute:** use shared no-op cleanup in getDevicePtr for GPU tensors ([a370d21](https://github.com/zerfoo/zerfoo/commit/a370d214d383c9a1374f06784347dfcea837c2b9))
* **compute:** virtual transpose and Q8 NT dispatch in MatMul ([145345b](https://github.com/zerfoo/zerfoo/commit/145345badbdb858bcf74050a1aabd3243c938b11))
* **compute:** wire GPU repeat kernel into GPUEngine.Repeat ([86fdff0](https://github.com/zerfoo/zerfoo/commit/86fdff0c107d255c2d6e16517561c7cc41c739f6))
* **cublas:** add SgemmStridedBatched for single-call batched MatMul ([2bbbeb1](https://github.com/zerfoo/zerfoo/commit/2bbbeb130dbae2f17ec2930002f88792ea599f9d))
* **cuda:** add MemPool singleton and hit/miss instrumentation ([fc39060](https://github.com/zerfoo/zerfoo/commit/fc39060636a2e08280122b67e9885fee2c5a2698))
* **cuda:** add optimized Q4 GEMV kernel with shared memory and warp shuffle ([38e9bf7](https://github.com/zerfoo/zerfoo/commit/38e9bf7e1953530a4f7bf6f3c90bf002470a2e11))
* **cuda:** add purego cuBLAS binding to eliminate CGo overhead ([edef0c0](https://github.com/zerfoo/zerfoo/commit/edef0c03490f58d4113a61b884fdd024cbbf5c3b))
* **cuda:** implement arena allocator for zero-malloc inference ([1200702](https://github.com/zerfoo/zerfoo/commit/12007023531ae8aa22e04322ec0e9a7517c3878e))
* **cuda:** implement power-of-2 bucket pooling for GPU memory reuse ([f0278f6](https://github.com/zerfoo/zerfoo/commit/f0278f65b612a1f84dfdf5d9c80596acbdec298d))
* **cuda:** optimize shared memory in attention and reduction kernels ([bc3285f](https://github.com/zerfoo/zerfoo/commit/bc3285f1635337858335dd162093bf0878922c31))
* **cuda:** tune register pressure for gemm_q4 and transpose kernels ([c2060e5](https://github.com/zerfoo/zerfoo/commit/c2060e5f4461918eb8fd223b9429a29136213728))
* **gemv_q4:** dual accumulators, FMA, and 2 rows per warp ([af2af0a](https://github.com/zerfoo/zerfoo/commit/af2af0a7b28783f194d0b9ac9d10ea236570ec3e))
* **generate:** add GPU counter to TensorCache for CUDA graph capture ([f5994ca](https://github.com/zerfoo/zerfoo/commit/f5994ca9ba4f3fb57663aaaa9979d51424cf3e30))
* **generate:** add runtime.LockOSThread for CUDA context affinity ([64d22f5](https://github.com/zerfoo/zerfoo/commit/64d22f53a22def3cb387d3098fa1f76e8c70d183))
* **generate:** add runtime.LockOSThread for CUDA context affinity ([959477d](https://github.com/zerfoo/zerfoo/commit/959477d4eef588b390a58146c40ae277c39664a0))
* **generate:** eliminate sampling allocation overhead ([0fa78c3](https://github.com/zerfoo/zerfoo/commit/0fa78c3f9351119a1ed47df217bdaf907694e2e8))
* **generate:** enable CUDA graph capture for decode loop ([d93fa82](https://github.com/zerfoo/zerfoo/commit/d93fa8293dcf07ac70236ce93728806cc2554746))
* **generate:** force GC between decode passes to reclaim GPU pool memory ([f8130a9](https://github.com/zerfoo/zerfoo/commit/f8130a9689934bc8839f66589b81d0c5e711b88c))
* **generate:** pre-allocate KV cache buffers to eliminate per-token cudaMalloc ([7e80e21](https://github.com/zerfoo/zerfoo/commit/7e80e21198252acc5a5febda2efaa3b3e4d6adff))
* **generate:** reset arena between decode tokens and persist KV cache ([67444ce](https://github.com/zerfoo/zerfoo/commit/67444ceb26caf79e3a6977f3fe070b43b69bf135))
* **generate:** reuse token input tensor across decode steps ([7f21001](https://github.com/zerfoo/zerfoo/commit/7f2100152de4553b34fdc6b2de3b60af117000e8))
* **generate:** sync GPU counter after decode loop completes ([79d9464](https://github.com/zerfoo/zerfoo/commit/79d9464858cd81429e1f472dacbc21e7a1ca5f04))
* **generate:** use GPU argmax for greedy decoding to skip logits D2H copy ([f47e3e9](https://github.com/zerfoo/zerfoo/commit/f47e3e9b82bea7759e2e2f45d9e6994dd7eb8413))
* **gguf:** re-quantize Q4_K/Q5_K/Q6_K to Q4_0 and lmHead Q8 to Q4 ([52190c9](https://github.com/zerfoo/zerfoo/commit/52190c935260eaaf8e79b634bf2366255cf014ac))
* **gguf:** re-quantize Q5_0 weights to Q4_0 for fast NEON GEMV ([39c4ca7](https://github.com/zerfoo/zerfoo/commit/39c4ca75c8c0c1ecc01681e756b39395a6309dcf))
* **gpuapi:** add cuBLAS profiling wrapper for SGEMM overhead analysis ([d4f34bf](https://github.com/zerfoo/zerfoo/commit/d4f34bf64cd04c9eaa1b32e34ec0bed63f4496d3))
* **gpu:** eliminate CPU tensor bottleneck in decode pipeline ([859e023](https://github.com/zerfoo/zerfoo/commit/859e0238dcf364b034147c2e8baa6ac208006060))
* **graph:** eliminate per-forward allocations in decode hot path ([6e4cdf8](https://github.com/zerfoo/zerfoo/commit/6e4cdf89e952e039d9afac6820779a3bdb21278e))
* **graph:** eliminate per-token heap allocations in ExecutionPlan.Run ([4655ed6](https://github.com/zerfoo/zerfoo/commit/4655ed6a3bb66f1b91cc2608b4a5770e7085fb10))
* **graph:** include all Parameter values in ConstantTensors for GPU upload ([f625c88](https://github.com/zerfoo/zerfoo/commit/f625c886f7cd36e008517d725a4779882c3368a0))
* **graph:** remove GQA from non-capturable ops list ([92e61ed](https://github.com/zerfoo/zerfoo/commit/92e61edbe80691960ef3b3a762a21949b0384fc6))
* **inference:** cache pre-transposed lmHead weight and use row-level Q8 dequant for embeddings ([37ebd83](https://github.com/zerfoo/zerfoo/commit/37ebd835e4e8e4fe806468a9394dad65dbab1989))
* **inference:** convert LM head to Q4 on GPU for 7x bandwidth reduction ([547f584](https://github.com/zerfoo/zerfoo/commit/547f584ad3df77c934131bdd660eb21a2bfb7fdc))
* **inference:** fuse residual Add + pre-FFN RMSNorm into single kernel ([5d22cd9](https://github.com/zerfoo/zerfoo/commit/5d22cd9a1d44ace3964ea546c9a3fdca7a0f0a3f))
* **inference:** integrate FusedAddRMSNorm into transformer graph ([8ebaab7](https://github.com/zerfoo/zerfoo/commit/8ebaab78e2882e69dc3b011dc31a1a92b3c1ab1b))
* **inference:** replace math.Tanh with fast float32 approximation and parallelize softcapping ([c7cefed](https://github.com/zerfoo/zerfoo/commit/c7cefed6564638aab8ef0fde1a0d803d349657ea))
* **inference:** skip Q8-to-Q4 lmHead conversion on GPU ([ccef647](https://github.com/zerfoo/zerfoo/commit/ccef647f6b26ae186dfc0cdb8950f47f5e81539f))
* **inference:** use virtual transpose for Q4 weights on GPU ([34877a6](https://github.com/zerfoo/zerfoo/commit/34877a677527ea7cf36a5f4b708edbba4eef9d98))
* **inference:** virtual transpose for quantized weights ([7babf3c](https://github.com/zerfoo/zerfoo/commit/7babf3cfc6a27372d899087dff52af9d4c9794e3))
* **kernels:** add fused GPU RoPE kernel to eliminate Split/Concat overhead ([e35cd2c](https://github.com/zerfoo/zerfoo/commit/e35cd2c844fad70448313efbe9b85888a0381ead))
* **kernels:** increase Q4K GEMV block size from 128 to 256 ([4e7f0fc](https://github.com/zerfoo/zerfoo/commit/4e7f0fc49804dd14fc7308c7d3b7b90250e0f35e))
* **kernels:** optimize Q8 GEMV with vectorized int4/float4 loads ([f8f2a52](https://github.com/zerfoo/zerfoo/commit/f8f2a52b6f45e01a32ac9c1498453cc12bf1f782))
* **kernels:** upgrade NVCC to -O3 --use_fast_math for faster GPU kernels ([d1ed26a](https://github.com/zerfoo/zerfoo/commit/d1ed26aed1ee0db0ff65f870e20c533af8f58430))
* **kernels:** vectorize Q4K GEMV loads and tile x-vector ([962f09d](https://github.com/zerfoo/zerfoo/commit/962f09d1f3533aa47a4797d88723a0cea188afc4))
* **layers:** use fused SwiGLU kernel eliminating 5 ops per FFN layer ([48ef18f](https://github.com/zerfoo/zerfoo/commit/48ef18fe348b2346e8806901d67263e1eafcc98f))
* **layers:** use GPU RoPE in GQA unfused path for CUDA graph capture ([b7ce7f9](https://github.com/zerfoo/zerfoo/commit/b7ce7f95f828d0aae9d5d32f6b0ca4121f061026))
* **layers:** use GPU RoPE selection in GQA decode fast path ([f4f7b3d](https://github.com/zerfoo/zerfoo/commit/f4f7b3de40992591d674994b699f46a2a43a325b))
* **rope:** expand cos/sin to match batch dimension for GPU-native Mul ([91e2218](https://github.com/zerfoo/zerfoo/commit/91e22180bad21c781e96fdeed2babf022f0d76dc))
* **tensor:** add refcounting to GPUStorage for immediate pool return ([276cc72](https://github.com/zerfoo/zerfoo/commit/276cc725150c881077b847f220ce3e63c22784ab))
* **tensor:** return pool-allocated GPU memory to pool instead of cudaFree ([399baf9](https://github.com/zerfoo/zerfoo/commit/399baf9b5674126b791936a0f5b744c9a0ad8150))


### Reverts

* **generate:** remove LockOSThread (caused 2.6% regression) ([062fc99](https://github.com/zerfoo/zerfoo/commit/062fc992ee1502e1a311da1e3d65c7f6446392dc))
* **inference:** remove FusedAddRMSNorm from graph builder ([71fc187](https://github.com/zerfoo/zerfoo/commit/71fc18751cf454eb1a7b79214d7bf7bf328b642c))
* **kernels:** restore original Q4K GEMV kernel (128 threads, scalar loads) ([97443aa](https://github.com/zerfoo/zerfoo/commit/97443aaa6e695f663c245c8d33879980c284ffb2))

## Changelog — dndungu/zerfoo Fork

Summary of 215 commits implemented on the dndungu/zerfoo fork, organized by phase.
Original branch preserved at tag/branch `backup-main-before-rebase` (commit `19b5820`).
Common ancestor with zerfoo/zerfoo: commit `2093b01` (UINT8 tensor support).

---

## Phase 1 — Architecture & Cleanup

Refactored the codebase from a Numerai/audacity-specific tool into a generic ML framework.

- **Architecture analysis**: cross-package dependency analysis, API design, migration plan
- **Numerai removal**: removed era-specific training code, Numerai references from training/model/data
- **Generic CLI framework**: plugin-based CLI with predict/tokenize commands
- **Generic data types**: replaced Numerai-specific types with generic `data.Sample`/`data.Batch`
- **Model adapter stubs**: round-trip model adapter tests
- **Audacity removal**: removed all audacity references from CLI, training, integration tests
- **Lint & formatting**: resolved critical linting issues, normalized formatting across codebase

## Phase 2 — Core ML Fixes & Model Serialization

Fixed fundamental correctness issues in the ML pipeline.

- **Binary tensor serialization**: replaced lossy `serializeTensorData` with binary `EncodeTensor`
- **RoPE embeddings**: fixed backward pass shape derivation from `dOut` tensor
- **RMSNorm**: reshaped gain gradient to match parameter shape
- **GQA attention**: rewrote backward with correct head-replication reversal
- **Transformer Block**: implemented `Block.Backward` with gradient tests
- **HRM module integration**: implemented `graph.Node` interface on `HModule`/`LModule`
- **ZMF parameter restore**: fixed Linear, SimpleRNN, and S4 parameter restoration from ZMF

## Phase 3 — Test Coverage (0% → 95%+ across packages)

Massive test coverage improvement across the entire codebase:

| Package | Before | After |
|---------|--------|-------|
| internal/xblas | 0% | 100% |
| layers/registry | 0% | 100% |
| pkg/tokenizer | 0% | 100% |
| layers/tokenizers | 66.7% | 100% |
| tensor | 58.5% | 98.9% |
| numeric | — | 98.5% |
| graph | 66.3% | 97.0% |
| layers/transpose | 66.7% | 97.2% |
| layers/activations | 59.0% | 97.1% |
| layers/recurrent | 62.3% | 96.7% |
| cmd/cli | 27.9% | 96.5% |
| layers/core | 56.1% | 96.0% |
| layers/reducesum | 0% | 95.9% |
| training | 27.5% | 95.7% |
| layers/attention | 52.3% | 95.1% |
| model | 47.5% | 95.4% |
| layers/gather | 0% | 91.7% |
| layers/normalization | 42.3% | 69.7% |

Additional error-path tests for: optimizer, distributed, embeddings, transformer, loss, components, features, coordinator, HRM, compute, data, gather.

## Phase 4 — GPU Engine (CUDA)

Full CUDA GPU compute engine implementation:

- **CUDA runtime bindings**: CGO bindings for CUDA runtime (alloc, memcpy, stream, device)
- **Device management**: CUDA allocator, device registration
- **GPU tensor storage**: `GPUStorage[T]` with `ToGPU`/`ToCPU` transfer, `Storage[T]` interface
- **cuBLAS integration**: CGO bindings with `Sgemm`, `GPUEngine[T]` with cuBLAS MatMul
- **Native CUDA kernels**: elementwise ops, Softmax with shared-memory reduction, SumAxis reduction
- **Stream-based execution**: CUDA stream support, async memcpy, size-bucketed memory pool
- **Device-resident pipeline**: `NewWithStorage`, `NewGPUStorageFromPtr` for zero-copy GPU ops
- **GPU integration tests**: linear layer forward/backward, chained ops, mixed storage

## Phase 5 — Production Infrastructure

Enterprise-grade production readiness features:

- **Structured logging**: leveled logger with package-wide integration
- **Configuration**: generic JSON config loader with env overrides and validation; Engine/Training/Distributed configs
- **Health checks**: HTTP health server with liveness/readiness probes, pprof endpoints
- **Graceful shutdown**: `Closer` interface and orderly shutdown coordinator
- **Signal handling**: SIGINT/SIGTERM handling wired into CLI
- **Metrics instrumentation**: counters and duration histograms for CPUEngine and AllReduceStrategy
- **TLS/mTLS**: TLS config with server/client credentials for distributed communication
- **CI pipeline**: coverage gate (93% threshold), benchmark regression detection (10% threshold), race detector
- **Distributed training**: gRPC-based AllReduce/Barrier/Broadcast with input validation, worker node lifecycle
- **Deployment docs**: runbook with config reference, troubleshooting guide

## Phase 6 — Open Weights Model Import (ONNX/ZMF)

Support for importing and running open-weight models:

- **Extended tensor decoding**: BFLOAT16, INT32, INT64, FLOAT64, UINT8 in `DecodeTensor`
- **Constant node handling**: ZMF constant nodes and `Attribute_Tensor` in model builder
- **New layers**: Softmax, Erf, Sigmoid, LayerNormalization, Slice, Pad, TopK
- **Vision layers**: Conv2d, GlobalAveragePool, Resize, BatchNormalization
- **MoE layers**: MoEGate, MixtureOfExperts with error-path tests
- **Layer registry**: all new layers registered in central registry
- **Parity tests**: Gemma 3 and SigLIP/Kimi-VL forward pass parity tests

## Phase 7 — Architecture Cleanup

Final cleanup and consolidation:

- **Doc consolidation**: merged gpu.md, runbook.md, troubleshooting.md into design.md
- **Dead code removal**: deleted empty `pkg/prelude`, dead test files (parity, numerics, wire stubs)
- **Exported builders**: exported `BuildFFN`, removed `init()` registration
- **Thread safety**: added `sync.Mutex` to protect memo map in graph `Forward`/`Backward`
- **FFN registration**: added FFN to `RegisterAll`

## New Layers Added

- S4 (diagonal state space model)
- Dropout (inverted)
- Softmax, Erf, Sigmoid
- LayerNormalization
- Slice, Pad, TopK
- Conv2d, GlobalAveragePool, Resize
- BatchNormalization
- MoEGate, MixtureOfExperts
- FFN (feed-forward network)

## New Packages/Commands

- `log/` — structured leveled logger
- `config/` — generic config loader
- `health/` — HTTP health server
- `shutdown/` — graceful shutdown coordinator
- `device/` — CUDA device management
- `cmd/bench-compare/` — benchmark regression detection
- `cmd/coverage-gate/` — per-package coverage enforcement
- `distributed/` — gRPC AllReduce strategy, worker node, TLS
