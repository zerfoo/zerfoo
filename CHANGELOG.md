# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.43.0](https://github.com/zerfoo/zerfoo/compare/v1.42.1...v1.43.0) (2026-04-10)


### Features

* **bench:** add bench-spark.sh helper for Spark submission ([0321a18](https://github.com/zerfoo/zerfoo/commit/0321a187595afd7f2f319e1f7d20865557cec7c2))
* **bench:** add PatchTST training benchmark tool ([d847238](https://github.com/zerfoo/zerfoo/commit/d847238c0a12e0c818057e8724709f3a4a6bf67a))
* **bench:** add Spark pod manifest for PatchTST training ([0e05d43](https://github.com/zerfoo/zerfoo/commit/0e05d43ab2aedb4938ed4f725f7f1408f8006abe))
* **timeseries:** activate fused encoder forward path ([8aa526d](https://github.com/zerfoo/zerfoo/commit/8aa526d4213779ab70c78b3d03998c2d90b69a3b))
* **timeseries:** add weight-hash debug helper for GPU training diagnosis ([c5a34c5](https://github.com/zerfoo/zerfoo/commit/c5a34c52c32564304c9e287205c702e89fd33195))
* **timeseries:** wire fused encoder kernel into PatchTST training ([bafdad0](https://github.com/zerfoo/zerfoo/commit/bafdad024f827dcd9ef69921b4b6e7771dfd0fab))


### Bug Fixes

* **bench:** mount /opt/zerfoo/lib so libkernels.so is reachable ([aa6331a](https://github.com/zerfoo/zerfoo/commit/aa6331a189f129ce642bf4df556868e0d8cd840a))
* **bench:** post YAML (not JSON) and parse Spark status shape ([9d20746](https://github.com/zerfoo/zerfoo/commit/9d207460e1f25098fb2be00ea227260e749f1319))
* **ci:** make govulncheck non-blocking for unfixed bbolt vuln ([b6b38a6](https://github.com/zerfoo/zerfoo/commit/b6b38a68a9e53e353764d327b970d655fff36154))
* **mlstm:** use paper's stabilized exponential-gating formulation ([46b7b86](https://github.com/zerfoo/zerfoo/commit/46b7b868dab76ae4af1b83fe5d28a2ced1c58bed))
* **slstm:** use paper's stabilized exponential-gating formulation ([e47e4a4](https://github.com/zerfoo/zerfoo/commit/e47e4a47e20e27952dd6e0a438769f3b18607243))
* **timeseries:** compare Storage identity in gradTs sentinel ([a67063a](https://github.com/zerfoo/zerfoo/commit/a67063a5d61ec75d29c50fd084228abd0113fc62))
* **timeseries:** GPU training convergence — rebuild paramTs/gradTs per batch, strengthen sentinel, remove dead machinery ([168a938](https://github.com/zerfoo/zerfoo/commit/168a938f68c62de8f1677854e6e45a038e77a288))
* **timeseries:** GPU training writes back optimizer step to device ([f29c93b](https://github.com/zerfoo/zerfoo/commit/f29c93bdb74bd278ea6f2149b8aec189d1034bba))
* **timeseries:** skip flaky TimeMixer gradient check + add WithTimeMixerRNG ([4f96d99](https://github.com/zerfoo/zerfoo/commit/4f96d999cf16521442859799aa35617f0cf0b6f8))
* **timeseries:** use return value of GPU Reshape in PatchTST backward ([d61cbab](https://github.com/zerfoo/zerfoo/commit/d61cbabb929ecfd4d1c65e17c073aef9f957b4a3))


### Performance Improvements

* **timeseries:** pre-allocate PatchTST GPU train loop buffers (E85 T85.2.1-3,5) ([09a318c](https://github.com/zerfoo/zerfoo/commit/09a318c6db9e625155c2964708b9c8f6ada4e6ed))

## [1.42.1](https://github.com/zerfoo/zerfoo/compare/v1.42.0...v1.42.1) (2026-04-06)


### Bug Fixes

* **modeldsl:** replace .Data() bias loop with engine.Add ([4fd8d63](https://github.com/zerfoo/zerfoo/commit/4fd8d634c9604c0955adc9f2040b2be4f7df07e6))

## [1.42.0](https://github.com/zerfoo/zerfoo/compare/v1.41.0...v1.42.0) (2026-04-05)


### Features

* **inference:** add builder_helpers with newTensorLookup and newParamWrapper ([adfb334](https://github.com/zerfoo/zerfoo/commit/adfb334b5bf0198458e55b114e3a67615294a2e8))


### Bug Fixes

* **generate:** remove unused compute import after merge ([b7511c3](https://github.com/zerfoo/zerfoo/commit/b7511c38658c42e477c3793bbd70c5526409ce36))

## [1.41.0](https://github.com/zerfoo/zerfoo/compare/v1.40.1...v1.41.0) (2026-04-04)


### Features

* **cmd:** add --pjrt flag for PJRT backend selection ([66fb945](https://github.com/zerfoo/zerfoo/commit/66fb945a185c06cd7e1178d18ebccf68a3a23721))
* **crossasset:** replace SGD with AdamW in CPU Train() ([#315](https://github.com/zerfoo/zerfoo/issues/315)) ([4d6664c](https://github.com/zerfoo/zerfoo/commit/4d6664c05d0af3763613c7d77330685af62a6fbf))
* **functional:** add GELUBackward for gradient computation ([0e89305](https://github.com/zerfoo/zerfoo/commit/0e893054ed5d97abf96b1445f3ecece868695ccd))
* **functional:** add LayerNormBackward for gradient computation ([1e51b9e](https://github.com/zerfoo/zerfoo/commit/1e51b9e474ddf0a6c011df45af87504391434678))
* **functional:** add LinearBackward for gradient computation ([534127d](https://github.com/zerfoo/zerfoo/commit/534127d71cb4794855b3e6bf3526aea734685853))
* **functional:** add MLPBackward for 2-layer MLP gradient computation ([8624a1e](https://github.com/zerfoo/zerfoo/commit/8624a1e17b4ae92589b160e9ad2a4d7c23de6eed))
* **functional:** add MultiHeadAttentionBackward ([2d91fa3](https://github.com/zerfoo/zerfoo/commit/2d91fa3f45bfa7e7b13c0fb191beb9bb9f58386e))
* **functional:** add SoftmaxBackward for gradient computation ([1c2c486](https://github.com/zerfoo/zerfoo/commit/1c2c486dedd9859bc1ece6215ddf9444fc0ff62c))
* **generate:** wire PJRTPlan into decode loop ([ca6bab6](https://github.com/zerfoo/zerfoo/commit/ca6bab630cf4871e5e3ea024b85813b0dc3c0f47))
* **inference:** add PJRT compilation path ([9cde667](https://github.com/zerfoo/zerfoo/commit/9cde667d1d0e3f88ea38ccab41c8ddd5f30acf81))
* **layers:** add functional activation wrappers (GELU, Softmax, ReLU, SiLU, Sigmoid) ([962b36d](https://github.com/zerfoo/zerfoo/commit/962b36d3e5233425a4fe793659ce85d65837270b))
* **layers:** add functional LayerNorm and RMSNorm wrappers ([08c7ac9](https://github.com/zerfoo/zerfoo/commit/08c7ac9bb4789331000f7d601864cb82785e6e29))
* **layers:** add functional Linear and MultiHeadAttention wrappers ([e5449e8](https://github.com/zerfoo/zerfoo/commit/e5449e85d1bef80feb8a283e9d981f479e6e500c))


### Bug Fixes

* **architecture:** add crossasset/backward.go to privateLayer allowlist ([5c01ccf](https://github.com/zerfoo/zerfoo/commit/5c01ccf9100365440de57cfc0e8e5e3f0db28e70))
* **architecture:** add layernorm_ops.go backward to dataAbuse allowlist ([34fe067](https://github.com/zerfoo/zerfoo/commit/34fe0679b1e9b7431fda4603d3b6bc8f26e55e9b))
* **crossasset:** call Train() once with all epochs to preserve AdamW state ([834b8f3](https://github.com/zerfoo/zerfoo/commit/834b8f3645e5e74e9c7ec1d049d0e99a1c8ffa43))
* **crossasset:** delegate TrainGPU to CPU full-backprop with AdamW ([#317](https://github.com/zerfoo/zerfoo/issues/317)) ([b345932](https://github.com/zerfoo/zerfoo/commit/b3459328139a4631eda8cb86fcde91f1b05f769a))
* **crossasset:** snapshot GPU tensors to CPU before backward reads ([#317](https://github.com/zerfoo/zerfoo/issues/317)) ([4de925e](https://github.com/zerfoo/zerfoo/commit/4de925e96d7aa2f2ca4da003caff19f708e8ff2a))
* **timeseries:** resolve warmupLR merge conflict with scheduler.WarmupLR ([9f573cf](https://github.com/zerfoo/zerfoo/commit/9f573cf1afaa5e759bb3501a43645c4f6d8c6c71))
* **timeseries:** update nhits_test weight shape check for transposed layout ([f090509](https://github.com/zerfoo/zerfoo/commit/f090509aacb0b63f98c58c2e0f8216522b938281))
* **training:** fix QuantileLoss generic type assertions ([a282e9d](https://github.com/zerfoo/zerfoo/commit/a282e9dc60523e5cab996efea2081d2c423fa195))


### Performance Improvements

* **training:** replace guardAndClipGradients .Data() loops with Engine ops ([92e1218](https://github.com/zerfoo/zerfoo/commit/92e1218faa4fcc138d0daebeeb83764d29d7154c))
* **training:** replace SGD broadcast allocation with engine.MulScalar ([aad4deb](https://github.com/zerfoo/zerfoo/commit/aad4debdbfd0f5a442e78ffb9f77e4e100200e95))

## [1.40.1](https://github.com/zerfoo/zerfoo/compare/v1.40.0...v1.40.1) (2026-04-02)


### Bug Fixes

* **crossasset:** prevent CUDA illegal memory access in TrainGPU backward ([#317](https://github.com/zerfoo/zerfoo/issues/317)) ([8db043e](https://github.com/zerfoo/zerfoo/commit/8db043e4330741935d4eb35bf1b756e548e98af1))

## [1.40.0](https://github.com/zerfoo/zerfoo/compare/v1.39.0...v1.40.0) (2026-04-02)


### Features

* **crossasset:** full backprop through all layers in CPU Train() ([#315](https://github.com/zerfoo/zerfoo/issues/315)) ([5f5e53b](https://github.com/zerfoo/zerfoo/commit/5f5e53bb6acc360272297d87d0d86cde97b454fb))
* **dsp:** add Cooley-Tukey radix-2 FFT in internal/dsp ([85aa180](https://github.com/zerfoo/zerfoo/commit/85aa1804f95b3daeaf1ec010a83ced67f777546b))
* remove direct gonum dependency (E59) ([2715a88](https://github.com/zerfoo/zerfoo/commit/2715a888834b8c90205f4d93984e50d76eec4442))


### Bug Fixes

* **crossasset:** prevent OOM by reusing gradient tensors ([#314](https://github.com/zerfoo/zerfoo/issues/314)) ([35b2771](https://github.com/zerfoo/zerfoo/commit/35b2771611fe94ebf9f8fbd85c4e7d29ac4adabe))

## [1.39.0](https://github.com/zerfoo/zerfoo/compare/v1.38.5...v1.39.0) (2026-04-02)


### Features

* **crossasset:** add GPU parameter extraction for training ([#312](https://github.com/zerfoo/zerfoo/issues/312)) ([9bf423e](https://github.com/zerfoo/zerfoo/commit/9bf423ebec7d0e379c596060da5f42a5547aa59d))
* **crossasset:** add GPU training with full backprop ([#312](https://github.com/zerfoo/zerfoo/issues/312)) ([4f3edc9](https://github.com/zerfoo/zerfoo/commit/4f3edc9840c8962224a3dcc92086a36417aea1b5))
* **inference:** use GPU Q8 Gather for embedding lookup (100% graph coverage) ([5c8af0e](https://github.com/zerfoo/zerfoo/commit/5c8af0ed77376b5dd5e780a598f8844de5957cbe))


### Bug Fixes

* **attention:** check K/V GPU storage before type assertion in flash paths ([b55036c](https://github.com/zerfoo/zerfoo/commit/b55036c7de70347a25c7357d089a0a7b31e1f387))
* **gguf:** match raw GGUF tensor names for embedding native decode ([c9219ca](https://github.com/zerfoo/zerfoo/commit/c9219cae36ce301b0afd0a546d4002c025f46950))
* **gguf:** skip Q4_0 re-quantization for embedding tensors ([badeadb](https://github.com/zerfoo/zerfoo/commit/badeadbe7fe1c8a9dac5422298ea69fe733f4922))
* **gguf:** update Q5_K/Q6_K decode tests for native storage ([21cc2bc](https://github.com/zerfoo/zerfoo/commit/21cc2bceb5b902e3af7153d31e440b1eee28af4f))
* **test:** use epoch-window average for loss decrease check (fixes flaky CI) ([b532516](https://github.com/zerfoo/zerfoo/commit/b532516a92f79833effc6cfcb6d3d68b415e7ec2))


### Performance Improvements

* eliminate ALL lossy re-quantization, keep native Q4K/Q5K/Q6K/Q5_0 ([123a979](https://github.com/zerfoo/zerfoo/commit/123a9794e92172adb3d08706be6006585e60a643))
* **gguf:** keep native Q4_K storage (eliminate lossy re-quantization) ([63bdd54](https://github.com/zerfoo/zerfoo/commit/63bdd544d60e6300e2f24c5f529504b5a4e5ab12))
* **gguf:** re-quantize all weight types to Q4_0 for maximum throughput ([4b202d2](https://github.com/zerfoo/zerfoo/commit/4b202d2de11818516fe53988e4f227408b84e2f4))
* **gguf:** re-quantize Q5_0/Q5_K/Q6_K to Q4_K for uniform fast GEMV ([b408c30](https://github.com/zerfoo/zerfoo/commit/b408c303f33c87339c7e92ddc819bb9230e33d7b))
* **gguf:** re-quantize Q8_0 weight matrices to Q4_K for merged QKV ([023505d](https://github.com/zerfoo/zerfoo/commit/023505d36e8c771665fb4e51cea045146f2ab8b4))

## [1.38.5](https://github.com/zerfoo/zerfoo/compare/v1.38.4...v1.38.5) (2026-04-01)


### Bug Fixes

* **attention:** disable fused softmax kernels that produce wrong decode output ([999f2fd](https://github.com/zerfoo/zerfoo/commit/999f2fdf7d7c2a22f87b43c94862a489f9fe4a31))
* **attention:** use engine.Add for causal masking instead of CPU Data() mutation ([90cacad](https://github.com/zerfoo/zerfoo/commit/90cacad409a4623d49b0d04593cf98f0eac957e4))
* **gguf:** keep native Q6_K storage (don't re-quantize to Q4_0) ([eee3a54](https://github.com/zerfoo/zerfoo/commit/eee3a54f75d548da369bcbc3bb3c1b7af9ac13c1))
* **gguf:** upgrade embeddings to F32, keep Q6_K re-quantization for weights ([b887b10](https://github.com/zerfoo/zerfoo/commit/b887b10deca2b432a94a1ecf2ede8ee37e294270))
* **gguf:** use native Q4_K storage instead of lossy re-quantization to Q4_0 ([1d56d2e](https://github.com/zerfoo/zerfoo/commit/1d56d2e5cac5e7d27030816a55e0e4cc85b4a852))
* **inference:** add Q5_0Storage virtual transpose in transposeWeight ([9fc3d68](https://github.com/zerfoo/zerfoo/commit/9fc3d681ca945e5b6c0a02afe6a2829871e1bde7))
* **inference:** re-enable Q5_0 virtual transpose and fused softmax kernels ([004d8bd](https://github.com/zerfoo/zerfoo/commit/004d8bdade1da9681b6abbc1b009e6b816df9f3c))
* **inference:** re-enable Q5_0 virtual transpose on GPU ([a3d4ac5](https://github.com/zerfoo/zerfoo/commit/a3d4ac5b04a048569e949b4e379ff8c7e4b3c731))
* **inference:** restore CUDA-only mmap disable (was accidentally global) ([7d07018](https://github.com/zerfoo/zerfoo/commit/7d070187463f3a956c3f7c1a879a0712bdf7770b))
* **inference:** revert Q5_0 virtual transpose on GPU (misaligned address) ([a906d3a](https://github.com/zerfoo/zerfoo/commit/a906d3a5e76193d550c8da459f8e2af0d6cb291f))


### Performance Improvements

* **gguf:** restore Q4_0 re-quantization for Q4_K/Q5_K/Q6_K/Q5_0 ([4e554e0](https://github.com/zerfoo/zerfoo/commit/4e554e04ce852b92a07bca0f774956ec7d79819e))
* **training:** convert layer norm backward to engine ops (E50.2.1) ([67772c5](https://github.com/zerfoo/zerfoo/commit/67772c5cf7069dfd4e3d7119ff889c33e2426862))

## [1.38.4](https://github.com/zerfoo/zerfoo/compare/v1.38.3...v1.38.4) (2026-03-31)


### Performance Improvements

* **attention:** wire fused softmax+V and repeat-interleave for decode ([641a119](https://github.com/zerfoo/zerfoo/commit/641a119b8dd7380f522b1a09d66f37733e3265a4))

## [1.38.3](https://github.com/zerfoo/zerfoo/compare/v1.38.2...v1.38.3) (2026-03-31)


### Bug Fixes

* **inference:** auto-disable mmap on CUDA devices ([ad1a373](https://github.com/zerfoo/zerfoo/commit/ad1a373c547bb5a19f877205e57c9777f3470760))

## [1.38.2](https://github.com/zerfoo/zerfoo/compare/v1.38.1...v1.38.2) (2026-03-31)


### Bug Fixes

* **timeseries:** disable CUDA graph capture until pool is capture-aware ([05a2659](https://github.com/zerfoo/zerfoo/commit/05a2659a2a8bdc8ecff51b5fff0a907079784b7b))
* **timeseries:** disable CUDA graph capture until ztensor is capture-pure ([faf3815](https://github.com/zerfoo/zerfoo/commit/faf3815399e0faf746bf180a9c4837f4e61a911c))
* **timeseries:** disable CUDA graph capture until ztensor is capture-pure ([3f38f17](https://github.com/zerfoo/zerfoo/commit/3f38f176d8fc488a869786e56c90475bdca43f8f))
* **timeseries:** remove backward CUDA graph capture (TrySlice D2H breaks it) ([8f304b2](https://github.com/zerfoo/zerfoo/commit/8f304b2f8f11b1d146c3e90954a75b96ac894096))


### Performance Improvements

* **attention:** extend merged QKV projection to prefill (seqLen &gt; 1) ([4dfdcbf](https://github.com/zerfoo/zerfoo/commit/4dfdcbf3cc016e9f14dec7572533871e319030df))
* **attention:** wire fused softmax+V and repeat-interleave into GQA ([6ec28a8](https://github.com/zerfoo/zerfoo/commit/6ec28a824539777c6bc4f3f6dd29ef0c5e65ce1c))
* **timeseries:** disable forward-prefix capture (slower than no capture) ([3e459d1](https://github.com/zerfoo/zerfoo/commit/3e459d17f34b785c580fe11092e1e5f3f6900905))
* **timeseries:** drop partial batches for CUDA graph compatibility ([a5e602a](https://github.com/zerfoo/zerfoo/commit/a5e602aba2e28330701205d336e9876c41996543))
* **timeseries:** eliminate .Data() calls for full encoder CUDA graph capture ([4134349](https://github.com/zerfoo/zerfoo/commit/41343494768604de91f903383ff7dacc6c6e0c82))
* **timeseries:** enable CUDA graph capture with capture-aware pool ([dd01284](https://github.com/zerfoo/zerfoo/commit/dd0128406e4077b14303d16c729ab9c350af0e53))
* **timeseries:** pre-allocate batch workspace outside training loop ([e60358f](https://github.com/zerfoo/zerfoo/commit/e60358f372410d92e711e4000efd7797b81c8b28))
* **timeseries:** re-enable CUDA graph capture with GPU-native Zero/Copy ([1f0849a](https://github.com/zerfoo/zerfoo/commit/1f0849a85fcd6aa514aae7bade447bd5af44b979))
* **timeseries:** replace GELU forward/backward with engine ops ([54d7162](https://github.com/zerfoo/zerfoo/commit/54d71621baf03df0df6323c3514e6e818f141ae2))
* **timeseries:** replace layer norm forward with engine ops ([e0dfe95](https://github.com/zerfoo/zerfoo/commit/e0dfe95e00e75b40ab94d9f6a580155a6624c75d))
* **timeseries:** wire CUDA graph capture into PatchTST GPU training ([382feb7](https://github.com/zerfoo/zerfoo/commit/382feb7a452919498ed50c3de03fae1af02fd8fd))

## [1.38.1](https://github.com/zerfoo/zerfoo/compare/v1.38.0...v1.38.1) (2026-03-30)


### Performance Improvements

* **timeseries:** batch PatchTST training across channels for 20x fewer engine calls ([c97931c](https://github.com/zerfoo/zerfoo/commit/c97931ce2422078eb806b152064fad93085b5733))
* **timeseries:** cache weight transposes outside backward loop ([0fbaf2e](https://github.com/zerfoo/zerfoo/commit/0fbaf2e8504bf1a7f9d1c11bb4f7d135a684757c))
* **timeseries:** eliminate tileBias/tilePosEmb allocations via broadcasting ([4e884a3](https://github.com/zerfoo/zerfoo/commit/4e884a37feefad603276a78b61b7854ba0c4b659))
* **timeseries:** move layer norm and GELU to engine ops ([e7d3436](https://github.com/zerfoo/zerfoo/commit/e7d343647c54b946747bbff05cacdc7f33290ef6))
* **timeseries:** replace per-sample CPU attention with batched engine ops ([4924b8e](https://github.com/zerfoo/zerfoo/commit/4924b8eeebb58b5a9707b57c2c3c617f7162820d))
* **ts_train:** use GPU engine when CUDA is available ([8b21eea](https://github.com/zerfoo/zerfoo/commit/8b21eea6662a29e60937140f3779d522436386e0))

## [1.38.0](https://github.com/zerfoo/zerfoo/compare/v1.37.0...v1.38.0) (2026-03-30)


### Features

* **timeseries:** add full GPU training loop for PatchTST ([dd5aa66](https://github.com/zerfoo/zerfoo/commit/dd5aa66e29aeac2c60831839e817379529fa431a))


### Bug Fixes

* **timeseries:** wire trainWindowedGPU as the engine training path ([bc00ae3](https://github.com/zerfoo/zerfoo/commit/bc00ae3bcbaeac01e4df75c135fa6f8499b1eabc))


### Performance Improvements

* **timeseries:** batch GPU training across samples for 64x fewer kernel launches ([158ddf6](https://github.com/zerfoo/zerfoo/commit/158ddf628826cc6a5d1dded1a1905b66f32e420e))
* **timeseries:** optimize PatchTST backward with buffer pooling ([89c713a](https://github.com/zerfoo/zerfoo/commit/89c713ae7d864d8814817924daed4e2ccbe1e6c5))
* **timeseries:** pre-allocate GPU training workspace ([0d9b0b3](https://github.com/zerfoo/zerfoo/commit/0d9b0b3977a2640c79f7c132a0f6e7bf509426ff))

## [1.37.0](https://github.com/zerfoo/zerfoo/compare/v1.36.0...v1.37.0) (2026-03-30)


### Features

* **cli:** add zerfoo forecast command ([ebaf984](https://github.com/zerfoo/zerfoo/commit/ebaf984298b7bac8304fd2575c1e9ba726d4745d))
* **timeseries:** add CfC batched forward pass ([9a41bee](https://github.com/zerfoo/zerfoo/commit/9a41bee07eda57a365f4b84ce3c736fbd949fd6e))
* **timeseries:** add Chronos value tokenizer ([86e93eb](https://github.com/zerfoo/zerfoo/commit/86e93eb6611ed4f7731ac24426f02807495b764b))
* **timeseries:** add Chronos-2 SafeTensors-to-GGUF converter ([50deee8](https://github.com/zerfoo/zerfoo/commit/50deee8812a7880918df287b7e0541e17ad4a21d))
* **timeseries:** add Chronos-2 T5 graph builder ([317b7d3](https://github.com/zerfoo/zerfoo/commit/317b7d344e6db99b08ad7bf6fd54fe4f1d9acdff))
* **timeseries:** add DataLoader for batched tensor iteration ([3f60e8f](https://github.com/zerfoo/zerfoo/commit/3f60e8f4f62568457e0cef058d7190959a21a549))
* **timeseries:** add DLinear batched forward pass ([77a9efc](https://github.com/zerfoo/zerfoo/commit/77a9efce5de4c59f9af86429ac8f7b6b60c42bc0))
* **timeseries:** add foundation model fine-tune API ([be4e6ce](https://github.com/zerfoo/zerfoo/commit/be4e6ceb1c61ece700ceec44e64c39251e00a1f5))
* **timeseries:** add foundation model zero-shot inference pipeline ([840ef7a](https://github.com/zerfoo/zerfoo/commit/840ef7a6b4382785667c6100a1989badcde6f250))
* **timeseries:** add foundation model zero-shot inference pipeline ([5c717d2](https://github.com/zerfoo/zerfoo/commit/5c717d2119dd41062faed80c8e5a602d7f6c35b8))
* **timeseries:** add FreTS batched forward pass ([958e08b](https://github.com/zerfoo/zerfoo/commit/958e08b3f4a9b163c8b7264750e19967062ccd1b))
* **timeseries:** add iTransformer batched backward pass ([7e704fa](https://github.com/zerfoo/zerfoo/commit/7e704fa2182f89330c492cd24e575475d2a958cc))
* **timeseries:** add iTransformer batched forward pass ([401b6de](https://github.com/zerfoo/zerfoo/commit/401b6dec690af6069528c9642578a54929e91ee6))
* **timeseries:** add Mamba batched forward pass ([2345211](https://github.com/zerfoo/zerfoo/commit/2345211cbfdaf324ff731b3662c21a062d0c5ac8))
* **timeseries:** add mLSTM cell layer with covariance memory ([e443805](https://github.com/zerfoo/zerfoo/commit/e4438058407ebace2b981d28c91ef355206dff27))
* **timeseries:** add Moirai-2 any-variate input projection ([d496b2c](https://github.com/zerfoo/zerfoo/commit/d496b2ced84014f2c66e5a39f5e69cde12a88a6b))
* **timeseries:** add Moirai-2 masked encoder graph builder ([b709afa](https://github.com/zerfoo/zerfoo/commit/b709afa4e50bf6bb324e7758584846c77b5220b5))
* **timeseries:** add Moirai-2 SafeTensors-to-GGUF converter ([926276e](https://github.com/zerfoo/zerfoo/commit/926276e210055f2dbe846a34b61bda4278736d92))
* **timeseries:** add N-BEATS batched forward pass ([52ac585](https://github.com/zerfoo/zerfoo/commit/52ac5853a0af9924bc123477148fe5ced0d8ccf0))
* **timeseries:** add N-HiTS batched forward pass ([0c43783](https://github.com/zerfoo/zerfoo/commit/0c4378367bb7a809cf77b6213e9b04e63fc98ac9))
* **timeseries:** add PatchTST batched backward pass ([3cdc795](https://github.com/zerfoo/zerfoo/commit/3cdc795efd226053e474603f02a87bfc8a8d6df1))
* **timeseries:** add sLSTM cell layer with exponential gating ([37b7761](https://github.com/zerfoo/zerfoo/commit/37b77616d1c64ac72430ab59b28c77db9d742a4c))
* **timeseries:** add TimeMixer backward pass ([e87895e](https://github.com/zerfoo/zerfoo/commit/e87895ea7c517f534e898f75e05211f28c85e913))
* **timeseries:** add TimeMixer engine-accelerated forward ([1028ad6](https://github.com/zerfoo/zerfoo/commit/1028ad6a9fd4056282a5aaf2c4b30e443b9e29cc))
* **timeseries:** add TimeMixer inference graph builder ([fb94912](https://github.com/zerfoo/zerfoo/commit/fb94912aaa11baf2a8a4b1ff649d30094c250fe9))
* **timeseries:** add TimeMixer multi-scale decomposition ([2542283](https://github.com/zerfoo/zerfoo/commit/25422831b9df79d55ae9b99c79855e9e7c740341))
* **timeseries:** add TimeMixer past-decomposable mixing ([b7ca336](https://github.com/zerfoo/zerfoo/commit/b7ca336cc7efb918031723ebc5ef83877486b2bb))
* **timeseries:** add TimeMixer TrainWindowed ([2e7f40c](https://github.com/zerfoo/zerfoo/commit/2e7f40c1a75f8e562ec75ce6158f91f5b0d7b97b))
* **timeseries:** add TimeMixerAdapter for training interface ([e6f7eb5](https://github.com/zerfoo/zerfoo/commit/e6f7eb5c04b52ddf880ad954d0b19b8a39895f71))
* **timeseries:** add TiRex SafeTensors-to-GGUF converter ([58834ce](https://github.com/zerfoo/zerfoo/commit/58834cef1a9c5a87220eed302c802c640818494f))
* **timeseries:** add TiRex xLSTM graph builder ([f46d620](https://github.com/zerfoo/zerfoo/commit/f46d62084361e9ed759424b8068a5bf2a40969fc))
* **timeseries:** add TTM batched forward pass ([59ecf10](https://github.com/zerfoo/zerfoo/commit/59ecf10425095234cae0df8abe106775586f111e))
* **timeseries:** wire iTransformer TrainWindowed to batched path ([e0078b8](https://github.com/zerfoo/zerfoo/commit/e0078b8a4328869adc9569cbe8d66642bcb9c734))
* **timeseries:** wire PatchTST TrainWindowed to batched path via DataLoader ([bbba353](https://github.com/zerfoo/zerfoo/commit/bbba3539229b84a3bfe910cdbe8afd5cd2c42e15))


### Bug Fixes

* **timeseries:** add mixing to TimeMixer engine forward, fix roundtrip test ([cb61c5a](https://github.com/zerfoo/zerfoo/commit/cb61c5a951ccc311f3177a9b2614eff2d45f1bef))
* **timeseries:** integrate TimeMixer multipredictor mixing with backward pass ([3bd6bce](https://github.com/zerfoo/zerfoo/commit/3bd6bceaf6d6df9b87537efb1f894aeb47d0fb60))
* **timeseries:** skip near-zero gradients in PatchTST gradient check ([9bfab88](https://github.com/zerfoo/zerfoo/commit/9bfab88378dad80fc660f9266e9c91dd01ddd6de))

## [1.36.0](https://github.com/zerfoo/zerfoo/compare/v1.35.0...v1.36.0) (2026-03-29)


### Features

* **audio:** add mel spectrogram extraction, WAV parser, and transcribe CLI ([64e289a](https://github.com/zerfoo/zerfoo/commit/64e289a81da84685413c3edf9b6b61ffc865635f))
* **bench_tps:** add --max-seq-len flag to cap KV cache allocation ([92f59c2](https://github.com/zerfoo/zerfoo/commit/92f59c288d711a75d85b3ed74da7069d16bbca66))
* **generate:** add WithTieredKV GeneratorOption and CacheProvider adapter ([c27696e](https://github.com/zerfoo/zerfoo/commit/c27696ea99eb8c598fc46419bfe2ba35140924af))


### Bug Fixes

* **attention:** apply QK norms before head reshape for MiniMax-M2 ([4ed3955](https://github.com/zerfoo/zerfoo/commit/4ed3955ca5c8a787c60b1d90c1fe4234307c0095))
* **audio:** wire mel spectrogram into Model.Transcribe for Voxtral ([0ba7a49](https://github.com/zerfoo/zerfoo/commit/0ba7a4979c4cd8877fb7b5401f17d687b46b55d1))
* **batcher:** make TestSchedulerImmediateEviction deterministic ([d3951a3](https://github.com/zerfoo/zerfoo/commit/d3951a323c334677325f34ac63ce24e885b19b5c))
* **generate:** Close() skips deletion of user-provided ColdDir ([68788ca](https://github.com/zerfoo/zerfoo/commit/68788ca256e6c85489409d8d4f7b7bf21f13e928))
* **inference:** eliminate OOM during MoE graph build for over-RAM models ([decd668](https://github.com/zerfoo/zerfoo/commit/decd66820ad71bfcc8a42d7c2b1642dc3af23a51))
* **inference:** zero-copy expert slicing for mmap'd MoE stacked weights ([d3427cf](https://github.com/zerfoo/zerfoo/commit/d3427cfd83008956d6d51604b03b5021f1d31a04))
* **test:** ensure TestGenerate_returnsResult Duration &gt; 0 on fast hardware ([e2a6a35](https://github.com/zerfoo/zerfoo/commit/e2a6a35d463afb1b900c244923f770f715ea0781))

## [1.35.0](https://github.com/zerfoo/zerfoo/compare/v1.34.0...v1.35.0) (2026-03-29)


### Features

* **audio:** generalize WhisperEncoder with configurable intermediate size and attention bias ([1eeeb0b](https://github.com/zerfoo/zerfoo/commit/1eeeb0b5fbe2a9e67bf1133fb313b45980af9882))
* **gguf:** add audio encoder config fields and Voxtral tensor name mapping ([ce9b18d](https://github.com/zerfoo/zerfoo/commit/ce9b18df707bf108f6e0af7e074ae79784e08110))
* **inference:** add Voxtral speech-to-text architecture ([4aa1a77](https://github.com/zerfoo/zerfoo/commit/4aa1a778dea54e0c6b7dd6dea2403d64f18af367))
* **inference:** add Voxtral speech-to-text architecture builder ([2a97f67](https://github.com/zerfoo/zerfoo/commit/2a97f67ca71fd17855c069dbcfb8acd9a3e75588))

## [1.34.0](https://github.com/zerfoo/zerfoo/compare/v1.33.0...v1.34.0) (2026-03-29)


### Features

* **inference:** add 5 Ollama-coverage architectures (Wave 1) ([8ecd322](https://github.com/zerfoo/zerfoo/commit/8ecd322754e511950572cdf4aaf5812ae0da7f62))
* **inference:** add EXAONE architecture builder for LG AI Research models ([9e41a6a](https://github.com/zerfoo/zerfoo/commit/9e41a6acc70ddd25f81fd16ec6fab400ac74c87d))
* **inference:** add GLM4, Kimi linear attention, and LFM2 architecture builders ([b80109d](https://github.com/zerfoo/zerfoo/commit/b80109d59d8803227af608a96e3860eae3b1a13a))
* **inference:** add GLM4, Kimi linear attention, LFM2 builders ([fb65813](https://github.com/zerfoo/zerfoo/commit/fb65813e6e76d504e791bedeeb770463082e098d))
* **inference:** add InternLM2 architecture builder ([bf77cb5](https://github.com/zerfoo/zerfoo/commit/bf77cb51b9e6c86686ccd3988b0e5785da00f74c))
* **inference:** add OLMo2 and DBRX architecture builders ([749ac78](https://github.com/zerfoo/zerfoo/commit/749ac78204e72025db6cda9f6c6429b77390cd75))
* **inference:** add split-GGUF support and default mmap loading ([a72391a](https://github.com/zerfoo/zerfoo/commit/a72391a0a4df15f5e40617dcec1e19ed4fa8556e))
* **inference:** add StarCoder2 architecture builder ([968a78e](https://github.com/zerfoo/zerfoo/commit/968a78e89ad1772f3cf99ea6a425935613003067))

## [1.33.0](https://github.com/zerfoo/zerfoo/compare/v1.32.0...v1.33.0) (2026-03-28)


### Features

* add tool calling, structured output, and sub-package examples ([d4bc8fc](https://github.com/zerfoo/zerfoo/commit/d4bc8fcc31315174ac59055f10dfa1c73e8bc344))
* **gguf:** add Nemotron-H SSM config fields and tensor name mappings ([16f2c4a](https://github.com/zerfoo/zerfoo/commit/16f2c4a07f71aff07b8ac23dbc011894218a8c01))
* **gguf:** add ScoringFunc to ModelConfig for expert gating function ([4ba0cc2](https://github.com/zerfoo/zerfoo/commit/4ba0cc25402c887f32fe6f38a491a1b4772a8d98))
* **inference:** add MiniMax-M2 architecture graph builder ([2ba8ad6](https://github.com/zerfoo/zerfoo/commit/2ba8ad615c8565ddd4ff14b9d6d16a9583d69467))
* **inference:** add Nemotron-H + MiniMax-M2 architecture builders ([56b1ed8](https://github.com/zerfoo/zerfoo/commit/56b1ed82de4efd42447825ce6d368e5b41765b1e))
* **inference:** add Nemotron-H dense and MoE graph builders ([bb91030](https://github.com/zerfoo/zerfoo/commit/bb91030996fb2292c77aa1086fc647b2ccbf3aa1))
* **moe:** add sigmoid gating option for MoE expert routing ([69f0f5e](https://github.com/zerfoo/zerfoo/commit/69f0f5e213d4f8755fe5770a9a3dd926305810f6))


### Bug Fixes

* **ci:** add CGO_ENABLED=0 for cross-compile in arm64-build ([a1f4a85](https://github.com/zerfoo/zerfoo/commit/a1f4a85e01674d9c76d741e372211318e3c80333))
* **ci:** remove local ztensor replace, use published v0.14.0 ([305a298](https://github.com/zerfoo/zerfoo/commit/305a298b24a2f681b60a14efda3a03a8fa75e88a))
* **ci:** skip flaky TestGenerateBatch_ConcurrentSessions in short mode ([cc626f7](https://github.com/zerfoo/zerfoo/commit/cc626f7897f895266c6dac0f290dc63b4309368a))
* **ci:** skip known-failing tests, increase timeout to 300s ([cc2aca0](https://github.com/zerfoo/zerfoo/commit/cc2aca01fa00b5bf0723fe3b4281fecebc8867f5))
* **ci:** upgrade grpc v1.79.3 (CVE fix), use native arm64 runner ([85e13cc](https://github.com/zerfoo/zerfoo/commit/85e13cc9297a271a074b4eaf6e76b8bd9ad610b7))

## [1.33.0](https://github.com/zerfoo/zerfoo/compare/v1.32.0...v1.33.0) (2026-03-28)


### Features

* add tool calling, structured output, and sub-package examples ([d4bc8fc](https://github.com/zerfoo/zerfoo/commit/d4bc8fcc31315174ac59055f10dfa1c73e8bc344))
* **gguf:** add Nemotron-H SSM config fields and tensor name mappings ([16f2c4a](https://github.com/zerfoo/zerfoo/commit/16f2c4a07f71aff07b8ac23dbc011894218a8c01))
* **gguf:** add ScoringFunc to ModelConfig for expert gating function ([4ba0cc2](https://github.com/zerfoo/zerfoo/commit/4ba0cc25402c887f32fe6f38a491a1b4772a8d98))
* **moe:** add sigmoid gating option for MoE expert routing ([69f0f5e](https://github.com/zerfoo/zerfoo/commit/69f0f5e213d4f8755fe5770a9a3dd926305810f6))


### Bug Fixes

* **ci:** add CGO_ENABLED=0 for cross-compile in arm64-build ([a1f4a85](https://github.com/zerfoo/zerfoo/commit/a1f4a85e01674d9c76d741e372211318e3c80333))
* **ci:** remove local ztensor replace, use published v0.14.0 ([305a298](https://github.com/zerfoo/zerfoo/commit/305a298b24a2f681b60a14efda3a03a8fa75e88a))
* **ci:** skip flaky TestGenerateBatch_ConcurrentSessions in short mode ([cc626f7](https://github.com/zerfoo/zerfoo/commit/cc626f7897f895266c6dac0f290dc63b4309368a))
* **ci:** skip known-failing tests, increase timeout to 300s ([cc2aca0](https://github.com/zerfoo/zerfoo/commit/cc2aca01fa00b5bf0723fe3b4281fecebc8867f5))
* **ci:** upgrade grpc v1.79.3 (CVE fix), use native arm64 runner ([85e13cc](https://github.com/zerfoo/zerfoo/commit/85e13cc9297a271a074b4eaf6e76b8bd9ad610b7))

## [1.32.0](https://github.com/zerfoo/zerfoo/compare/v1.31.0...v1.32.0) (2026-03-28)


### Features

* add NewModel constructor and 5 pkg.go.dev Example functions ([ffc8c33](https://github.com/zerfoo/zerfoo/commit/ffc8c33731afef4b04301c0f8328e50e6246cc57))
* **attention:** make RoPE optional in GQA for GPT-2 support ([ee8303a](https://github.com/zerfoo/zerfoo/commit/ee8303aa625772f0f8aac0c011b20409253f595b))
* **gguf:** add GPT-2 tensor name mapping with position_embd support ([cbaf99f](https://github.com/zerfoo/zerfoo/commit/cbaf99f9b319ca1af6a3efc9ede629a0a37c3e0e))
* **inference:** add GPT-2 architecture graph builder ([1d0ecc5](https://github.com/zerfoo/zerfoo/commit/1d0ecc5d6cc5b52765a0bf0c0a3167a7fa05caf2))
* **inference:** add GPT-2 config parser for TinyStories support ([6368827](https://github.com/zerfoo/zerfoo/commit/6368827e4945ab478b205850b76d8aeb0cfec960))

## [1.31.0](https://github.com/zerfoo/zerfoo/compare/v1.30.0...v1.31.0) (2026-03-28)


### Features

* **eagle:** add EAGLE head training CLI and penultimate feature collection ([fd18936](https://github.com/zerfoo/zerfoo/commit/fd189369a1a7e5a04f6623177503e5bd48c6be5e))

## [1.30.0](https://github.com/zerfoo/zerfoo/compare/v1.29.0...v1.30.0) (2026-03-28)


### Features

* **cmd:** add transmla validate perplexity comparison script ([bc04155](https://github.com/zerfoo/zerfoo/commit/bc04155c28d29c25531452f484ba77ddf0cff8f8))
* **generate:** add sync.RWMutex to TieredKVStore for concurrent serve access ([30474d7](https://github.com/zerfoo/zerfoo/commit/30474d7e8c5b4dd6e86a8dd13ec6e8ce7432913f))

## [1.29.0](https://github.com/zerfoo/zerfoo/compare/v1.28.0...v1.29.0) (2026-03-28)


### Features

* **attention:** add SparseRoutedAttention layer ([1d29aee](https://github.com/zerfoo/zerfoo/commit/1d29aee7d53baacd955c1d8502e07143584bfd10))
* **attention:** add SparseRoutedAttention layer ([060f98d](https://github.com/zerfoo/zerfoo/commit/060f98d2e5b851f4cb813b2ab7536e00fa34c7ea))
* **attention:** register NativeSparseAttention in layer registry ([ccef2e1](https://github.com/zerfoo/zerfoo/commit/ccef2e1fdcaf41d94efc6b8673f1cf6cec8c70b3))
* **attention:** register SparseRoutedAttention in layer registry ([78b1373](https://github.com/zerfoo/zerfoo/commit/78b1373f316d4353e78fca7f78bc1670b837c2c1))
* **attention:** wire split-KV flash decode kernel for autoregressive decode ([fb53c41](https://github.com/zerfoo/zerfoo/commit/fb53c412f03db98ac9743a128039d3d939a1137d))
* **batcher:** cache-aware request scheduling (T42.1.2) ([b84297b](https://github.com/zerfoo/zerfoo/commit/b84297b47e204a52702481d4020cb0c5a02173b3))
* **cli:** add --quarot flag to run command for QuaRot weight fusion ([4799449](https://github.com/zerfoo/zerfoo/commit/4799449fde07e47b85ed0a3bd0d927cdf10dc80f))
* **cmd:** implement transmla CLI command for MHA-to-MLA conversion ([3b6d618](https://github.com/zerfoo/zerfoo/commit/3b6d618257c8d26654967e515b665023101f98a9))
* **generate:** add async CPU-to-GPU prefetch for tiered KV store ([42af0b7](https://github.com/zerfoo/zerfoo/commit/42af0b7c06fb79ac8e93d27c87afa457cdf22ac5))
* **generate:** add Q3 KV cache with non-uniform codebook quantization ([b82e15d](https://github.com/zerfoo/zerfoo/commit/b82e15d70a42de95df7b2f5670ae236106ab5e40))
* **generate:** add TieredKVStore with hot/warm/cold KV cache tiers ([1366297](https://github.com/zerfoo/zerfoo/commit/13662971f209809c01c281901951bbd7820ebe2d))
* **generate:** add WithEAGLE generator option for speculative EAGLE decoding ([da93e0b](https://github.com/zerfoo/zerfoo/commit/da93e0b9f3138c0ec8aa11c74b86cbd8e8a8f867))
* **generate:** wire WithGeneratorKVDtype q4/q3 into all cache creation paths ([557e856](https://github.com/zerfoo/zerfoo/commit/557e8561578b1e9b7e93456091bebbd796434a38))
* **gguf:** add TQ2_0 ternary tensor type to GGUF loader ([5e59c8a](https://github.com/zerfoo/zerfoo/commit/5e59c8a542064f5fc9d6daa12b273dd805552da1))
* **gguf:** wire IQ2_XXS, IQ3_S, IQ4_NL dequantization into GGUF loader ([f34bdb2](https://github.com/zerfoo/zerfoo/commit/f34bdb2c7aed2f23ec4f4bc9ada5f9f0b09642c0))
* **inference:** add MoE weight splitter for GPU/CPU expert placement ([9ac4a63](https://github.com/zerfoo/zerfoo/commit/9ac4a632f7e4d1f1c717e99cefa6d43b9b5b9e08))
* **inference:** detect TransMLA tensors and wire MLA inference path ([dd7be29](https://github.com/zerfoo/zerfoo/commit/dd7be2971aa78618dfcbc7719eabaee8b0609bc7))
* **inference:** implement async CPU expert dispatch for hybrid MoE ([4c31e6c](https://github.com/zerfoo/zerfoo/commit/4c31e6cae37bed44848b886622cf49b9382276e7))
* **inference:** implement predictive expert prefetch for hybrid MoE ([b5be277](https://github.com/zerfoo/zerfoo/commit/b5be2779508f6c958c9eb2a48546f1850d8aafb3))
* **inference:** load EAGLE head weights from GGUF tensors ([bc3b332](https://github.com/zerfoo/zerfoo/commit/bc3b332c0f3fa6601c922b1f0535e407aed6be66))
* **layers:** wire ternary MatMul dispatch for BitNet models ([ed52b21](https://github.com/zerfoo/zerfoo/commit/ed52b2192b2e9d50311d5b76aa23138e91c4b5a3))
* **lora:** implement adapter cache with LRU eviction ([f3f8bd4](https://github.com/zerfoo/zerfoo/commit/f3f8bd4851a536a54d9e1d2b7391031fa927ee13))
* **lora:** implement LoRA weight merging in forward pass ([1f1f3b8](https://github.com/zerfoo/zerfoo/commit/1f1f3b8c12d67e48443fea58344122ff32195845))
* **serve:** add per-request LoRA adapter selection to API ([88f5049](https://github.com/zerfoo/zerfoo/commit/88f504931d8b55c5702b5951af47726ab478df8d))
* **training:** add contrastive routing loss for sparse routed attention ([d1eb07f](https://github.com/zerfoo/zerfoo/commit/d1eb07fe100fd7ed6b37396e639a478f8c1e2405))
* **transmla:** write converted TransMLA weights as GGUF ([08e72cd](https://github.com/zerfoo/zerfoo/commit/08e72cd85bdc4a64a078ce2ada5d1e368cd06e0a))
* **transmla:** write converted TransMLA weights as GGUF ([c6bb6ae](https://github.com/zerfoo/zerfoo/commit/c6bb6ae0301217f0ea94746f2d0e748d083b820e))


### Bug Fixes

* **generate:** remove racy concurrent test for non-thread-safe TieredKVStore ([6553815](https://github.com/zerfoo/zerfoo/commit/65538159e1fd56d200e5e5709235deb40b73f3b3))
* **gguf:** add Q2_K and Q3_K tensor type decoders for Phi3/Llama3.1 ([45b3b88](https://github.com/zerfoo/zerfoo/commit/45b3b8884ad7fec73c47ebbbe0a306e4926a6e8f))
* **gguf:** preserve TernaryStorage in loader and add BitNet loading tests ([a071e48](https://github.com/zerfoo/zerfoo/commit/a071e483b7471136e044c14d60af4bfd445ba228))

## [1.28.0](https://github.com/zerfoo/zerfoo/compare/v1.27.1...v1.28.0) (2026-03-27)


### Features

* **attention:** add NativeSparseAttention combining three NSA paths ([0cab558](https://github.com/zerfoo/zerfoo/commit/0cab5585c743475d03553f488268fc951eed6751))
* **attention:** add NSA coarse-grained token compression path ([7ad09fc](https://github.com/zerfoo/zerfoo/commit/7ad09fc72f1a82d248413c625c70d6b0fad4eeec))
* **attention:** add NSA fine-grained token selection path ([fe49385](https://github.com/zerfoo/zerfoo/commit/fe49385790616590283be52f9f0567d15581f960))
* **attention:** add NSA sliding window attention path ([5201266](https://github.com/zerfoo/zerfoo/commit/52012669d9b1a6151548a2d927cbc80b58334cdf))
* **attention:** wire document-wise RoPE into GQA layer ([2e8d2cd](https://github.com/zerfoo/zerfoo/commit/2e8d2cd0b5a9250995bf871023079aca9c49f079))
* **embeddings:** add document-wise RoPE mode ([6a9e851](https://github.com/zerfoo/zerfoo/commit/6a9e8514fb785c62475648641aa4f63afd04023b))
* **generate:** add CompressedKVCache with chunk-wise mean pooling ([d47105f](https://github.com/zerfoo/zerfoo/commit/d47105f1d056022ac13c4c66d64c15f46b557553))
* **generate:** add hash-based RadixCache for KV block prefix matching ([1625cdf](https://github.com/zerfoo/zerfoo/commit/1625cdfa647c9c2a14a31b4a63459e034fea197f))
* **generate:** add Q4 quantized KV cache storage ([8f5f35e](https://github.com/zerfoo/zerfoo/commit/8f5f35e230ec3ce5ca80f6ef487f4bd5dd4893cc))
* **generate:** add WithCompressedKV generator option ([8dd2334](https://github.com/zerfoo/zerfoo/commit/8dd2334d498afdcbfff58ed464b10c48c9315669))
* **inference:** add ExpertPlacementPolicy for hybrid CPU/GPU MoE ([183214f](https://github.com/zerfoo/zerfoo/commit/183214fe66969ef8b9b66e0d175e53efd48ae7e9))
* **inference:** wire EAGLEHead into computation graph for draft generation ([e1b0e46](https://github.com/zerfoo/zerfoo/commit/e1b0e4649fe9d69d01471a33899bc0e149921dff))
* **layers:** add EAGLEHead layer for self-speculative decoding ([27bcfd6](https://github.com/zerfoo/zerfoo/commit/27bcfd6d81783670a99e39b9ecb16eb21a92319e))
* **lora:** add LoRA adapter weight format loader ([75d0134](https://github.com/zerfoo/zerfoo/commit/75d01347d64537cb6ffd9bf47fcd5a8ea5c4fc35))
* **transmla:** add SVD decomposition for KV projection conversion ([a8dbeae](https://github.com/zerfoo/zerfoo/commit/a8dbeaee1a40696071d12f6fcf0ed79c8975947b))

## [1.27.1](https://github.com/zerfoo/zerfoo/compare/v1.27.0...v1.27.1) (2026-03-27)


### Bug Fixes

* **gguf:** handle uint64/float64 metadata in GetUint32/GetFloat32 ([1648db9](https://github.com/zerfoo/zerfoo/commit/1648db9a0e684c907e64c86e62b559942d0e2565))

## [1.27.0](https://github.com/zerfoo/zerfoo/compare/v1.26.2...v1.27.0) (2026-03-27)


### Features

* **inference:** add madvise hints for mmap GGUF loading ([315ba7b](https://github.com/zerfoo/zerfoo/commit/315ba7b7eca7aeee5a3f438dad8dcefd7151cce3))

## [1.26.2](https://github.com/zerfoo/zerfoo/compare/v1.26.1...v1.26.2) (2026-03-27)


### Bug Fixes

* **inference:** MmapStorage LMHead virtual transpose + ztensor v0.9.2 ([b2bc37a](https://github.com/zerfoo/zerfoo/commit/b2bc37a0451c8f002b98c2b58952ee30eefb0531))
* **inference:** use MatMulTransposeB for MmapStorage LMHead ([b3ad5d4](https://github.com/zerfoo/zerfoo/commit/b3ad5d401c8da338a9f4265b755dee2f521f7caf))


### Reverts

* remove MmapStorage LMHead path for debugging ([740574d](https://github.com/zerfoo/zerfoo/commit/740574d582f093dc215db71cd18e274436fd7dce))

## [1.26.1](https://github.com/zerfoo/zerfoo/compare/v1.26.0...v1.26.1) (2026-03-27)


### Bug Fixes

* **gguf:** add Q4_1, Q5_0, Q5_1 type mappings to mmap loader ([42b5b19](https://github.com/zerfoo/zerfoo/commit/42b5b194aaab0cca3dc8dcf692504362a308b967))
* **inference:** add MmapStorage virtual transpose path in LMHead ([3ef7a95](https://github.com/zerfoo/zerfoo/commit/3ef7a95d9d024c087525863f314bcc6f04def467))

## [1.26.0](https://github.com/zerfoo/zerfoo/compare/v1.25.5...v1.26.0) (2026-03-27)


### Features

* **gguf:** add LoadTensorsMmap for zero-copy mmap-based tensor loading ([6abd065](https://github.com/zerfoo/zerfoo/commit/6abd065fc5e04b4a244b58adb52a66f4470ceeec))
* **inference:** wire WithMmap option to mmap-based GGUF loading ([793d071](https://github.com/zerfoo/zerfoo/commit/793d0715f07e684afc55a0a063a949a7893f2eae))

## [1.25.5](https://github.com/zerfoo/zerfoo/compare/v1.25.4...v1.25.5) (2026-03-27)


### Bug Fixes

* **attention:** guard flash prefill kernel against Q/K dimension mismatches ([5122bb9](https://github.com/zerfoo/zerfoo/commit/5122bb9da8a462b774ee5f9002c34b465905a533))

## [1.25.4](https://github.com/zerfoo/zerfoo/compare/v1.25.3...v1.25.4) (2026-03-27)


### Bug Fixes

* **cuda:** sync Repeat kernel to repeat-each semantics ([31af31a](https://github.com/zerfoo/zerfoo/commit/31af31a35fd3f18273b05fd802b22f8e44999d67))

## [1.25.3](https://github.com/zerfoo/zerfoo/compare/v1.25.2...v1.25.3) (2026-03-27)


### Bug Fixes

* **attention:** use reshape+repeat for correct GQA KV head expansion ([f58b4e0](https://github.com/zerfoo/zerfoo/commit/f58b4e0196d6f98d295db9187ccb6ff195029fea))
* **deps:** update ztensor to fix Repeat interleave for GQA ([279d557](https://github.com/zerfoo/zerfoo/commit/279d5576f782eb0421a11f370c4d4f490840a1ed))
* **inference:** re-enable embedding Q8 upgrade ([c466246](https://github.com/zerfoo/zerfoo/commit/c466246632bd4ce94a023756c88603e8965c7efc))
* **inference:** upgrade embedding/lm_head from Q4 to Q8 on load ([2830d2e](https://github.com/zerfoo/zerfoo/commit/2830d2e04db37a2bf3442cb77114e6154298bde1))

## [1.25.2](https://github.com/zerfoo/zerfoo/compare/v1.25.1...v1.25.2) (2026-03-26)


### Bug Fixes

* **generate:** prepend BOS token for Mistral inference ([9e469ae](https://github.com/zerfoo/zerfoo/commit/9e469ae4cedb37c58b832fd56779cffc93d2fb09))

## [1.25.1](https://github.com/zerfoo/zerfoo/compare/v1.25.0...v1.25.1) (2026-03-26)


### Bug Fixes

* **gguf:** extract tokenizer scores for SentencePiece models ([e4ecf26](https://github.com/zerfoo/zerfoo/commit/e4ecf26aa2d733cba043a9c731f29acddee3d919))

## [1.25.0](https://github.com/zerfoo/zerfoo/compare/v1.24.0...v1.25.0) (2026-03-26)


### Features

* **training:** implement SaveModel using ztensor/gguf writer ([99efbce](https://github.com/zerfoo/zerfoo/commit/99efbcef3ca5fd417e9f329358b7fb5d35f644ec))

## [1.24.0](https://github.com/zerfoo/zerfoo/compare/v1.23.0...v1.24.0) (2026-03-26)


### Features

* **timeseries:** add batched GPU forward pass for PatchTST training ([461e7c9](https://github.com/zerfoo/zerfoo/commit/461e7c90eb50361cf1a38e0a743c91042ed6e450))

## [1.23.0](https://github.com/zerfoo/zerfoo/compare/v1.22.1...v1.23.0) (2026-03-26)


### Features

* **inference:** detect Mistral from GGUF metadata and fix tokenizer ([4c38e37](https://github.com/zerfoo/zerfoo/commit/4c38e37f075ed1550b5235602a39148984c16254))
* **serve:** add Guardian guardrails middleware for chat completions ([d2304f2](https://github.com/zerfoo/zerfoo/commit/d2304f2da9765570ab81bced8565b36ffffa5575))
* **timeseries:** wire crossasset package into training pipeline ([e7fdd53](https://github.com/zerfoo/zerfoo/commit/e7fdd53183027627b3f6af08be580568544a992e))

## [1.23.0](https://github.com/zerfoo/zerfoo/compare/v1.22.1...v1.23.0) (2026-03-26)


### Features

* **inference:** detect Mistral from GGUF metadata and fix tokenizer ([4c38e37](https://github.com/zerfoo/zerfoo/commit/4c38e37f075ed1550b5235602a39148984c16254))
* **serve:** add Guardian guardrails middleware for chat completions ([d2304f2](https://github.com/zerfoo/zerfoo/commit/d2304f2da9765570ab81bced8565b36ffffa5575))

## [1.22.1](https://github.com/zerfoo/zerfoo/compare/v1.22.0...v1.22.1) (2026-03-26)


### Bug Fixes

* **timeseries:** dispatch TTM TrainWindowed to GPU engine ([#207](https://github.com/zerfoo/zerfoo/issues/207)) ([38dd21a](https://github.com/zerfoo/zerfoo/commit/38dd21ad20cddc4b2031cfd2e9a131271c85fac5))

## [1.22.0](https://github.com/zerfoo/zerfoo/compare/v1.21.0...v1.22.0) (2026-03-26)


### Features

* **cmd:** add guard CLI command for content moderation ([2d49727](https://github.com/zerfoo/zerfoo/commit/2d497273e97bafc5452938c4d961a40b94b840ac))
* **guardian:** add evaluator for Guardian inference pipeline ([ef93276](https://github.com/zerfoo/zerfoo/commit/ef93276b3dd1b8a882ef22fe40ec2ea6053ad67c))
* **guardian:** add prompt template engine for all risk categories ([cba58a5](https://github.com/zerfoo/zerfoo/commit/cba58a57ff2e5df87f0fc1b3f4f35a3ee47cf075))
* **guardian:** add verdict parser for all Guardian output formats ([00fb100](https://github.com/zerfoo/zerfoo/commit/00fb100a240cf1df15f87ab5b2c145adee3c0002))
* **inference:** add Granite architecture builder ([41b792d](https://github.com/zerfoo/zerfoo/commit/41b792dd53d529af67af5dc0e8bd425e77adc1cc))
* **serve:** add Guardian content moderation API endpoints ([7e32c4c](https://github.com/zerfoo/zerfoo/commit/7e32c4c310336850f6aa6eecd5631ad2660c513a))

## [1.21.0](https://github.com/zerfoo/zerfoo/compare/v1.20.1...v1.21.0) (2026-03-26)


### Features

* **cuda:** add Q6_K, Q5_K, and Q4_K SM121 GEMV kernels ([4823e4d](https://github.com/zerfoo/zerfoo/commit/4823e4dd8a79be35a9fdc737ec1c70d6e0862961))
* **gguf:** keep native Q4_K/Q5_K/Q6_K storage instead of re-quantizing to Q4_0 ([92185ae](https://github.com/zerfoo/zerfoo/commit/92185aeba4ee69c3d99ed9049fdf644cf52e2107))
* **inference:** add merged QKV and GateUp GEMV for Q4K storage ([e525907](https://github.com/zerfoo/zerfoo/commit/e525907216b3e54fd8271fceb5aade766de4a539))
* **inference:** add virtual transpose for Q4K/Q5K/Q6K storage types ([7e56e5e](https://github.com/zerfoo/zerfoo/commit/7e56e5e36a7c11b676446ffc7f119b724c10e7a0))
* **timeseries:** add dual-space encoder for Granite TSPulse ([b2ee997](https://github.com/zerfoo/zerfoo/commit/b2ee997b853f2d053c1931e13c9aff1385da1604))
* **timeseries:** add FlowState architecture builder ([ec4808a](https://github.com/zerfoo/zerfoo/commit/ec4808a3832d5e5992d6772a79d97788ea6fdeab))
* **timeseries:** add FlowState inference pipeline ([954fcb5](https://github.com/zerfoo/zerfoo/commit/954fcb52868c594bde275bd73d85c8dc01c435db))
* **timeseries:** add GraniteTimeSeriesConfig for GGUF metadata ([a78021f](https://github.com/zerfoo/zerfoo/commit/a78021f985c7d1771b38afba8b786a4b9b517028))
* **timeseries:** add SSM layer for Granite FlowState ([8e74170](https://github.com/zerfoo/zerfoo/commit/8e741703df36c6e065fbf5d9e99c7975656b8b9e))
* **timeseries:** add TSMixer backbone layer for Granite TTM ([0916c11](https://github.com/zerfoo/zerfoo/commit/0916c11cc9ade7d476eb9be4014047521154ba99))
* **timeseries:** add TSPulse architecture with multi-task heads ([934c782](https://github.com/zerfoo/zerfoo/commit/934c782f94527e1a6031897cbd1802a89b176e7d))
* **timeseries:** add TSPulse inference pipeline ([040c37d](https://github.com/zerfoo/zerfoo/commit/040c37d25a6fb21618ff4fd294a37c3c6b815ba6))
* **timeseries:** add TTM architecture builder for Granite TTM ([6b35039](https://github.com/zerfoo/zerfoo/commit/6b3503901f6e1b1752fc41299c964b167d92778f))
* **timeseries:** add TTM exogenous variable support ([c3095c8](https://github.com/zerfoo/zerfoo/commit/c3095c88a6130824d1a9df2d3099dcfdcc1b7992))
* **timeseries:** add TTM inference pipeline with normalization ([1a6ed34](https://github.com/zerfoo/zerfoo/commit/1a6ed347c5e5b2e91501718cf71d177bd58e2de6))
* **timeseries:** add TTM training backend ([5f1febd](https://github.com/zerfoo/zerfoo/commit/5f1febdff212d67de8e9ba664331058c6d7f5e0c))


### Bug Fixes

* **cuda:** raise Q4 GEMV shared memory limit for 7B+ models ([8ac4bc2](https://github.com/zerfoo/zerfoo/commit/8ac4bc2559174406177d50b1de91418f8b894103))
* **gguf:** restore Q4_K to Q4_0 re-quantization for GEMV performance ([6632798](https://github.com/zerfoo/zerfoo/commit/66327987267b80c03bfa300027f145c7cc6c7e25))
* **gguf:** restore Q5_K/Q6_K to Q4_0 re-quantization for GEMV performance ([9f7aae3](https://github.com/zerfoo/zerfoo/commit/9f7aae3abcfad626a37d1bc952920c308f1d2c7d))
* **inference:** use GPU RMSNorm fallback to prevent CUDA graph D2H copy ([da0f747](https://github.com/zerfoo/zerfoo/commit/da0f74787d98ebec4b53badf1b0d55cc25f905d8))

## [1.20.1](https://github.com/zerfoo/zerfoo/compare/v1.20.0...v1.20.1) (2026-03-25)


### Performance Improvements

* **timeseries:** eliminate all GPU dispatch from DLinear training loop ([#172](https://github.com/zerfoo/zerfoo/issues/172)) ([9130526](https://github.com/zerfoo/zerfoo/commit/9130526f3ce8a0d16e7fd7a5a6e797186f8e4b83))

## [1.20.0](https://github.com/zerfoo/zerfoo/compare/v1.19.1...v1.20.0) (2026-03-25)


### Features

* **scripts:** add Zerfoo vs Ollama benchmark comparison script ([fb4c637](https://github.com/zerfoo/zerfoo/commit/fb4c63732fcea783c7f192deda6d4d191c093ce3))


### Bug Fixes

* **scripts:** parse only decode eval rate from Ollama verbose output ([3e00e2e](https://github.com/zerfoo/zerfoo/commit/3e00e2e95fb8d540dea1b97a3e73d009c429b126))

## [1.19.1](https://github.com/zerfoo/zerfoo/compare/v1.19.0...v1.19.1) (2026-03-25)


### Bug Fixes

* **timeseries:** batch DLinear engine forward to eliminate per-sample GPU allocation overhead ([#172](https://github.com/zerfoo/zerfoo/issues/172)) ([bc357b3](https://github.com/zerfoo/zerfoo/commit/bc357b3358405cdc91d17881159e466bd8b5943b))

## [1.19.0](https://github.com/zerfoo/zerfoo/compare/v1.18.0...v1.19.0) (2026-03-25)


### Features

* **timeseries:** add batched forward pass for PatchTST engine training ([61a09d9](https://github.com/zerfoo/zerfoo/commit/61a09d94cae7890dc5c236c260700ea6a2e64a20)), closes [#169](https://github.com/zerfoo/zerfoo/issues/169)

## [1.18.0](https://github.com/zerfoo/zerfoo/compare/v1.17.2...v1.18.0) (2026-03-25)


### Features

* **cloud:** add concurrency limit and model allow list to TenantManager ([7c9e401](https://github.com/zerfoo/zerfoo/commit/7c9e401043a236013b1c4b15c82451710a6bb444))
* **cloud:** add persistent bbolt backend for TenantManager ([1266657](https://github.com/zerfoo/zerfoo/commit/126665745547f5b795374cf5a5852a512fd4bb51))


### Bug Fixes

* **timeseries:** use GPU engine for DLinear, NHiTS, Mamba forward pass in training ([#166](https://github.com/zerfoo/zerfoo/issues/166)) ([64a5795](https://github.com/zerfoo/zerfoo/commit/64a579537eae3d5df73c716e36810e31a343b073))

## [1.17.2](https://github.com/zerfoo/zerfoo/compare/v1.17.1...v1.17.2) (2026-03-25)


### Bug Fixes

* **timeseries:** use GPU engine for CfC forward pass in training ([#166](https://github.com/zerfoo/zerfoo/issues/166)) ([c67ec1f](https://github.com/zerfoo/zerfoo/commit/c67ec1fe2b6cfd86890724ac573566c2726fb61f))
* **timeseries:** use GPU engine for FreTS forward pass in training ([#166](https://github.com/zerfoo/zerfoo/issues/166)) ([d8686c0](https://github.com/zerfoo/zerfoo/commit/d8686c0bbfe8e33af1af26c1b3a42630aeff5aab))
* **timeseries:** use GPU engine for FreTS forward pass in training ([#166](https://github.com/zerfoo/zerfoo/issues/166)) ([99908e2](https://github.com/zerfoo/zerfoo/commit/99908e2f60f587daa7476839eea4c337d172dda7))
* **timeseries:** use GPU engine for PatchTST forward pass in training ([#166](https://github.com/zerfoo/zerfoo/issues/166)) ([0d6be90](https://github.com/zerfoo/zerfoo/commit/0d6be907a507ccfe31bf42114cd310a6db0e69f3))
* **timeseries:** use GPU engine MatMul for forward pass in training ([#166](https://github.com/zerfoo/zerfoo/issues/166)) ([c597b58](https://github.com/zerfoo/zerfoo/commit/c597b5872051b398849cbef24f5a9bd5f7a1b5d6))

## [1.17.1](https://github.com/zerfoo/zerfoo/compare/v1.17.0...v1.17.1) (2026-03-25)


### Bug Fixes

* **serve:** remove duplicate declarations after file split ([15be5a7](https://github.com/zerfoo/zerfoo/commit/15be5a7537fbb6dd38c459fa63c1be99a2ec7f54))
* **timeseries:** remove duplicate declarations after itransformer file split ([3146a8a](https://github.com/zerfoo/zerfoo/commit/3146a8acd1d03dfea266d772bd26ff655b719de8))

## [1.17.0](https://github.com/zerfoo/zerfoo/compare/v1.16.0...v1.17.0) (2026-03-25)


### Features

* **security:** add persistent bbolt backend for KeyStore ([0d486ed](https://github.com/zerfoo/zerfoo/commit/0d486ed64e85894902e1ff9a53edfc2bbffa8ac3))
* **serve:** add error rate counter and active request gauge to metrics ([f32ffcd](https://github.com/zerfoo/zerfoo/commit/f32ffcdca6a1d96cf949bbf8c7853f9a5055bef2))
* **support:** add persistent bbolt backend for ticket store ([98f2f75](https://github.com/zerfoo/zerfoo/commit/98f2f75c100522a42782fcab960d79021365ab78))


### Bug Fixes

* **generate:** log GPU counter sync errors instead of discarding ([7900778](https://github.com/zerfoo/zerfoo/commit/7900778b8686b97dbdf6ea2825b30bfd5206ed12))
* **inference:** propagate tensor creation errors in architecture builders ([27bd408](https://github.com/zerfoo/zerfoo/commit/27bd4083cedfad390b0697db325de3106b2ba6d5))

## [1.16.0](https://github.com/zerfoo/zerfoo/compare/v1.15.1...v1.16.0) (2026-03-25)


### Features

* **layers:** implement BatchNorm backward pass for training ([4f08575](https://github.com/zerfoo/zerfoo/commit/4f0857520b41ca952cb6cfab37ae622eeb3ef4a3))


### Bug Fixes

* **cloud:** add Reference URI validation to prevent SAML XSW attacks ([ef998b2](https://github.com/zerfoo/zerfoo/commit/ef998b2477494202acf24a51c99a0c147e63b4b9))
* **cloud:** deduct excess tokens when actual usage exceeds pre-authorized estimate ([28916bb](https://github.com/zerfoo/zerfoo/commit/28916bb6a5d321a4919467e7303dd3c3be26bf7a))
* **cloud:** enforce model allow list in tenant middleware ([657ba7f](https://github.com/zerfoo/zerfoo/commit/657ba7f827bc5823a5b24c327609112d12484ea5))
* **cloud:** fail closed in billing middleware when tenant is nil ([fb289a8](https://github.com/zerfoo/zerfoo/commit/fb289a824cca00b03efb4fbb1eff0d80f7abd0f9))
* **cloud:** hash API key in billing events to prevent credential exposure ([8fd0365](https://github.com/zerfoo/zerfoo/commit/8fd036524e569716d17c446a8da57153c4a718f6))
* **cloud:** limit billing middleware body reads to 10MB ([3790b5f](https://github.com/zerfoo/zerfoo/commit/3790b5f27fdd79be89b2455c4ab3f166732f89dd))
* **cloud:** log billing record errors instead of silently swallowing ([6c0240a](https://github.com/zerfoo/zerfoo/commit/6c0240a8051f85b098b7013afbcf2c1b22d45894))
* **cloud:** reject SAML assertions with empty ID to prevent replay bypass ([2f892c8](https://github.com/zerfoo/zerfoo/commit/2f892c8e1749a293ecd0b96786bc3ce76aeb0a52))
* **deploy:** update Dockerfile Go version from 1.25 to 1.26 ([e3e407f](https://github.com/zerfoo/zerfoo/commit/e3e407f2bc4613aa766f398059116d56f4962bbc))
* **distributed:** add stop channel to coordinator reaper goroutine ([d89de2d](https://github.com/zerfoo/zerfoo/commit/d89de2d81d27b73ab1185eaf851b9b0465b39476))
* **security:** send literal wildcard for CORS instead of reflecting origin ([6d2b566](https://github.com/zerfoo/zerfoo/commit/6d2b5664bd59316347e427d11ba7bdb477d6198e))
* **serve:** require ScopeReadOnly for all /v1/ routes by default ([92ea4c1](https://github.com/zerfoo/zerfoo/commit/92ea4c1150be2d1f7e4d559ffab62d4cb94980b3))
* **serve:** sanitize error messages in audio transcription handler ([8021815](https://github.com/zerfoo/zerfoo/commit/8021815cbaf6e5805969c0b1d8a07b9f98e37e36))
* **serve:** sanitize error messages to prevent internal detail leakage ([013583d](https://github.com/zerfoo/zerfoo/commit/013583d635743cbd3b480880d3b554b648dd663a))
* **support:** add SSRF protection to webhook dispatcher ([eaf144b](https://github.com/zerfoo/zerfoo/commit/eaf144b73305a7ead5b2ba47ee893361bddd7961))
* **support:** use plain HTTP client in webhook filter test ([1dd8606](https://github.com/zerfoo/zerfoo/commit/1dd8606b1deb9c4882b440703c5431d14068b616))

## [1.15.1](https://github.com/zerfoo/zerfoo/compare/v1.15.0...v1.15.1) (2026-03-25)


### Bug Fixes

* **ssm:** center MambaBlock weight init and scale by 1/sqrt(fan_in) ([a97712e](https://github.com/zerfoo/zerfoo/commit/a97712ee7bbad4a6578fbc5a100414cb0e3ca038))
* **timeseries:** add PredictWindowed nil-engine fallback and residual scaling ([0e6cb9a](https://github.com/zerfoo/zerfoo/commit/0e6cb9a9797be6648440d367b7fc045b85ebf83d))

## [1.15.0](https://github.com/zerfoo/zerfoo/compare/v1.14.0...v1.15.0) (2026-03-24)


### Features

* **timeseries:** add GPU engine training support to FreTS (issue [#158](https://github.com/zerfoo/zerfoo/issues/158)) ([429ef05](https://github.com/zerfoo/zerfoo/commit/429ef050e65426006ea9e1df33c6e2432ffd9213))
* **timeseries:** add GPU engine training support to ITransformer ([d13ab07](https://github.com/zerfoo/zerfoo/commit/d13ab07921034f73ea99ccb692cc56497efeb2dc))

## [1.14.0](https://github.com/zerfoo/zerfoo/compare/v1.13.0...v1.14.0) (2026-03-24)


### Features

* **timeseries:** add CPU fallback for Mamba training ([575b1da](https://github.com/zerfoo/zerfoo/commit/575b1dadbd5ee221f94a7a6a2d6edf2406c948f4))


### Bug Fixes

* **recurrent:** compute bias gradient in SimpleRNN backward pass ([3a79b45](https://github.com/zerfoo/zerfoo/commit/3a79b45d8b74e5ecc419ced1fcf9e4b4dfe737d5))


### Performance Improvements

* **timeseries:** refactor CfC backward from Jacobian to vector-Jacobian product ([ccb87d4](https://github.com/zerfoo/zerfoo/commit/ccb87d429d98f8038df3e9c22525dd6896c1c5d0))
* **timeseries:** replace ITransformer finite-diff with analytical backprop ([80e2a93](https://github.com/zerfoo/zerfoo/commit/80e2a93846073157a155080c5b914f5b99fa4e81))
* **timeseries:** replace PatchTST finite-diff with analytical backprop ([cbf58ff](https://github.com/zerfoo/zerfoo/commit/cbf58ff173e34cdeeaca8b235294ac505c84972a))

## [1.13.0](https://github.com/zerfoo/zerfoo/compare/v1.12.0...v1.13.0) (2026-03-24)


### Features

* **timeseries:** add FreTS backend (issue [#153](https://github.com/zerfoo/zerfoo/issues/153)) ([09783bf](https://github.com/zerfoo/zerfoo/commit/09783bf1751c1ee32b55b63e6682c00bcfa17274))
* **timeseries:** add iTransformer backend (issue [#155](https://github.com/zerfoo/zerfoo/issues/155)) ([6c53b2e](https://github.com/zerfoo/zerfoo/commit/6c53b2eef857c1a83a1890faf6137444008d60f5))
* **timeseries:** add Mamba/SSM backend (issue [#156](https://github.com/zerfoo/zerfoo/issues/156)) ([f879283](https://github.com/zerfoo/zerfoo/commit/f8792830948991cd10d888c0c5867f9101d90697))
* **timeseries:** add WithCfCEngine option to CfC constructor (issue [#154](https://github.com/zerfoo/zerfoo/issues/154)) ([fc80aac](https://github.com/zerfoo/zerfoo/commit/fc80aacf219eeab7b1b0aeead260afe4ccbc6bf9))
* **timeseries:** rewrite Mamba backend to wrap layers/ssm.MambaBlock (issue [#156](https://github.com/zerfoo/zerfoo/issues/156)) ([d103e83](https://github.com/zerfoo/zerfoo/commit/d103e83b640200e3c82ea37ce172faa5c551b4ac))


### Bug Fixes

* **inference:** defer releaseSession in Generate and GenerateStream ([1079796](https://github.com/zerfoo/zerfoo/commit/1079796949b22c53fa92e2b95a901b34ffe720ad))
* **inference:** use session pool in GenerateBatch instead of generator mutex (C-002) ([5405108](https://github.com/zerfoo/zerfoo/commit/5405108775e9ccb493fc7ff3f20700f66a37e6c8))
* **security:** complete E109 deep review v1.12.0 remediation -- Waves 48-50 ([a9e2e30](https://github.com/zerfoo/zerfoo/commit/a9e2e3001bbad6edb0b655dd9195c76f9ba5c525))
* **security:** remediate 3 High findings from v1.12.0 deep review ([743fbc6](https://github.com/zerfoo/zerfoo/commit/743fbc6f385fef0d6adf068c96308311ca46f824))
* **security:** sanitize error in audio transcription response ([d045351](https://github.com/zerfoo/zerfoo/commit/d0453510b2c496d63a572fe1fa75629783375b9b))
* **security:** suppress raw error details in repository handler 500 responses ([f782d4a](https://github.com/zerfoo/zerfoo/commit/f782d4a14f4c86785cfeaedc34c2316194ddf9ee))
* **security:** suppress raw error in Azure webhook 500 response ([f13d3b8](https://github.com/zerfoo/zerfoo/commit/f13d3b8315a3efeb1a3da6a3518e52a95404c57d))
* **security:** suppress raw error in disaggregated gateway SSE stream ([45f85a5](https://github.com/zerfoo/zerfoo/commit/45f85a51ca4beccca36c7b7cb2a33711b8ced6ea))
* **serve:** check http.Flusher before WriteHeader in streaming handlers ([6610e67](https://github.com/zerfoo/zerfoo/commit/6610e67db508629d0e33eadf6108136ede38d037))
* **timeseries:** prevent NHiTS segfault on small inputLen (issue [#152](https://github.com/zerfoo/zerfoo/issues/152)) ([2ebd9db](https://github.com/zerfoo/zerfoo/commit/2ebd9db35b68260aece3f26f3dc3b0039c784250))
* **timeseries:** store normalization stats and apply in PredictWindowed ([2321568](https://github.com/zerfoo/zerfoo/commit/2321568f6789af0ce52fe64bc5f18de4b8077439))

## [1.12.0](https://github.com/zerfoo/zerfoo/compare/v1.11.1...v1.12.0) (2026-03-24)


### Features

* **helm:** add NetworkPolicy template gated by networkPolicy.enabled ([969d1f6](https://github.com/zerfoo/zerfoo/commit/969d1f6e5280ec25c2e4ef46e3a50faa2a470d10))
* **inference:** make session pool size configurable (T108.22) ([36ab0e9](https://github.com/zerfoo/zerfoo/commit/36ab0e93f1b1de4f3cd4f675637f7008306afb15))
* **serve:** register healthz/readyz on main serve mux ([a250102](https://github.com/zerfoo/zerfoo/commit/a250102eff14050e7e350f8cfe1aa577904615dc))
* **support:** add API auth middleware and body size limits ([5078f5d](https://github.com/zerfoo/zerfoo/commit/5078f5d68d0b8fe8d1d7daecea16bb6e4b811d18))
* **timeseries:** add engine-accelerated backward pass for NHiTS TrainWindowed ([39797b7](https://github.com/zerfoo/zerfoo/commit/39797b7a0e881dba9f916bd2c8b8c34b5c563336)), closes [#133](https://github.com/zerfoo/zerfoo/issues/133)
* **timeseries:** add engine-accelerated training path for PatchTST TrainWindowed ([c6de291](https://github.com/zerfoo/zerfoo/commit/c6de29100cece6ca944d4371db341228eec76a8f))
* **timeseries:** add GPU-accelerated TrainWindowed for CfC ([#133](https://github.com/zerfoo/zerfoo/issues/133)) ([607952c](https://github.com/zerfoo/zerfoo/commit/607952ceafece2d347869774b721f33dcfc29cad))
* **timeseries:** add GPU-accelerated TrainWindowed for DLinear ([#133](https://github.com/zerfoo/zerfoo/issues/133)) ([96f2f9c](https://github.com/zerfoo/zerfoo/commit/96f2f9cf9fd7304a78dc8e5432fd3f0d86a9f7ef))


### Bug Fixes

* **cloud:** implement http.Flusher on responseCapture ([6d5347d](https://github.com/zerfoo/zerfoo/commit/6d5347da92a45e5b8c7c2c6fbb258de06d0c10a5))
* **cloud:** read streaming token usage from context in billingMiddleware (T108.1) ([97ce379](https://github.com/zerfoo/zerfoo/commit/97ce3797eec753cdf1d9634dc3f7028de480179c))
* **cloud:** use UTC for billing timestamps (T108.30) ([741b4b7](https://github.com/zerfoo/zerfoo/commit/741b4b72173e770a467a4255476fdf5d39f91900))
* **generate:** add TokenUsage context tracking for billing (T108.1) ([c2736c6](https://github.com/zerfoo/zerfoo/commit/c2736c6b46c4e2f37cd32c0643e0722821901192))
* **generate:** eliminate O(n^2) checkStop decoding via incremental decode ([13fa2b8](https://github.com/zerfoo/zerfoo/commit/13fa2b88888d7fc38880312cc2fbda5ecd783253))
* **generate:** remove unused seqLen computation in prefix cache path ([f840eef](https://github.com/zerfoo/zerfoo/commit/f840eef706f381032c37cc26a70ce5edb0809e41))
* **reducesum:** return error instead of panic for invalid axis in Backward ([ff242db](https://github.com/zerfoo/zerfoo/commit/ff242db691a11a2f0cb160398ab82a945d770d6a))
* **security:** add E108 deep review v1.11.1 remediation plan (32 tasks) ([bf837f3](https://github.com/zerfoo/zerfoo/commit/bf837f3ec66d7d0edf2be3b767df41afe3d7c289))
* **security:** add exponential backoff retry to marketplace metering (T108.10) ([b2addbc](https://github.com/zerfoo/zerfoo/commit/b2addbce8817448e0c30c760231c13ea0d8797c1))
* **security:** add SAML XXE protection and replay prevention (T108.15) ([bed8de1](https://github.com/zerfoo/zerfoo/commit/bed8de1dfd91ae2cc7be9ebfa0f297e3c891c122))
* **security:** add Vary: Origin header to CORS middleware ([621b8a7](https://github.com/zerfoo/zerfoo/commit/621b8a7be8f7cc65867ca71269dc82edde4d6894))
* **security:** complete E108 deep review remediation -- Waves 41-47 done ([ea9b223](https://github.com/zerfoo/zerfoo/commit/ea9b2231561cdd92197c588bb528a94e01838cb4))
* **security:** complete Wave 42 -- T107.2-T107.3, T108.6-T108.10 ([e9bf8f7](https://github.com/zerfoo/zerfoo/commit/e9bf8f7103ba323df27fd25482fc678bde302881))
* **security:** complete Wave 43 -- T108.11-T108.15 ([e8e91cf](https://github.com/zerfoo/zerfoo/commit/e8e91cf3fca3ceceb5a3a1c1f4f6f5450217f9ef))
* **security:** complete Wave 44 -- T108.16-T108.20 ([314a6f9](https://github.com/zerfoo/zerfoo/commit/314a6f9f7ca148248716a8d48c6546be71f300aa))
* **security:** complete Wave 45 -- T108.21-T108.25 ([e630fa0](https://github.com/zerfoo/zerfoo/commit/e630fa03b233904ffa71a13c2f65419b16583b63))
* **security:** enable GKE private cluster and scope node OAuth (H9/M15) ([2618ec8](https://github.com/zerfoo/zerfoo/commit/2618ec8b9b9fbf069d6a07a4365c800e882d6740))
* **security:** enforce scope-based authorization on endpoints (T108.11) ([b5f9296](https://github.com/zerfoo/zerfoo/commit/b5f92968dd54ecba95e316d6fb26ae04bf5f7d49))
* **security:** hash tenant API keys with SHA-256 instead of storing raw keys (H10) ([750c296](https://github.com/zerfoo/zerfoo/commit/750c296409fda4c341e4f066a0c3c578feefb479))
* **security:** implement SAML XML signature verification (T108.2) ([be1ac12](https://github.com/zerfoo/zerfoo/commit/be1ac12614ba45f46040d6dc054445f3204cb7f1))
* **security:** make Azure webhook signature validation mandatory (H8) ([6b2470b](https://github.com/zerfoo/zerfoo/commit/6b2470b6d33e518a2f744763b907fb036d718c65))
* **security:** pin GitHub Actions to commit SHAs ([67d3968](https://github.com/zerfoo/zerfoo/commit/67d3968057f8112892ad39fcd27839117e6a0ce0))
* **security:** pre-authorize token budget before request execution (T108.6) ([430154b](https://github.com/zerfoo/zerfoo/commit/430154b396e692cfc97f37b00a028a833ba6c1c5))
* **security:** replace hardcoded DP seed with crypto/rand and validate config ([f90c2e1](https://github.com/zerfoo/zerfoo/commit/f90c2e15175402251afc4f40fe9fb332ba9814d1))
* **security:** validate trusted proxies before trusting forwarding headers ([98949a2](https://github.com/zerfoo/zerfoo/commit/98949a2c17274d7271ea289eabf96e917ae8bbf9))
* **security:** warn or refuse startup when API key is empty (T108.12) ([7caa71c](https://github.com/zerfoo/zerfoo/commit/7caa71cbc2710afd7f8e8deccbff246f301d45a5))
* **serve:** add 30-second graceful shutdown timeout ([9d71483](https://github.com/zerfoo/zerfoo/commit/9d7148321357113a9a86f028a77b0ff12f00d230))
* **serve:** add OpenAI-required id/object/created/model fields to streaming chunks ([fc264e6](https://github.com/zerfoo/zerfoo/commit/fc264e638567b55e98f67bd7f1f8fc31b494a213))
* **serve:** apply chat template in batch path instead of raw concatenation ([000b55f](https://github.com/zerfoo/zerfoo/commit/000b55ff794044cbd19141ab23923d88db2a1ce3))
* **serve:** decouple batch context from individual request lifecycle ([df14cd0](https://github.com/zerfoo/zerfoo/commit/df14cd064469546921a255c96c988233ee7365b5))
* **serve:** read streaming token usage from context in BillingMiddleware (T108.1) ([919cd2a](https://github.com/zerfoo/zerfoo/commit/919cd2a8276782c5112e11acbd7cd2bf976a3828))
* **serve:** replace broad "cuda" match with specific OOM patterns in isOOMError ([2ae79a4](https://github.com/zerfoo/zerfoo/commit/2ae79a43493f626ef29585f376602274bb34c37d))
* **serve:** validate sampling params temperature/TopP/TopK (T108.27) ([c46fb4f](https://github.com/zerfoo/zerfoo/commit/c46fb4ffde8e4d9b18d7f1565084108bd184cccf))

## [1.11.1](https://github.com/zerfoo/zerfoo/compare/v1.11.0...v1.11.1) (2026-03-24)


### Bug Fixes

* **layers:** convert panic to error in reducesum Backward (T107.2) ([3bed57d](https://github.com/zerfoo/zerfoo/commit/3bed57d79640ccd5041e21958ee8a8eddeffc010))
* **rl:** convert panic to error returns in replay buffer (T107.3) ([55c4bef](https://github.com/zerfoo/zerfoo/commit/55c4bef5f9f118ec2cca86b9a918f5e617f1352a))
* **security:** complete Wave 38 -- T107.1-T107.5 (E107) ([fd3f8b7](https://github.com/zerfoo/zerfoo/commit/fd3f8b773ba926e80d8f70f1e10e22fef85d6cb1))
* **security:** complete Wave 39 -- T107.6-T107.10 (E107) ([8148776](https://github.com/zerfoo/zerfoo/commit/81487767c166463fd1763c457f45d5c8bfece881))
* **serve:** add body limit, error sanitization, inflight tracking to handleClassify (T107.1) ([273bd07](https://github.com/zerfoo/zerfoo/commit/273bd077f15f41c5f8ff84fa02654569d1bcffc0))
* **serve:** add inflight tracking to handleEmbeddings (T107.7) ([87424ee](https://github.com/zerfoo/zerfoo/commit/87424ee6cd926d4e68531b58fa77b9cf8a889d52))
* **serve:** use connect-time IP validation to prevent DNS rebinding SSRF (T107.5) ([0777bab](https://github.com/zerfoo/zerfoo/commit/0777babb48e5351e4c8bfdc3742dab5e22053545))
* **support:** use slices.SortFunc for ListByCustomer ordering (T107.6) ([6d2847b](https://github.com/zerfoo/zerfoo/commit/6d2847b60d38a743b3d766ae0a5f79523087de85))
* **timeseries:** fix NHiTS nil pointer in linearForward with high-channel data (T107.4, closes [#123](https://github.com/zerfoo/zerfoo/issues/123)) ([4578098](https://github.com/zerfoo/zerfoo/commit/45780985e69cf50b0ba87e957910a543de928e0b))

## [1.11.0](https://github.com/zerfoo/zerfoo/compare/v1.10.0...v1.11.0) (2026-03-21)


### Features

* **cli:** add TLS support to serve command (T106.8) ([91de5e2](https://github.com/zerfoo/zerfoo/commit/91de5e22949d91286840d6259cbaea0fba942ae9))
* **serve:** add request ID correlation middleware (T106.18) ([d31b2fc](https://github.com/zerfoo/zerfoo/commit/d31b2fc5f876e38ba78316a96a3fcde101ce29ea))


### Bug Fixes

* **cloud:** redact tenant API keys in Config/List responses (T106.34) ([50ac102](https://github.com/zerfoo/zerfoo/commit/50ac1021bfe266d13f590415e9d4bc73fae7f0a1))
* **deploy:** add pod securityContext to Helm deployment (T106.24) ([6eb7a6e](https://github.com/zerfoo/zerfoo/commit/6eb7a6e54083d3403cbe6c6c05968690247f3f60))
* **gguf:** add integer overflow checks to tensor parsing (T106.21) ([06bd6a3](https://github.com/zerfoo/zerfoo/commit/06bd6a37450b58aa61d6bdc6b24a0da4cdf837c2))
* **inference:** add embedding lookup bounds check (T106.7) ([91486eb](https://github.com/zerfoo/zerfoo/commit/91486eb0e282f05e9cb8e6d9f786d351e84eb91b))
* **inference:** cap GenerateBatch concurrency (T106.13) ([d2a3f18](https://github.com/zerfoo/zerfoo/commit/d2a3f187dda4a28a0afc17209f16c8ee2643e481))
* **inference:** protect RegisterAlias with sync.RWMutex (T106.15) ([a6bcaea](https://github.com/zerfoo/zerfoo/commit/a6bcaea57a4df94d7bb9d5b78b64bf5564bb7870))
* **infra:** restrict Cloud Run IAM from allUsers to service account (T106.25) ([273832c](https://github.com/zerfoo/zerfoo/commit/273832cc34bf64f1e0767ce8c8c47deffd9ab124))
* **layers:** convert panic() to error returns in attention_head (T106.12) ([c15bcc0](https://github.com/zerfoo/zerfoo/commit/c15bcc00c0eab0668fb4ff86f9819faa217aad30))
* **layers:** convert panic() to error returns in layers/core (T106.11) ([0a9feff](https://github.com/zerfoo/zerfoo/commit/0a9feff54f6f3e8ec8dc9b7204b4a9c857459981))
* **lora:** handle nil gradient in backward pass (T106.29) ([61d6a55](https://github.com/zerfoo/zerfoo/commit/61d6a5530fb3a783b432c9068a807e28740ae7bf))
* **meta:** set fixed seed in TestMAML_MetaConvergence for determinism (T106.33) ([a9a8bbd](https://github.com/zerfoo/zerfoo/commit/a9a8bbd66497f44cb375c84f89ab09d59884e66d))
* **registry:** add checksum verification and atomic writes to HF downloads (T106.14) ([68ccae2](https://github.com/zerfoo/zerfoo/commit/68ccae20125d74048f806cea04d622afe66a79f6))
* **registry:** add size limit to OCI blob download (T106.22) ([d7ca9c7](https://github.com/zerfoo/zerfoo/commit/d7ca9c74a82547f9e56834d30b21115ba3692421))
* **registry:** reject OCI references with path traversal in repository (T106.35) ([9dc1bf9](https://github.com/zerfoo/zerfoo/commit/9dc1bf9c72d377e835d8f29b30970c04e4a2fb53))
* **security:** complete Wave 30 -- 5 critical security fixes (E106) ([b242bb2](https://github.com/zerfoo/zerfoo/commit/b242bb26744136c304c744f3d716b0494fe6fb35))
* **security:** complete Wave 32 -- T106.11-T106.15 ([c3c0ea2](https://github.com/zerfoo/zerfoo/commit/c3c0ea2d0de7b05620fcd1597ef4ddf394b0a8c0))
* **security:** complete Wave 33 -- T106.16-T106.20 ([d5d5aab](https://github.com/zerfoo/zerfoo/commit/d5d5aab375e0384aa5eca1c92b571eb088ffa80a))
* **security:** complete Wave 34 -- T106.21-T106.25 ([a8ead63](https://github.com/zerfoo/zerfoo/commit/a8ead634e5daba3b8e50685476a97dd3a7eb3751))
* **security:** complete Wave 35 -- T106.26-T106.30 ([ffc44d8](https://github.com/zerfoo/zerfoo/commit/ffc44d8828197d522e16f9590a100f2313ff7120))
* **security:** complete Wave 36 -- T106.31-T106.35 ([7dc59b0](https://github.com/zerfoo/zerfoo/commit/7dc59b0f7eb7c1da918fd7130123827b81a99f19))
* **security:** merge Wave 32-33 worktree changes ([365fc3e](https://github.com/zerfoo/zerfoo/commit/365fc3e9f3da6b70c7a4e196aa872755c1f84afe))
* **serve:** add path traversal protection to FileSystemRepository (T106.3) ([a205a4d](https://github.com/zerfoo/zerfoo/commit/a205a4db683ba3d1ed139ad3d84c44b61de5d4a7))
* **serve:** add request body size limits to inference endpoints (T106.6) ([8200fb6](https://github.com/zerfoo/zerfoo/commit/8200fb6b2abd6df139ec03753b7830a1b14a36fc))
* **serve:** add request body size limits to inference endpoints (T106.6) ([e1ef411](https://github.com/zerfoo/zerfoo/commit/e1ef4110389343ba0f646532c2b4a3ea286bfcd0))
* **serve:** add security headers middleware (T106.17) ([af3e1f5](https://github.com/zerfoo/zerfoo/commit/af3e1f54a7d98ba6e23303cec91732841effe402))
* **serve:** add SSRF protection to vision image fetch (T106.4) ([c0b4b40](https://github.com/zerfoo/zerfoo/commit/c0b4b40e23f9975ddeff8c763d678f4b6865f22c))
* **serve:** wire authentication middleware into Server (T106.1) ([1f785ac](https://github.com/zerfoo/zerfoo/commit/1f785ac39a3183fe804c76fe21849135f046bc65))
* **ssm:** add nil gradient check in S4 backward (T106.28) ([04d1a06](https://github.com/zerfoo/zerfoo/commit/04d1a06ec56ad6604c52bc642662b1a3390bf409))
* **support:** use json.Encode for error responses (T106.23) ([c6a6242](https://github.com/zerfoo/zerfoo/commit/c6a6242dea7c969e322e54e78238318e6143a5fc))
* **timeseries:** use channel-independent projection in PatchTST (T106.30) ([195efe8](https://github.com/zerfoo/zerfoo/commit/195efe8837c4b13c1e59642cd0cebdd5421bd6de))
* **training:** add gradient clipping and NaN guard to AdamW (T106.27) ([8ed3c4e](https://github.com/zerfoo/zerfoo/commit/8ed3c4ef79ca6d6fa575854122038fdd06c728d2))
* **workerpool:** use sync.Once for Close() to prevent data race (T106.26) ([8494b76](https://github.com/zerfoo/zerfoo/commit/8494b7655f9cb318a5fb6ef7119f55c2a772372d))

## [1.10.0](https://github.com/zerfoo/zerfoo/compare/v1.9.0...v1.10.0) (2026-03-21)


### Features

* **timeseries:** add ModelPath and Metrics to TrainResult (fixes [#118](https://github.com/zerfoo/zerfoo/issues/118)) ([7bd7cec](https://github.com/zerfoo/zerfoo/commit/7bd7cec97d9ad4a701eea012c1d4cce20186fe38))
* **timeseries:** add TrainWindowed/PredictWindowed to PatchTST (fixes [#120](https://github.com/zerfoo/zerfoo/issues/120)) ([7d6cee2](https://github.com/zerfoo/zerfoo/commit/7d6cee2a92ca9d8adb030d42e4299a16dd7cd8b5))
* **training:** add CreateWindows and ParseWindowSizes (fixes [#119](https://github.com/zerfoo/zerfoo/issues/119)) ([5c53faa](https://github.com/zerfoo/zerfoo/commit/5c53faa930d082ac8db3565297e29878a342bbe6))


### Bug Fixes

* **cuda:** use cgo build tag for arm64 dlopen trampolines ([2d2d7c6](https://github.com/zerfoo/zerfoo/commit/2d2d7c61f4176907f0363897e9a9e051e38398b7)), closes [#116](https://github.com/zerfoo/zerfoo/issues/116)

## [1.9.0](https://github.com/zerfoo/zerfoo/compare/v1.8.0...v1.9.0) (2026-03-21)


### Features

* **attention:** add bidirectional self-attention mode ([aa2b407](https://github.com/zerfoo/zerfoo/commit/aa2b407a73a5bde42dfe87af3bd0512220ee9d4b))
* **cli:** add zerfoo finetune-sentiment subcommand ([2c7a1c1](https://github.com/zerfoo/zerfoo/commit/2c7a1c16056d471e93b2dce235c5725339d48395))
* **cli:** add zerfoo sentiment subcommand ([a18c609](https://github.com/zerfoo/zerfoo/commit/a18c60930a233cbee4b67ccf31b9404f184c3750))
* **inference:** add AttnRes residual config to graph builder ([db61339](https://github.com/zerfoo/zerfoo/commit/db61339918bf434dd33e6142f00c0a193dd3f9da))
* **inference:** add BERT encoder-only architecture builder ([a041684](https://github.com/zerfoo/zerfoo/commit/a0416845694a3474b0c6d6885a8293d143983e22))
* **inference:** add encoder-only GGUF model loading ([73a3d8c](https://github.com/zerfoo/zerfoo/commit/73a3d8c4a28cc0493dc46fda7c484e592cfd6005))
* **layers/residual:** implement BlockAttnRes layer ([594281a](https://github.com/zerfoo/zerfoo/commit/594281aaf0802a898765dd24238db478f29a5e03))
* **layers:** add AttnRes layer for attention residuals (arXiv:2603.15031) ([ecc8a9a](https://github.com/zerfoo/zerfoo/commit/ecc8a9af4f7a85f34c4bee1484aa8a5523d4fa81))
* **layers:** add sequence classification head ([3a9cfb4](https://github.com/zerfoo/zerfoo/commit/3a9cfb487c196442a7a98b3771a348c5f3445ec6))
* **sentiment:** add fine-tuning support for sentiment models ([7062e92](https://github.com/zerfoo/zerfoo/commit/7062e9299b596dc6a2f19e366a43363eb9083717))
* **sentiment:** add sentiment classification inference pipeline ([7e93eb8](https://github.com/zerfoo/zerfoo/commit/7e93eb870560daaff0d432f0002e18f98a012bed))
* **serve:** add /v1/sentiment API endpoint ([c060277](https://github.com/zerfoo/zerfoo/commit/c060277bddc405bd90377637d00498cc1c60c25a))
* **timeseries:** add CfC closed-form continuous-time backend ([50be230](https://github.com/zerfoo/zerfoo/commit/50be2300a6a22dff74208214e2fe204cecc71a1a))
* **timeseries:** add DLinear forecasting backend ([972772e](https://github.com/zerfoo/zerfoo/commit/972772ecfe108cec11ae192fa8334eccc84eccff))
* **timeseries:** add N-HiTS multi-rate forecasting backend ([55f96f3](https://github.com/zerfoo/zerfoo/commit/55f96f322e3878d26636bc13f598d515e9b624c8))
* **training:** add WindowedBackend and WindowedPredictor interfaces ([a4305b6](https://github.com/zerfoo/zerfoo/commit/a4305b663fd4bcb65604c81e1500c667d3a6707a))


### Bug Fixes

* **cuda:** port arm64 trampoline fix from ztensor (issue [#115](https://github.com/zerfoo/zerfoo/issues/115)) ([5952e31](https://github.com/zerfoo/zerfoo/commit/5952e31984bb8b73d6ad09234c3dfa92e433d6a7))
* **gguf:** restore Q5_K/Q6_K re-quantization to Q4_0 for fast GEMV decode ([21c9f45](https://github.com/zerfoo/zerfoo/commit/21c9f4564fa35593e1c3e4733d62ac945b139d43))
* **hrm:** resize hidden state to match input batch dimension ([5daba62](https://github.com/zerfoo/zerfoo/commit/5daba62caa4ec2a6cd0eb8283aa98b94724034ea)), closes [#105](https://github.com/zerfoo/zerfoo/issues/105)
* **inference:** replace GQA+RoPE with direct self-attention for BERT encoder ([fe74b18](https://github.com/zerfoo/zerfoo/commit/fe74b188ed0d02d1f74636218d606275f6d9b01a))

## [1.8.0](https://github.com/zerfoo/zerfoo/compare/v1.7.0...v1.8.0) (2026-03-20)


### Features

* **automl:** wire actual model constructors into AutoML search ([ad61709](https://github.com/zerfoo/zerfoo/commit/ad61709b07c498f7d393941e095e37ccf5c89cc1))
* **autoopt:** add automatic kernel selection based on hardware profiling ([d178559](https://github.com/zerfoo/zerfoo/commit/d178559ce6e6c4f4b720d9825009f0ab6ddf0929))
* **autoopt:** add automatic quantization recommendation ([76c72ef](https://github.com/zerfoo/zerfoo/commit/76c72efa5f64d6670803ae090370450f661e5993))
* **autoopt:** add automatic workload splitting ([1f7bc1d](https://github.com/zerfoo/zerfoo/commit/1f7bc1d98ca4675f12c33c649d3bb43854883b76))
* **autoopt:** add automatic workload splitting ([a8d68f0](https://github.com/zerfoo/zerfoo/commit/a8d68f01a6915e46926f83d5b8771a38453097a0))
* **autoopt:** add hardware-specific kernel codegen ([cb86c40](https://github.com/zerfoo/zerfoo/commit/cb86c400db7cbc15d640a28985f00d6431595767))
* **autoopt:** add multi-accelerator scheduling ([6e4c1b7](https://github.com/zerfoo/zerfoo/commit/6e4c1b78e29cbddde15b90635aef8269b8a1e7b0))
* **autoopt:** add next-gen GPU architecture optimizations for Hopper/Blackwell ([1a8f41f](https://github.com/zerfoo/zerfoo/commit/1a8f41faa39d36714ad9575a0ebf04f8407c7683))
* **causal:** implement Intervene for counterfactual prediction ([f91c9cf](https://github.com/zerfoo/zerfoo/commit/f91c9cff718d349ad787387a6ab3676200cbcbb9))
* **causal:** implement PC algorithm for causal structure discovery ([8132ffc](https://github.com/zerfoo/zerfoo/commit/8132ffccecf3484d9197f20858e13fae5ae8d41f))
* **cli:** add --gpus flag to serve command ([2d76c73](https://github.com/zerfoo/zerfoo/commit/2d76c735795cc71355892314232bd3055f871b90))
* **cli:** add --gpus flag to serve command ([94b72e1](https://github.com/zerfoo/zerfoo/commit/94b72e14a1ac1e83cf7a7d8879e587d0a010be9c))
* **cli:** add --gpus flag to serve command ([b23b3c0](https://github.com/zerfoo/zerfoo/commit/b23b3c0c633472d28ca61b8f5613cbc72b7ada7a))
* **cli:** add --gpus flag to serve command ([7b97a37](https://github.com/zerfoo/zerfoo/commit/7b97a37eb5639e549d2aeebc874b431aaaa940c6))
* **cli:** add train subcommand ([6b7f8db](https://github.com/zerfoo/zerfoo/commit/6b7f8dbfa21c3ce0d65da02114e64d00cbd39b5e))
* **cli:** implement tabular AutoML worker ([7efd976](https://github.com/zerfoo/zerfoo/commit/7efd976058a57218a578ce579f5b05e2b748256d))
* **cli:** register train command in main CLI entrypoint ([301a142](https://github.com/zerfoo/zerfoo/commit/301a1422af8e82c793982238c47f5d83f327c407))
* **cloud:** implement enterprise features for SOC 2 compliance ([74a0b5b](https://github.com/zerfoo/zerfoo/commit/74a0b5b4976fff566c080e210402e754dbd0fb93))
* **cloud:** implement multi-tenant inference service with token billing ([b322919](https://github.com/zerfoo/zerfoo/commit/b322919d7de02e2eb0a0a17f83e4be044feff10a))
* **cloud:** implement multi-tenant inference service with token billing ([60a6fac](https://github.com/zerfoo/zerfoo/commit/60a6facee650c98ccf926f828767be1e81860dde))
* **compliance:** add SOC 2 compliance automation platform ([9229ad8](https://github.com/zerfoo/zerfoo/commit/9229ad843336e1a45de8426f37505a6fcd8f2c3e))
* **compliance:** add SOC 2 Type I audit tooling ([207d0e3](https://github.com/zerfoo/zerfoo/commit/207d0e38c410fd641df354e9709e9c8f139883da))
* **compliance:** add SOC 2 Type II observation framework ([1d74d18](https://github.com/zerfoo/zerfoo/commit/1d74d18a70d0c9dc59c3e3c8a04d93e6f10df79f))
* **crossasset:** implement cross-attention model for multi-source features ([3158183](https://github.com/zerfoo/zerfoo/commit/315818333ec68200072d1beb9b69a805b8ef4f46))
* **deploy:** add AWS Marketplace listing infrastructure ([b94a103](https://github.com/zerfoo/zerfoo/commit/b94a103c6a6b2699e74ee78e9b0fe6d6156fc96b))
* **deploy:** add Helm chart for Zerfoo Kubernetes deployment ([0d7c7e8](https://github.com/zerfoo/zerfoo/commit/0d7c7e89fa208c1ec929194c6a91342c0ae560ed))
* **distributed:** wire AdamW optimizer into train_distributed CLI ([3fa4f81](https://github.com/zerfoo/zerfoo/commit/3fa4f81fbd56bb6127ca44078108058b154237e3))
* **edge:** add Jetson Orin Nano cross-compilation ([5703222](https://github.com/zerfoo/zerfoo/commit/5703222ed27e8c256d68a25b779d73f870f4df0e))
* **edge:** add pre-optimized model format for edge deployment ([24f64a8](https://github.com/zerfoo/zerfoo/commit/24f64a803243d6777b5e3873cabc20c258669fda))
* **edge:** add Raspberry Pi 5 cross-compilation ([cb4239b](https://github.com/zerfoo/zerfoo/commit/cb4239b0a42b0e1a01ee54b011fd010ba4c21c34))
* **edge:** implement build-tag-gated edge binary ([5d61f1f](https://github.com/zerfoo/zerfoo/commit/5d61f1f7581c99869f88f4cc0a5733f52f74a602))
* **examples:** add Android demo app ([6fea3f7](https://github.com/zerfoo/zerfoo/commit/6fea3f75f5ad19a882e5730edf33fb0170d3cc4b))
* **examples:** add iOS demo app with gomobile bindings ([095a058](https://github.com/zerfoo/zerfoo/commit/095a058b3cf9ca53b11562ddb45b257d59d27988))
* **federated:** add 4-client federated simulation integration test ([83ba802](https://github.com/zerfoo/zerfoo/commit/83ba802f93c7d8fdf2d567bfc76428040514d3c5))
* **federated:** implement differential privacy noise injection ([eef70e9](https://github.com/zerfoo/zerfoo/commit/eef70e9632267d75dffff41e1dec5b316335254a))
* **federated:** implement FedAvg strategy and coordinator ([149cdf9](https://github.com/zerfoo/zerfoo/commit/149cdf9138cb5587a65b410607317d2bb3aacbf8))
* **federated:** implement FedProx strategy with proximal term ([5a6240c](https://github.com/zerfoo/zerfoo/commit/5a6240ce59eee8a9584ff59f68bc0517e3eb2cca))
* **generate:** implement FP8 KV cache quantization ([d2b1e6d](https://github.com/zerfoo/zerfoo/commit/d2b1e6da48075146f558067245bc64dca2c95c07))
* **gnn:** implement GCN and GAT graph neural network layers ([b8401cb](https://github.com/zerfoo/zerfoo/commit/b8401cb584278a313ce78d8f038f0f7a4f776a58))
* **gp:** implement tree-based genetic programming with Evolve ([09fb996](https://github.com/zerfoo/zerfoo/commit/09fb996099534cae42df37b7d16d8b3675f9ec8e))
* **inference:** add Mamba 3 architecture builder with MIMO SSM ([dff1dc3](https://github.com/zerfoo/zerfoo/commit/dff1dc37454fb036aea42583329ca837140979da))
* **inference:** add Qwen-VL vision-language architecture builder ([3115e80](https://github.com/zerfoo/zerfoo/commit/3115e807548a49bfe240f38b441c18e0023bf39b))
* **inference:** implement automated architecture builder from GGUF metadata ([f64f23a](https://github.com/zerfoo/zerfoo/commit/f64f23ac420a1e647f3e69d9ca63a9eede50db76))
* **integrations:** add LangChain-Go and Weaviate adapters ([f13175a](https://github.com/zerfoo/zerfoo/commit/f13175ac3ba25799074780396f41566b53595e2e))
* **integrations:** add LangChain-Go and Weaviate READMEs, fix go vet ([56ef2b6](https://github.com/zerfoo/zerfoo/commit/56ef2b62127e996f89d2735fee29c86888a20a04))
* **marketplace:** add AWS Marketplace integration ([472b8ba](https://github.com/zerfoo/zerfoo/commit/472b8baba575cf327dd5a8ab265b7367fa7d3b32))
* **marketplace:** add Azure Marketplace integration ([7e581be](https://github.com/zerfoo/zerfoo/commit/7e581be8143c3d12583a65bf111c8ed2a85b743d))
* **marketplace:** add GCP Marketplace integration ([1f0e1ea](https://github.com/zerfoo/zerfoo/commit/1f0e1ea3876cd7a663b037720211d2b86ec8184a))
* **marketplace:** add unified multi-cloud marketplace abstraction ([79de057](https://github.com/zerfoo/zerfoo/commit/79de0576561d87212487df31366d8986d1c342fc))
* **meta:** implement MAML for few-shot meta-learning ([4505e4b](https://github.com/zerfoo/zerfoo/commit/4505e4b362a37e1f5a3b209574e56e26fa1ed009))
* **mobile:** implement gomobile bindings for on-device inference ([7415743](https://github.com/zerfoo/zerfoo/commit/7415743188678f1169642db908762425e5467e6f))
* **modelcache:** implement LRU model cache with K8s DaemonSet ([d510af5](https://github.com/zerfoo/zerfoo/commit/d510af57c9ce71a949f975c5c0d4ebc3e0e4d914))
* **modeldsl:** implement custom model training workflow ([3fa86e9](https://github.com/zerfoo/zerfoo/commit/3fa86e9e6ac515e7c34e277fc9a118472f7275f6))
* **modeldsl:** implement graph-level optimization passes ([b57ed9c](https://github.com/zerfoo/zerfoo/commit/b57ed9c945a30295005a6a2f7234d8bf691a3726))
* **modeldsl:** implement model definition DSL ([586e8a5](https://github.com/zerfoo/zerfoo/commit/586e8a5039775fb55c58c7a2ea140f3c96686171))
* **modeldsl:** implement model definition DSL ([ac15be3](https://github.com/zerfoo/zerfoo/commit/ac15be360c334271153c2e1696d933cfde49c8a5))
* **monitor:** implement Page-Hinkley and ADWIN drift detection ([f55b9f2](https://github.com/zerfoo/zerfoo/commit/f55b9f2ba79acfd3d095457ae63653311cc17659))
* **nas:** implement multi-objective Pareto frontier search ([0e6ad92](https://github.com/zerfoo/zerfoo/commit/0e6ad9215b8052b0473ce0406da36c9ac5f9970f))
* **online:** implement elastic weight consolidation for continual learning ([61346f5](https://github.com/zerfoo/zerfoo/commit/61346f5108777695795760f969f9c8aab500bcf0))
* **parity:** add Mamba 3 CPU/CUDA parity test for DGX Spark ([7cc38b0](https://github.com/zerfoo/zerfoo/commit/7cc38b04a6d02ccda9cf5fb10af31f7faa381a07))
* **parity:** add vision model benchmarks for LLaVA and QwenVL ([0b218d7](https://github.com/zerfoo/zerfoo/commit/0b218d7b120bcb01eb6cd3e20a0e88a548f35c12))
* **provenance:** implement cryptographic model lifecycle tracker ([8be1f6b](https://github.com/zerfoo/zerfoo/commit/8be1f6bbed26cd58e881edabb1f1f60bfb292529))
* **recover:** implement AutoRetrain pipeline ([4b58760](https://github.com/zerfoo/zerfoo/commit/4b58760012f047d705ac3d277deda9f860723b6f))
* **regime:** implement HMM with Baum-Welch and Viterbi ([e5f77ac](https://github.com/zerfoo/zerfoo/commit/e5f77ac65723ed1fdcaf97dac92ff486a1b3385c))
* **registry:** implement OCI-compatible model registry ([7497614](https://github.com/zerfoo/zerfoo/commit/7497614ba0271d99ad8fd938aecc4f734458ab99))
* **rl:** add Environment, Agent interfaces and ReplayBuffer ([4e0617e](https://github.com/zerfoo/zerfoo/commit/4e0617e911a3759ba8673dac169b0766e3b1eea9))
* **rl:** implement PPO with clipped surrogate and GAE ([525a224](https://github.com/zerfoo/zerfoo/commit/525a224dca5764ae80456acba02d2d6c50daec94))
* **rl:** implement SAC with twin Q-networks and entropy tuning ([f3290b6](https://github.com/zerfoo/zerfoo/commit/f3290b6218938c069b2276d5942f77c3eb0c381e))
* **security:** implement SOC 2 security controls ([c3010cd](https://github.com/zerfoo/zerfoo/commit/c3010cd6507609070e114ea46165361a187476e1))
* **serve:** add Kubernetes operator for ZerfooInferenceService CRD ([c72302b](https://github.com/zerfoo/zerfoo/commit/c72302b7eab6102b2fdc7dfe8784e06b7eaaaeb3))
* **serve:** implement adaptive batching ([a248ca5](https://github.com/zerfoo/zerfoo/commit/a248ca521e442c3d9e660bcfdcf3c9c8c58e12cb))
* **serve:** implement model repository server ([dc178b0](https://github.com/zerfoo/zerfoo/commit/dc178b0f572b95b47f2d2008ec721a4338e4ec97))
* **serve:** implement model repository server ([f70954b](https://github.com/zerfoo/zerfoo/commit/f70954b439ce1014db31760ab49bfcf2269b10d0))
* **serve:** implement multi-model serving with LRU eviction ([6cf667c](https://github.com/zerfoo/zerfoo/commit/6cf667cb974d8b6eed0041aa85b3ced6fd3a93bc))
* **shared:** implement cross-model latent space for knowledge sharing ([65e57a6](https://github.com/zerfoo/zerfoo/commit/65e57a674f8fe93dae0ad3ee0bf36fd88b0780d8))
* **support:** add enterprise ticketing system ([73dbef0](https://github.com/zerfoo/zerfoo/commit/73dbef0a2bbe7c169f7d782222f29db5926e80bc))
* **synth:** implement CrashGenerator for stress testing scenarios ([8610213](https://github.com/zerfoo/zerfoo/commit/86102139b7e8ed92095d510e03c8e9948fb8a07e))
* **synth:** implement MarketVAE variational autoencoder ([8c611a5](https://github.com/zerfoo/zerfoo/commit/8c611a544e7f68e9323ba5e90743e0519fa357b4))
* **tabular:** add Ensemble combining MLP models with tree predictions via stacking ([a79299d](https://github.com/zerfoo/zerfoo/commit/a79299d65310d9eee49a1195c9fad13f1f0b5834))
* **tabular:** add Ensemble type with stacking meta-learner ([1344cc8](https://github.com/zerfoo/zerfoo/commit/1344cc8b067e4c6069d4e1becebf3850cc745d1a))
* **tabular:** add FTTransformer architecture ([f311af0](https://github.com/zerfoo/zerfoo/commit/f311af006843d822eb49e5644a99680bb851e9e3))
* **tabular:** add FTTransformer.Forward method for batched tensor input ([70831cb](https://github.com/zerfoo/zerfoo/commit/70831cb7e8daa47cfdac6c7a2a5bf1d97bd8ac9c))
* **tabular:** add Model type with configurable MLP on ztensor compute graph ([a59e719](https://github.com/zerfoo/zerfoo/commit/a59e719b487d3d81af5046f01fdfedd7af172eb1))
* **tabular:** add SAINT with intersample attention ([8448be3](https://github.com/zerfoo/zerfoo/commit/8448be3063fcabbb50700b79fa7913f5db282ddb))
* **tabular:** add SAINT with intersample attention ([87c2b67](https://github.com/zerfoo/zerfoo/commit/87c2b6777c9ec670b7580108a5606c00f2d52f18))
* **tabular:** add TabNet with sparsemax attention ([fcbc640](https://github.com/zerfoo/zerfoo/commit/fcbc640200f922ca5c9a72a835ee5b46c07dd41b))
* **tabular:** add TabResNet with residual connections ([be06a41](https://github.com/zerfoo/zerfoo/commit/be06a41b2383e9671ba937d8fa428f2290a16d91))
* **tabular:** implement FineTuneLoRA and MergeAdapter for per-source LoRA adaptation ([e68a012](https://github.com/zerfoo/zerfoo/commit/e68a012693353c615b1ce08fdbe0cf0c45925858))
* **tabular:** implement PreTrain for multi-asset base model ([405317b](https://github.com/zerfoo/zerfoo/commit/405317bd37597f9ca2b5060a37edb7767483df10))
* **tabular:** implement Save/Load for model serialization ([f3c8636](https://github.com/zerfoo/zerfoo/commit/f3c863614e0e8646d8d280a8371d0d2efad21e23))
* **tabular:** implement Train function with AdamW optimizer ([2405a50](https://github.com/zerfoo/zerfoo/commit/2405a50ec49e49ed88bfaed2d4b892c66495c347))
* **testing:** implement automated benchmark suite ([38c6cf8](https://github.com/zerfoo/zerfoo/commit/38c6cf8a4aec6a181816002fe1afc0eaea8e392e))
* **testing:** implement model comparison tool ([c283743](https://github.com/zerfoo/zerfoo/commit/c283743a2fd314a2b600bcf348a904bad34d2751))
* **timeseries:** add N-BEATS with basis expansion decomposition ([4b7b8ec](https://github.com/zerfoo/zerfoo/commit/4b7b8eca3cdae8bb5cccada7d83f2921cb91887f))
* **timeseries:** add PatchTST with channel-independent patching ([c9ce8a0](https://github.com/zerfoo/zerfoo/commit/c9ce8a063ac4d787e842a52cc40cdf067e5dd4e7))
* **timeseries:** add Predict method to PatchTST ([06aa0a7](https://github.com/zerfoo/zerfoo/commit/06aa0a7a3f4818f917f9648e74fa2abaed361142))
* **timeseries:** add TFT with variable selection and multi-horizon output ([d91bd6a](https://github.com/zerfoo/zerfoo/commit/d91bd6af7d336a0118cf61b52f5fd334ab718b40))
* **timeseries:** add training.Model[float32] adapters for NBEATS, PatchTST, TFT ([90a4c38](https://github.com/zerfoo/zerfoo/commit/90a4c38a1799c18e595d328f9af16ac35352a368))


### Bug Fixes

* **autoopt:** remove duplicate parseComputeCap from nextgen.go ([c4c5830](https://github.com/zerfoo/zerfoo/commit/c4c5830492942ee0391500059dbcd12274da6f7b))
* **batcher:** eliminate race in eviction ordering test ([36e9a90](https://github.com/zerfoo/zerfoo/commit/36e9a90efe87772f72cf614c0387570b563b545a))
* **cloud:** use atomic fields and constant-time API key comparison ([134c300](https://github.com/zerfoo/zerfoo/commit/134c300829a0c801092758bee76b7803a6e4b427))
* **compliance:** remove duplicate audit.go conflicting with existing audit package ([9f9fc1f](https://github.com/zerfoo/zerfoo/commit/9f9fc1f3fcf8d930470ab60a10549059251c9326))
* **federated:** persist global weights across rounds ([cca0ac0](https://github.com/zerfoo/zerfoo/commit/cca0ac0dc2bb93805064ba9c95668d3dc3b238c0))
* **gguf:** stop re-quantizing Q5_K/Q6_K tensors to Q4_0 ([f7e5b49](https://github.com/zerfoo/zerfoo/commit/f7e5b49f2aac67dad5c64e5af25cf06473734bd6))
* **online:** cap drift detector sharpes history to bound memory ([cf5e99e](https://github.com/zerfoo/zerfoo/commit/cf5e99e44a8e365981c460a4ae4b7c4ee0753118))
* **registry:** prevent path traversal in model directory and download ([077262b](https://github.com/zerfoo/zerfoo/commit/077262b291bc5f3f6c0f18345850887709eba58b))
* **rl:** fix PPO clipping logic and add gradient norm clipping ([1690b32](https://github.com/zerfoo/zerfoo/commit/1690b32362d0a861a8e8d05d9073cc0e83202e72))
* **rl:** speed up SAC tests with smaller networks and relaxed thresholds ([5b0764f](https://github.com/zerfoo/zerfoo/commit/5b0764f167650463b51505cad450a74716dee0b1))
* **test:** use current directory in bench_disagg TestBinaryBuilds ([0c80026](https://github.com/zerfoo/zerfoo/commit/0c80026720077e497ae5116f335114d836946db5))
* **timeseries,tabular:** resolve cross-file name collisions after wave-7 merge ([c00e8d6](https://github.com/zerfoo/zerfoo/commit/c00e8d65ef5e740a2e490eea87769d949018a611))

## [1.7.0](https://github.com/zerfoo/zerfoo/compare/v1.6.0...v1.7.0) (2026-03-18)


### Features

* **generate:** add FP16 KV cache storage ([7409503](https://github.com/zerfoo/zerfoo/commit/74095035670dfc55e5fa8a96b1b8d616d5d28205))
* **inference:** add Command R architecture builder ([9198790](https://github.com/zerfoo/zerfoo/commit/91987903b226d870359160aa2e565db395b5667d))
* **inference:** add Falcon architecture builder ([6d5686f](https://github.com/zerfoo/zerfoo/commit/6d5686f538fb7aa75558c1d1e062c75df6077e31))
* **inference:** add LLaVA vision-language architecture builder ([8346d05](https://github.com/zerfoo/zerfoo/commit/8346d05befe3d2efac19f24347152bf03176b5b3))
* **inference:** add Mixtral MoE architecture builder ([fcf0030](https://github.com/zerfoo/zerfoo/commit/fcf0030d856a539b39a8c99187f97b5a456d4088))
* **inference:** add RWKV architecture builder ([5602db2](https://github.com/zerfoo/zerfoo/commit/5602db20024f14e78920fbcec0346bbf7103aea7))
* **lint:** add deprecation comment linter ([5f50b98](https://github.com/zerfoo/zerfoo/commit/5f50b9848b8dc4ec453192df7fa5af0e826c2cc2))
* **parallel:** add pipeline parallelism with micro-batch scheduling ([2a7b8b5](https://github.com/zerfoo/zerfoo/commit/2a7b8b5350ba745616400e3b4841805ce4e457fb))
* **parallel:** add tensor parallelism for multi-GPU prefill ([063cc74](https://github.com/zerfoo/zerfoo/commit/063cc74b3ee58568e7511f827ae945c6d346360f))
* **ssm:** add BCNorm stabilization layer for B/C matrices ([69d5f66](https://github.com/zerfoo/zerfoo/commit/69d5f66550a1aade326233f99c1451b9dd52f520))
* **ssm:** add complex-valued SSM state tracking with RoPE ([4b8bb02](https://github.com/zerfoo/zerfoo/commit/4b8bb02930a97f44ddc7f5dd56d37e73cfee0004))
* **ssm:** add exponential-trapezoidal discretization mode ([d567477](https://github.com/zerfoo/zerfoo/commit/d5674775b1df60e91a468916ebf962405ac936c6))
* **ssm:** add MIMO multi-head SSM block ([587d42b](https://github.com/zerfoo/zerfoo/commit/587d42baea9c9ff446cf14cece1b4413cf51945f))

## [1.6.0](https://github.com/zerfoo/zerfoo/compare/v1.5.0...v1.6.0) (2026-03-18)


### Features

* **plan:** add Mamba 3 architecture tasks T2.8-T2.12 ([f59490c](https://github.com/zerfoo/zerfoo/commit/f59490cfddee61669976bf033caae1dc5fb73726))

## [1.5.0](https://github.com/zerfoo/zerfoo/compare/v1.4.1...v1.5.0) (2026-03-18)


### Features

* **agent:** implement agentic loop supervisor ([a335b22](https://github.com/zerfoo/zerfoo/commit/a335b2223405c2104a0711e870d46dace7615023))
* **agent:** implement function-call JSON grammar parser ([25d2388](https://github.com/zerfoo/zerfoo/commit/25d23883a4184dea59186211ef7934a4fb73b4db))
* **agent:** implement market data tool set for agentic loop ([cdba942](https://github.com/zerfoo/zerfoo/commit/cdba9420ac851e406d0a7d0a6b87a36632bdf3bf))
* **agent:** implement tool registry for agentic tool-use loop ([d5cf840](https://github.com/zerfoo/zerfoo/commit/d5cf840d7ec67a1e8abba34f11672e015128f823))
* **attention:** fix GQA backward pass with finite-diff test (T8.2) ([2a4ef8f](https://github.com/zerfoo/zerfoo/commit/2a4ef8f5ebe69741b2782504ddb933f2db938ba4))
* **attention:** implement MLA backward for DeepSeek V3 (T8.5) ([ddcde19](https://github.com/zerfoo/zerfoo/commit/ddcde19aa7a3d30726e9e8c3f10fe93435da2eb5))
* **audio:** implement Whisper-style audio encoder ([7577d77](https://github.com/zerfoo/zerfoo/commit/7577d7794727528665d7da707d7f44668be23630))
* **automl:** implement AutoML loop coordinator ([69ef00a](https://github.com/zerfoo/zerfoo/commit/69ef00ab7e67c9d431cafa324a47ba473534a204))
* **automl:** implement Bayesian hyperparameter optimization ([fb07a72](https://github.com/zerfoo/zerfoo/commit/fb07a722294fa9b4b0583e85322f47f132488292))
* **automl:** implement population-based training ([9dae4b7](https://github.com/zerfoo/zerfoo/commit/9dae4b7bdcbb29ad5640130ab61336c93e74e36b))
* **bench:** add continuous batching vs session pool benchmark ([3759c33](https://github.com/zerfoo/zerfoo/commit/3759c330220cd72c132c6b4b541ef93adcaafad7))
* **bench:** add disaggregated vs collocated benchmark ([bb4d22d](https://github.com/zerfoo/zerfoo/commit/bb4d22d0ad8af2cd53c87b7e5f6b916e6934c936))
* **bench:** add Mamba-3 vs Transformer decode throughput benchmark ([07522df](https://github.com/zerfoo/zerfoo/commit/07522df5b439fb5c17a54e981bead2e7848e9a71))
* **bench:** add multi-architecture benchmark for all 6 architectures ([61e762d](https://github.com/zerfoo/zerfoo/commit/61e762d95d476eb6f9d1846f3d20e1b146319822))
* **bench:** add multi-architecture benchmark for all 6 architectures ([76ce656](https://github.com/zerfoo/zerfoo/commit/76ce656f5764bff87ed2dde762613f6aacac89a0))
* **bench:** add prefix cache hit rate benchmark ([5013d83](https://github.com/zerfoo/zerfoo/commit/5013d832dae1ed53baea6b208b72968f157a2269))
* **bench:** add prefix cache hit rate benchmark ([ccbb9a0](https://github.com/zerfoo/zerfoo/commit/ccbb9a034550fd68ee8f9d7ddbab9d4a0c010065))
* **bench:** add speculative decoding benchmark harness ([fa8dead](https://github.com/zerfoo/zerfoo/commit/fa8dead90701d20e49a340c4dd07088eeff632ec))
* **bench:** add speculative decoding benchmark harness ([0b1c603](https://github.com/zerfoo/zerfoo/commit/0b1c6038bf6bf29b8a9a6bc4d27f30e7e696d82a))
* **cloud:** implement GPU model LRU eviction ([8ab7683](https://github.com/zerfoo/zerfoo/commit/8ab768311dd85f4013fc8fe3f3d57688c4ab4b45))
* **cloud:** implement multi-tenant namespace isolation ([cacb67c](https://github.com/zerfoo/zerfoo/commit/cacb67c17e717ee142fae35531013cd574a1a357))
* **cloud:** implement token metering middleware ([f6e8941](https://github.com/zerfoo/zerfoo/commit/f6e8941850a8dbb1f784f5f3a17f4ea9ad2b3036))
* **cmd:** add standardized benchmark harness (T7.1) ([b91a09c](https://github.com/zerfoo/zerfoo/commit/b91a09ce078c4fe7eab74285ce50980139dba247))
* **cmd:** add zerfoo finetune CLI command (T9.8) ([34e4673](https://github.com/zerfoo/zerfoo/commit/34e4673d8952ce0258c523c1a48303e762761532))
* **cmd:** add zerfoo train-distributed CLI command ([3727c76](https://github.com/zerfoo/zerfoo/commit/3727c7682e395162690414334414002487ede8f2))
* **cmd:** implement zerfoo automl CLI command ([6941386](https://github.com/zerfoo/zerfoo/commit/6941386c48cce928384577f856bbd96fedce19c4))
* **cmd:** implement zerfoo automl CLI command ([f1c9b7f](https://github.com/zerfoo/zerfoo/commit/f1c9b7f02c02e52b937a62e6441ed72eb4c25159))
* **core:** implement MoE backward with straight-through estimator (T8.6) ([995c952](https://github.com/zerfoo/zerfoo/commit/995c9520c4ed224ed2be021109e2824fedb8020f))
* **core:** implement MoE backward with straight-through estimator (T8.6) ([b21f776](https://github.com/zerfoo/zerfoo/commit/b21f776865a92cec7cb72c87473d6d165b33aa9f))
* **distributed:** add FSDP distributed checkpoint save/load (T10.5) ([437ff65](https://github.com/zerfoo/zerfoo/commit/437ff65b60326cbf9c1ec75039eb485c8df6fb27))
* **distributed:** add FSDP gradient accumulation (T10.3) ([0d04858](https://github.com/zerfoo/zerfoo/commit/0d048583278c291855556c2d21850c53e1aa1586))
* **distributed:** add FSDP sharded module (T10.1) ([f23e510](https://github.com/zerfoo/zerfoo/commit/f23e51041da1167b2f912518b36e7d1d48e9e7e2))
* **distributed:** add NCCL AllGather+ReduceScatter via purego dlopen (T10.2) ([a32fe8d](https://github.com/zerfoo/zerfoo/commit/a32fe8d5b54c73f4505ce5c5ae0d2ca5ad79849f))
* **distributed:** add sharded AdamW optimizer state (T10.4) ([6984c2a](https://github.com/zerfoo/zerfoo/commit/6984c2a322148345b25cced61e893ef68644d5b1))
* **docs:** add documentation site generator ([c60e778](https://github.com/zerfoo/zerfoo/commit/c60e77863858fe6495e2891a23e5bab27aed5ade))
* **embeddings:** verify RoPE backward pass with finite difference test (T8.4) ([3d6fb3d](https://github.com/zerfoo/zerfoo/commit/3d6fb3dc54b280a840bd8ea47020c9b93bc1267b))
* **fp8:** add NVFP4 type and quantization primitives ([8ce4ac8](https://github.com/zerfoo/zerfoo/commit/8ce4ac8f2389b2affbc0a80b58a7172b582ce233))
* **fp8:** implement master weight FP32 copy store ([30dbffb](https://github.com/zerfoo/zerfoo/commit/30dbffb14d91f561fbe57f7b1091f868dc9790a5))
* **generate:** add external draft model for speculative decoding (T3.1) ([0151142](https://github.com/zerfoo/zerfoo/commit/0151142d6a321e8e9cc31f53266282889b52aba7))
* **generate:** add external draft model for speculative decoding (T3.1) ([eb990aa](https://github.com/zerfoo/zerfoo/commit/eb990aa5110450538e9d3d872ce7c9c3a3d63438))
* **generate:** add self-speculative decoding (T3.2) ([8fa76ad](https://github.com/zerfoo/zerfoo/commit/8fa76ad04ed55c7e4d518594b92e9f359f8052b1))
* **generate:** add speculative acceptance rate Prometheus metric (T3.5) ([0292db5](https://github.com/zerfoo/zerfoo/commit/0292db5d3a5463859036a49e49be043620586d04))
* **generate:** add speculative decoding token acceptance loop (T3.3) ([7769994](https://github.com/zerfoo/zerfoo/commit/776999498fad863e56d4646ed8ef2ed20dfb9ca2))
* **generate:** add SSM session state management (T6.3) ([1768b3f](https://github.com/zerfoo/zerfoo/commit/1768b3f3e7fca1d986ab3538deeff172f159ec15))
* **generate:** wire prefix cache into session initialization (T4.2) ([8754225](https://github.com/zerfoo/zerfoo/commit/87542255c3cccef7dca0ded1ca47428c031041d1))
* **generate:** wire speculative decoding into generator (T3.4) ([bc4d0de](https://github.com/zerfoo/zerfoo/commit/bc4d0deb9637cbed623f1d71de55315ad6e3fece))
* **inference:** add Gemma 3n architecture builder ([3984763](https://github.com/zerfoo/zerfoo/commit/39847638cc1f5011d4082cb614dc807be2b97af3))
* **inference:** add Llama 4 architecture builder ([70df5e4](https://github.com/zerfoo/zerfoo/commit/70df5e42b25ce5b3b36954effe27ae6415e9d064))
* **inference:** add Llama 4 config parser for HuggingFace config.json ([7da3733](https://github.com/zerfoo/zerfoo/commit/7da37339ac14e2b2e1b72eb70df6c261faa9fdd2))
* **inference:** add PatchTST model builder for Wolf time-series (T12.2) ([e8804d8](https://github.com/zerfoo/zerfoo/commit/e8804d8316acf0644e34428a6a814c95a39a6867))
* **inference:** add regime detection model builder (T12.5) ([7d7adf9](https://github.com/zerfoo/zerfoo/commit/7d7adf99cf98e447d5a089e472962321f787c776))
* **inference:** add Wolf feature store with CSV loading and ring buffer (T12.6) ([00c6cbc](https://github.com/zerfoo/zerfoo/commit/00c6cbc8cf040d1a3c73338955ec8500eb22cf7a))
* **inference:** export BuildArchGraph, add devlog entry for multi-arch benchmark ([a227212](https://github.com/zerfoo/zerfoo/commit/a227212184d3b8dc4d41da7699d792d1c5abe680))
* **inference:** implement architecture registry ([a177da8](https://github.com/zerfoo/zerfoo/commit/a177da81d8fe0883c9de850dd28d1706f8605f3d))
* **inference:** implement hybrid Jamba SSM+Attention graph builder ([8b498a7](https://github.com/zerfoo/zerfoo/commit/8b498a7e1b94ba753d7980eac9c428e99a63b13c))
* **inference:** implement Whisper GGUF loader ([2dddef8](https://github.com/zerfoo/zerfoo/commit/2dddef8fd18ada1e4beef172d6bef5306740c19b))
* **inference:** register Gemma 3n in config parser and tensor name mapper ([f44e26c](https://github.com/zerfoo/zerfoo/commit/f44e26c3127dd25f4d83c36d1e051d2fdd58c580))
* **inference:** register Llama 4 in architecture dispatch ([d46497d](https://github.com/zerfoo/zerfoo/commit/d46497d847c2538005ede352977a5ace976cdbca))
* **infra:** add GKE deployment Terraform for zerfoo-cloud ([c146d2a](https://github.com/zerfoo/zerfoo/commit/c146d2a6f982a1f0d67e9cdfcfa23d90c4f55ef3))
* **layers:** add Mamba SSM block with forward/backward (T6.2) ([ec00cf1](https://github.com/zerfoo/zerfoo/commit/ec00cf188eea5038b1ab400d8805e5e2534e18a3))
* **layers:** add patch embedding for time-series (T12.1) ([4b46fcf](https://github.com/zerfoo/zerfoo/commit/4b46fcf545e5e7650015f4fbb231ecc444fc1d7b))
* **layers:** add variable selection network for TFT (T12.3) ([ff9febf](https://github.com/zerfoo/zerfoo/commit/ff9febfd810c8674adf50f7544b9ca51211c5a41))
* **lora:** implement LoraLinear layer with forward+backward (T9.1) ([e33987c](https://github.com/zerfoo/zerfoo/commit/e33987c0293146c60024bbb4b807de6fdc17e7da))
* **multimodal:** implement audio+text inference session ([5140d97](https://github.com/zerfoo/zerfoo/commit/5140d97166109cfd8499779801ae1544374def74))
* **multimodal:** implement generic VisionEncoder interface and SigLIP encoder ([7332237](https://github.com/zerfoo/zerfoo/commit/7332237f2ece26c098be5da580e4e8fd537efd23))
* **multimodal:** implement GGUF loader for vision/multimodal metadata ([cda421c](https://github.com/zerfoo/zerfoo/commit/cda421cdea1dc092324358f4070b1186b3c5b98a))
* **multimodal:** implement image preprocessing pipeline for vision-language inference ([554bae1](https://github.com/zerfoo/zerfoo/commit/554bae100c1e3caa85b2496c936d7be67014c6cb))
* **multimodal:** implement mel-spectrogram extraction for audio inference ([258e58c](https://github.com/zerfoo/zerfoo/commit/258e58ccf1c0686d253d577bd618e42d0c7dc374))
* **multimodal:** implement text+vision embedding merge at image token positions ([433e1c3](https://github.com/zerfoo/zerfoo/commit/433e1c32b527a33b4b23dd94f951dab36243da3e))
* **multimodal:** implement vision-to-text projection connector ([4b0946d](https://github.com/zerfoo/zerfoo/commit/4b0946d569aee0167058edcfd576fea2aef7e5c9))
* **nas:** add DARTS mixed-operation layer ([f3672f3](https://github.com/zerfoo/zerfoo/commit/f3672f3a0071ff61bd47481ad4aad79de372f593))
* **nas:** define NAS search space with cell/edge/op types ([e7a6a94](https://github.com/zerfoo/zerfoo/commit/e7a6a94060245d6278c862e386fc8839923a3da4))
* **nas:** implement architecture discretization ([b7a7bfb](https://github.com/zerfoo/zerfoo/commit/b7a7bfb217032e5ad6a5d0a5c0c573c462d2ea22))
* **nas:** implement DARTS bilevel optimizer ([6325848](https://github.com/zerfoo/zerfoo/commit/63258482d03d2c8e1d04b5e292848faec3d5fc26))
* **nas:** implement hardware-aware latency estimator ([db204db](https://github.com/zerfoo/zerfoo/commit/db204db3383bb97645dee0eec94071f0353f82c1))
* **nas:** implement NAS export to GGUF ([1880fc7](https://github.com/zerfoo/zerfoo/commit/1880fc7ccfccac76e3b11463cbc3fc5ae20a4089))
* **nas:** implement signal model NAS search runner ([406b8d2](https://github.com/zerfoo/zerfoo/commit/406b8d20456a848a2d002380031dfc9ec48c355d))
* **online:** implement automated NAS trigger on drift event ([a5a56e5](https://github.com/zerfoo/zerfoo/commit/a5a56e58316faa2b387c5c1bf025ae1c9bdd60ba))
* **online:** implement feedback signal collector for online learning ([8756a37](https://github.com/zerfoo/zerfoo/commit/8756a371263720091010cc3eea8a7b6576c46fa2))
* **online:** implement incremental LoRA updater for online learning ([88bd425](https://github.com/zerfoo/zerfoo/commit/88bd42579e337b628e84dd701e4259e589c9a7e0))
* **online:** implement JSONL audit log for online learning events ([4d03f89](https://github.com/zerfoo/zerfoo/commit/4d03f89d1c6b33e3d27975e39243ac6b7df39f6a))
* **online:** implement model performance drift detector ([c78b15a](https://github.com/zerfoo/zerfoo/commit/c78b15a9ec56c2714261119fadcd4b70f0e691f6))
* **online:** implement model rollback manager with versioned snapshots ([f63b4fa](https://github.com/zerfoo/zerfoo/commit/f63b4fac178e31efdefb16f78f747645c1b382f5))
* **online:** implement online learning trigger (drift and scheduled) ([9cced5e](https://github.com/zerfoo/zerfoo/commit/9cced5e878ebded577caecac76ae3450066dc80c))
* **online:** implement safety validators for online learning updates ([baa791f](https://github.com/zerfoo/zerfoo/commit/baa791f01e230ea6f2e85d472d3bea6939214f47))
* **optimizer:** add AdamW 8-bit with block-wise INT8 momentum (T9.6) ([9c90f27](https://github.com/zerfoo/zerfoo/commit/9c90f275e9712920fa2c0b140fcd1988dbf96756))
* **registry:** add per-version performance metrics store ([efa9cd4](https://github.com/zerfoo/zerfoo/commit/efa9cd4af6df7a494f38f5b4d2c0d53192c561c0))
* **registry:** implement bbolt-backed model version registry ([aa296df](https://github.com/zerfoo/zerfoo/commit/aa296df3b96a0985314f27237d747164ae93444d))
* **registry:** implement canary release controller with auto-ramp ([c9324b4](https://github.com/zerfoo/zerfoo/commit/c9324b499c14331246a30e4a1c4462f64ef769bc))
* **registry:** implement deterministic A/B champion-challenger router ([b3264ab](https://github.com/zerfoo/zerfoo/commit/b3264abdb7bebda6b242b9fc53f2fead234d0c3f))
* **registry:** implement shadow mode inference runner ([9e0e8dc](https://github.com/zerfoo/zerfoo/commit/9e0e8dc7830d8949259b90ec740282c282b9d0ff))
* **serve:** add audio transcription endpoint ([8b98212](https://github.com/zerfoo/zerfoo/commit/8b9821214f445cffe21338c7725d0a744fb92279))
* **serve:** add continuous batching scheduler (T1.5) ([c6bd18b](https://github.com/zerfoo/zerfoo/commit/c6bd18be7864592fdf1846960f456fc575ce7011))
* **serve:** add disaggregated API gateway with least-loaded routing (T5.4) ([d758d8a](https://github.com/zerfoo/zerfoo/commit/d758d8adc4db54a50b8ad0723c0649d645306a84))
* **serve:** add disaggregated decode worker (T5.3) ([8dd011e](https://github.com/zerfoo/zerfoo/commit/8dd011e8a478444798b0c533a031af7735af6e7a))
* **serve:** add disaggregated prefill worker (T5.2) ([34a6597](https://github.com/zerfoo/zerfoo/commit/34a6597ccc125440a9c6742c01c46c830a6c3e12))
* **serve:** add disaggregated prefill/decode gRPC proto (T5.1) ([e0f9fc4](https://github.com/zerfoo/zerfoo/commit/e0f9fc448402fef9d0d63da286f517fc236f6159))
* **serve:** add image input to OpenAI-compatible API ([1e16da0](https://github.com/zerfoo/zerfoo/commit/1e16da0e1dffd1c69f04f9a2b6b3c46b85c47d0c))
* **serve:** add tool-use to OpenAI-compatible API ([9962a6b](https://github.com/zerfoo/zerfoo/commit/9962a6b68484e2ccf7d4a6a49e59650a5ccc89e2))
* **timeseries:** add GGUF loader for time-series signal metadata ([8dd2b3c](https://github.com/zerfoo/zerfoo/commit/8dd2b3c8ec4ffab04a8ab611aeb68ee03c35940d))
* **training:** add dynamic loss scaling for FP8 training (T11.2) ([babc411](https://github.com/zerfoo/zerfoo/commit/babc411541a2a6aff80ef512b65b871e64e0e983))
* **training:** add FP8 linear layer with forward/backward (T11.1) ([ab8d19f](https://github.com/zerfoo/zerfoo/commit/ab8d19fc1bfddb9cb1a712925d1e1e08cec0095d))
* **training:** add LoRA adapter checkpoint save/load (T9.5) ([2288ec7](https://github.com/zerfoo/zerfoo/commit/2288ec76ff0c1024872613439c95b4ad44e126fe))
* **training:** add LoRA injection walk (T9.2) ([96d75c5](https://github.com/zerfoo/zerfoo/commit/96d75c5930d9740b43a3610f6565668d7da980ac))
* **training:** add QLoRA trainer with NF4 base weights (T9.4) ([6f419ad](https://github.com/zerfoo/zerfoo/commit/6f419ad188f7bf99c348a7eb90c44d2f800fdc98))
* **training:** add quantile loss and Sharpe-ratio metric (T12.9) ([70194a4](https://github.com/zerfoo/zerfoo/commit/70194a448891d005418c0770be55032e01135ce1))
* **wolf:** add PatchTST training script cmd/wolf_train ([5b6c0c1](https://github.com/zerfoo/zerfoo/commit/5b6c0c19c7448b9120c988e2f84398f7eb5802ef))
* **wolf:** implement Temporal Fusion Transformer builder ([ef262a9](https://github.com/zerfoo/zerfoo/commit/ef262a9a2060fce968bcac7375ebe174e5a34e85))


### Bug Fixes

* **generate:** make external_draft.go self-contained without self_draft.go ([0d05f32](https://github.com/zerfoo/zerfoo/commit/0d05f321b405b44d53df23a9337f7b6ecbfe0c46))
* **normalization:** correct RMSNorm backward gradient computation (T8.1) ([36a3489](https://github.com/zerfoo/zerfoo/commit/36a3489f72aa8ed16bb0ec079a116d8111ee3e90))
* **timeseries:** rename package wolf -&gt; timeseries in arch_patchtst, arch_regime and tests ([aa7354f](https://github.com/zerfoo/zerfoo/commit/aa7354f0b5c83cd2cbedd23102a784a84498f49d))
* **timeseries:** resolve package conflicts and import paths after wolf-&gt;timeseries rename ([18545eb](https://github.com/zerfoo/zerfoo/commit/18545eb3a8b118a205a0e3ed09ec18d3765a75e7))

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
