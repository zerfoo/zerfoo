# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
