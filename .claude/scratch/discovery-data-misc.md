# Discovery: Data Model, Timeseries, and Miscellaneous Subsystems

Audit of the zerfoo ML framework's non-core subsystems, covering timeseries models,
data model structures, operational infrastructure, and security-relevant patterns.

---

## 1. Timeseries Models (`inference/timeseries/`)

### 1.1 PatchTST (`arch_patchtst.go`)

**Config:** `PatchTSTConfig` -- PatchLen, Stride, NumLayers, NumHeads, DModel, Horizon, NumVars.

**Architecture:** Channel-independent PatchTST. Each variable is processed independently through:
1. Patch embedding (shared `timeseries.PatchEmbed[T]` weights across variables)
2. Stacked transformer encoder layers (pre-norm self-attention with GELU FFN)
3. Final RMSNorm + mean pooling over patches
4. Projection head mapping to `[batch, horizon, numVars]`

**Key data structures:**
- `patchTSTNode[T]` -- single graph node wrapping entire forward pass
- `encoderLayer[T]` -- holds RMSNorm, Q/K/V/O projection parameters, FFN1/FFN2 parameters
- All parameters are `*graph.Parameter[T]` from ztensor

**State:** No mutable state beyond parameters. Implements `graph.Node[T]` interface.
Backward returns nil (inference-only).

**Security note:** Manual `extractVariable` does raw slice indexing with bounds derived from shape metadata -- no explicit bounds checking beyond shape validation at entry.

### 1.2 TFT - Temporal Fusion Transformer (`arch_tft.go`)

**Config:** `TFTConfig` -- NumStaticFeatures, NumTemporalFeatures, HiddenDim, NumHeads, NumLSTMLayers, HorizonLen, Quantiles.

**Architecture:** Full TFT pipeline:
1. Static covariate encoder: Linear + GRN (from `layers/timeseries`)
2. Temporal VSN (Variable Selection Network) per timestep
3. Stacked LSTM encoder (custom `lstmLayer[T]` with full gate parameters)
4. Multi-head self-attention over LSTM outputs + residual
5. Mean pooling + quantile projection head -> `[batch, H, numQuantiles]`

**Key data structures:**
- `tftNode[T]` -- holds staticProj, staticGRN, temporalVSN, lstmLayers, attention params, quantileProj
- `lstmLayer[T]` -- 12 parameters per layer (Wi/Ui/bi for each of 4 gates), forget gate bias initialized to 1
- `sigmoidFn` -- composed from Exp/AddScalar/Div (no native sigmoid kernel)

**State:** LSTM hidden/cell states are initialized to zeros each forward pass (no persistent state).

### 1.3 RegimeDetector (`arch_regime.go`)

**Config:** `RegimeConfig` -- InputDim, HiddenDim, NumLayers, SeqLen, NumClasses (default 4).

**Architecture:** Stacked GRU layers + linear classifier + softmax.
- `gruLayer[T]` -- 6 parameters (Wz/Uz, Wr/Ur, Wn/Un) per layer
- `classifierHead[T]` -- weights + bias
- Outputs softmax probabilities for bull/bear/sideways/volatile regimes

**Shared utilities:** `newParam[T]` uses `rand.Float64()*0.1 - 0.05` for parameter init (seeded by global rand). `extractTimestep`, `expandHidden`, `makeOnes` are shared helpers.

### 1.4 GGUF Loader (`gguf_loader.go`)

**Config:** `TimeSeriesSignalConfig` -- PatchLen, Stride, InputFeatures, HiddenDim, NumHeads, NumLayers, HorizonLen.

GGUF metadata keys use `ts.signal.*` prefix. Required keys: `ts.signal.patch_len`, `ts.signal.input_features`. All others have defaults.

`LoadPatchTSTFromGGUF[T]` parses GGUF, extracts metadata, delegates graph construction to a `PatchTSTBuilder[T]` callback.

**Security note:** File path is sanitized via `filepath.Clean`. `getMetaInt` handles uint32/int/int64/uint64/float64/float32 type switches for GGUF metadata values.

---

## 2. Cross-Asset Module (`crossasset/`)

**Config:** `Config` -- NSources, FeaturesPerSource, DModel, NHeads, NLayers, DropoutRate, LearningRate.

**Architecture:** Pure float64 cross-attention transformer (NOT using ztensor). Each source attends to ALL other sources via multi-head cross-attention. Classification head maps to 3 classes (Long/Short/Flat).

**Key data structures:**
- `Model` -- layers, inputW/inputB (per-source input projections), headW/headB (3-class classifier)
- `layer` -- qW/kW/vW/outW, lnGamma/lnBeta, ffnW1/ffnB1/ffnW2/ffnB2/ffnGamma/ffnBeta
- All weights stored as flat `[]float64` slices

**Training:** SGD with cross-entropy loss. Only backprops through the classification head; transformer layers are NOT trained (frozen feature extractor).

**Methods:** Forward, Predict (directions + confidences), Train, AttentionWeights (returns [n_sources][n_sources] matrix).

---

## 3. Monitor / Drift Detection (`monitor/`)

Two drift detectors implementing `DriftDetector` interface (`Observe(float64) bool`):

### 3.1 Page-Hinkley Test (`PageHinkley`)
- Detects mean shifts in sequential streams
- State: n, sum, cumSum, minSum (mutex-protected)
- Parameters: Delta (tolerance, default 0.005), Lambda (threshold, default 50)

### 3.2 ADWIN (`ADWIN`)
- Adaptive windowing with Hoeffding bound
- State: window slice, sum (mutex-protected)
- Drops older sub-window when drift detected
- Parameters: Confidence (default 0.002)

**Security note:** Both detectors are thread-safe (sync.Mutex). ADWIN window grows unbounded -- no maximum window size cap.

---

## 4. Model DSL (`modeldsl/`)

Declarative model definition with validation, graph compilation, optimization passes, and training.

### 4.1 Definition Layer (`dsl.go`)
- `ModelDef` -> Parse -> `ModelGraph` with topological sort, cycle detection
- Layer types: linear, rmsnorm, silu, softmax, attention
- `ConnectionDef` specifies directed edges; inputs/outputs auto-detected from parentless/childless nodes

### 4.2 Graph (`graph.go`)
- `ModelGraph.Build(inputDim, outputDim)` -> `Model` with resolved dimensions
- Dimension propagation: linear uses output_dim param or inherits; attention/elementwise preserve

### 4.3 Model Runtime (`model.go`)
- `Model.Forward(input)` executes layers in topological order
- Executable layer types: linearLayer, rmsnormLayer, siluLayer, softmaxLayer, attentionLayer
- All operate on `[]float64` vectors

### 4.4 Optimization Passes (`optimize.go`)
Three passes: `ConstantFolding` (idempotent op removal), `DeadNodeElimination` (backward reachability from output), `OperatorFusion` (RMSNorm+SiLU fusion pairs). All produce new `ModelGraph` instances (immutable transforms).

### 4.5 Training (`train.go`)
- `Model.Train(config, samples)` -- MSE loss, SGD, per-sample forward-backward
- `trainableLayer` interface extends `execLayer` with `backward` and `params`
- `linearLayerT` caches lastInput for backward; softmaxLayerT caches lastOutput
- `BuildTrainable` constructs model with training-capable layers
- Attention training simplified: single-token case means Q/K gradients zeroed

---

## 5. Auto-Optimization (`autoopt/`)

### 5.1 Hardware Profile (`profile.go`)
`HardwareProfile` -- CPU (cores, SIMD flags, cache sizes, RAM) + GPU (backend, name, memory, compute cap, multi-GPU).

### 5.2 Kernel Selection (`kernel.go`)
10 kernel classes (GEMM, GEMV, Attention, RMSNorm, Softmax, RoPE, SiLU, Elementwise, QuantGEMM, QuantDot).
11 implementations (GenericCPU, NEON, AVX2, AVX512, CUDA, CUDAFused, ROCm, ROCmFused, Metal, OpenCL).

`SelectKernels(hw)` returns `KernelSelection` with per-class impl mapping. Logic:
- CUDA sm70+ enables fused ops; sm80+ enables flash attention
- VRAM < 8 GiB triggers quantized GEMM preference
- ROCm enables fused + flash attention
- Metal: basic fused, no flash attention
- OpenCL: standard only
- CPU: SIMD-based (AVX512 > AVX2 > NEON > generic)

### 5.3 Quantization Recommendation (`quantrec.go`)
8 quant formats: NVFP4, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP8, BF16, FP16.
`RecommendQuant(hw, model, pref)` -- picks best format that fits in memory. NVFP4 requires CUDA compute cap >= 8.9.

### 5.4 Codegen (`codegen.go`)
`KernelCodegen` generates hardware-tuned `KernelConfig` (tile sizes, unroll, shared mem, grid/block dims).
Templates: GEMMTemplate, GEMVTemplate, ElementwiseTemplate.
CUDA Ampere+ uses 128x128x32 tiles, 48 KiB shared mem, 8 warps.

### 5.5 Scheduler (`scheduler.go`)
Multi-device workload scheduling with 3 strategies:
- `RoundRobin` -- mutex-protected counter, skips memory-insufficient devices
- `LoadBalanced` -- scores by utilization + 0.1*queueDepth
- `Priority` -- ordered device type preference with LoadBalanced fallback

`Scheduler` tracks assignments, supports `AutoMigrate(threshold)` for workload migration.

### 5.6 Workload Splitting (`split.go`)
`WorkloadSplitter` with cost model (compute-bound vs memory-bound estimate).
`Split(ops)` assigns each op to the device minimizing execution time + transfer cost.
Transfer cost: 16 GB/s PCIe Gen4 default.

### 5.7 Next-Gen GPU Support (`nextgen.go`, `tma.go`, `wgmma.go`)
- GPU generations: Ampere (SM8x), Hopper (SM9x), Blackwell (SM10x+)
- Hopper: TMA, WGMMA, FP8 native, cluster size 8, 228 KiB shared mem
- Blackwell: +FP4, cluster primitives, max cluster 16, 256 KiB shared mem
- Execution paths: Standard, TMA, WGMMA, TMA+WGMMA, FP4Cluster
- `TMAConfig` validation: 2D/3D, box dims [1,256], row alignment 16B, max 128 KiB
- `WGMMAConfig` validation: M=64 fixed, N multiples of 8 [8,256], K=16 (FP16/BF16) or 32 (FP8/INT8)

---

## 6. Model Cache (`modelcache/`)

LRU file cache for GGUF models on K8s nodes (DaemonSet use case).

**Key structures:**
- `Cache` -- dir, maxSize, entries map (mutex-protected)
- `entry` -- ref, path, size, lastUsed

**Operations:** Get (touch lastUsed), Put (copy file, evict if needed), Evict (LRU by lastUsed), Prefetch (batch download).

**Security note:** `sanitize(ref)` replaces non-alphanumeric chars (except `-_.`) with `_` to prevent path traversal in filenames. File permissions: not explicitly set on copy (inherits from os.Create default).

---

## 7. Registry (`registry/`)

### 7.1 Local Registry (`registry.go`)
`LocalRegistry` -- filesystem-backed model cache at `~/.zerfoo/models/`.
- `ModelInfo` struct: ID, Path, Architecture, VocabSize, MaxSeqLen, Size
- Directory layout: `<cacheDir>/<org>/<model>/config.json`
- `PullFunc` callback pattern for download delegation

**Security:** `modelDir` validates containment -- resolved path must have cachePrefix. Directory permissions 0o750, config.json 0o600.

### 7.2 HuggingFace Pull (`pull.go`)
Downloads GGUF models from HuggingFace Hub.
- `HFPullOptions` -- APIURL, CDNURL, Token (from env `HF_TOKEN`), Quant filter
- `resolveGGUFByQuant` -- two-pass matching (exact segment, then substring)
- `shouldDownload` -- filters to .gguf, .onnx, tokenizer files, config.json
- Progress reporting via `ProgressFunc` callback

**Security:** `downloadFile` validates filenames against `..` path traversal AND verifies resolved path stays within targetDir. Auth via Bearer token header. Token read from environment variable.

### 7.3 OCI Registry (`oci.go`, `manifest.go`)
Full OCI distribution spec client for push/pull of GGUF models as OCI artifacts.
- Custom media types: `application/vnd.zerfoo.model.gguf.v1`, `application/vnd.zerfoo.model.config.v1+json`
- `Push` -- blob upload + config + manifest PUT
- `Pull` -- manifest resolve + GGUF layer download
- `parseReference` -- handles `registry/repo:tag` and `@sha256:...` digest formats

**Security:** Basic auth via `SetBasicAuth`. Credentials passed via `WithCredentials` option. No TLS certificate pinning. `os.ReadFile` on model path for push (potential large file in memory). SHA256 digest computation for blob integrity.

---

## 8. Shutdown Coordination (`shutdown/`)

`Coordinator` -- reverse-order cleanup via `Closer` interface.
- `Register(Closer)` -- appends to list (no-op after shutdown)
- `Shutdown(ctx)` -- idempotent (done flag), reverse iteration, collects errors
- Thread-safe (sync.Mutex), releases lock during close calls

---

## 9. Recovery / Auto-Retrain (`recover/`)

Pipeline: Detect -> Rollback -> Retrain -> Validate -> Redeploy.
- `AutoRetrainConfig` -- callback functions for each phase (RetrainFn required)
- `AutoRetrain.Run(detector, stream)` -- feeds stream to drift detector, triggers pipeline on drift
- `PipelineError` wraps errors with phase context
- Integrates with `monitor.DriftDetector` interface

---

## 10. Regime Detection (`regime/`)

Full Gaussian HMM implementation with Baum-Welch EM training.
- `HMM` -- Initial probs, Transition matrix, Means, Variances (Gaussian emissions)
- `Fit` -- Baum-Welch with scaled forward-backward, convergence checking
- `Viterbi` -- most likely state sequence (log-space dynamic programming)
- `Predict` -- posterior probabilities for final observation
- `initEmissionsFromData` -- percentile-based initial means for better convergence

**State:** All parameters mutable during Fit. Uses `math/rand` with configurable seed.

---

## 11. Reinforcement Learning (`rl/`)

### 11.1 Core Types (`rl.go`)
State = []float64, Action = []float64, Experience (SARS+done), Environment/Agent interfaces.

### 11.2 PPO (`ppo.go`)
Full analytical-gradient PPO with GAE:
- `policyNet` -- 2-layer MLP (tanh activation), outputs mean + state-independent logStd
- `valueNet` -- 2-layer MLP (tanh), outputs scalar value
- Clipped surrogate objective with gradient norm clipping (max 1.0)
- Multi-epoch minibatch updates with advantage normalization

### 11.3 SAC (`sac.go`)
Soft Actor-Critic with twin critics and automatic entropy tuning:
- Actor: 3-layer MLP (ReLU), outputs mean+logStd, tanh squashing with reparameterization trick
- Twin critics: 3-layer MLPs taking (state, action) concatenation
- Target critics with Polyak averaging (tau=0.005 default)
- logStd clamped to [-20, 2]
- Full analytical backprop through critic for actor gradient

### 11.4 Replay Buffer (`replay.go`)
Ring buffer with FIFO eviction. Supports uniform sampling and prioritized sampling (binary search on cumulative weights).

---

## 12. Graph Neural Networks (`gnn/`)

### 12.1 GCN (`gcn.go`)
Multi-layer Graph Convolutional Network:
- Normalized adjacency: D_tilde^{-1/2} A_tilde D_tilde^{-1/2} (self-loops added)
- ReLU activations, optional dropout (inverted scaling)
- Training: cross-entropy loss, full backprop through all layers

### 12.2 GAT (`gat.go`)
Multi-head Graph Attention Network:
- Per-head: W_head, a_left, a_right attention vectors
- LeakyReLU(0.2) attention scores with masked softmax (neighbors only)
- Output projection from concat(heads) to output dim
- Training: cross-entropy, approximate gradient for attention vectors

---

## 13. Gaussian Processes / Genetic Programming (`gp/`)

Tree-based genetic programming (symbolic regression):
- `Primitive` -- Name, Arity, Func
- `Node` -- expression tree with recursive Evaluate
- `Program` -- Root + Fitness
- `Evolve` -- generational GP with tournament selection, subtree crossover, mutation, elitism
- Ramped half-and-half initialization
- Built-in primitives: Add, Sub, Mul, ProtectedDiv, Sin, Cos, Variable, Constant

---

## 14. Tabular Models (`tabular/`)

MLP for tabular prediction built on ztensor engine:
- `Model` -- configurable hidden layers with ReLU or GELU activation
- 3-class output (Long/Short/Flat) with softmax
- He (Kaiming) weight initialization
- Uses `compute.Engine[float32]` for MatMul/Add/Softmax/UnaryOp

Additional files (not fully read): TabNet, FT-Transformer, SAINT, ResNet, ensemble, LoRA, pretraining, save/load, training with GPU support.

---

## 15. Federated Learning (`federated/`)

### 15.1 Core (`federated.go`)
- `Strategy` interface: Aggregate + SelectClients
- `Client` interface: Train + ID
- `Coordinator` manages rounds: select clients, distribute weights, collect updates, aggregate

### 15.2 FedAvg (`fedavg.go`)
Sample-weighted averaging. Validates weight dimension consistency and positive sample counts.

### 15.3 FedProx (`fedprox.go`)
Extends FedAvg with proximal term `(mu/2) * ||w - w_global||^2` (applied client-side during training, not during aggregation). `ProximalLoss` helper function provided.

### 15.4 Differential Privacy (`dp.go`)
`DPStrategy` wraps any Strategy:
- L2 norm clipping per client update
- Gaussian or Laplacian noise injection after aggregation
- `PrivacyAccountant` tracks cumulative (epsilon, delta) via basic composition
- Gaussian sigma: `ClipNorm * sqrt(2*ln(1.25/delta)) / epsilon`

**Security:** DP is correctly layered (clip -> aggregate -> noise). Accountant uses basic composition (not advanced/Renyi). Random seed fixed at 42 -- not cryptographically random.

---

## 16. Synthetic Data (`synth/`)

### 16.1 MarketVAE (`vae.go`)
Variational Autoencoder for synthetic data generation:
- Encoder: input -> hidden layers -> (mu, logvar) heads
- Decoder: latent -> reversed hidden layers -> output (linear final)
- Training: ELBO loss (MSE reconstruction + KL divergence), mini-batch SGD
- Reparameterization trick: z = mu + exp(0.5*logvar) * eps
- Full backprop through encoder and decoder

### 16.2 CrashGenerator (`crash.go`)
Extends MarketVAE for extreme tail scenario generation:
- Severity [1.0, 10.0] scales latent sampling toward tails
- Correlated shock: shared latent direction across timesteps
- Temporal decay: exponential fade over duration
- `sampleTailLatent` -- adds signed offset proportional to severity

---

## 17. Feature Engineering (`features/`)

Three transformers implementing `Transformer` interface (Transform on `*data.Dataset`):
- `LaggedTransformer` -- adds lagged features at specified lag offsets
- `RollingTransformer` -- adds rolling mean and standard deviation per feature over a window
- `FFTTransformer` -- adds top-k FFT frequency magnitudes per feature

External dependency: `gonum.org/v1/gonum/dsp/fourier` for FFT computation.

---

## 18. Shared Latent Space (`shared/`)

`LatentSpace` -- shared embedding space for multi-model alignment:
- `Register(name, inputDim)` -- adds model with Xavier-initialized projection/reconstruction matrices
- `Project(name, features)` -- maps to shared latent space
- `Retrieve(name, latent)` -- maps back to model-specific space
- `TrainProjections(data, config)` -- learns alignments from paired data
  - Joint loss: reconstruction MSE + alignment (pull corresponding samples together)
  - Learning rate decay: 0.999 per epoch

**State:** Projection matrices stored per model, mutex-protected for concurrent access.

---

## 19. Support Ticketing System (`support/`)

Full enterprise support portal:

### 19.1 Ticket (`ticket.go`)
- `Ticket` -- ID, CustomerID, Subject, Body, Priority (P0-P3), Status, AssignedTo, Comments, timestamps
- Status FSM: Open -> Triaged -> InProgress -> Resolved -> Closed (with valid transitions)
- `Store` -- thread-safe in-memory store with sequential ID generation

### 19.2 Router (`router.go`)
Priority-based routing rules with fallback assignee. Auto-transitions to Triaged on route.

### 19.3 SLA Tracker (`sla.go`)
- Default SLAs: P0=15min response/4hr resolution, P1=1hr/24hr, P2=4hr/72hr, P3=24hr/168hr
- `Check` evaluates response+resolution breaches
- `BreachHandler` callback pattern for alerts

### 19.4 Webhooks (`webhook.go`)
- Events: ticket.created, triaged, resolved, closed, comment_added, sla.breach
- `WebhookDispatcher` sends JSON to registered URLs (10s timeout)
- Event filtering per target

### 19.5 API (`api.go`)
HTTP handlers: POST /support/tickets, GET /support/tickets, GET /support/tickets/{id},
POST /support/tickets/{id}/comments, POST /support/tickets/{id}/close.

**Security note:** Error messages in CloseTicket embed raw error string in JSON response (`err.Error()` directly). No authentication/authorization on API endpoints. No CSRF protection. No rate limiting.

---

## 20. Meta-Learning (`meta/`)

MAML (Model-Agnostic Meta-Learning) implementation:
- `Task` interface: TrainData (support set) + TestData (query set)
- Inner loop: task-specific adaptation via gradient descent
- Outer loop: meta-parameter updates from averaged task gradients
- First-order MAML approximation (no second derivatives)
- `Adapt(task, steps)` -> `AdaptedModel` for few-shot prediction

---

## 21. Mobile Deployment (`mobile/`)

gomobile-compatible bindings:
- `Engine` wraps `inference.Model` with mutex protection
- gomobile-safe API: no slices/maps/channels in public interface
- `Tokenize` returns JSON string of token IDs
- `GenerateWithConfig` supports temperature, topP, topK, maxTokens

---

## 22. Causal Inference (`causal/`)

### 22.1 Graph Structure (`causal.go`)
`CausalGraph` with Edges (directed), AdjacencyMatrix, Parents/Children, IsDAG (Kahn's algorithm).

### 22.2 PC Algorithm (`pc.go`, `independence.go`)
Full PC algorithm implementation:
1. Skeleton discovery: conditional independence tests with increasing conditioning set sizes
2. V-structure orientation: X -> Z <- Y when Z not in sepSet(X,Y)
3. Meek's rules (all 4): iterative edge orientation

Statistical tests: Pearson partial correlation + Fisher z-test.
Normal quantile approximation: Beasley-Springer-Moro algorithm.

### 22.3 Interventions (`intervene.go`)
do-calculus via graph mutilation:
- OLS regression estimates linear causal coefficients
- Topological propagation through mutilated graph
- `Intervene(graph, intervention)` returns downstream effects
- Gaussian elimination with partial pivoting for normal equations

---

## 23. Integration Tests (`integration/`)

Two test files: `gemma3_quantized_test.go`, `production_smoke_test.go`.

---

## 24. Third-Party Integrations (`integrations/`)

### 24.1 LangChain Adapter (`integrations/langchain/`)
HTTP client adapter matching LangChain-Go's `schema.LLM` interface:
- `Call(ctx, prompt, stop...)` -- single prompt completion
- `Generate(ctx, prompts, stop...)` -- batch prompts
- Uses `/v1/chat/completions` endpoint
- No LangChain-Go dependency (interface-compatible)

### 24.2 Weaviate Adapter (`integrations/weaviate/`)
Embedding adapter for Weaviate vector DB:
- `EmbedQuery(ctx, text)` -- single vector
- `EmbedDocuments(ctx, texts)` -- batched embedding
- Uses `/v1/embeddings` endpoint
- BatchSize configurable (default 32)
- Results re-ordered by index for correctness

---

## Security Summary

### Path Traversal Protections
- `registry/registry.go`: modelDir validates containment within cacheDir
- `registry/pull.go`: downloadFile validates `..` in filename AND resolved path containment
- `modelcache/cache.go`: sanitize replaces unsafe chars in filenames

### Authentication
- HuggingFace: Bearer token from `HF_TOKEN` env var
- OCI Registry: Basic auth via `WithCredentials` option
- Support API: **No authentication** on any endpoint

### Thread Safety
- monitor/drift: Mutex-protected detectors
- shutdown/coordinator: Mutex with idempotent shutdown
- modelcache: Mutex-protected cache operations
- registry: RWMutex-protected model operations
- shared/latent: RWMutex-protected projections
- support/ticket: RWMutex-protected store
- autoopt/scheduler: RWMutex-protected scheduler

### Potential Concerns
1. **ADWIN unbounded memory**: Window grows without cap
2. **Support API unauthenticated**: No auth, CSRF, or rate limiting
3. **DP random seed fixed**: `rand.New(rand.NewSource(42))` in DPStrategy -- not cryptographically secure
4. **OCI push reads entire model into memory**: `os.ReadFile` on potentially multi-GB files
5. **Error message leakage**: Support API CloseTicket embeds raw error strings in response
6. **crossasset training**: Only trains classification head, transformer layers frozen
7. **Timeseries extractVariable**: Raw slice indexing relies on shape correctness
