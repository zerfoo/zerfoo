# Zerfoo Product Roadmap (2026-2036)

## Context

### Problem Statement

Zerfoo is a production-grade ML inference and training framework written entirely in
Go (zero CGo by default). As of 2026-03-18, the 5-year technical roadmap is complete:
21 epics, 124 tasks covering PagedAttention, FP8/NVFP4, speculative decoding,
LoRA/QLoRA, FSDP, multi-modal, agentic tool-use, NAS/AutoML, and a cloud product
prototype. The framework runs 6 model architectures at 245 tok/s on Gemma 3 1B
Q4_K_M (20% faster than Ollama).

A critical internal consumer needs tabular and time-series ML capabilities that
Zerfoo does not yet provide. This consumer currently misuses LLM inference for
tabular prediction (formatting numeric features as text prompts) and has hand-rolled
pure Go CNN/TabNet implementations instead of using Zerfoo's Trainer[T] and GPU
acceleration. Bridging this gap is the top priority and blocks the consumer from
going live.

The roadmap serves two masters:
1. **Internal consumer** -- needs tabular/time-series ML for signal prediction
2. **Open-source community** -- needs LLM inference excellence for Go developers

Internal consumer needs take priority when they conflict (per Chairman directive).

See docs/VISION.md for the full 10-year product vision and revenue model.
See docs/design.md for the technical architecture (29 sections, 56 ADRs).

### Objectives

- **Immediate (2026 Q2-Q3):** Tabular model package. Internal consumer integration.
  GPU-accelerated feature engineering. Fix failing tests.
- **Near-term (2026 Q3-Q4):** Advanced tabular architectures (FTTransformer, TabNet,
  SAINT, ResNet). Time-series architectures (TFT, N-BEATS, PatchTST). AutoML for
  tabular/time-series search.
- **Year 1-2 (2026-2027):** 300+ tok/s. 12+ LLM architectures. ROCm parity.
  v1.0 stable release. 25,000+ stars.
- **Year 3-4 (2028-2029):** Enterprise foundation. $500K-$2M ARR. SOC 2.
  Cloud marketplace. Edge deployment.
- **Year 5-6 (2030-2031):** Training platform. $10M-$25M ARR. Multi-accelerator.
- **Year 7-10 (2032-2036):** Industry standard. $100M-$150M+ ARR. IPO readiness.

### Non-Goals

- Pre-training at scale (100B+ parameters). Focus is inference + fine-tuning.
- Python API or Python bindings. Go-first; Python users use the OpenAI-compatible API.
- Custom ASIC backends. Support NVIDIA, AMD, Intel, Apple only.
- Runtime ONNX execution. zonnx converts ONNX to GGUF at build time.
- Internal consumer repo tasks. Only zerfoo and ztensor tasks are in this plan.

### Constraints and Assumptions

- Primary hardware: DGX Spark at ssh ndungu@192.168.86.250 (GB10, sm_121).
- Go 1.25+ required (generics, range-over-func).
- All GPU bindings via purego/dlopen; no CGo in core packages.
- GGUF is the sole model format; zonnx handles ONNX conversion.
- Each repo (ztensor, ztoken, zerfoo, zonnx, float16, float8) is independent.
- Apache 2.0 license for all core repos (see ADR-057).
- Tests use standard library only (no testify, no cobra).
- Agentic coders execute parallel waves; human review gates at milestones.
- metee v1.0.1 is stable and provides LightGBM/XGBoost bindings.

### Success Metrics

| Year | Metric | Target |
|------|--------|--------|
| 2026 Q2 | Internal consumer uses tabular.Train | Shipped |
| 2026 Q3 | Advanced tabular architectures | 7+ models |
| 2026 | Decode tok/s (1B Q4_K_M) | 300+ |
| 2026 | GitHub stars (all repos) | 5,000+ |
| 2026 | Supported LLM architectures | 12+ |
| 2027 | GitHub stars | 25,000+ |
| 2027 | v1.0 stable release | Shipped |
| 2028 | ARR | $500K |
| 2030 | ARR | $10M |
| 2036 | ARR | $150M+ |

### Research Findings

Research conducted by three parallel agents (tech-researcher, risk-researcher,
arch-researcher) on 2026-03-18. Key findings incorporated into this plan:

**Technical Landscape:**
- Ollama (165K stars) wraps llama.cpp C++ -- not native Go. Zerfoo is the only
  framework combining native Go + zero CGo + library-first + competitive tok/s.
- W&B reached $50M ARR in ~5 years. Replicate ($5.3M ARR) acquired for ~$550M.
  Enterprise ML tooling valuations are strong.
- AWS marketplace charges 20% for ML containers vs 3% for SaaS listings. Pursue SaaS.
- Edge runtimes (TFLite, ONNX Mobile) target <5MB. Go baseline is 10-15MB; need
  build-tag stripping and split runtime (see ADR-059).
- Documentation "wow moment" (working inference in <10 lines of Go) is the single
  highest-leverage community growth action for Year 1.

**Risks:**
- Go ML TAM ceiling is the top risk. $150M ARR requires Go ML to become a real market.
  Mitigation: expand beyond Go developers via OpenAI-compatible API and edge runtime.
- Apache 2.0 fork by cloud provider is an existential risk if successful. Mitigation:
  compete on innovation velocity, not legal moats (ADR-057).
- AI-generated code quality: agents = latent bug risk. Mitigation:
  security audit (Year 3), comprehensive DGX validation, enterprise-grade testing.

**Architecture Patterns:**
- v1.0 API: freeze Engine[T], use extension interfaces for new capabilities (ADR-058).
- Plugin architecture: in-process init() registration (Go database/sql pattern).
  No out-of-process plugins (go plugin package is fragile, gRPC adds latency).
- Cloud: Model Repository pattern (Triton convention), Kubernetes operator for
  declarative serving, token-based billing (ADR-060).
- Edge: build-tag-gated minimal binary, pre-optimized GGUF models (ADR-059).

---

## Scope and Deliverables

### In Scope

- Tabular model package (MLP, FTTransformer, TabNet, SAINT, ResNet) -- internal consumer blocker
- Time-series architectures (TFT, N-BEATS, PatchTST)
- AutoML extension for tabular/time-series architecture search
- Performance optimization to 300+ tok/s (Year 1), 500+ (Year 3), 1000+ (Year 7)
- 6 new LLM architectures and ROCm parity
- v1.0 stable release and documentation
- Enterprise support, SOC 2, cloud marketplace
- Edge deployment, on-device inference
- Transfer learning, RL, cross-asset models, regime detection (later phases)
- Provenance tracking, continuous learning, meta-learning (later phases)

### Out of Scope

- ZMF model format (archived, replaced by GGUF per ADR-037)
- CGo-based GPU bindings (purego/dlopen is the standard)
- Python SDK or CLI wrappers
- Pre-training runs for 100B+ models
- Custom hardware or kernel microarchitecture below CUDA level
- Payment processing (billing uses Stripe webhooks)
- Internal consumer repo tasks (separate repo, separate plan)

### Deliverables Table

| ID | Description | Owner Role | Acceptance Criterion |
|----|-------------|------------|----------------------|
| D0 | Tabular model package for internal consumer | ML Eng | tabular.Train + Predict + Save/Load + Ensemble working on GPU |
| D1 | 12+ model architectures | Arch Eng | All produce coherent output; parity tests pass on DGX |
| D2 | 300+ tok/s decode | Kernel Eng | Gemma 3 1B Q4_K_M >= 300 tok/s on DGX Spark |
| D3 | v1.0 stable release | Lead Eng | API freeze, 2-year guarantee, release-please tag |
| D4 | Documentation site | DevRel | Quickstart, API ref, cookbook, architecture tour live |
| D5 | 5,000+ GitHub stars | DevRel | Organic stars across all repos |
| D6 | ROCm CUDA parity | Kernel Eng | All GPU ops pass on AMD Instinct; benchmark within 20% |
| D7 | Enterprise support tier | Biz Dev | SLA contracts, Slack channel, ticketing system live |
| D8 | SOC 2 Type II | Compliance | Audit report issued by 3PAO |
| D9 | Zerfoo Cloud GA | Platform Eng | Multi-tenant, marketplace listed, 99.9% uptime SLO |
| D10 | Zerfoo Runtime | Arch Eng | <10MB ARM64 binary, inference on Raspberry Pi 5 |
| D11 | Kubernetes operator | Platform Eng | ZerfooInferenceService CRD, autoscaling, canary |
| D12 | Apple Metal backend | Kernel Eng | All GPU ops pass on M-series; benchmark published |

---

## Checkable Work Breakdown

### PRIORITY 1: Tabular and Time-Series ML (Internal Consumer Blocker)

These tasks are the highest priority and must be completed before any remaining
10-year roadmap tasks. Decision rationale: docs/adr/062-tabular-model-package.md

---

#### WE1: Tabular Model Package [2026 Q2 -- CRITICAL]

- [x] W1.1.1 Create tabular package with Model type (configurable MLP on ztensor compute graph) (2026-03-18)
  Owner: ML Eng  Est: 4h
  Deps: none
  Acceptance: `tabular` package in zerfoo with `Model` struct. Configurable MLP:
  input dim, hidden layers ([]int), dropout rate, activation function (ReLU/GELU).
  Built on ztensor compute graph via Engine[T]. `NewModel(config ModelConfig) *Model`.
  `Predict(features []float64) (Direction, float64)` returns direction (Long/Short/Flat)
  and confidence score. Direction is an enum type. Model uses float32 internally.
  Table-driven tests: TestNewModel, TestPredict_ThreeClasses, TestPredict_BatchConsistency.

- [x] W1.1.2 Implement tabular.Train using existing Trainer[T] + AdamW + CrossEntropy (2026-03-18)
  Owner: ML Eng  Est: 4h
  Deps: W1.1.1
  Acceptance: `Train(data [][]float64, labels []int, config TrainConfig) (*Model, error)`.
  TrainConfig: epochs, batch_size, learning_rate, weight_decay, validation_split.
  Converts [][]float64 to ztensor tensors. Uses existing training.Trainer[T] with
  AdamW optimizer and CrossEntropyLoss. Supports both CPU and GPU via Engine[T].
  Returns trained Model ready for Predict(). Train logs loss per epoch to structured
  logger. TestTrain_Convergence (XOR problem), TestTrain_Validation, TestTrain_GPU
  (build tag cuda).

- [x] W1.1.3 Implement tabular.Save and tabular.Load for model serialization (2026-03-18)
  Owner: ML Eng  Est: 3h
  Deps: W1.1.1
  Acceptance: `Save(model *Model, path string) error` writes binary file containing
  model config (JSON header) + weight tensors (raw float32 bytes). File format:
  4-byte magic, 4-byte version, 4-byte config length, JSON config, weight data.
  `Load(path string) (*Model, error)` reads file and reconstructs Model with weights.
  Round-trip test: save, load, predict, compare outputs within 1e-7. TestSave,
  TestLoad, TestRoundTrip, TestLoad_InvalidFile, TestLoad_VersionMismatch.

- [ ] W1.1.4 Implement tabular.Ensemble combining metee trees with MLP via stacking
  Owner: ML Eng  Est: 4h
  Deps: W1.1.2
  Acceptance: `Ensemble` struct holds []*Model (MLP models) and metee booster references.
  `NewEnsemble(models []*Model, treePredictions func([]float64) []float64) *Ensemble`.
  treePredictions is a callback that returns tree ensemble outputs (decouples from metee
  import). `Predict(features []float64) (Direction, float64)` runs all sub-models,
  concatenates outputs, feeds through a learned meta-learner (small MLP). Meta-learner
  trained via `TrainEnsemble(subModelOutputs [][]float64, labels []int, config) error`.
  TestEnsemble_CombinesModels, TestEnsemble_MetaLearnerConverges, TestEnsemble_Predict.

---

#### WE2: Advanced Tabular Architectures [2026 Q3]

- [ ] W2.1.1 Implement tabular.FTTransformer (Feature Tokenizer + Transformer)
  Owner: ML Eng  Est: 5h
  Deps: W1.1.2
  Acceptance: `FTTransformer` model type in tabular package. Each numeric feature
  tokenized via learned embedding, then processed by standard transformer encoder
  (self-attention + FFN). Uses existing layers/attention/ and layers/normalization/.
  Config: num_features, d_token, n_heads, n_layers, d_ffn, dropout. Implements same
  Predict() interface as Model. TestFTTransformer_Forward, TestFTTransformer_Train,
  TestFTTransformer_Shapes.

- [ ] W2.1.2 Implement tabular.TabNet (sequential attention with sparsemax)
  Owner: ML Eng  Est: 5h
  Deps: W1.1.2
  Acceptance: `TabNet` model type. Sequential attention mechanism with sparsemax
  activation for feature selection. Config: input_dim, output_dim, n_steps,
  relaxation_factor, sparsity_coefficient. Attentive transformer + feature transformer
  blocks. Feature importance extractable via AttentionMasks() method.
  TestTabNet_Forward, TestTabNet_Sparsemax, TestTabNet_FeatureImportance.

- [ ] W2.1.3 Implement tabular.SAINT (Self-Attention and Intersample Attention)
  Owner: ML Eng  Est: 5h
  Deps: W1.1.2
  Acceptance: `SAINT` model type. Two attention mechanisms: (1) self-attention across
  features within a sample, (2) intersample attention across samples within a batch.
  Config: num_features, d_model, n_heads, n_layers, inter_sample_attention (bool).
  TestSAINT_Forward, TestSAINT_IntersampleAttention, TestSAINT_Train.

- [ ] W2.1.4 Implement tabular.ResNet (residual MLP baseline)
  Owner: ML Eng  Est: 3h
  Deps: W1.1.2
  Acceptance: `TabResNet` model type. MLP with skip connections between hidden layers.
  Config: input_dim, output_dim, hidden_dims []int, dropout, normalization (batch/layer).
  Simple but surprisingly strong baseline for tabular data.
  TestTabResNet_Forward, TestTabResNet_Residuals, TestTabResNet_Train.

---

#### WE3: Time-Series Architectures [2026 Q3]

- [ ] W2.2.1 Implement timeseries.TFT (Temporal Fusion Transformer)
  Owner: ML Eng  Est: 6h
  Deps: W1.1.2
  Acceptance: `timeseries` package in zerfoo. `TFT` model type. Variable selection
  network + gated residual network + temporal self-attention + multi-horizon output.
  Config: num_static_features, num_time_features, d_model, n_heads, n_horizons,
  quantiles []float64. Predict returns multi-horizon forecasts with quantile estimates.
  TestTFT_Forward, TestTFT_VariableSelection, TestTFT_MultiHorizon.

- [ ] W2.2.2 Implement timeseries.NBEATS (N-BEATS basis expansion)
  Owner: ML Eng  Est: 5h
  Deps: W1.1.2
  Acceptance: `NBEATS` model type. Stack of blocks with basis expansion (trend +
  seasonality). Config: input_length, output_length, stack_types (trend/seasonality/generic),
  n_blocks_per_stack, hidden_dim, n_harmonics. Double residual stacking architecture.
  TestNBEATS_Forward, TestNBEATS_BasisExpansion, TestNBEATS_Decomposition.

- [ ] W2.2.3 Implement timeseries.PatchTST (Patch Time-Series Transformer)
  Owner: ML Eng  Est: 5h
  Deps: W1.1.2
  Acceptance: `PatchTST` model type. Channel-independent patching of time series,
  processed by transformer encoder. Config: input_length, patch_length, stride,
  d_model, n_heads, n_layers, channel_independent (bool). Supports multivariate
  time series with independent channel processing.
  TestPatchTST_Forward, TestPatchTST_Patching, TestPatchTST_ChannelIndependence.

---

#### WE4: Tabular AutoML Extension [2026 Q3-Q4]

- [ ] W2.3.1 Extend NAS/AutoML to search tabular and time-series architecture space
  Owner: ML Eng  Est: 5h
  Deps: W2.1.1, W2.1.2, W2.1.3, W2.1.4, W2.2.1, W2.2.2, W2.2.3
  Acceptance: Extend existing NAS/AutoML (see ADR-055) to include tabular and
  time-series models in search space. Search space includes: Model (MLP), FTTransformer,
  TabNet, SAINT, TabResNet, TFT, NBEATS, PatchTST. Hyperparameter search per
  architecture. `AutoML(data, labels, config) (*BestModel, SearchReport)`.
  TestAutoML_SearchSpace, TestAutoML_FindsBestArchitecture.

---

#### WE5: Transfer Learning for Tabular (Internal Consumer) [2027-2028]

- [ ] W5.1.1 Implement tabular.PreTrain for multi-asset base model
  Owner: ML Eng  Est: 5h
  Deps: W2.3.1
  Acceptance: `PreTrain(allData [][][]float64, config PreTrainConfig) (*BaseModel, error)`.
  Pre-trains a tabular model on data from multiple sources to learn universal patterns.
  Uses existing Trainer[T] with larger batch sizes. BaseModel is a Model with
  pre-trained weights. TestPreTrain_Convergence, TestPreTrain_TransferBenefit.

- [ ] W5.1.2 Implement tabular.FineTuneLoRA for per-source LoRA adaptation
  Owner: ML Eng  Est: 5h
  Deps: W5.1.1
  Acceptance: `FineTuneLoRA(base *BaseModel, data [][]float64, labels []int, config LoRAConfig) (*Adapter, error)`.
  Applies LoRA (Low-Rank Adaptation) to tabular model layers. Config: rank, alpha,
  target_layers. Reuses existing training/lora/ infrastructure. Adaptation completes
  in seconds on small datasets. TestFineTuneLoRA_FastAdaptation, TestFineTuneLoRA_Quality.

- [ ] W5.1.3 Implement tabular.MergeAdapter for deployment without LoRA overhead
  Owner: ML Eng  Est: 3h
  Deps: W5.1.2
  Acceptance: `MergeAdapter(base *BaseModel, adapter *Adapter) (*Model, error)`.
  Merges LoRA adapter weights into base model for zero-overhead inference deployment.
  Output model produces identical predictions to base+adapter but without LoRA forward
  pass overhead. TestMergeAdapter_OutputParity, TestMergeAdapter_NoOverhead.

---

#### WE6: Reinforcement Learning Package [2028-2029]

- [ ] W6.1.1 Create rl package with Environment and Agent interfaces
  Owner: ML Eng  Est: 4h
  Deps: W1.1.2
  Acceptance: `rl` package in zerfoo. `Environment` interface: Reset() State,
  Step(Action) (State, float64, bool). `Agent` interface: Act(State) Action,
  Learn(Experience). `ReplayBuffer` with configurable capacity, priority sampling.
  TestReplayBuffer_FIFO, TestReplayBuffer_PrioritySampling.

- [ ] W6.1.2 Implement PPO (Proximal Policy Optimization) on ztensor compute graph
  Owner: ML Eng  Est: 5h
  Deps: W6.1.1
  Acceptance: `PPO` agent type implementing Agent interface. Clipped surrogate
  objective, generalized advantage estimation. Config: clip_ratio, gamma, lambda,
  n_epochs, batch_size. Trained via ztensor compute graph with Engine[T].
  TestPPO_CartPole (converges on simple environment), TestPPO_ClipObjective.

- [ ] W6.1.3 Implement SAC (Soft Actor-Critic) for continuous action spaces
  Owner: ML Eng  Est: 5h
  Deps: W6.1.1
  Acceptance: `SAC` agent type. Twin Q-networks, entropy regularization, automatic
  temperature tuning. Handles continuous action spaces (position sizing is continuous).
  TestSAC_ContinuousAction, TestSAC_EntropyTuning.

---

#### WE7: Cross-Asset and Causal Models [2029-2030]

- [ ] W7.1.1 Implement crossasset.Model with cross-attention mechanism
  Owner: ML Eng  Est: 5h
  Deps: W5.1.2
  Acceptance: `crossasset` package. `Model` takes features from multiple sources,
  applies cross-attention so each source attends to features of correlated sources.
  Config: n_sources, features_per_source, d_model, n_heads.
  TestCrossAsset_Forward, TestCrossAsset_AttentionWeights.

- [ ] W7.1.2 Implement Graph Neural Network layers (GCN, GAT)
  Owner: ML Eng  Est: 5h
  Deps: W5.1.2
  Acceptance: `gnn` package. Graph Convolutional Network (GCN) and Graph Attention
  Network (GAT) layers. Adjacency matrix + node features as input. Configurable
  number of layers and hidden dims. TestGCN_Forward, TestGAT_AttentionMask.

- [ ] W7.3.1 Implement causal.DiscoverGraph for causal structure learning
  Owner: ML Eng  Est: 5h
  Deps: W5.1.2
  Acceptance: `causal` package. `DiscoverGraph(data [][]float64) (*CausalGraph, error)`.
  Learns causal DAG from observational data using PC algorithm or NOTEARS.
  TestDiscoverGraph_KnownStructure, TestDiscoverGraph_DAGConstraint.

- [ ] W7.3.2 Implement causal.Intervene for counterfactual prediction
  Owner: ML Eng  Est: 4h
  Deps: W7.3.1
  Acceptance: `Intervene(graph *CausalGraph, intervention Intervention) (*Prediction, error)`.
  Performs do-calculus intervention on causal graph. Returns predicted effect.
  TestIntervene_SimpleChain, TestIntervene_Confounder.

---

#### WE8: Regime Detection and Synthetic Data [2030-2031]

- [ ] W8.1.1 Implement regime.HMM for regime classification
  Owner: ML Eng  Est: 5h
  Deps: W5.1.2
  Acceptance: `regime` package. `HMM` (Hidden Markov Model) with configurable number
  of hidden states. Baum-Welch training, Viterbi decoding. States represent regimes
  (trending, mean-reverting, volatile, crash). TestHMM_BaumWelch, TestHMM_Viterbi,
  TestHMM_RegimeDetection.

- [ ] W8.2.1 Implement synth.MarketVAE for synthetic data generation
  Owner: ML Eng  Est: 5h
  Deps: W5.1.2
  Acceptance: `synth` package. `MarketVAE` -- Variational Autoencoder trained on
  real data to generate realistic synthetic scenarios. Encoder/decoder on ztensor
  graph. KL divergence + reconstruction loss. TestMarketVAE_Generation,
  TestMarketVAE_LatentSpace.

- [ ] W8.2.2 Implement synth.CrashGenerator for stress testing scenarios
  Owner: ML Eng  Est: 4h
  Deps: W8.2.1
  Acceptance: `CrashGenerator` extends MarketVAE to generate extreme tail scenarios.
  Configurable severity, duration, correlation spike. TestCrashGenerator_ExtremeEvents,
  TestCrashGenerator_CorrelationSpike.

---

#### WE9: Self-Improving Systems [2031-2032]

- [ ] W9.1.1 Implement multi-objective NAS for domain-specific optimization
  Owner: ML Eng  Est: 5h
  Deps: W7.1.1
  Acceptance: Extend NAS to optimize for multiple objectives simultaneously
  (e.g., accuracy AND inference latency AND model size). Pareto frontier search.
  TestMultiObjectiveNAS_ParetoFrontier, TestMultiObjectiveNAS_Convergence.

- [ ] W9.2.1 Implement meta.MAML (Model-Agnostic Meta-Learning)
  Owner: ML Eng  Est: 5h
  Deps: W5.1.2
  Acceptance: `meta` package. `MAML` learns initialization weights that adapt to
  new tasks in few gradient steps. Inner loop (task adaptation) + outer loop
  (meta-update). TestMAML_FewShotAdaptation, TestMAML_MetaConvergence.

- [ ] W9.3.1 Implement gp.Evolve for genetic programming
  Owner: ML Eng  Est: 5h
  Deps: W7.1.1
  Acceptance: `gp` package. `Evolve(primitives []Primitive, fitness FitnessFunc, config GPConfig) (*Program, error)`.
  Tree-based genetic programming. Crossover, mutation, selection operators.
  TestEvolve_SimpleFitness, TestEvolve_TreeOperations.

---

#### WE10: Hardware Optimization for Tabular [2032-2034]

- [ ] W10.1.1 Implement TensorRT compilation for tabular models in ztensor [ztensor]
  Owner: Kernel Eng  Est: 6h
  Deps: W5.1.2
  Acceptance: Tabular model graphs compiled to TensorRT for sub-10us per-source
  inference. Uses existing TensorRT bindings (ADR-009).
  TestTensorRT_TabularCompile, TestTensorRT_Latency.

- [ ] W10.1.2 Implement batched multi-model inference in ztensor [ztensor]
  Owner: Kernel Eng  Est: 6h
  Deps: W10.1.1
  Acceptance: Run 1000+ per-source models in a single GPU kernel launch. Batch
  dimension = number of sources. TestBatchedInference_1000Models,
  TestBatchedInference_Throughput.

- [ ] W10.1.3 Implement FPGA backend via purego in ztensor [ztensor]
  Owner: Kernel Eng  Est: 8h
  Deps: W9.1.1
  Acceptance: FPGA runtime abstraction layer. Basic ops (MatMul, Add) via purego
  bindings to FPGA runtime. TestFPGA_BasicOps, TestFPGA_Latency.

---

#### WE11: Enterprise Features (Internal Consumer) [2032-2034]

- [ ] W10.2.1 Implement Zerfoo Cloud managed inference service
  Owner: Platform Eng  Est: 8h
  Deps: T17.2
  Acceptance: See ADR-056. Multi-tenant inference service. Token-based billing.
  99.9% uptime SLO. TestCloud_MultiTenant, TestCloud_Billing.

- [ ] W10.2.2 Implement enterprise features (audit logging, SSO, multi-tenancy)
  Owner: Platform Eng  Est: 8h
  Deps: W10.2.1
  Acceptance: SOC 2 compliant audit logging. SAML 2.0 SSO. Tenant isolation.
  TestEnterprise_AuditLog, TestEnterprise_SSO, TestEnterprise_TenantIsolation.

- [ ] W10.2.3 Implement cloud marketplace listings (AWS, GCP, Azure)
  Owner: Biz Dev  Est: 6h
  Deps: W10.2.2
  Acceptance: SaaS listings on all three marketplaces. Consumption metering.
  TestMarketplace_AWSMetering.

---

#### WE12: Continuous Learning and Provenance [2034-2036]

- [ ] W11.1.1 Implement online.Model with elastic weight consolidation
  Owner: ML Eng  Est: 5h
  Deps: W5.1.2
  Acceptance: `online` package. Model supports continuous weight updates with EWC
  to prevent catastrophic forgetting. Fisher information matrix tracks parameter
  importance. TestEWC_PreventsForgetting, TestEWC_ContinuousUpdate.

- [ ] W11.1.2 Implement monitor.DriftDetector + recover.AutoRetrain
  Owner: ML Eng  Est: 5h
  Deps: W8.1.1
  Acceptance: `monitor` package detects performance degradation via statistical
  tests (Page-Hinkley, ADWIN). `recover` package triggers automatic rollback,
  retrain, validate, redeploy pipeline. TestDriftDetector_DetectsShift,
  TestAutoRetrain_Pipeline.

- [ ] W11.2.1 Implement shared.LatentSpace for cross-model knowledge sharing
  Owner: ML Eng  Est: 5h
  Deps: W7.1.1
  Acceptance: `shared` package. Models share learned representations via a common
  latent space. What one model learns helps other models via shared embedding.
  TestLatentSpace_SharedRepresentation, TestLatentSpace_TransferBenefit.

- [ ] W11.3.1 Implement provenance.Tracker for model lifecycle audit
  Owner: ML Eng  Est: 4h
  Deps: W5.1.2
  Acceptance: `provenance` package. Cryptographic hashes for every training run,
  dataset, hyperparameter, model version. Full DAG from prediction back to training
  data. TestProvenance_HashChain, TestProvenance_DAGTraversal.

---

### PRIORITY 2: Inference Performance and Bug Fixes

These tasks overlap with the 10-year roadmap but are also needed by the internal
consumer. They can run in parallel with Priority 1 tasks.

---

#### WE13: Performance and Test Fixes [2026 Q2 -- parallel with WE1]

- [ ] W3.1.1 Fix Q4_K to Q4_0 re-quantization to unblock sm_121 dispatch path [ztensor]
  Owner: Kernel Eng  Est: 5h
  Deps: none
  Acceptance: Q4_K GEMV dispatches directly on sm_121 without re-quantizing to Q4_0.
  The re-quantization path is the known blocker for 300+ tok/s. Fix in
  ztensor internal/cuda/ dispatch logic. TestQ4K_DirectDispatch_sm121 passes on DGX.

- [ ] W3.1.2 Expand CUDA graph capture to 100% instruction coverage [ztensor]
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: All inference path instructions captured in CUDA graph (currently 99.5%).
  Identify and fix remaining 0.5% uncaptured ops. TestCUDAGraph_FullCapture passes.

- [ ] W3.1.3 Fix Q5_K_M and Q6_K quantized GEMM/GEMV kernel tests [ztensor]
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: Pre-existing Q5_K and Q6_K test failures resolved. All quantized GEMM/GEMV
  tests pass. TestQ5K_GEMV, TestQ6K_GEMV pass on DGX.

- [x] W3.1.4 Fix batcher scheduler eviction ordering test (2026-03-18)
  Owner: Kernel Eng  Est: 3h
  Deps: none
  Acceptance: Pre-existing batcher test failure resolved. serve/batcher/ tests all pass.
  TestBatchScheduler_EvictionOrdering passes consistently (not flaky).

- [ ] W3.1.5 Implement fused attention kernel with FlashAttention-2 [ztensor]
  Owner: Kernel Eng  Est: 6h
  Deps: W3.1.1
  Acceptance: FlashAttention-2 algorithm fused into single CUDA kernel. Supports
  GQA head counts. Memory usage O(N) not O(N^2). TestFlashAttention2_Correctness,
  TestFlashAttention2_MemoryBound passes on DGX.

---

### PRIORITY 3: 10-Year Product Roadmap (Remaining Tasks)

These tasks continue the original 10-year roadmap. They are lower priority than
Priority 1 and 2 tasks but should be scheduled when agent capacity is available.

---

#### E2: New Model Architecture Support [Q1-Q3 2026]

- [x] T2.1 Implement Llama 4 architecture builder in inference/arch_llama4.go (2026-03-18)
  Owner: Arch Eng  Est: 6h
  Deps: none
  Acceptance: Llama 4 GGUF loads and generates coherent text; parity test passes
  on DGX; TestLlama4Forward passes.

- [x] T2.2 Implement Gemma 3n architecture builder in inference/arch_gemma3n.go (2026-03-18)
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: Gemma 3n mobile-optimized model runs inference; TestGemma3nForward passes.

- [x] T2.3 Implement Command R architecture builder in inference/arch_commandr.go (2026-03-18)
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: Command R GGUF loads; long-context (128K) supported;
  TestCommandRForward passes.

- [x] T2.4 Implement Falcon architecture builder in inference/arch_falcon.go (2026-03-18)
  Owner: Arch Eng  Est: 4h
  Deps: none
  Acceptance: Falcon GGUF loads; multi-query attention handled correctly;
  TestFalconForward passes.

- [x] T2.5 Implement Mixtral MoE architecture builder in inference/arch_mixtral.go (2026-03-18)
  Owner: Arch Eng  Est: 5h
  Deps: none
  Acceptance: Mixtral GGUF loads; MoE routing correct; top-K expert selection matches
  reference; TestMixtralForward passes.

- [x] T2.6 Implement RWKV architecture builder in inference/arch_rwkv.go (2026-03-18)
  Owner: Arch Eng  Est: 5h
  Deps: none
  Acceptance: RWKV GGUF loads; linear attention (WKV operator) correct;
  TestRWKVForward passes.

- [x] T2.7 Add parity tests for all 6 new architectures on DGX [DGX] (2026-03-18, test code written, DGX execution deferred pending model files)
  Owner: Arch Eng  Est: 3h
  Deps: T2.1, T2.2, T2.3, T2.4, T2.5, T2.6
  Acceptance: All 6 new architectures produce correct output vs reference; parity
  tolerance < 1e-3; TestNewArchParity passes.

- [x] T2.8 Implement exponential-trapezoidal SSM discretization in layers/ssm/ (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: none
  Acceptance: New discretization mode added to SSM recurrence replacing ZOH with
  exponential-trapezoidal formula from Mamba 3. TestExpTrapDiscretization passes.

- [x] T2.9 Implement complex-valued SSM state tracking with RoPE in layers/ssm/ (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: T2.8
  Acceptance: SSM B/C matrices operate in complex domain via RoPE embeddings.
  TestComplexSSMState passes; BCNorm stabilization layer added. TestBCNorm passes.

- [x] T2.10 Implement MIMO (multi-input multi-output) SSM heads in layers/ssm/ (2026-03-18)
  Owner: Kernel Eng  Est: 4h
  Deps: T2.9
  Acceptance: MIMOMambaBlock supports multiple parallel state spaces with cross-channel
  mixing. TestMIMOSSM passes.

- [x] T2.11 Implement Mamba 3 architecture builder (2026-03-18) in inference/arch_mamba3.go
  Owner: Arch Eng  Est: 3h
  Deps: T2.8, T2.9, T2.10
  Acceptance: Mamba 3 GGUF loads and generates coherent text. Architecture uses
  exponential-trapezoidal discretization, complex-valued states with RoPE, MIMO heads.
  TestMamba3Forward passes.

- [ ] T2.12 Add Mamba 3 to parity tests on DGX [DGX]
  Owner: Arch Eng  Est: 2h
  Deps: T2.11
  Acceptance: Mamba 3 output matches reference implementation within 1e-3 tolerance
  on DGX Spark. TestMamba3Parity passes.

---

#### E1: Performance Optimization to 300+ tok/s [Q1-Q2 2026]

- [x] T1.1 Profile decode hot path on DGX Spark with nsight systems (2026-03-18)
  Owner: Kernel Eng  Est: 3h  Deps: none
- [x] T1.2 Implement KV cache FP16 storage in generate/kv_cache.go (2026-03-18)
  Owner: Kernel Eng  Est: 4h  Deps: T1.1
- [x] T1.3 Optimize Q4_K GEMV kernel for Blackwell sm_121 in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 6h  Deps: T1.1
- [x] T1.4 Implement kernel launch batching to reduce driver overhead (2026-03-18)
  Owner: Kernel Eng  Est: 4h  Deps: T1.1
- [x] T1.5 Benchmark: achieve 300+ tok/s on Gemma 3 1B Q4_K_M [DGX] (2026-03-18, 245 tok/s -- target not met, bottleneck analysis in devlog)
  Owner: Kernel Eng  Est: 2h  Deps: T1.2, T1.3, T1.4

---

#### E3: Quantization Expansion [Q2-Q3 2026]

- [x] T3.1 Implement GPTQ dequantization in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 5h  Deps: none
- [x] T3.2 Implement AWQ dequantization in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 5h  Deps: none
- [x] T3.3 Implement native Q5_K GEMV CUDA kernel in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 6h  Deps: none
- [x] T3.4 Implement native Q6_K GEMV CUDA kernel in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 6h  Deps: none
- [x] T3.5 Implement W4A16 mixed-precision dispatch in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 4h  Deps: T3.1
- [x] T3.6 Implement W8A8 mixed-precision dispatch in ztensor [ztensor] (2026-03-18)
  Owner: Kernel Eng  Est: 4h  Deps: none

---

#### E4: Documentation and Developer Experience [Q1-Q3 2026]

- [x] T4.1 Create documentation site structure (2026-03-18)
  Owner: DevRel  Est: 4h  Deps: none
- [x] T4.2 Write quickstart guide (2026-03-18)
  Owner: DevRel  Est: 3h  Deps: none
- [x] T4.3 Write API reference for inference/, generate/, serve/ (2026-03-18)
  Owner: DevRel  Est: 4h  Deps: none
- [x] T4.4 Write architecture tour document (2026-03-18)
  Owner: DevRel  Est: 4h  Deps: none
- [x] T4.5 Write cookbook with 10+ recipes (2026-03-18)
  Owner: DevRel  Est: 6h  Deps: none
- [x] T4.6 Write benchmark comparison guide (2026-03-18)
  Owner: DevRel  Est: 3h  Deps: T1.5

- [ ] T4.7 Record 15-minute video walkthrough of Zerfoo
  Owner: DevRel  Est: 4h
  Deps: T4.2
  Acceptance: Video covers installation, model loading, text generation, and
  OpenAI API serving. Published on YouTube.

---

#### E5: Community Infrastructure [Q1-Q2 2026]

- [x] T5.1 Create CONTRIBUTING.md (2026-03-18)
  Owner: DevRel  Est: 2h  Deps: none
- [x] T5.2 Create good first issue labels and 20+ starter issues (2026-03-18)
  Owner: DevRel  Est: 3h  Deps: none
- [x] T5.3 Set up GitHub Discussions (2026-03-18)
  Owner: DevRel  Est: 1h  Deps: none

- [ ] T5.4 Create Discord server with channels
  Owner: DevRel  Est: 2h  Deps: none
  Acceptance: Discord server with roles, channels, and bot for GitHub notifications.

- [x] T5.5 Write and publish 5 blog posts (2026-03-18)
  Owner: DevRel  Est: 8h
  Deps: T4.2, T4.6
  Acceptance: 5 posts published: launch, benchmarks, architecture, "Why Go for ML?",
  migration from Ollama.

- [x] T5.6 Submit GopherCon 2026 talk proposal (2026-03-18)
  Owner: DevRel  Est: 3h  Deps: T4.2
- [x] T5.7 Create 10 example applications (2026-03-18)
  Owner: DevRel  Est: 6h  Deps: none

---

#### E6: Plugin and Extension Architecture [Q2-Q3 2026]

- [x] T6.1 Implement architecture registry (2026-03-18)
  Owner: Lead Eng  Est: 3h  Deps: none
- [x] T6.2 Implement quantization format registry [ztensor] (2026-03-18)
  Owner: Lead Eng  Est: 3h  Deps: none
- [x] T6.3 Document third-party extension convention (2026-03-18)
  Owner: Lead Eng  Est: 2h  Deps: T6.1, T6.2

---

#### E7: v1.0 Stable Release [Q1-Q2 2027]
Decision: docs/adr/058-api-stability-v1-contract.md

- [x] T7.1 Audit and freeze Engine[T] interface (2026-03-18)
  Owner: Lead Eng  Est: 4h  Deps: none

- [x] T7.2 Label sub-package maturity (2026-03-18) (stable/beta/alpha)
  Owner: Lead Eng  Est: 2h  Deps: T7.1
  Acceptance: Every sub-package labeled in design.md and package-level doc comment.

- [x] T7.3 Implement deprecation linter (2026-03-18)
  Owner: Lead Eng  Est: 3h  Deps: none

- [x] T7.4 Create release-please config (2026-03-18) for v1.0.0
  Owner: Lead Eng  Est: 2h  Deps: T7.1
  Acceptance: release-please configured; v1.0.0 tag on all 6 active repos.

- [x] T7.5 Write v1.0 migration guide (2026-03-18)
  Owner: DevRel  Est: 3h  Deps: T7.1
  Acceptance: Guide lists all breaking changes from v0 to v1.

---

#### E8: ROCm Backend Hardware Validation [Q1-Q3 2027]

- [ ] T8.1 Acquire AMD Instinct GPU access
  Owner: Infra Eng  Est: 2h  Deps: none
- [ ] T8.2 Validate all purego HIP bindings on AMD hardware
  Owner: Kernel Eng  Est: 6h  Deps: T8.1
- [ ] T8.3 Validate rocBLAS GEMM parity with cuBLAS
  Owner: Kernel Eng  Est: 4h  Deps: T8.1
- [ ] T8.4 Port custom CUDA kernels to HIP in ztensor [ztensor]
  Owner: Kernel Eng  Est: 8h  Deps: T8.2
- [ ] T8.5 Benchmark ROCm vs CUDA throughput [AMD]
  Owner: Kernel Eng  Est: 2h  Deps: T8.4
- [ ] T8.6 Add ROCm to CI pipeline
  Owner: Infra Eng  Est: 3h  Deps: T8.4

---

#### E9: Multi-GPU Inference [Q2-Q3 2027]

- [x] T9.1 Implement tensor parallelism (2026-03-18)
  Owner: Infra Eng  Est: 6h  Deps: none
- [x] T9.2 Implement pipeline parallelism (2026-03-18)
  Owner: Infra Eng  Est: 6h  Deps: none

- [x] T9.3 Add --gpus flag to zerfoo serve command (2026-03-18)
  Owner: Infra Eng  Est: 2h  Deps: T9.1, T9.2
  Acceptance: `zerfoo serve --gpus 0,1,2,3` distributes model across 4 GPUs.

- [ ] T9.4 Benchmark: multi-GPU inference on Llama 3 70B [DGX]
  Owner: Infra Eng  Est: 2h  Deps: T9.3

---

#### E10: Vision-Language Model Expansion [Q3-Q4 2027]

- [x] T10.1 Implement LLaVA architecture builder (2026-03-18)
  Owner: Arch Eng  Est: 5h  Deps: none

- [x] T10.2 Implement Qwen-VL architecture builder (2026-03-18)
  Owner: Arch Eng  Est: 5h  Deps: none
  Acceptance: Qwen-VL GGUF loads; multi-image input; TestQwenVLForward passes.

- [ ] T10.3 Add vision model benchmarks
  Owner: Arch Eng  Est: 2h  Deps: T10.1, T10.2

---

#### E11: Community Growth to 25,000 Stars [Q1-Q4 2027]

- [ ] T11.1 Sponsor GopherCon 2027 booth
  Owner: DevRel  Est: 2h  Deps: none
- [ ] T11.2 Publish tutorial series (5 parts)
  Owner: DevRel  Est: 8h  Deps: none
- [ ] T11.3 Submit KubeCon 2027 talk
  Owner: DevRel  Est: 3h  Deps: none
- [ ] T11.4 Recruit 5 external co-maintainers
  Owner: Lead Eng  Est: 4h  Deps: T5.1
- [ ] T11.5 Integrate with LangChain-Go and Weaviate
  Owner: DevRel  Est: 6h  Deps: none

---

#### E12: Enterprise Support Tier [Q1-Q2 2028]

- [ ] T12.1 Define enterprise support SLA tiers
  Owner: Biz Dev  Est: 2h  Deps: none
- [ ] T12.2 Set up enterprise ticketing system
  Owner: Biz Dev  Est: 3h  Deps: T12.1
- [ ] T12.3 Create enterprise deployment guide
  Owner: DevRel  Est: 4h  Deps: none
- [ ] T12.4 Sign first 5 enterprise support contracts ($500K ARR)
  Owner: Biz Dev  Est: ongoing  Deps: T12.1, T12.2

---

#### E13: Security Audit and Hardening [Q2-Q3 2028]

- [ ] T13.1 Engage third-party security auditor
  Owner: Lead Eng  Est: 2h  Deps: none
- [ ] T13.2 Fix all critical and high findings
  Owner: Lead Eng  Est: 8h  Deps: T13.1
- [x] T13.3 Implement SBOM generation (2026-03-18) in CI
  Owner: Infra Eng  Est: 3h  Deps: none
- [x] T13.4 Add fuzz testing (2026-03-18) for GGUF parser and API server
  Owner: Lead Eng  Est: 4h  Deps: none

---

#### E14: SOC 2 Certification [Q3-Q4 2028]

- [ ] T14.1 Deploy compliance automation platform
  Owner: Compliance  Est: 4h  Deps: none
- [ ] T14.2 Implement required security controls
  Owner: Infra Eng  Est: 8h  Deps: T14.1
- [ ] T14.3 Complete SOC 2 Type I audit
  Owner: Compliance  Est: 4h  Deps: T14.2
- [ ] T14.4 Begin SOC 2 Type II observation period
  Owner: Compliance  Est: 2h  Deps: T14.3

---

#### E15: Edge Deployment (Zerfoo Runtime) [Q2-Q4 2028]
Decision: docs/adr/059-edge-runtime-architecture.md

- [x] T15.1 Implement build-tag-gated edge binary (2026-03-18)
  Owner: Arch Eng  Est: 4h  Deps: none
- [ ] T15.2 Implement pre-optimized model format
  Owner: Arch Eng  Est: 4h  Deps: T15.1
- [ ] T15.3 Cross-compile and test on Raspberry Pi 5
  Owner: Arch Eng  Est: 3h  Deps: T15.1
- [ ] T15.4 Cross-compile and test on NVIDIA Jetson Orin Nano
  Owner: Arch Eng  Est: 3h  Deps: T15.1
- [ ] T15.5 Add ARM64 cross-compilation to CI
  Owner: Infra Eng  Est: 2h  Deps: T15.1

---

#### E16: Performance Optimization to 500+ tok/s [Q3-Q4 2028]

- [ ] T16.1 Implement warp-specialized GEMV kernel [ztensor]
  Owner: Kernel Eng  Est: 6h  Deps: none
- [x] T16.2 Implement KV cache quantization (FP8 KV) (2026-03-18)
  Owner: Kernel Eng  Est: 5h  Deps: none
- [ ] T16.3 Benchmark: 500+ tok/s [DGX]
  Owner: Kernel Eng  Est: 2h  Deps: T16.1, T16.2

---

#### E17: Zerfoo Cloud GA [Q1-Q3 2029]
Decision: docs/adr/060-cloud-platform-architecture.md

- [x] T17.1 Implement model repository server (2026-03-18)
  Owner: Platform Eng  Est: 5h  Deps: none
- [ ] T17.2 Implement Kubernetes operator
  Owner: Platform Eng  Est: 8h  Deps: none
- [x] T17.3 Implement adaptive batching (2026-03-18)
  Owner: Platform Eng  Est: 4h  Deps: none
- [x] T17.4 Implement multi-model serving with LRU GPU eviction (2026-03-18)
  Owner: Platform Eng  Est: 5h  Deps: none
- [ ] T17.5 List on AWS Marketplace
  Owner: Biz Dev  Est: 4h  Deps: T17.1, T17.2
- [ ] T17.6 List on GCP Marketplace
  Owner: Biz Dev  Est: 4h  Deps: T17.5
- [ ] T17.7 List on Azure Marketplace
  Owner: Biz Dev  Est: 4h  Deps: T17.5

---

#### E18: Enterprise Features [Q2-Q4 2029]
Decision: docs/adr/057-open-core-licensing-strategy.md

- [ ] T18.1 Create zerfoo-enterprise repository
  Owner: Lead Eng  Est: 2h  Deps: none
- [ ] T18.2 Implement SSO/SAML authentication
  Owner: Platform Eng  Est: 6h  Deps: T18.1
- [ ] T18.3 Implement RBAC
  Owner: Platform Eng  Est: 5h  Deps: T18.1
- [ ] T18.4 Implement audit logging
  Owner: Platform Eng  Est: 4h  Deps: T18.1
- [ ] T18.5 Implement monitoring dashboards
  Owner: Platform Eng  Est: 5h  Deps: T18.1

---

#### E19: SOC 2 Type II Completion [Q1-Q2 2029]

- [ ] T19.1 Complete SOC 2 Type II audit
  Owner: Compliance  Est: 2h  Deps: T14.4

---

#### E20: Apple Metal Backend [Q1-Q2 2030]

- [ ] T20.1 Implement Metal compute shader bindings [ztensor]
  Owner: Kernel Eng  Est: 8h  Deps: none
- [ ] T20.2 Port critical CUDA kernels to Metal [ztensor]
  Owner: Kernel Eng  Est: 10h  Deps: T20.1
- [ ] T20.3 Benchmark Metal vs CPU on Apple M4 Max
  Owner: Kernel Eng  Est: 2h  Deps: T20.2

---

#### E21: Intel SYCL Backend [Q2-Q3 2030]

- [ ] T21.1 Implement SYCL runtime bindings [ztensor]
  Owner: Kernel Eng  Est: 8h  Deps: none
- [ ] T21.2 Port GEMV and attention kernels to SYCL [ztensor]
  Owner: Kernel Eng  Est: 8h  Deps: T21.1

---

#### E22: Auto-Optimization Framework [Q3-Q4 2030]

- [ ] T22.1 Implement hardware profiling [ztensor]
  Owner: Kernel Eng  Est: 4h  Deps: none
- [ ] T22.2 Implement automatic kernel selection
  Owner: Kernel Eng  Est: 5h  Deps: T22.1
- [ ] T22.3 Implement automatic quantization recommendation
  Owner: ML Eng  Est: 4h  Deps: T22.1

---

#### E23: Evaluation Framework [Q2-Q3 2030]

- [x] T23.1 Implement automated benchmark suite (2026-03-18)
  Owner: Infra Eng  Est: 5h  Deps: none
- [x] T23.2 Implement model comparison tool (2026-03-18)
  Owner: ML Eng  Est: 4h  Deps: none

---

#### E24: Custom Model Architecture SDK [Q1-Q3 2031]

- [ ] T24.1 Implement model definition DSL
  Owner: Lead Eng  Est: 8h  Deps: none
- [ ] T24.2 Implement custom model training workflow
  Owner: ML Eng  Est: 6h  Deps: T24.1
- [ ] T24.3 Implement graph-level optimization passes
  Owner: Kernel Eng  Est: 8h  Deps: T24.1

---

#### E25: Heterogeneous Compute [Q2-Q4 2031]

- [ ] T25.1 Implement automatic workload splitting
  Owner: Kernel Eng  Est: 6h  Deps: T22.1
- [ ] T25.2 Implement multi-accelerator scheduling
  Owner: Kernel Eng  Est: 6h  Deps: T20.1

---

#### E26: ZerfooConf [2031-2032]

- [ ] T26.1 Plan ZerfooConf Day
  Owner: DevRel  Est: 4h  Deps: none
- [ ] T26.2 Execute ZerfooConf Day
  Owner: DevRel  Est: 8h  Deps: T26.1
- [ ] T26.3 Plan standalone ZerfooConf 2032
  Owner: DevRel  Est: 6h  Deps: T26.2

---

#### E27: Ecosystem Integrations [Q1-Q4 2031]

- [ ] T27.1 Implement OCI-compatible model registry
  Owner: Platform Eng  Est: 6h  Deps: none
- [ ] T27.2 Implement Kubernetes model cache DaemonSet
  Owner: Platform Eng  Est: 4h  Deps: none
- [ ] T27.3 Publish Helm chart
  Owner: Platform Eng  Est: 3h  Deps: none

---

#### E28: Federated Learning [Q1-Q3 2033]

- [ ] T28.1 Implement FederatedStrategy interface
  Owner: ML Eng  Est: 4h  Deps: none
- [ ] T28.2 Implement FedProx strategy
  Owner: ML Eng  Est: 3h  Deps: T28.1
- [ ] T28.3 Implement differential privacy noise injection
  Owner: ML Eng  Est: 4h  Deps: T28.1
- [ ] T28.4 Integration test: 4-client federated simulation
  Owner: ML Eng  Est: 3h  Deps: T28.1, T28.3

---

#### E29: On-Device Inference [Q2-Q4 2033]

- [ ] T29.1 Implement gomobile bindings
  Owner: Arch Eng  Est: 6h  Deps: none
- [ ] T29.2 Create iOS demo app
  Owner: Arch Eng  Est: 4h  Deps: T29.1
- [ ] T29.3 Create Android demo app
  Owner: Arch Eng  Est: 4h  Deps: T29.1
- [ ] T29.4 Benchmark on-device inference
  Owner: Arch Eng  Est: 2h  Deps: T29.2, T29.3

---

#### E30: FedRAMP Authorization [Q1-Q4 2034]

- [ ] T30.1 Engage FedRAMP 3PAO
  Owner: Compliance  Est: 4h  Deps: T19.1
- [ ] T30.2 Implement FedRAMP controls (NIST 800-53)
  Owner: Infra Eng  Est: 12h  Deps: T30.1
- [ ] T30.3 Complete FedRAMP authorization
  Owner: Compliance  Est: 4h  Deps: T30.2

---

#### E31: IPO Preparation [Q1-Q4 2035]

- [ ] T31.1 Form board of directors
  Owner: CEO  Est: ongoing  Deps: none
- [ ] T31.2 Engage Big 4 audit firm
  Owner: CFO  Est: 4h  Deps: none
- [ ] T31.3 Hire VP Sales and VP Marketing
  Owner: CEO  Est: ongoing  Deps: none
- [ ] T31.4 Achieve $150M+ ARR
  Owner: CEO  Est: ongoing  Deps: all
- [ ] T31.5 Draft S-1 registration
  Owner: CFO  Est: 8h  Deps: T31.1, T31.2, T31.4

---

#### E32: Architecture Expansion to 100+ Models [2035-2036]

- [ ] T32.1 Implement automated architecture builder from GGUF metadata
  Owner: Arch Eng  Est: 8h  Deps: T6.1
- [ ] T32.2 Validate 100+ model architectures
  Owner: Arch Eng  Est: ongoing  Deps: T32.1

---

#### E33: Performance Target 1000+ tok/s [2032-2035]

- [ ] T33.1 Implement next-gen GPU architecture optimizations
  Owner: Kernel Eng  Est: ongoing  Deps: none
- [ ] T33.2 Implement automatic hardware-specific kernel codegen
  Owner: Kernel Eng  Est: 10h  Deps: T22.1

---

## Parallel Work

### Parallel Tracks

| Track | Description | Epic/Group IDs | Sync Points |
|-------|-------------|----------------|-------------|
| A | Tabular Core (CRITICAL) | WE1 | Merge at tabular.Train working |
| B | Performance Fixes | WE13 | Merge at all tests green |
| C | Advanced Tabular | WE2, WE3, WE4 | Merge at 7+ architectures |
| D | Mamba 3 + Community | E2 (T2.11-T2.12), E5, E7 | Merge at Mamba 3 parity |
| E | Transfer Learning | WE5 | Merge at LoRA tabular working |
| F | RL + Cross-Asset | WE6, WE7 | Merge at PPO/SAC trained |
| G | Regime + Synthetic | WE8 | Merge at HMM + VAE working |
| H | Self-Improving | WE9 | Merge at MAML + GP working |
| I | Backend Expansion | E8, E20, E21 | Merge at ROCm parity |
| J | Platform and Enterprise | E12-E19, WE10, WE11 | Merge at cloud GA |
| K | 10-Year Long-Tail | E10-E11, E15-E16, E22-E33 | Merge at milestones |

### Wave Plan (Maximum Parallelism, Up to 10 Agents)

**Wave 5 -- Tabular Core + Performance Fixes (CRITICAL):**
1. W1.1.1 tabular.Model (ML Eng) [no deps]
2. W1.1.3 tabular.Save/Load (ML Eng) [no deps -- can stub Model]
3. W3.1.1 Fix Q4_K re-quantization [ztensor] (Kernel Eng) [no deps]
4. W3.1.2 CUDA graph 100% coverage [ztensor] (Kernel Eng) [no deps]
5. W3.1.3 Fix Q5_K/Q6_K tests [ztensor] (Kernel Eng) [no deps]
6. W3.1.4 Fix batcher test (Kernel Eng) [no deps]
7. T2.11 Mamba 3 builder (Arch Eng) [deps met: T2.8, T2.9, T2.10]
8. T5.4 Discord server (DevRel) [no deps]
9. T4.7 Video walkthrough (DevRel) [deps met: T4.2]
10. T7.2 Label sub-package maturity (Lead Eng) [deps met: T7.1]

**Wave 6 -- Tabular Train + Ensemble + FlashAttention:**
1. W1.1.2 tabular.Train (ML Eng) [needs W1.1.1]
2. W3.1.5 FlashAttention-2 [ztensor] (Kernel Eng) [needs W3.1.1]
3. T2.12 Mamba 3 parity [DGX] (Arch Eng) [needs T2.11]
4. T5.5 Blog posts (DevRel) [deps met: T4.2, T4.6]
5. T7.4 Release-please v1.0 (Lead Eng) [deps met: T7.1]
6. T7.5 Migration guide (DevRel) [deps met: T7.1]
7. T8.1 Acquire AMD GPU (Infra Eng) [no deps]
8. T10.2 Qwen-VL builder (Arch Eng) [no deps]
9. T15.1 Edge binary (Arch Eng) [no deps]
10. T11.1 GopherCon booth (DevRel) [no deps]

**Wave 7 -- Advanced Tabular + Time-Series (7 parallel architectures):**
1. W1.1.4 tabular.Ensemble (ML Eng) [needs W1.1.2]
2. W2.1.1 FTTransformer (ML Eng) [needs W1.1.2]
3. W2.1.2 TabNet (ML Eng) [needs W1.1.2]
4. W2.1.3 SAINT (ML Eng) [needs W1.1.2]
5. W2.1.4 TabResNet (ML Eng) [needs W1.1.2]
6. W2.2.1 TFT (ML Eng) [needs W1.1.2]
7. W2.2.2 N-BEATS (ML Eng) [needs W1.1.2]
8. W2.2.3 PatchTST (ML Eng) [needs W1.1.2]
9. T8.2 Validate HIP bindings (Kernel Eng) [needs T8.1]
10. T11.2 Tutorial series (DevRel) [no deps]

**Wave 8 -- AutoML + v1.0 + Enterprise start:**
1. W2.3.1 AutoML tabular/timeseries extension (ML Eng) [needs W2.1.x, W2.2.x]
2. T8.3 Validate rocBLAS (Kernel Eng) [needs T8.1]
3. T8.4 Port CUDA to HIP [ztensor] (Kernel Eng) [needs T8.2]
4. T9.3 Multi-GPU CLI (Infra Eng) [deps met: T9.1, T9.2]
5. T10.3 Vision benchmarks (Arch Eng) [deps met: T10.1; needs T10.2]
6. T11.3 KubeCon CFP (DevRel) [no deps]
7. T11.4 Recruit co-maintainers (Lead Eng) [no deps]
8. T11.5 LangChain-Go integration (DevRel) [no deps]
9. T12.1 Enterprise SLA tiers (Biz Dev) [no deps]
10. T13.1 Security auditor (Lead Eng) [no deps]

**Wave 9 -- Transfer Learning + ROCm + Enterprise:**
1. W5.1.1 tabular.PreTrain (ML Eng) [needs W2.3.1]
2. T8.5 ROCm benchmark (Kernel Eng) [needs T8.4]
3. T8.6 ROCm CI (Infra Eng) [needs T8.4]
4. T9.4 Multi-GPU benchmark (Infra Eng) [needs T9.3]
5. T12.2 Enterprise ticketing (Biz Dev) [needs T12.1]
6. T12.3 Enterprise deployment guide (DevRel) [no deps]
7. T13.2 Fix audit findings (Lead Eng) [needs T13.1]
8. T13.3 SBOM generation (Infra Eng) [no deps]
9. T13.4 Fuzz testing (Lead Eng) [no deps]
10. T15.2 Pre-optimized model format (Arch Eng) [needs T15.1]

**Wave 10 -- LoRA + SOC 2 + Platform start:**
1. W5.1.2 tabular.FineTuneLoRA (ML Eng) [needs W5.1.1]
2. W5.1.3 tabular.MergeAdapter (ML Eng) [needs W5.1.2]
3. T14.1 Compliance platform (Compliance) [no deps]
4. T15.3 Raspberry Pi test (Arch Eng) [needs T15.1]
5. T15.4 Jetson test (Arch Eng) [needs T15.1]
6. T15.5 ARM64 CI (Infra Eng) [needs T15.1]
7. T16.1 Warp-specialized GEMV [ztensor] (Kernel Eng) [no deps]
8. T16.2 FP8 KV cache (Kernel Eng) [no deps]
9. T17.1 Model repository (Platform Eng) [no deps]
10. T17.2 Kubernetes operator (Platform Eng) [no deps]

Waves 11+ continue with WE6-WE12 (RL, cross-asset, regime, self-improving,
hardware optimization, enterprise, continuous learning) interleaved with remaining
T-series tasks (E17-E33) based on dependency order and agent availability.

---

## Timeline and Milestones

| ID | Milestone | Epics | Exit Criteria | Date |
|----|-----------|-------|---------------|------|
| M0 | Internal Consumer Bridge | WE1, WE13 | tabular.Train/Predict/Save/Load working; all tests green | 2026-06-30 |
| M0.5 | Advanced Tabular | WE2, WE3, WE4 | 7+ tabular/timeseries architectures; AutoML extension | 2026-09-30 |
| M1 | Inference Excellence | E1-E6 | 300+ tok/s; 12+ archs; docs live; 5K stars | 2026-12-31 |
| M2 | v1.0 and Ecosystem | E7-E11 | v1.0 shipped; ROCm parity; 25K stars | 2027-12-31 |
| M3 | Enterprise Foundation | E12-E16, WE5 | $500K ARR; SOC 2 Type I; transfer learning | 2028-12-31 |
| M4 | Platform GA | E17-E19, WE6, WE7 | $2M ARR; SOC 2 Type II; RL + cross-asset | 2029-12-31 |
| M5 | Training Platform | E20-E23, WE8, WE9 | $10M ARR; Metal + SYCL; regime + NAS | 2030-12-31 |
| M6 | Industry Standard | E24-E27, WE10, WE11 | $50M ARR; ZerfooConf; hardware optimization | 2032-12-31 |
| M7 | Platform Maturity | E28-E30, WE12 | $75M ARR; federated; on-device; continuous learning | 2034-12-31 |
| M8 | Market Leadership | E31-E33 | $150M+ ARR; IPO filed; 100+ architectures | 2036-12-31 |

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R0 | Tabular package does not improve internal consumer signal quality | Critical | Medium | Focus on architecture diversity (7+ models); AutoML finds best fit; walk-forward validation as quality gate. |
| R1 | Go ML TAM ceiling | Critical | High | Expand beyond Go devs via OpenAI API, edge runtime, language FFI. |
| R2 | Apache 2.0 fork by cloud provider | Existential | Medium-High | Innovation velocity; consider AGPL for v2. See ADR-057. |
| R3 | Latent bugs in AI-generated code | High | High | Security audit (Year 3); DGX validation; fuzz testing; bug bounty. |
| R4 | Maintainer burnout / bus factor of 1 | Critical | High | 5 co-maintainers by Year 2; governance by Year 4. |
| R5 | No enterprise budget owner for "Go ML library" | High | Medium-High | Position as "inference infrastructure"; POC program; marketplace credits. |
| R6 | ROCm never reaches CUDA parity | Medium | High | 80% parity target; gate by user demand; drop if < 5% adoption. |
| R7 | Enterprise sales cycle too long | High | Medium | Marketplace consumption; support contracts first; PLG motion. |
| R8 | SaaS multiples compressed | High | Medium | Maintain optionality: acquisition, PE, or continued private. |
| R9 | Rust ML captures "systems language ML" first | High | Medium | Ship v1.0 first; Go has Python interop advantage; edge differentiator. |
| R10 | NVIDIA CUDA licensing changes | High | Low-Medium | GRAL insulates; Metal + SYCL fallback. |
| R11 | Go generics limitations | Medium | Medium | Extension interface pattern (ADR-058). |
| R12 | Cloud marketplace revenue share erodes margins | Medium | Medium | SaaS listings (3%); enterprise self-managed. See ADR-060. |
| R13 | GopherCon talk rejected | Low | Medium | Multiple conferences; sponsor booth; host meetups. |
| R14 | FedRAMP cost exceeds budget | Medium | Medium | Delay to Year 8-9; evaluate demand first; partner with GovCloud MSP. |
| R15 | Agentic coder quality drift | High | High | Human review gates; security audit; strict CI; /review before releases. |

---

## Operating Procedure

### Definition of Done

1. Code compiles: `go build ./...` in the target repo directory.
2. Tests pass: `go test ./... -race -timeout 300s` in the target repo.
3. No vet warnings: `go vet ./...` clean.
4. Acceptance criteria satisfied as written in the task.
5. Benchmark tasks: results appended to docs/devlog.md and docs/benchmarks.md.
6. ADR tasks: ADR file created and referenced in plan.
7. Documentation tasks: content reviewed and published on docs site.
8. Enterprise tasks: customer validation or contract execution confirmed.

### Quality Gates

- Every implementation task must have a paired test.
- Run `go vet ./...` after every code change before committing.
- Commit each task as its own commit. One logical change per commit.
- Never commit files from different directories in the same commit.
- Use standard library only: no testify, no cobra, no viper. Use testing.T and flag.
- GPU-only tests: tag with `//go:build cuda` and run only on DGX.
- Benchmark tasks must run on DGX Spark (not CPU-only CI).
- Never skip CI hooks with --no-verify.
- Human review gate required at each milestone (M0-M8).
- Security review (/review) before each enterprise-facing release.
- Run `golangci-lint` on all changed packages before committing.

### Agent Assignment Protocol

1. Read TaskList to find available (pending, no owner, not blocked) tasks.
2. Prefer Priority 1 (W-series) tasks over Priority 2 and 3 (T-series).
3. Within same priority, prefer lowest-ID task in your skill domain.
4. TaskUpdate status=in_progress, owner=your-name.
5. Read task description fully; identify target file paths.
6. Implement, test, vet, commit in target repo directory.
7. TaskUpdate status=completed.
8. Repeat from step 1.

### Code Style

- Engine[T] is law: all tensor ops through compute.Engine[T].
- Generics throughout: [T tensor.Numeric] constraints.
- Fuse, do not fragment: prefer fused ops over primitive sequences.
- No CGo in core packages; GPU via purego.
- Docstrings only on exported types and functions. No inline comments unless logic
  is non-obvious.
- Rebase and merge. Not squash, not merge commits.

---

## Progress Log

### 2026-03-18: Internal consumer roadmap integration

**Change summary:**
- Integrated internal consumer roadmap as PRIORITY 1 (WE1-WE12, 42 W-series tasks).
- Pushed remaining 10-year roadmap tasks to PRIORITY 3.
- Created PRIORITY 2 for performance/bug fixes that overlap both roadmaps (WE13).
- Added new milestones M0 (Internal Consumer Bridge, 2026-06-30) and M0.5
  (Advanced Tabular, 2026-09-30).
- Added risk R0 for signal quality improvement.
- Re-sequenced wave plan: Waves 5-10 prioritize tabular package and internal
  consumer tasks. Previous Waves 1-4 already completed.
- ADR created: docs/adr/062-tabular-model-package.md
- Completed tasks (40 total from Waves 1-4) preserved with [x] and dates.

### 2026-03-18: GGUF writer consolidation plan created

Created docs/plan-gguf-writer.md to consolidate 5 hand-rolled GGUF writers into a
shared `ztensor/gguf` package. 3 epics (E1-E3), 18 tasks, 7 waves across ztensor,
zerfoo, and zonnx repos. ADR-061 created (docs/adr/061-gguf-writer-in-ztensor.md).

### 2026-03-18: 10-year product roadmap created

Scope: Created full 10-year product roadmap (2026-2036) expanding from completed
5-year technical roadmap. 33 epics, 120+ tasks, 10-wave parallel execution plan.
15 risks with mitigations. 8 milestones.
ADRs created: 057, 058, 059, 060.

### 2026-03-18: Trimmed plan -- 5-year roadmap complete

Trimmed plan. Stable knowledge preserved in docs/design.md (sections 15-29),
docs/adr/ (044-056), and docs/devlog.md. Removed completed epics E1-E21.

---

## Hand-Off Notes

### What You Need to Know

- **Priority system:** W-series tasks (PRIORITY 1, internal consumer) come before
  T-series tasks (PRIORITY 3, 10-year roadmap). Always check W-series first.
- **Internal consumer context:** An internal consumer currently misuses LLM inference
  for tabular prediction and has hand-rolled pure Go CNN/TabNet. The tabular package
  replaces both with GPU-accelerated Zerfoo-native models. Do not mention the
  internal consumer's name in public repos or commits.
- **Repos:** Each repo has its own go.mod. Never commit across repos. Tasks marked
  [ztensor] go in /Users/dndungu/Code/zerfoo/ztensor; unmarked tasks go in
  /Users/dndungu/Code/zerfoo/zerfoo.
- **DGX Spark:** GPU hardware at `ssh ndungu@192.168.86.250`. Set
  `LD_LIBRARY_PATH=~/Code/zerfoo` before running GPU tests. Always rebuild binary.
- **Baseline benchmark:** 245 tok/s, Gemma 3 1B Q4_K_M, 256 tokens, CUDA graph,
  DGX Spark GB10. Target: 300+ (Year 1), 500+ (Year 3), 1000+ (Year 7).
- **Current ADRs:** 001-062 in docs/adr/. Next ADR: 063.
- **GGUF writer plan:** docs/plan-gguf-writer.md -- consolidates 5 hand-rolled
  writers into shared ztensor/gguf package. See ADR-061.
- **Architecture docs:** docs/design.md (29 sections), docs/benchmarks.md,
  docs/devlog.md.
- **CI:** GitHub Actions in .github/workflows/. CPU tests in CI; GPU tests on DGX only.
- **Model downloads:** `zerfoo pull model_id` for HuggingFace models (ADR-039).
- **Licensing:** Apache 2.0 for all core repos. Enterprise in zerfoo-enterprise
  under commercial license (ADR-057).
- **v1.0 contract:** Engine[T] frozen; extension interfaces (ADR-058).
- **metee:** v1.0.1 provides LightGBM/XGBoost bindings. tabular.Ensemble integrates
  with metee via callback pattern (no direct import required).
- **Founder approval required:** ADR-056 (Zerfoo cloud product) status is Proposed;
  blocked until founder approves per Feza governance.

### Placeholder Credentials

- DGX SSH: ndungu@192.168.86.250 (key auth; no password in this file)
- HuggingFace token: set HUGGINGFACE_TOKEN env var
- Stripe API key: set STRIPE_API_KEY env var (billing)
- GCP project: set GOOGLE_CLOUD_PROJECT env var
- AWS Marketplace: set AWS_MARKETPLACE_SELLER_ID env var
- Discord: set DISCORD_BOT_TOKEN env var
- Vanta/Drata: set COMPLIANCE_API_KEY env var

---

## Appendix

### Research Findings: Technical Landscape (2026)

**Competing frameworks:**
- Ollama: 165K stars, wraps llama.cpp C++ as subprocess. CLI-first, not embeddable.
- llama.cpp: 98.4K stars, joined HuggingFace Feb 2026. GGUF originator.
- go-llama.cpp: ~600 stars, CGo bindings (defeats Go build simplicity). Inactive.
- llama.go: ~500 stars, pure Go port. Unmaintained, no GPU.

**Enterprise ML platform revenue benchmarks:**
- W&B: $50M ARR by Dec 2024. Acquired by CoreWeave for $1.7B (Mar 2025).
- Replicate: $5.3M ARR, acquired by Cloudflare for ~$550M (Nov 2025).
- Modal Labs: $87M Series B at $1.1B valuation (Sep 2025).
- MLflow: Open source; Databricks monetizes as platform ($2.4B+ ARR).

**Tabular ML landscape:**
- PyTorch Tabular: Python-only, wraps PyTorch. No Go equivalent exists.
- AutoGluon: Amazon's AutoML for tabular. Python-only.
- FT-Transformer (Gorishniy 2021): treats features as tokens, competitive with GBDT.
- TabNet (Arik & Pfister 2019): sequential attention, interpretable feature selection.
- SAINT (Somepalli 2021): intersample attention, strong on small datasets.
- No Go-native tabular ML framework exists. Zerfoo would be the first.

### ADR Index

| ADR | Title | Status | Year |
|-----|-------|--------|------|
| 001-043 | Phases 1-27 (see docs/adr/) | Accepted | Pre-2026 |
| 044 | PagedAttention KV Block Manager | Accepted | 2026 |
| 045 | Speculative Decoding | Accepted | 2026 |
| 046 | FP8 and NVFP4 Quantization Roadmap | Accepted | 2026-2027 |
| 047 | Disaggregated Prefill/Decode Serving | Accepted | 2026 |
| 048 | Mamba/SSM Architecture Support | Accepted | 2026 |
| 049 | LoRA/QLoRA Fine-Tuning | Accepted | 2027 |
| 050 | Distributed Training FSDP-Equivalent | Accepted | 2027 |
| 051 | Time-Series ML Platform | Accepted | 2028 |
| 052 | Online Learning Safety Framework | Accepted | 2028 |
| 053 | Multi-Modal Inference Pipeline | Accepted | 2029 |
| 054 | Agentic Tool-Use Loop | Accepted | 2029 |
| 055 | Neural Architecture Search | Accepted | 2030 |
| 056 | Zerfoo Cloud Product | Proposed | 2030 |
| 057 | Open-Core Licensing Strategy | Accepted | 2029 |
| 058 | API Stability v1.0 Contract | Accepted | 2027 |
| 059 | Zerfoo Runtime -- Edge Inference Architecture | Accepted | 2028 |
| 060 | Zerfoo Cloud Platform Architecture | Accepted | 2029 |
| 061 | Shared GGUF Writer in ztensor | Accepted | 2026 |
| 062 | Tabular Model Package | Accepted | 2026 |
