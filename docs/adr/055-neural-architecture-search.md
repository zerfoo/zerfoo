# ADR 055: Neural Architecture Search for Wolf Trading Models

## Status
Accepted

## Date
2026-03-17

## Context
Wolf's trading models (signal generator, regime detector) are hand-designed
architectures (PatchTST, TFT) chosen based on literature. Neural Architecture
Search (NAS) can discover architectures that outperform hand-crafted ones by
5-15% on target benchmarks by optimizing architecture jointly with hardware
constraints. DARTS (Differentiable Architecture Search) reduces search cost
by up to 90% vs RL-based methods via continuous relaxation. Hardware-aware NAS
(ProxylessNAS) optimizes for target hardware latency directly.

By 2030, Zerfoo should provide automated architecture discovery for Wolf's
financial time-series domain, eliminating the need to manually track academic
ML architecture developments.

## Decision
Implement NAS in training/nas/ targeting Wolf's time-series tasks:

Search Space (training/nas/search_space.go):
- Layer types: Attention, MLP, Conv1D, SSMBlock, Linear
- Connectivity: dense, skip connections, gated
- Hyperparameters: hidden_dim in {64, 128, 256, 512}, num_heads in {2, 4, 8},
  depth in {2, 4, 6, 8, 12}
- Constraints: max_params configurable (default 10M for Wolf signal models)

DARTS Search (training/nas/darts.go):
- Continuous relaxation: each edge in the computation graph is a softmax-weighted
  mixture of candidate operations
- Bilevel optimization: architecture parameters alpha updated by validation loss;
  weight parameters w updated by training loss (alternating gradient steps)
- After search: discretize by selecting argmax operation per edge

Hardware-Aware Evaluation (training/nas/hardware_eval.go):
- Proxy: estimate latency from operation counts + memory bandwidth model
  (calibrated against DGX Spark measurements)
- Latency budget: target <1ms per inference step for Wolf real-time signal generation
- Pareto frontier: report accuracy vs latency tradeoff, let Wolf team select

Architecture Export (training/nas/export.go):
- Selected architecture exported as a GGUF model (weights from final training run)
- Architecture config stored in GGUF metadata for reproducibility
- Exported model is a standard Zerfoo inference model; no NAS runtime dependency

## Consequences
Positive:
- Automates the architecture design loop; Wolf models improve without manual research
- Hardware-aware search ensures discovered architectures meet latency SLAs
- DARTS is well-understood and has open reference implementations to validate against

Negative:
- DARTS bilevel optimization is numerically unstable; requires careful learning rate
  scheduling and regularization (DARTS+ stabilization may be needed)
- Search cost: even with DARTS, a full search takes 1-4 GPU days on DGX Spark
- Architecture space must be carefully constrained for financial time-series; applying
  NAS designed for image classification without adaptation produces poor results
- AutoML-Zero (discovering from scratch) is too expensive for production use; DARTS
  with constrained search space is the pragmatic choice
