# ADR 071: Sigmoid MoE Gating with Routing Bias

## Status
Accepted

## Date
2026-03-28

## Context
MiniMax-M2 uses sigmoid gating instead of softmax for MoE expert routing,
with an additional per-expert routing bias tensor (exp_probs_b). The existing
MoEGate only supported softmax.

## Decision
Extended MoEGate with WithSigmoidGating[T]() and WithRoutingBias[T](bias)
options. Added ScoringFunc field to ModelConfig (parsed from GGUF
{arch}.expert_gating_func key, default "softmax").

## Consequences
- MiniMax-M2 (256 experts, sigmoid) and all future sigmoid-gated MoE models supported.
- Existing softmax MoE architectures (Mixtral, DeepSeek, DBRX) unaffected.
- Top-k selection and weight normalization shared between both gating modes.
