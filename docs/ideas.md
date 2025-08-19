Here is a clean “parts list” from smallest to whole, showing how pieces compose into a complete Gemma-3 + HRM model. Each level is built from items below it.

# Level 1: Neuron

* Weighted sum, bias, nonlinearity.
* Typical activations used later: SiLU, GELU.
  Composes into vectors and layers.

# Level 2: Vector and basic ops

* Vectors, matrices, tensors.
* Core ops: matmul, add, mul, softmax, normalization stats.
  Composes into linear layers, norms, attention logits.

# Level 3: Parameterized layer primitives

* Linear layer: y = xW + b.
* Elementwise gate: z = σ(xWg) ⊙ (xWp).
* Normalization: RMSNorm or LayerNorm.
* Positional mixing: Rotary position embedding applied to Q and K.
  Composes into MLPs and attention heads.

# Level 4: Feed-forward unit (MLP blocklet)

* Two linears with activation or gate: e.g., SwiGLU or GeGLU.
* Residual connection around the unit.
  Composes into the Transformer MLP sublayer.

# Level 5: Attention head

* Projections: Q = xWq, K = xWk, V = xWv.
* Scaled dot-product attention: softmax(QKᵀ/√d) V.
* Optional masking and locality windowing.
  Multiple heads compose into multi-head attention.

# Level 6: Multi-head attention (MHA)

* H parallel heads, concatenated, then output projection Wo.
* Residual connection and pre-norm.
  Composes into the Transformer attention sublayer.

# Level 7: Transformer block (vanilla)

* Pre-norm, MHA, residual.
* Pre-norm, MLP (from Level 4), residual.
* Dropout and init details as needed.
  Composes into local or global variants.

# Level 8: Local Transformer block (Gemma-style local)

* Same as Level 7, but attention is restricted to a sliding window W around each token.
* Maintains a compact KV cache per window.
  Composes into local stacks for efficient long context.

# Level 9: Global Transformer block (Gemma-style global)

* Same as Level 7, but attention spans the full context or a large subset.
* Refreshes long-range interactions and summary tokens.
* Larger or separate KV cache region.
  Composes with local blocks to balance range and cost.

# Level 10: Local-Global group (Gemma pattern)

* k local blocks followed by 1 global block form a group.
* The group is the fundamental efficiency rhythm: local steps for short-range processing, a global step for long-range mixing.
  Multiple groups compose the core text stack.

# Level 11: Token and embedding stage

* Text tokenizer with vocabulary and special tokens.
* Token embedding matrix and output LM head (often tied weights).
* Positional encoding via rotary mixing already handled in attention projections.
  Feeds tokens into the Local-Global stack.

# Level 12: Vision pathway (Gemma multimodal)

* Image preprocessor and cropper. Pan-and-scan generates multiple high-res crops if needed.
* Vision encoder, e.g., a SigLIP-style model that outputs patch tokens.
* Projection layer maps vision tokens into the text residual space.
  Produces “image tokens” that are merged with text tokens.

# Level 13: Residual stream and KV caches

* Unified residual stream that carries interleaved text and image tokens.
* KV caches: local windows for local blocks, larger spans for global blocks.
* Streaming decode support for inference.
  Backbone state over which HRM will “think.”

# Level 14: HRM state variables and controllers

* Low-level recurrent state s\_L that updates frequently.
* High-level recurrent state s\_H that updates less often.
* Halting head that predicts continue or stop for a segment.
* Cycle controller that schedules inner (L) and outer (H) updates per segment.
  These wrap groups to add iterative “thinking.”

# Level 15: HRM Low-level module L

* Defines a fast inner update over a Local-Global group.
* Runs for T\_L inner cycles without changing s\_H.
* Can reset or decay its own state between groups.
  Provides rapid refinement of token representations.

# Level 16: HRM High-level module H

* Defines a slower outer update that reads aggregate signals from recent L cycles.
* Updates s\_H, which conditions the next round of L cycles.
* Exposes the state used by the halting head and planning logic.
  Coordinates longer-horizon reasoning.

# Level 17: HRM cycle wrapper around a group

* One “HRM CycleBlock” = repeat L for T\_L steps, then one H update.
* The wrapper sits around the Level 10 Local-Global group.
* At test time, T\_L or the number of cycles can be raised to spend more compute.
  Multiple CycleBlocks compose the HRM-augmented stack.

# Level 18: Stack assembly

* Repeat: \[CycleBlock over Local-Global group] × G.
* Early blocks focus on lexical and local visual grounding. Later blocks focus on global integration and planning via s\_H.
* Checkpointable and shardable across devices.
  This is the full multimodal HRM-augmented Transformer.

# Level 19: Output heads

* Language modeling head for next-token logits.
* Optional value head for RL post-training.
* Halting head from Level 14 for compute control.
  Provide probabilities, values, and stop decisions.

# Level 20: Training objectives

* Cross-entropy on next token for text and multimodal corpora.
* Auxiliary halting loss and compute-cost regularizer.
* Distillation loss from a stronger teacher if used.
* RL or preference-based post-training for instruction, safety, and math.
  Optimized together under a schedule.

# Level 21: HRM training mechanics

* One-step fixed-point gradient or similar approximation through L and H updates to keep memory O(1) in cycle depth.
* Deep supervision on intermediate states if helpful.
* Stability helpers: pre-norm, RMSNorm, weight decay, gradient clipping.
  Enables many inner “thought” steps without large memory growth.

# Level 22: Optimizer and schedules

* AdamW or variant with decoupled weight decay.
* Learning rate warmup and cosine or step decay.
* Mixed precision, gradient accumulation, gradient checkpointing.
  Drives scalable training.

# Level 23: Data and curriculum

* Tokenization pipeline, text and image preprocessing, pan-and-scan policy.
* Mixture: web text, code, math, diagrams, charts, documents.
* Curriculum that includes algorithmic tasks, ARC-style induction, Sudoku, mazes, multimodal QA.
  Teaches both pattern induction and deliberate search.

# Level 24: Inference runtime and policies

* Dynamic halting: stop cycles when the halting head is confident.
* Adjustable compute budget: max cycles, per-query knobs.
* Pan-and-scan scheduling for images, retrieval for long context, streaming decode.
  Lets users trade speed for accuracy.

# Level 25: System and serving

* Tensor and pipeline parallelism across nodes.
* Quantization and KV cache offloading.
* Checkpointing, versioning, and safe rollout.
  Delivers the model in production.

---

If you want this mapped to package boundaries for an implementation, I can turn the hierarchy into a concrete module layout with interfaces for Attention, MLP, LocalBlock, GlobalBlock, CycleBlock, HRMController, VisionProjector, Tokenizer, KVCache, and the training and inference loops.
