# ADR 049: LoRA/QLoRA Fine-Tuning Infrastructure

## Status
Accepted

## Date
2026-03-17

## Context
Wolf requires domain-adapted models fine-tuned on financial data. Full fine-tuning
of 7B+ models is prohibitively expensive (requires 40GB+ GPU memory per replica).
LoRA (Low-Rank Adaptation) reduces trainable parameters by 99%+ by injecting
low-rank adapter matrices into linear layers. QLoRA extends this with 4-bit
quantized base weights, enabling 7B fine-tuning on a single 24GB GPU and 70B
fine-tuning on 2x H100.

Zerfoo's training package already has AdamW/SGD and backpropagation through the
computation graph. LoRA is the natural next step.

## Decision
Implement LoRA in training/lora/ as a composable wrapper over existing layers:

LoraLinear (training/lora/linear.go):
- Wraps any existing Linear layer: y = Wx + (alpha/r) * B*A*x
- A is [r x in_features], B is [out_features x r], initialized A~N(0,1), B=0
- Rank r and alpha are configurable per layer group
- Merged mode: folds A*B into W for zero-overhead inference (training/lora/merge.go)

Injection (training/lora/inject.go):
- Walks the model graph and replaces target Linear nodes with LoraLinear
- Target modules configurable: default ["q_proj", "v_proj", "k_proj", "o_proj"]
- Freezes all base model parameters; only A and B matrices are trainable

QLoRA (training/lora/qlora.go):
- Base model weights kept in NF4 (Normal Float 4) quantization
- Forward pass dequantizes NF4 to BF16 for compute
- Backward pass computes gradients in BF16 through dequant op
- NF4 quantization implemented in tensor/quantized.go

Training Loop (training/lora/trainer.go):
- Extends existing Trainer[T] with LoRA adapter management
- Checkpoints save only adapter weights (A, B matrices) as GGUF tensors
- Adapter GGUF uses convention: "lora.{layer_name}.weight_a", "lora.{layer_name}.weight_b"

## Consequences
Positive:
- 7B fine-tuning on single 24GB GPU (QLoRA); 70B on 2x H100
- Adapters are small (10-100MB vs 14GB for 7B model); cheap to version and A/B test
- Merge mode enables zero-overhead inference after fine-tuning

Negative:
- NF4 dequantization on backward pass is not CUDA graph capturable (dynamic shape)
- Gradient flow through quantization introduces numerical noise; requires careful
  learning rate tuning (1e-4 to 1e-5 range recommended)
- LoRA cannot adapt embedding layers or layernorm parameters without modification;
  full fine-tuning still required for tokenizer-level adaptation
