# ADR 046: FP8 and NVFP4 Quantization Roadmap

## Status
Accepted

## Date
2026-03-17

## Context
Zerfoo currently supports Q4_K_M, Q5_K, Q6_K, Q8, FP16, and FP8 weight storage
via GGUF. FP8 E4M3FN is stored but dequantized to FP32/FP16 before compute, losing
the throughput benefit. NVIDIA Blackwell (B200/GB10) introduces NVFP4 (4-bit
floating point with block-scale factors, block size 16) achieving 3.5x memory
reduction vs FP16 and 2.2x end-to-end speedup. AMD will support MXFP4. Training
requires FP8 forward+backward passes to scale to 671B parameter models
(validated by DeepSeek-V3).

## Decision
Three-phase quantization strategy:

Phase 1 (2026 Q1-Q2) -- Inference FP8:
- Implement dynamic FP8 E4M3FN GEMM kernel using cuBLAS LT fp8 matmul API
- Per-tensor dynamic quantization: compute amax at runtime, scale weights and
  activations to FP8 range, run cuBLAS fp8 GEMM, dequantize output to FP16
- Acceptance criterion: perplexity within 0.5 of FP16 baseline on Gemma 3 1B

Phase 2 (2026 Q3-Q4) -- NVFP4 for Blackwell:
- Implement NVFP4 E2M1 weight storage in tensor/quantized.go
- Block-scaled NVFP4 GEMV using cuBLAS LT fp4 API (Blackwell-only)
- Fallback to FP8 on Hopper/Ampere; fallback to FP16 on older hardware
- Runtime hardware detection via cuda/runtime.go ComputeCapability()

Phase 3 (2027 Q4) -- FP8 Training:
- FP8 gradient computation for linear layers
- Loss scaling with dynamic scale factor per-tensor
- Master weights stored in FP32 for optimizer step; FP8 used only for
  forward/backward compute
- Validated on LoRA fine-tuning of Gemma 3 1B on DGX Spark

Post-training quantization tools (AWQ, GPTQ) are out of scope for the core
framework; zonnx handles model conversion.

## Consequences
Positive:
- FP8 inference: 30-50% throughput improvement over FP16 on Hopper+
- NVFP4: 3.5x memory reduction enables serving 3B models in Blackwell's
  previously 1B-capacity memory budget
- FP8 training: enables LoRA fine-tuning of 70B+ models on 2x DGX Spark

Negative:
- NVFP4 is Blackwell-only; creates divergent code paths by GPU generation
- Dynamic quantization adds ~0.5ms overhead per forward pass for amax computation
- FP8 training requires careful loss scaling; numerical instability on low-LR
  fine-tuning tasks is a known risk
