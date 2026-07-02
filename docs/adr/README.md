# Architecture Decision Records

This directory holds the Architecture Decision Records (ADRs) for Zerfoo. Each
ADR captures a single significant decision -- its context, the decision itself,
and the consequences -- so the reasoning behind the framework's shape stays
discoverable long after the change lands.

## Status legend

- **Accepted** -- decision is in force.
- **Proposed** -- under consideration, not yet adopted.
- **Superseded** -- replaced by a later ADR (noted in the title cell when the file states it).
- **Rejected** -- considered and declined.

## Adding a new ADR

1. Copy the next number in sequence (the last entry below is ADR 093).
2. Name the file `NNN-short-kebab-title.md`.
3. Include `# ADR-NNN: Title`, a `## Status`, `## Date`, `## Context`,
   `## Decision`, and `## Consequences` section.
4. Add a row to the table below (kept sorted by number).

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-enterprise-production-readiness.md) | Enterprise Production Readiness | Accepted | 2026-03-01 |
| [002](002-distributed-training-protocol.md) | Distributed Training Protocol | Accepted | 2026-03-01 |
| [003](003-open-weights-model-import.md) | Open Weights Model Import | Accepted | 2026-03-02 |
| [004](004-embeddable-inference-library.md) | Embeddable Inference Library | Accepted | 2026-03-02 |
| [005](005-multi-architecture-support.md) | Multi-Architecture Support | Accepted | 2026-03-02 |
| [006](006-gpu-engine-architecture.md) | GPU Engine Architecture | Accepted | 2026-03-01 |
| [007](007-multi-gpu-architecture.md) | Multi-GPU Architecture | Accepted | 2026-03-03 |
| [008](008-cudnn-integration.md) | cuDNN Integration | Accepted | 2026-03-03 |
| [009](009-tensorrt-integration.md) | TensorRT Integration | Accepted | 2026-03-03 |
| [010](010-cutlass-flash-attention.md) | CUTLASS Flash Attention | Accepted | 2026-03-03 |
| [011](011-gpu-runtime-abstraction-layer.md) | GPU Runtime Abstraction Layer (GRAL) | Accepted | 2026-03-03 |
| [012](012-amd-rocm-backend.md) | AMD ROCm Backend | Accepted | 2026-03-03 |
| [013](013-opencl-backend.md) | OpenCL Backend | Accepted | 2026-03-03 |
| [014](014-cudnn-backward-pass.md) | cuDNN Backward Pass | Accepted | 2026-03-03 |
| [015](015-cutlass-quantized-gemm.md) | CUTLASS Quantized GEMM Kernels | Accepted | 2026-03-03 |
| [016](016-tensorrt-dynamic-shapes.md) | TensorRT Dynamic Shape Support | Accepted | 2026-03-03 |
| [017](017-dgx-spark-hardware-validation.md) | DGX Spark Hardware Validation | Accepted | 2026-03-03 |
| [018](018-model-parity-testing.md) | Model Parity Testing on DGX Spark | Accepted | 2026-03-04 |
| [019](019-phase22-bf16-unified-siglip.md) | Phase 22 -- BF16 GEMM, Unified Memory, SigLIP Fix | Accepted | 2026-03-05 |
| [020](020-q4-quantized-dot-product.md) | Q4 Quantized Dot Product | Accepted | 2026-03-06 |
| [021](021-graph-compilation-worker-pool.md) | Graph Compilation and Persistent Worker Pool | Accepted | 2026-03-06 |
| [022](022-gpu-first-inference-pipeline.md) | GPU-First Inference Pipeline | Accepted | 2026-03-06 |
| [023](023-gpu-scalar-ops-d2h-elimination.md) | GPU Scalar Ops and D2H Elimination Strategy | Accepted | 2026-03-06 |
| [024](024-cuda-graph-fused-kernels.md) | CUDA Graph Capture and Fused Kernels for Inference Acceleration | Accepted | 2026-03-06 |
| [025](025-purego-cuda-bindings.md) | Replace CGo with purego for CUDA Bindings | Accepted | 2026-03-06 |
| [026](026-megakernel-decode.md) | Megakernel Code Generation for Single-Launch Decode | Accepted (revised) | 2026-03-06 |
| [027](027-composition-prerequisite.md) | Enforce Layer Composition Before Megakernel | Accepted | 2026-03-07 |
| [028](028-tracing-compiler.md) | Tracing Compiler for Automatic Primitive Op Decomposition | Accepted | 2026-03-07 |
| [029](029-neon-simd-cpu-acceleration.md) | NEON SIMD CPU Acceleration for Inference Parity with llama.cpp | Accepted | 2026-03-07 |
| [030](030-ollama-performance-parity.md) | Ollama Performance Parity Strategy | Accepted | 2026-03-11 |
| [031](031-openai-server-in-zerfoo.md) | OpenAI-Compatible Inference Server Lives in Zerfoo | Accepted | 2026-03-12 |
| [032](032-gpu-resident-position-counter.md) | GPU-Resident Position Counter for CUDA Graph Capture | Accepted | 2026-03-14 |
| [033](033-how-we-beat-ollama.md) | How Zerfoo Surpassed Ollama — 241 tok/s on DGX Spark GB10 | Accepted | 2026-03-14 |
| [034](034-gqa-aware-flash-attention-decode.md) | GQA-Aware Flash Attention Decode Kernel | Accepted | 2026-03-14 |
| [035](035-gemma3-architecture-parameterization.md) | Gemma 3 Architecture Parameterization | Accepted | 2026-03-11 |
| [036](036-ztensor-ztok-repo-extraction.md) | Extract ztensor and ztoken as Independent Repositories | Accepted | 2026-03-15 |
| [037](037-gguf-only-drop-zmf-model-format.md) | GGUF as Sole Model Format, Drop ZMF for Model Storage | Accepted | 2026-03-15 |
| [038](038-structured-output-grammar-guided-decoding.md) | Structured Output via Grammar-Guided Decoding | Accepted | 2026-03-15 |
| [039](039-huggingface-model-download.md) | HuggingFace Model Download via zerfoo pull | Accepted | 2026-03-15 |
| [040](040-native-gguf-gemv-kernels.md) | Native GGUF GEMV Kernels Instead of Re-Quantization | Accepted | 2026-03-17 |
| [041](041-fp16-weight-dequantization.md) | FP16 Weight Dequantization Instead of Native GEMV Kernels | Accepted | 2026-03-17 |
| [042](042-dp4a-int8-q4k-gemv.md) | dp4a INT8 Q4_K GEMV with FP32 FMA Fallback | Accepted | 2026-03-17 |
| [043](043-arena-free-list-tensor-lifetime.md) | Arena Free-List with Tensor Lifetime Analysis | Accepted | 2026-03-17 |
| [044](044-paged-attention-kv-block-manager.md) | PagedAttention KV Block Manager | Accepted | 2026-03-17 |
| [045](045-speculative-decoding.md) | Speculative Decoding Implementation Strategy | Accepted | 2026-03-17 |
| [046](046-fp8-nvfp4-quantization-roadmap.md) | FP8 and NVFP4 Quantization Roadmap | Accepted | 2026-03-17 |
| [047](047-disaggregated-prefill-decode-serving.md) | Disaggregated Prefill/Decode Serving | Accepted | 2026-03-17 |
| [048](048-mamba-ssm-architecture-support.md) | Mamba/SSM Architecture Support | Accepted | 2026-03-17 |
| [049](049-lora-qlora-finetuning.md) | LoRA/QLoRA Fine-Tuning Infrastructure | Accepted | 2026-03-17 |
| [050](050-distributed-training-fsdp.md) | Distributed Training FSDP-Equivalent | Accepted | 2026-03-17 |
| [051](051-time-series-ml-platform.md) | Wolf Time-Series ML Platform | Accepted | 2026-03-17 |
| [052](052-online-learning-safety-framework.md) | Online Learning Safety Framework | Accepted | 2026-03-17 |
| [053](053-multimodal-inference-pipeline.md) | Multi-Modal Inference Pipeline | Accepted | 2026-03-17 |
| [054](054-agentic-tool-use-loop.md) | Agentic Tool-Use and Loop Supervisor | Accepted | 2026-03-17 |
| [055](055-neural-architecture-search.md) | Neural Architecture Search for Wolf Trading Models | Accepted | 2026-03-17 |
| [056](056-zerfoo-cloud-product.md) | Zerfoo Cloud Product Architecture | Proposed | 2026-03-17 |
| [057](057-open-core-licensing-strategy.md) | Open-Core Licensing Strategy | Accepted | 2026-03-18 |
| [058](058-api-stability-v1-contract.md) | API Stability v1.0 Contract | Accepted | 2026-03-18 |
| [059](059-edge-runtime-architecture.md) | Zerfoo Runtime -- Edge Inference Architecture | Accepted | 2026-03-18 |
| [060](060-cloud-platform-architecture.md) | Zerfoo Cloud Platform Architecture | Accepted | 2026-03-18 |
| [061](061-gguf-writer-in-ztensor.md) | Shared GGUF Writer in ztensor | Accepted | 2026-03-18 |
| [062](062-tabular-model-package.md) | Tabular Model Package | Accepted | 2026-03-18 |
| [063](063-financial-sentiment-analysis.md) | Financial Sentiment Analysis Architecture | Accepted | 2026-03-20 |
| [064](064-docs-site-hugo.md) | Use Hugo for Documentation Site | Accepted | 2026-03-21 |
| [065](065-security-middleware-integration.md) | Security Middleware Integration | Accepted | 2026-03-21 |
| [066](066-cpu-training-backprop.md) | Replace finite-difference gradients with analytical backprop in timeseries CPU training | Accepted | 2026-03-24 |
| [067](067-msa-sparse-attention-memory.md) | MSA-Inspired Sparse Attention for Scalable Memory | Accepted | 2026-03-27 |
| [068](068-research-driven-inference-priorities.md) | Research-Driven Inference Optimization Priorities | Accepted | 2026-03-27 |
| [069](069-transmla-mha-to-mla-conversion.md) | TransMLA -- Retrofit MLA onto MHA/GQA Models | Accepted | 2026-03-27 |
| [070](070-rope-optional-gqa.md) | Make RoPE Optional in GQA for Non-Rotary Architectures | Accepted | 2026-03-28 |
| [071](071-sigmoid-moe-gating.md) | Sigmoid MoE Gating with Routing Bias | Accepted | 2026-03-28 |
| [072](072-kimi-linear-attention.md) | Kimi Linear Attention Layer | Accepted | 2026-03-28 |
| [073](073-mmap-default-split-gguf.md) | mmap Default Loading and Split-GGUF Support | Accepted | 2026-03-28 |
| [074](074-satellite-libraries-v1-release-policy.md) | Satellite Libraries v1 Release Policy | Accepted | 2026-03-29 |
| [075](075-batched-training-kernel-fusion.md) | Batched Training with Kernel Fusion for Time Series Backends | Accepted | 2026-03-29 |
| [076](076-native-foundation-model-inference.md) | Native Go Foundation Model Inference via GGUF | Accepted | 2026-03-29 |
| [077](077-cuda-graph-training-capture.md) | CUDA Graph Capture for Time Series Training | Accepted | 2026-03-30 |
| [078](078-remove-gonum-dependency.md) | Remove gonum.org/v1/gonum Dependency | Accepted | 2026-04-01 |
| [079](079-pjrt-multi-accelerator-backend.md) | PJRT Multi-Accelerator Backend via purego | Accepted | 2026-04-01 |
| [080](080-pjrt-kv-cache-explicit-io.md) | KV Cache as Explicit I/O for PJRT Execution Path | Accepted | 2026-04-01 |
| [081](081-stablehlo-text-generation.md) | StableHLO Text Generation over HLO Protobuf | Accepted | 2026-04-01 |
| [082](082-composition-remediation-strategy.md) | Composition Remediation Strategy | Accepted | 2026-04-02 |
| [083](083-spark-bench-runner.md) | Spark as the bench runner for DGX GPU workloads | Accepted | 2026-04-07 |
| [084](084-extract-crossasset-to-wolf.md) | Extract crossasset package from zerfoo to wolf | Accepted | 2026-04-12 |
| [085](085-gemma4-architecture-support.md) | Gemma 4 Architecture Support | Accepted | 2026-04-13 |
| [086](086-gemma4-edge-ple-architecture.md) | Adopt the Canonical Shared-PLE-Plus-Per-Layer-Proj Layout for Gemma 4 Edge | Accepted | 2026-04-13 |
| [087](087-external-kv-for-shared-kv-attention.md) | External K/V Input Path for GroupedQueryAttention | Accepted | 2026-04-13 |
| [088](088-gemma4-ple-cuda-graph-capture.md) | CUDA Graph Capture Compatibility for Gemma 4 Edge PLE Combiner | Accepted | 2026-04-15 |
| [089](089-lmhead-cuda-graph-capture.md) | CUDA Graph Capture Compatibility for LMHead | Proposed (awaiting ztensor ship) | 2026-04-16 |
| [090](090-zerfoo-oss-scope-cloud-marketplace-compliance.md) | OSS scope of cloud/, marketplace/, and compliance/ in zerfoo | Accepted | 2026-04-27 |
| [091](091-gradcheck-pytorch-oracle-verification.md) | Per-op verification harnesses -- gradcheck, engine parity under arena stress, and PyTorch as oracle | Accepted | 2026-06-10 |
| [092](092-ltx2-diffusion-dit-first.md) | LTX-2 diffusion audio/video inference -- DiT-first, via general primitives | Proposed | 2026-06-16 |
| [093](093-h2-2026-trust-then-traction-strategy.md) | H2 2026 Product Strategy -- Trust, then Traction | Accepted | 2026-07-02 |
