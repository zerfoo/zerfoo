# Architecture Decision Records

| ADR | Title | Category |
|-----|-------|----------|
| [001](001-enterprise-production-readiness.md) | Enterprise Production Readiness | Infrastructure |
| [002](002-distributed-training-protocol.md) | Distributed Training Protocol | Training |
| [003](003-open-weights-model-import.md) | Open Weights Model Import | Models |
| [004](004-embeddable-inference-library.md) | Embeddable Inference Library | Architecture |
| [005](005-multi-architecture-support.md) | Multi-Architecture Support | Models |
| [006](006-gpu-engine-architecture.md) | GPU Engine Architecture | GPU |
| [007](007-multi-gpu-architecture.md) | Multi-GPU Architecture | GPU |
| [008](008-cudnn-integration.md) | cuDNN Integration | GPU |
| [009](009-tensorrt-integration.md) | TensorRT Integration | GPU |
| [010](010-cutlass-flash-attention.md) | CUTLASS Flash Attention | GPU |
| [011](011-gpu-runtime-abstraction-layer.md) | GPU Runtime Abstraction Layer (GRAL) | GPU |
| [012](012-amd-rocm-backend.md) | AMD ROCm Backend | GPU |
| [013](013-opencl-backend.md) | OpenCL Backend | GPU |
| [014](014-cudnn-backward-pass.md) | cuDNN Backward Pass | GPU |
| [015](015-cutlass-quantized-gemm.md) | CUTLASS Quantized GEMM | GPU |
| [016](016-tensorrt-dynamic-shapes.md) | TensorRT Dynamic Shapes | GPU |
| [017](017-dgx-spark-hardware-validation.md) | DGX Spark Hardware Validation | Hardware |
| [018](018-model-parity-testing.md) | Model Parity Testing | Testing |
| [019](019-phase22-bf16-unified-siglip.md) | BF16 Unified + SigLIP | Precision |
| [020](020-q4-quantized-dot-product.md) | Q4 Quantized Dot Product | Quantization |
| [021](021-graph-compilation-worker-pool.md) | Graph Compilation Worker Pool | Performance |
| [022](022-gpu-first-inference-pipeline.md) | GPU-First Inference Pipeline | Performance |
| [023](023-gpu-scalar-ops-d2h-elimination.md) | GPU Scalar Ops D2H Elimination | Performance |
| [024](024-cuda-graph-fused-kernels.md) | CUDA Graph + Fused Kernels | Performance |
| [025](025-purego-cuda-bindings.md) | Purego CUDA Bindings (No CGo) | Architecture |
| [026](026-megakernel-decode.md) | Megakernel Decode | Codegen |
| [027](027-composition-prerequisite.md) | Composition Prerequisite | Architecture |
| [028](028-tracing-compiler.md) | Tracing Compiler | Codegen |
| [029](029-neon-simd-cpu-acceleration.md) | NEON SIMD CPU Acceleration | Performance |
| [030](030-ollama-performance-parity.md) | Ollama Performance Parity | Benchmarks |
| [031](031-openai-server-in-zerfoo.md) | OpenAI Server in Zerfoo | Serving |
| [032](032-gpu-resident-position-counter.md) | GPU-Resident Position Counter | Performance |
| [033](033-how-we-beat-ollama.md) | How We Beat Ollama | Benchmarks |
| [034](034-gqa-aware-flash-attention-decode.md) | GQA-Aware Flash Attention Decode | Attention |
| [035](035-gemma3-architecture-parameterization.md) | Gemma 3 Architecture Parameterization | Models |
| [036](036-ztensor-ztok-repo-extraction.md) | Extract ztensor and ztoken Repositories | Architecture |
| [037](037-gguf-only-drop-zmf-model-format.md) | GGUF as Sole Model Format, Drop ZMF | Architecture |
