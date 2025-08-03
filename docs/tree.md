zerfoo/                       ← Root of the project (Go module: github.com/zerfoo/zerfoo)
├── go.mod                    ← Go module file
├── go.sum
├── README.md
├── internal/                 ← Internal-only utilities (optional)
│   └── utils/                ← Generic internal helpers (logging, etc.)
│
├── tensor/                   ← Core tensor struct & basic ops
│   ├── tensor.go             ← Tensor struct (shape, data, dtype)
│   ├── numeric.go            ← Numeric interface/constraints for generics
│   ├── indexing.go           ← At/Set methods, slicing
│   ├── ops_basic.go          ← Elementwise add, mul, div
│   ├── ops_broadcast.go      ← Broadcasting logic
│   ├── ops_reduce.go         ← Sum, mean, max reductions
│   └── ops_linalg.go         ← Dot product, matmul
│
├── device/                   ← Hardware device abstractions
│   ├── device.go             ← Device interface and management
│   └── allocator.go          ← Pluggable memory allocators
│
├── compute/                  ← Hardware abstraction (Engine)
│   ├── engine.go             ← Engine interface definition
│   ├── cpu_engine.go         ← Pure Go CPU implementation
│   ├── blas_engine.go        ← Optional BLAS-based CPU engine (Gonum)
│   ├── debug_engine.go       ← Engine for logging/debugging
│   └── xla_engine.go         ← Future OpenXLA integration
│
├── graph/                    ← Computation graph construction and autograd
│   ├── graph.go              ← Graph DAG definition
│   ├── builder.go            ← Graph builder (Node registration)
│   ├── node.go               ← Node interface
│   ├── executor.go           ← Forward & backward traversal executor
│   └── autograd.go           ← Gradient calculation and storage
│
├── layers/                   ← Neural network layers
│   ├── core/
│   │   ├── layer.go          ← Layer interface
│   │   ├── dense.go          ← Dense layer (Linear)
│   │   ├── dropout.go        ← Dropout layer
│   │   ├── layernorm.go      ← Layer Normalization
│   │   └── sequential.go     ← Sequential container
│   │
│   ├── activations/
│   │   └── activations.go    ← Activation layers (ReLU, GELU, etc.)
│   │
│   ├── attention/
│   │   ├── mha.go            ← Multi-Head Attention
│   │   ├── masking.go        ← Causal & padding masks
│   │   └── scaled_dot.go     ← Scaled dot-product attention
│   │
│   ├── transformer/
│   │   ├── encoder.go        ← Transformer encoder block
│   │   ├── decoder.go        ← Transformer decoder block
│   │   └── stack.go          ← Full Transformer stack
│   │
│   ├── embedding/
│   │   ├── embedding.go      ← Token embedding
│   │   └── positional.go     ← Positional encoding (sinusoidal)
│   │
│   └── convolution/          ← Only if needed for vision
│       ├── conv2d.go
│       └── pooling.go
│
├── training/                 ← Training loop orchestration
│   ├── trainer.go            ← Trainer struct (epoch/batch loops)
│   ├── gradient_clip.go      ← Gradient clipping utility
│   ├── evaluator.go          ← Eval mode helpers
│   ├── logger.go             ← Training metrics logging
│   │
│   ├── loss/                 ← Loss functions
│   │   ├── loss.go           ← Loss interface
│   │   ├── cross_entropy.go  ← Cross-entropy loss + softmax combo
│   │   └── mse.go            ← Mean Squared Error
│   │
│   ├── optimizer/            ← Training optimizers
│   │   ├── optimizer.go      ← Optimizer interface
│   │   ├── sgd.go            ← SGD + momentum
│   │   └── adam.go           ← Adam optimizer
│   │
│   └── scheduler/            ← LR schedulers (optional)
│       ├── lr_scheduler.go   ← Interface for schedulers
│       ├── warmup.go         ← Warmup scheduler
│       └── cosine_decay.go   ← Cosine LR decay
│
├── distributed/              ← Distributed training strategies
│   ├── strategy.go           ← DistributedStrategy interface
│   ├── all_reduce.go         ← All-Reduce strategy implementation
│   ├── param_server.go       ← Parameter Server strategy implementation
│   └── rpc/                  ← gRPC definitions for communication
│       └── zerfoo.proto
│
├── model/                    ← Model abstraction & serialization
│   ├── model.go              ← Model interface (Forward, Parameters)
│   ├── save_load.go          ← Save/load model weights
│   └── init.go               ← Weight initialization strategies
│
├── pkg/                      ← External integrations
│   ├── onnx/                 ← ONNX import/export support
│   │   ├── importer.go
│   │   └── exporter.go
│   └── tokenizer/            ← Text tokenization
│       ├── tokenizer.go      ← Base tokenizer interface
│       ├── sentencepiece.go  ← SentencePiece/BPE wrapper
│       └── vocab.go          ← Vocabulary utilities
│
└── examples/                 ← Example usage and models
    └── gemma/                ← Gemma 2 example
        ├── config.go         ← Gemma model configs
        ├── fine_tune.go      ← Fine-tuning script
        └── inference.go      ← Inference script
