zerfoo/                       ← Root of the project (Go module: github.com/zerfoo/zerfoo)
├── go.mod                    ✅ Go module file with local float8/float16 dependencies
├── go.sum                    ✅ Dependency checksums
├── README.md                 ✅ Project documentation
├── LICENSE                   ✅ Project license
├── Makefile                  ✅ Build automation
├── .golangci.yml             ✅ Linter configuration
├── .gitignore                ✅ Git ignore patterns
│
├── numeric/                  ✅ Foundation: precision types & conversions (100% coverage)
│   ├── float8.go             ✅ Wraps github.com/dndungu/float8 with local replacement
│   ├── float16.go            ✅ Wraps github.com/dndungu/float16 with local replacement  
│   ├── float32.go            ✅ Basic float32 operations
│   ├── float64.go            ✅ Basic float64 operations
│   ├── arithmetic.go         ✅ Arithmetic operations for all types
│   └── [tests]               ✅ Comprehensive test coverage with generic helpers
│
├── tensor/                   ✅ Core tensor struct & basic ops (77% coverage)
│   ├── tensor.go             ✅ Tensor struct (shape, data, dtype)
│   ├── indexing.go           ✅ At/Set methods, slicing
│   ├── broadcast.go          ✅ Broadcasting logic
│   ├── ops.go                ✅ Basic tensor operations
│   ├── utils.go              ✅ Utility functions
│   └── [tests]               ✅ Good test coverage
│
├── device/                   ✅ Hardware device abstractions (100% coverage)
│   ├── device.go             ✅ Device interface and management
│   ├── allocator.go          ✅ Pluggable memory allocators
│   └── [tests]               ✅ Complete test coverage
│
├── compute/                  ✅ Hardware abstraction (Engine interface)
│   ├── cpu_engine.go         ✅ Pure Go CPU implementation
│   ├── [broadcast tests]     ✅ Broadcasting functionality tests
│   └── [coverage tests]      ✅ Comprehensive test coverage
│
├── graph/                    ✅ Computation graph construction (88% coverage)
│   ├── builder.go            ✅ Graph builder with NodeHandle references
│   ├── node.go               ✅ Node interface implementation
│   ├── no_parameters.go      ✅ No-parameter node implementation
│   └── [tests]               ✅ Parameter and gradient tests
│
├── layers/                   ✅ Neural network layers (98.6% coverage in core)
│   ├── core/                 ✅ Core layer implementations
│   │   ├── linear.go         ✅ Linear/Dense layer with functional options
│   │   ├── linear_v2.go      ✅ Component-based Linear layer (demo)
│   │   ├── dense.go          ✅ Dense layer (Linear + Bias)
│   │   ├── dropout.go        ✅ Dropout layer
│   │   ├── bias.go           ✅ Bias layer
│   │   ├── ffn.go            ✅ Feed-Forward Network with functional options
│   │   ├── polynomial_expansion.go ✅ Polynomial expansion with functional options
│   │   └── [tests]           ✅ Comprehensive test coverage (98.6%)
│   │
│   ├── components/           ✅ Reusable layer components
│   │   ├── weight_initializer.go ✅ Xavier, He, Uniform initializers
│   │   ├── matrix_multiplier.go  ✅ Matrix multiplication operations
│   │   ├── gradient_computer.go  ✅ Gradient computation components
│   │   └── [tests]           ✅ 100% test coverage for all components
│   │
│   ├── activations/          ✅ Activation functions
│   │   ├── base_activation.go ✅ Base activation interface
│   │   ├── relu.go           ✅ ReLU activation
│   │   ├── leaky_relu.go     ✅ Leaky ReLU activation
│   │   ├── gelu.go           ✅ GELU activation
│   │   ├── fast_gelu.go      ✅ Fast GELU approximation
│   │   ├── sigmoid.go        ✅ Sigmoid activation
│   │   ├── tanh.go           ✅ Tanh activation
│   │   ├── softmax.go        ✅ Softmax activation
│   │   ├── swish.go          ✅ Swish/SiLU activation
│   │   └── [tests]           ✅ Comprehensive activation tests
│   │
│   ├── attention/            ✅ Attention mechanisms
│   │   ├── attention_head.go ✅ Single attention head
│   │   ├── multi_head_attention.go ✅ Multi-head attention
│   │   ├── global_attention.go ✅ Global attention mechanism
│   │   ├── causal_mask.go    ✅ Causal masking
│   │   ├── attention_mask.go ✅ General attention masking
│   │   └── [tests]           ✅ Attention mechanism tests
│   │
│   ├── embeddings/           ✅ Embedding layers
│   │   ├── token_embedding.go ✅ Token embedding with functional options
│   │   ├── rotary_positional_embedding.go ✅ RoPE with functional options
│   │   └── [tests]           ✅ Embedding layer tests
│   │
│   ├── normalization/        ✅ Normalization layers (75% coverage)
│   │   ├── layer_normalization.go ✅ Layer normalization with functional options
│   │   ├── rms_norm.go       ✅ RMS normalization with functional options
│   │   └── [tests]           ✅ Comprehensive normalization tests
│   │
│   ├── pooling/              ✅ Pooling operations
│   │   ├── max_pooling.go    ✅ Max pooling layer
│   │   ├── average_pooling.go ✅ Average pooling layer
│   │   └── [tests]           ✅ Pooling layer tests
│   │
│   ├── recurrent/            ✅ Recurrent layers
│   │   ├── lstm_cell.go      ✅ LSTM cell implementation
│   │   ├── gru_cell.go       ✅ GRU cell implementation
│   │   └── [tests]           ✅ Recurrent layer tests
│   │
│   └── transformer/          ✅ Transformer components
│       ├── encoder_block.go  ✅ Transformer encoder block
│       ├── decoder_block.go  ✅ Transformer decoder block
│       └── [tests]           ✅ Transformer component tests
│
├── training/                 ✅ Training loop orchestration (100% coverage)
│   ├── trainer.go            ✅ Trainer struct with comprehensive error handling
│   ├── model.go              ✅ Training model interface
│   │
│   ├── loss/                 ✅ Loss functions (66% coverage)
│   │   ├── loss.go           ✅ Loss interface
│   │   ├── cross_entropy_loss.go ✅ Cross-entropy loss implementation
│   │   ├── mse_loss.go       ✅ Mean Squared Error loss
│   │   └── [tests]           ✅ Loss function tests with edge cases
│   │
│   └── optimizer/            ✅ Training optimizers
│       ├── optimizer.go      ✅ Optimizer interface
│       ├── adamw.go          ✅ AdamW optimizer implementation
│       ├── sgd.go            ✅ SGD optimizer (duplication eliminated)
│       └── [tests]           ✅ Optimizer tests with generic helpers
│
├── distributed/              ⚠️  Distributed training strategies (90% tests passing)
│   ├── coordinator/          ✅ Coordinator service
│   │   ├── coordinator.go    ✅ Coordinator implementation
│   │   └── [tests]           ✅ Coordinator tests
│   │
│   ├── pb/                   ✅ Protocol buffer definitions
│   │   ├── coordinator.proto ✅ gRPC service definitions
│   │   ├── coordinator.pb.go ✅ Generated protobuf code
│   │   └── coordinator_grpc.pb.go ✅ Generated gRPC code
│   │
│   ├── all_reduce.go         ⚠️  All-Reduce strategy (2 failing tests)
│   ├── all_reduce_internal_test.go ✅ Internal test utilities
│   ├── network_manager.go    ✅ Network management
│   ├── server_manager.go     ✅ Server management
│   └── [tests]               ⚠️  18/20 tests passing (gradient averaging issue)
│
├── model/                    ✅ Model abstraction & serialization
│   ├── builder.go            ✅ Model builder interface
│   ├── model.go              ✅ Model interface implementation
│   ├── registry.go           ✅ Model registry
│   └── [tests]               ✅ Model tests
│
├── pkg/                      ✅ External integrations
│   ├── importer/             ✅ Model import utilities
│   │   ├── importer.go       ✅ Base importer interface
│   │   ├── registry.go       ✅ Importer registry
│   │   └── [tests]           ✅ Import functionality tests
│   │
│   └── tokenizer/            ✅ Text tokenization
│       └── tokenizer.go      ✅ Base tokenizer interface
│
├── testing/                  ✅ Test utilities
│   └── testutils/            ✅ Custom test helpers
│       ├── custom_mocks.go   ✅ Mock implementations
│       └── test_helpers.go   ✅ Generic test helper functions
│
├── examples/                 ✅ Example usage and models
│   ├── distributed/          ✅ Distributed training examples
│   │   └── xor_float16_test.go ✅ XOR training with float16
│   │
│   ├── gemma/                ✅ Gemma model example
│   │   └── gemma_integration_test.go ✅ Integration test
│   │
│   └── linear_decomposition_demo.go ✅ Component decomposition demo
│
├── scripts/                  ✅ Build and utility scripts
│   ├── generate_docs.sh      ✅ Documentation generation
│   └── run_distributed.sh    ✅ Distributed training runner
│
├── docs/                     ✅ Project documentation
│   ├── design.md             ✅ Complete architectural design document
│   └── tree.md               ✅ This project structure overview
│
└── .github/                  ✅ CI/CD configuration
    └── workflows/
        └── ci.yml            ✅ Continuous integration pipeline
