# Zerfoo Documentation

Welcome to the comprehensive documentation for Zerfoo, a high-performance, scalable Go framework for machine learning!

## Quick Start

New to Zerfoo? Start here:

1. **[Installation](#installation)** - Get Zerfoo running in minutes
2. **[Quick Start Guide](#quick-start-guide)** - Build your first model
3. **[Developer Guide](DEVELOPER_GUIDE.md)** - Comprehensive learning resource

## Documentation Structure

### 📚 **Learning Resources**
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Complete guide to using Zerfoo
- **[Examples & Tutorials](EXAMPLES.md)** - Hands-on learning with practical examples
- **[API Reference](API_REFERENCE.md)** - Detailed API documentation

### 🏗️ **Architecture & Design**
- **[Architecture Design](design.md)** - Deep dive into Zerfoo's architecture
- **[Project Goals](goal.md)** - Vision and objectives
- **[Project Structure](tree.md)** - Codebase organization

### 🤝 **Contributing**
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to Zerfoo
- **[Code Style](CONTRIBUTING.md#code-style-and-conventions)** - Style guidelines
- **[Testing](CONTRIBUTING.md#testing-guidelines)** - Testing best practices

## Installation

### Prerequisites

- **Go 1.24+**: Zerfoo requires Go 1.24 or later for generics support
- **Git**: For cloning and contributing

### Install Zerfoo

```bash
go get github.com/zerfoo/zerfoo
```

### Verify Installation

Create a simple test file:

```go
// test_zerfoo.go
package main

import (
    "fmt"
    "github.com/zerfoo/zerfoo/tensor"
)

func main() {
    // Create a simple tensor
    t, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Zerfoo is working! Tensor shape: %v\n", t.Shape())
    fmt.Printf("Tensor data: %v\n", t.Data())
}
```

Run it:
```bash
go run test_zerfoo.go
```

## Quick Start Guide

### Your First Model in 5 Minutes

```go
package main

import (
    "fmt"
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
    "github.com/zerfoo/zerfoo/tensor"
)

func main() {
    // 1. Create compute engine
    engine := compute.NewCPUEngine[float32]()

    // 2. Build model
    builder := graph.NewBuilder[float32](engine)
    input := builder.Input([]int{1, 2})
    hidden := builder.AddNode(core.NewDense[float32](2, 4, engine), input)
    activation := builder.AddNode(activations.NewReLU[float32](engine), hidden)
    output := builder.AddNode(core.NewDense[float32](4, 1, engine), activation)

    // 3. Compile graph
    forward, _, err := builder.Build(output)
    if err != nil {
        panic(err)
    }

    // 4. Run inference
    inputData, _ := tensor.New[float32]([]int{1, 2}, []float32{1.0, 2.0})
    result := forward(map[graph.NodeHandle]*tensor.Tensor[float32]{
        input: inputData,
    })

    fmt.Printf("Input: %v -> Output: %v\n", inputData.Data(), result.Data())
}
```

**What this does:**
- Creates a 2-layer neural network (2 → 4 → 1)
- Uses ReLU activation
- Runs a forward pass with sample data

## Core Concepts

### 🔢 Tensors
Multi-dimensional arrays with type safety:
```go
tensor, _ := tensor.New[float32]([]int{3, 4}, nil) // 3x4 matrix
scalar, _ := tensor.NewScalar[float32](42.0)       // Single value
```

### 📊 Computational Graphs
Static graphs for optimal performance:
```go
builder := graph.NewBuilder[float32](engine)
input := builder.Input([]int{10})
output := builder.AddNode(someLayer, input)
forward, backward, _ := builder.Build(output)
```

### ⚙️ Engines
Hardware abstraction layer:
```go
cpu := compute.NewCPUEngine[float32]()      // CPU computing
debug := compute.NewDebugEngine[float32](cpu) // Debug wrapper
```

### 🧱 Layers
Composable building blocks:
```go
dense := core.NewDense[float32](10, 20, engine)
relu := activations.NewReLU[float32](engine)
dropout := core.NewDropout[float32](0.5)
```

## Package Overview

| Package | Purpose | Key Types |
|---------|---------|-----------|
| **tensor** | N-dimensional arrays | `Tensor[T]`, `Numeric` |
| **graph** | Computational graphs | `Node[T]`, `Builder[T]`, `Parameter[T]` |
| **compute** | Hardware abstraction | `Engine[T]`, `CPUEngine[T]` |
| **layers** | Neural network layers | `Dense[T]`, `ReLU[T]`, `Dropout[T]` |
| **training** | Training utilities | `Optimizer[T]`, `Loss[T]` |
| **distributed** | Multi-node training | `DistributedStrategy` |
| **model** | Model serialization | `Model[T]`, Save/Load functions |

## Learning Path

### 1. **Beginner** (New to ML or Zerfoo)
- Read the [Developer Guide](DEVELOPER_GUIDE.md) introduction
- Try the [Basic Examples](EXAMPLES.md#quick-start-examples)
- Build your first model following [Tutorial 1](EXAMPLES.md#tutorial-1-building-your-first-neural-network)

### 2. **Intermediate** (Some ML experience)
- Explore [custom layers](EXAMPLES.md#tutorial-2-custom-layers)
- Compare [different optimizers](EXAMPLES.md#tutorial-3-training-with-different-optimizers)
- Study the [API Reference](API_REFERENCE.md)

### 3. **Advanced** (ML practitioners)
- Build [complex architectures](EXAMPLES.md#advanced-examples)
- Implement [distributed training](DEVELOPER_GUIDE.md#distributed-training)
- Contribute to the [project](CONTRIBUTING.md)

## Common Use Cases

### 🎯 **Classification**
```go
// Binary classification with sigmoid output
output := builder.AddNode(activations.NewSigmoid[float32](engine), dense)

// Multi-class with softmax output  
output := builder.AddNode(activations.NewSoftmax[float32](engine), dense)
```

### 📈 **Regression**
```go
// Direct output for regression
output := builder.AddNode(core.NewDense[float32](hidden, 1, engine), lastLayer)
```

### 📊 **Time Series**
```go
// Sequential data processing
input := builder.Input([]int{1, sequenceLength})
// Add temporal layers (RNN/LSTM when available)
```

### 🔍 **Feature Learning**
```go
// Autoencoder architecture
encoder := buildEncoder(builder, input)
decoder := buildDecoder(builder, encoder)
```

## Performance Tips

### ✅ **Do**
- Use appropriate numeric types (`float32` vs `float64`)
- Reuse tensors when possible
- Batch operations for better throughput
- Profile your code with `go tool pprof`

### ❌ **Don't**
- Create tensors in tight loops unnecessarily
- Use debug engine in production
- Ignore shape mismatches (caught at build time)
- Skip testing your custom layers

## Troubleshooting

### Common Issues

**Build Errors:**
```bash
# Update dependencies
go mod tidy

# Clean module cache
go clean -modcache
```

**Shape Mismatches:**
- Check tensor dimensions carefully
- Use `tensor.Shape()` to debug
- Zerfoo catches most issues at graph build time

**Performance Issues:**
- Use CPU engine for development
- Consider batch size optimization
- Profile memory allocations

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/zerfoo/zerfoo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zerfoo/zerfoo/discussions)
- **Documentation**: You're reading it! 📖

## What's Next?

### 🚀 **Start Building**
1. Try the [Quick Start](#quick-start-guide) example
2. Read the [Developer Guide](DEVELOPER_GUIDE.md)
3. Explore [Examples](EXAMPLES.md)

### 🤝 **Get Involved**
1. Read the [Contributing Guide](CONTRIBUTING.md)
2. Check [open issues](https://github.com/zerfoo/zerfoo/issues)
3. Join discussions

### 📚 **Deep Dive**
1. Study the [Architecture](design.md)
2. Understand the [API](API_REFERENCE.md)
3. Read the [source code](https://github.com/zerfoo/zerfoo)

## Version Information

- **Current Version**: Pre-release (actively in development)
- **Go Version**: 1.24+
- **Stability**: API is not yet stable

## License

Zerfoo is licensed under the **Apache 2.0 License**. See [LICENSE](../LICENSE) for details.

---

**Ready to build amazing ML models with Go?** 

Start with the [Developer Guide](DEVELOPER_GUIDE.md) or dive into [Examples](EXAMPLES.md)! 🚀