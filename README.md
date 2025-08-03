# Zerfoo: A High-Performance, Scalable, and Idiomatic Go Framework for Machine Learning

[![Go Reference](https://pkg.go.dev/badge/github.com/zerfoo/zerfoo.svg)](https://pkg.go.dev/github.com/zerfoo/zerfoo)
[![Go Report Card](https://goreportcard.com/badge/github.com/zerfoo/zerfoo)](https://goreportcard.com/report/github.com/zerfoo/zerfoo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Zerfoo** is a machine learning framework built from the ground up in Go. It is designed for performance, scalability, and developer experience, enabling everything from practical deep learning tasks to large-scale AGI experimentation.

By leveraging Go's strengths—simplicity, strong typing, and best-in-class concurrency—Zerfoo provides a robust and maintainable foundation for building production-ready ML systems.

> **Status**: Pre-release — actively in development.

---

## Quick Start

Define, train, and run a simple model in just a few lines of idiomatic Go.

```go
package main

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training"
)

func main() {
	// 1. Create a compute engine
	engine := compute.NewCPUEngine()

	// 2. Define the model architecture using a graph builder
	builder := graph.NewBuilder[float32](engine)
	input := builder.Input([]int{1, 10})
	dense1 := builder.AddNode(core.NewDense(10, 32), input)
	act1 := builder.AddNode(activations.NewReLU(), dense1)
	output := builder.AddNode(core.NewDense(32, 1), act1)

	// 3. Build the computational graph
	forward, backward, err := builder.Build(output)
	if err != nil {
		panic(err)
	}

	// 4. Create an optimizer
	optimizer := training.NewAdamOptimizer[float32](0.01)

	// 5. Generate dummy data
	inputTensor, _ := tensor.NewTensor(engine, []int{1, 10})
	targetTensor, _ := tensor.NewTensor(engine, []int{1, 1})

	// 6. Run the training loop
	for i := 0; i < 100; i++ {
		// Forward pass
		predTensor := forward(map[graph.NodeHandle]*tensor.Tensor[float32]{input: inputTensor})

		// Compute loss (dummy loss for this example)
		loss := predTensor.Data()[0] - targetTensor.Data()[0]
		grad := tensor.NewScalar(engine, 2*loss)

		// Backward pass to compute gradients
		backward(grad, map[graph.NodeHandle]*tensor.Tensor[float32]{input: inputTensor})

		// Update weights
		optimizer.Step(builder.Parameters())
	}

	fmt.Println("Training complete!")
}
```

## Why Zerfoo?

Zerfoo is designed to address the limitations of existing ML frameworks by embracing Go's philosophy.

*   ✅ **Idiomatic and Simple**: Build models using clean, readable Go. We favor composition over inheritance and explicit interfaces over magic.
*   🚀 **High-Performance by Design**: A static graph execution model, pluggable compute engines (CPU, GPU planned), and minimal Cgo overhead ensure your code runs fast.
*   ⛓️ **Robust and Type-Safe**: Leverage Go's strong type system to catch errors at compile time, not runtime. Shape mismatches and configuration issues are caught before training even begins.
*   🌐 **Scalable from the Start**: With first-class support for distributed training, Zerfoo is architected to scale from a single laptop to massive compute clusters.
*   🧩 **Modular and Extensible**: A clean, layered architecture allows you to extend any part of the framework—from custom layers to new hardware backends—by implementing well-defined interfaces.

## Core Features

-   **Declarative Graph Construction**: Define models programmatically with a `Builder` API or declaratively using Go structs and tags.
-   **Static Execution Graph**: The graph is built and validated once, resulting in error-free forward/backward passes and significant performance optimizations.
-   **Pluggable Compute Engines**: A hardware abstraction layer allows Zerfoo to target different backends. The default is a pure Go engine, with BLAS and GPU (CUDA) engines planned.
-   **Automatic Differentiation**: Gradients are computed efficiently using reverse-mode AD (backpropagation).
-   **First-Class Distributed Training**: A `DistributedStrategy` interface abstracts away the complexity of multi-node training, with support for patterns like All-Reduce and Parameter Server.
-   **Multi-Precision Support**: Native support for `float32` and `float64`, with `float16` and `float8` for cutting-edge, low-precision training.
-   **ONNX Interoperability**: Export models to the Open Neural Network Exchange (ONNX) format for deployment in any compatible environment.

## Architectural Vision

Zerfoo is built on a clean, layered architecture that separates concerns, ensuring the framework is both powerful and maintainable.

![High-Level Architecture](docs/images/high-level-architecture.svg)

1.  **Composition Layer**: Define models as a Directed Acyclic Graph (DAG) of nodes (layers, activations). This layer is hardware-agnostic.
2.  **Execution Layer**: A pluggable `Engine` performs the actual tensor computations on specific hardware (CPU, GPU). This allows the same model to run on different devices without code changes.

For a deep dive into the design philosophy, core interfaces, and technical roadmap, please read our **[Architectural Design Document](docs/design.md)**.

## Getting Started

*This project is in a pre-release state. The API is not yet stable.*

To install Zerfoo, use `go get`:
```sh
go get github.com/zerfoo/zerfoo
```

## Contributing

Zerfoo is an ambitious project, and we welcome contributions from the community! Whether you're an expert in machine learning, compilers, or distributed systems, or a Go developer passionate about AI, there are many ways to get involved.

Please read our **[Architectural Design Document](docs/design.md)** to understand the project's vision and technical foundations.

## License

Zerfoo is licensed under the **Apache 2.0 License**.