---
title: "Why Go for ML? Making the Case for Go in Machine Learning"
date: 2026-03-18
author: Zerfoo Team
tags: [go, ml, opinion, inference, production]
---

# Why Go for ML? Making the Case for Go in Machine Learning

The ML ecosystem is dominated by Python. PyTorch, TensorFlow, JAX, Hugging Face — the tools that define modern machine learning are all Python-first. So when we tell people we built a production ML inference framework in Go, the first reaction is usually skepticism: "Why would you use Go for ML?"

Here's our answer.

## The Deployment Gap

Most ML code is written in Python, but most production services are written in Go, Java, Rust, or C++. This creates a deployment gap: the language you train in is not the language you serve from.

In practice, this gap is bridged with one of three approaches:

1. **Sidecar process.** Run a Python inference server (vLLM, TGI, Triton) alongside your Go service. Your Go code sends HTTP/gRPC requests to the sidecar for each inference call.

2. **External service.** Deploy a separate inference cluster and call it over the network. Your Go service becomes a client of this service.

3. **CGo bindings.** Wrap a C/C++ inference runtime (ONNX Runtime, llama.cpp) in CGo bindings and call it from Go.

Each approach works, but each adds complexity:

- **Sidecar:** Two processes to manage, monitor, and scale. Cold start latency. Resource contention. Failure modes multiply.
- **External service:** Network latency on every inference call. One more service to operate, version, and deploy. Authentication, load balancing, circuit breaking.
- **CGo:** Build complexity (C toolchain required). Debugging across the Go/C boundary. Per-call overhead (~200ns) that accumulates across thousands of CUDA API calls.

What if you could just `import "github.com/zerfoo/zerfoo"` and run inference directly in your Go process?

## Go's Strengths for Inference

Go is not an obvious choice for ML, but it has properties that are surprisingly well-suited for production inference.

### Compilation and Deployment

Go produces statically-linked binaries. Your inference server is a single file — no virtual environments, no dependency resolution at deploy time, no "works on my machine." This matters enormously for production reliability.

```bash
go build -o server ./cmd/zerfoo
scp server prod-host:/usr/local/bin/
```

Compare this to deploying a Python inference server with its constellation of dependencies: PyTorch, CUDA toolkit, transformers, tokenizers, accelerate, and their transitive dependencies.

### Concurrency

Go's goroutines and channels are purpose-built for serving workloads. An inference server needs to handle concurrent requests, manage session pools, stream tokens to multiple clients, and coordinate batch scheduling. Go's concurrency model handles all of this natively:

```go
// Session pooling with channels
sessionPool := make(chan *InferenceSession, poolSize)
for i := 0; i < poolSize; i++ {
    sessionPool <- newSession()
}

// Handle requests concurrently
http.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
    session := <-sessionPool
    defer func() { sessionPool <- session }()
    // ... generate tokens ...
})
```

Python's GIL makes true concurrency difficult. Async frameworks help, but they add complexity and don't parallelize CPU-bound work.

### Memory Control

Go gives you control over memory layout and allocation without the complexity of manual memory management. This matters for inference because:

- Tensor buffers can be pre-allocated and reused across requests
- KV cache memory can be managed with custom allocators (arena allocation, paged caches)
- GPU memory pools can be reset between tokens without triggering GC

Zerfoo's decode loop allocates zero bytes per token — the input tensor is pre-allocated and updated in-place, and intermediate GPU buffers are managed by a memory arena.

### Type Safety with Generics

Go 1.18+ generics let us write type-safe tensor operations without sacrificing performance:

```go
type Engine[T tensor.Numeric] interface {
    MatMul(a, b *tensor.Tensor[T]) (*tensor.Tensor[T], error)
    Add(a, b *tensor.Tensor[T]) (*tensor.Tensor[T], error)
    Softmax(t *tensor.Tensor[T], dim int) (*tensor.Tensor[T], error)
    // ...
}
```

The `tensor.Numeric` constraint covers float32, float64, float16, bfloat16, float8, and quantized types. Incorrect type usage is caught at compile time, not runtime.

## "But Go Is Slow for Math"

This is the most common objection, and it misunderstands where inference time is spent.

Modern LLM inference is GPU-bound. The decode step is a sequence of matrix operations (GEMV, softmax, normalization) executed on the GPU. The host language's role is to orchestrate these operations — launch kernels, manage memory, handle I/O. The actual computation happens in CUDA kernels written in PTX/SASS, not in Go.

Here's the time breakdown for a single decode step in Zerfoo on a DGX Spark:

- **GPU kernel execution:** ~95% of wall time
- **Memory management (arena reset, buffer setup):** ~3%
- **Host orchestration (engine dispatch, graph traversal):** ~1.5%
- **Token sampling and decoding:** ~0.5%

Go's performance in the host orchestration layer is more than sufficient. And by using purego instead of CGo for GPU bindings, we eliminate the ~200ns per-call overhead that would otherwise dominate the host-side cost.

For CPU-only inference, Zerfoo uses hand-written ARM NEON and x86 AVX2 SIMD assembly for the critical inner loops (quantized GEMV, normalization). Go's assembler support makes this possible without CGo.

## What Go Gets Wrong for ML (And How We Work Around It)

We should be honest about Go's limitations for ML work:

### No Operator Overloading

Python's `__add__`, `__mul__`, and `__matmul__` make tensor expressions read like math. In Go, `a + b` becomes `engine.Add(a, b)`. This is more verbose, but it makes the compute flow explicit — you always know which engine is executing the operation and on which device.

### Smaller ML Ecosystem

Python has Hugging Face, PyTorch, scikit-learn, and thousands of ML libraries. Go has fewer. But for inference specifically, the ecosystem requirements are narrower: you need a model loader (GGUF), a tokenizer (BPE), and a compute engine. Zerfoo provides all three.

For training, the ecosystem gap is larger. Zerfoo supports training with backpropagation, AdamW/SGD, and distributed gradient exchange, but large-scale pre-training is still Python's domain. Our focus is inference and fine-tuning.

### No Dynamic Shapes (Without Recompilation)

Python frameworks like PyTorch handle dynamic tensor shapes seamlessly. In Zerfoo, the computation graph is compiled for a specific shape (sequence length 1 for decode). Handling variable-length prefill requires either running uncompiled or recompiling for different lengths.

In practice, this isn't a limitation for the decode loop (always sequence length 1), and prefill runs uncompiled anyway since it only executes once per request.

## The Production Argument

The strongest case for Go in ML isn't about Go's intrinsic properties — it's about where ML inference runs in production.

If your production stack is Go (and many are — Kubernetes, Docker, Prometheus, CockroachDB, and hundreds of backend services are written in Go), then running inference in-process eliminates an entire class of operational complexity. No sidecar. No external service. No CGo. Just a library call.

```go
import "github.com/zerfoo/zerfoo"

func handleChat(w http.ResponseWriter, r *http.Request) {
    reply, err := model.Chat(r.Context(), userMessage)
    // ...
}
```

Your inference scales with your service. Your monitoring is unified. Your deployment is a single binary. Your team doesn't need to learn a second language ecosystem.

## When NOT to Use Go for ML

To be clear, Go is not the right choice for every ML task:

- **Research and prototyping.** Python's interactive REPL, Jupyter notebooks, and vast library ecosystem make it the clear winner for exploratory work.
- **Large-scale pre-training.** Training billion-parameter models from scratch requires PyTorch/JAX's mature distributed training infrastructure and massive GPU cluster support.
- **Rapid model iteration.** If you're changing model architectures daily, Python's dynamism is an advantage.

Go's sweet spot is production inference and fine-tuning — workloads where reliability, deployment simplicity, and operational efficiency matter more than prototyping speed.

## Try It

If you're running Go in production and using LLMs, give Zerfoo a try:

```bash
go get github.com/zerfoo/zerfoo@latest
```

Seven lines of code to run inference. One binary to deploy. 245 tokens per second on a DGX Spark.

The question isn't whether Go can do ML. The question is why your production inference is still running in a different language than the rest of your stack.
