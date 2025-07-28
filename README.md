# Zerfoo

**Zerfoo** is a modular, accelerator-ready machine learning framework written entirely in Go.  
It is being built to train and serve state-of-the-art models like Transformers using float8/16/32/64 precision.  
The framework will support ONNX model import, dynamic graph construction, and low-level optimization primitives to enable AGI-scale systems.

> **Status**: Pre-release — actively in development.  
> **Goal**: Build a powerful, production-ready ML stack in Go from the ground up.

---

## 🌍 Vision

While Python dominates the ML landscape, it wasn't designed for low-latency, modular, and concurrent systems.  
Zerfoo will bring:

- ✅ Strong typing and compile-time safety
- ✅ Native concurrency (goroutines, channels)
- ✅ Fine-grained memory control
- ✅ Performance via custom backends, BLAS, and GPU acceleration
- ✅ Native support for low-precision formats (float8, float16)

We believe Go is the ideal language to scale intelligent systems beyond the limitations of Pythonic stacks.

---

## 🔧 What We're Building

| Feature                        | Description                                      |
|-------------------------------|--------------------------------------------------|
| `float8`, `float16` support   | Native + fallback implementations                |
| Pure Go tensor library        | With autograd and broadcasting                   |
| Modular execution engine      | CPU, BLAS, and accelerator backends              |
| ONNX import support           | For Transformer-class architectures              |
| Model graph API               | Build custom models programmatically             |
| Training + Inference engines  | Batch optimization, scheduling, gradient flow    |
| Gemma 3 compatibility         | First flagship model to validate the platform    |

---

## 🧱 Components

This project will be organized as:

