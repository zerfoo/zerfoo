# Third-Party Extension Convention

This document describes how third-party developers can extend Zerfoo by
registering custom model architectures and quantization formats. Extensions
use Go's `init()` function and the blank-import pattern so that importing a
package is sufficient to activate it — no explicit wiring required.

## Extension Points

Zerfoo exposes two registries that accept third-party extensions:

| Registry | Package | Function | Interface |
|----------|---------|----------|-----------|
| Architecture | `github.com/zerfoo/zerfoo/inference` | `RegisterArchitecture(name, builder)` | `inference.ArchBuilder` |
| Quantization format | `github.com/zerfoo/ztensor/tensor` | `RegisterQuantType(name, dequantizer)` | `tensor.Dequantizer` |

Both registries are concurrency-safe and panic on duplicate or empty names,
so registration errors surface immediately at program startup.

## Naming Convention

Extension packages should follow the naming pattern:

```
github.com/<user>/zerfoo-ext-<name>
```

Examples:

- `github.com/acme/zerfoo-ext-starcoder` — StarCoder architecture
- `github.com/acme/zerfoo-ext-nf4` — NF4 quantization format

The `zerfoo-ext-` prefix makes extensions discoverable via GitHub search and
clearly signals compatibility with the Zerfoo ecosystem.

## Import Pattern

End users activate an extension with a blank import. This causes the
package's `init()` function to run, which registers the extension with the
appropriate registry.

```go
import (
    "github.com/zerfoo/zerfoo/inference"

    // Activate the StarCoder extension.
    _ "github.com/acme/zerfoo-ext-starcoder"
)
```

After this import, `inference.GetArchitecture("starcoder")` returns the
registered builder, and models with `general.architecture = starcoder` in
their GGUF metadata load automatically.

## Writing an Architecture Extension

An architecture extension registers an `inference.ArchBuilder` — a function
that builds a computation graph from GGUF tensors.

### ArchBuilder Signature

```go
type ArchBuilder func(
    tensors map[string]*tensor.TensorNumeric[float32],
    cfg     *gguf.ModelConfig,
    engine  compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)
```

The builder receives the model's tensors (keyed by GGUF tensor name), the
parsed model configuration, and a compute engine. It returns the compiled
computation graph and the embedding table tensor.

### Example

```go
// Package starcoder registers the StarCoder architecture with Zerfoo.
package starcoder

import (
    "github.com/zerfoo/ztensor/compute"
    "github.com/zerfoo/ztensor/graph"
    "github.com/zerfoo/ztensor/tensor"
    "github.com/zerfoo/zerfoo/inference"
    "github.com/zerfoo/zerfoo/model/gguf"
)

func init() {
    inference.RegisterArchitecture("starcoder", buildStarCoderGraph)
}

func buildStarCoderGraph(
    tensors map[string]*tensor.TensorNumeric[float32],
    cfg     *gguf.ModelConfig,
    engine  compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
    g := graph.New[float32](engine)
    embedTensor := tensors["token_embd.weight"]
    if embedTensor == nil {
        return nil, nil, fmt.Errorf("starcoder: missing token_embd.weight")
    }

    // Build the transformer layers using cfg and tensors...
    // (See inference/arch_llama.go for a full reference implementation.)

    return g, embedTensor, nil
}
```

### Architecture Name

The name passed to `RegisterArchitecture` must match the
`general.architecture` value stored in the GGUF file. This is how Zerfoo
looks up the correct builder at model-load time.

## Writing a Quantization Format Extension

A quantization format extension registers a `tensor.Dequantizer` — an
interface that decodes quantized weight data back to float32.

### Dequantizer Interface

```go
type Dequantizer interface {
    // Dequantize decodes quantized bytes in src into float32 values in dst.
    Dequantize(src []byte, dst []float32) error

    // BlockSize returns the number of elements per quantization block.
    BlockSize() int

    // BitsPerWeight returns the effective number of bits per weight element.
    BitsPerWeight() int
}
```

### Example

```go
// Package nf4 registers the NF4 quantization format with ztensor.
package nf4

import "github.com/zerfoo/ztensor/tensor"

func init() {
    tensor.RegisterQuantType("NF4", nf4Dequantizer{})
}

type nf4Dequantizer struct{}

func (nf4Dequantizer) Dequantize(src []byte, dst []float32) error {
    // Decode NF4 blocks from src into dst...
    return nil
}

func (nf4Dequantizer) BlockSize() int     { return 32 }
func (nf4Dequantizer) BitsPerWeight() int { return 4 }
```

### Quantization Type Name

The name passed to `RegisterQuantType` must match the quantization type
string stored in GGUF tensor metadata. Built-in formats include `Q4_0`,
`Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`, `FP8_E4M3`, and `FP8_E5M2`.

## Testing Requirements

Extensions must include tests that verify correct registration and
functional behavior. Use the patterns below as a starting point.

### Registration Test

Verify that the `init()` function registers the extension and that lookup
succeeds.

```go
package starcoder_test

import (
    "testing"

    "github.com/zerfoo/zerfoo/inference"

    // Blank import triggers init() registration.
    _ "github.com/acme/zerfoo-ext-starcoder"
)

func TestRegistered(t *testing.T) {
    builder, ok := inference.GetArchitecture("starcoder")
    if !ok {
        t.Fatal("starcoder architecture not registered")
    }
    if builder == nil {
        t.Fatal("starcoder builder is nil")
    }
}
```

### Functional Test

For architecture extensions, construct a minimal set of tensors and verify
the builder produces a valid graph without error.

```go
func TestBuildGraph(t *testing.T) {
    engine := compute.NewCPUEngine[float32]()
    tensors := createMinimalTensors(t, engine) // helper that creates stub tensors
    cfg := &gguf.ModelConfig{
        NumLayers:      2,
        EmbeddingSize:  64,
        NumHeads:       4,
        HeadDim:        16,
        VocabSize:      256,
    }
    g, embed, err := inference.GetArchitecture("starcoder")(tensors, cfg, engine)
    if err != nil {
        t.Fatalf("build graph: %v", err)
    }
    if g == nil {
        t.Fatal("graph is nil")
    }
    if embed == nil {
        t.Fatal("embedding tensor is nil")
    }
}
```

For quantization format extensions, round-trip a known input through
quantization and dequantization, then verify the output is within an
acceptable tolerance.

```go
func TestDequantize(t *testing.T) {
    d, ok := tensor.GetQuantType("NF4")
    if !ok {
        t.Fatal("NF4 not registered")
    }

    src := encodeNF4Block([]float32{0.1, -0.2, 0.3, /* ... */})
    dst := make([]float32, 32)
    if err := d.Dequantize(src, dst); err != nil {
        t.Fatalf("dequantize: %v", err)
    }

    // Verify values are within quantization tolerance.
    for i, want := range expected {
        if abs(dst[i]-want) > 0.05 {
            t.Errorf("dst[%d] = %f, want %f (±0.05)", i, dst[i], want)
        }
    }
}
```

## Listing Registered Extensions

At runtime, applications can inspect which extensions are loaded:

```go
// List all registered architectures.
for _, name := range inference.ListArchitectures() {
    fmt.Println("arch:", name)
}

// List all registered quantization formats.
for _, name := range tensor.ListQuantTypes() {
    fmt.Println("quant:", name)
}
```

## Summary

| Aspect | Convention |
|--------|-----------|
| Package naming | `github.com/<user>/zerfoo-ext-<name>` |
| Activation | Blank import: `_ "github.com/<user>/zerfoo-ext-<name>"` |
| Registration | Call `RegisterArchitecture` or `RegisterQuantType` in `init()` |
| Architecture name | Must match GGUF `general.architecture` value |
| Quant type name | Must match GGUF tensor quantization type string |
| Testing | Registration test + functional test (graph build or round-trip dequant) |
