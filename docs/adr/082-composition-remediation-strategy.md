# ADR 082: Composition Remediation Strategy

## Status
Accepted

## Date
2026-04-02

## Context
A 5-agent audit (docs/dirty-architecture.md, 2026-04-02) found that 7 packages
violate the composition principle: timeseries, crossasset, tabular, gnn, modeldsl,
parts of inference (6 architecture builders), and ztensor (gpu_engine.go god object).

The violations share a root cause: training backends were written before the layers/
package existed. Each backend reimplements its own softmax (15 copies), GELU (13),
sigmoid (7), layerNorm (67 references), matmul (33), and AdamW optimizer (5 copies).

Prior work (E50, E52, E53, E60) addressed the worst duplication in timeseries/ and
crossasset/ by extracting shared helpers and moving some operations to engine ops.
However, the fundamental pattern of bypassing layers/ persists across all training
backends.

Additionally, 6 inference architecture builders (arch_rwkv, arch_bert, arch_gpt2,
arch_llava, arch_falcon, arch_llama) define 31 custom graph nodes with inline math
instead of composing from layers/. And gpu_engine.go in ztensor contains 16
nearly-identical quantized matmul methods (1,562 lines of copy-paste).

## Decision
Remediate in 5 phases, each as an independent epic:

1. **E61: Inference Builder Composition.** Migrate 6 architecture builders to
   compose from layers/ instead of inline math. Priority: arch_rwkv (worst),
   arch_bert, arch_gpt2, then arch_llava, arch_falcon, arch_llama.

2. **E62: Auxiliary Training Package Composition.** Migrate tabular, gnn, and
   modeldsl to use layers/ or engine ops. Convert gnn from [][]float64 to tensor
   representation. Delete private math reimplementations.

3. **E63: Quantized MatMul Consolidation (ztensor).** Replace 16 copy-paste
   matmul methods in gpu_engine.go with a single generic dequantMatMul dispatcher
   that takes storage type and block size as parameters. Target: eliminate ~1,400
   lines.

4. **E64: GPU Engine File Decomposition (ztensor).** Split gpu_engine.go (4,318
   lines, 94 methods) into focused files: matmul, elementwise, reduction, memory,
   and core lifecycle. Does not change any API -- pure file reorganization.

5. **E65: MoE Layer Composition Fix.** Replace raw .Data() access in
   layers/core/moe.go with engine ops for bias addition, sigmoid, and softmax.
   Keep top-K routing as-is (no engine op equivalent).

Phase ordering: E61 and E62 are independent and can run in parallel. E63 must
precede E64 (consolidation before splitting). E65 is independent.

Do NOT create an internal/mathops/ package. E52 established that shared helpers
live alongside their consumers (timeseries/math_ops.go, timeseries/layernorm_ops.go).
The long-term goal is to eliminate these helpers entirely by composing from layers/.

## Consequences
Positive:
- All packages compose from layers/ and engine ops, enabling transparent CPU/GPU switching
- Bug fixes and optimizations to layers/ benefit all training backends automatically
- Reduced codebase by ~3,000+ lines of duplicated math
- gpu_engine.go becomes maintainable (from 4,318 to ~800 lines per file)
- CUDA graph capture and megakernel codegen work across all code paths

Negative:
- Migration risk: each backend needs parity testing (output match within tolerance)
- gnn [][]float64 to tensor conversion may require API changes in gnn package
- Some engine dispatch overhead for operations previously done as raw loops
- E63/E64 are in ztensor repo, requiring coordinated releases
