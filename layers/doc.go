// Package layers provides neural network layer implementations for the Zerfoo
// ML framework. It is organized into sub-packages by functional category, with
// a central registry that wires every layer into the model builder. (Stability: stable)
//
// # Sub-packages by category
//
// Activations:
//
//   - [github.com/zerfoo/zerfoo/layers/activations] — Activation functions
//     (GELU, FastGELU, Sigmoid, Softmax, Tanh, Erf).
//
// Attention:
//
//   - [github.com/zerfoo/zerfoo/layers/attention] — Attention mechanisms
//     (Grouped Query Attention, Global Attention, Multi-Head Latent Attention).
//
// Core math and tensor operations:
//
//   - [github.com/zerfoo/zerfoo/layers/core] — Arithmetic (Add, Sub, Mul, Div,
//     Pow, Neg, Sqrt, Mod), comparison (Equal, Greater, LessOrEqual, Or, Where),
//     shape manipulation (Reshape, Unsqueeze, Squeeze, Expand, Concat, Slice,
//     Pad, Tile, Shape, Cast), linear algebra (MatMul, Gemm, Conv2d,
//     GlobalAveragePool), rotary embeddings, FFN, Mixture of Experts, and more.
//   - [github.com/zerfoo/zerfoo/layers/gather] — Gather operation for index-based
//     element selection.
//   - [github.com/zerfoo/zerfoo/layers/reducesum] — ReduceSum along specified axes.
//   - [github.com/zerfoo/zerfoo/layers/transpose] — Transpose for axis permutation.
//
// Embeddings:
//
//   - [github.com/zerfoo/zerfoo/layers/embeddings] — Token and positional embedding
//     layers.
//
// Normalization:
//
//   - [github.com/zerfoo/zerfoo/layers/normalization] — RMSNorm, LayerNormalization,
//     SimplifiedLayerNormalization, SkipSimplifiedLayerNormalization, and
//     BatchNormalization.
//
// Regularization:
//
//   - [github.com/zerfoo/zerfoo/layers/regularization] — Dropout and FeatureDropout.
//
// Transformer:
//
//   - [github.com/zerfoo/zerfoo/layers/transformer] — Transformer building blocks
//     (encoder/decoder Block).
//
// State space models:
//
//   - [github.com/zerfoo/zerfoo/layers/ssm] — State space model layers
//     (Mamba, RWKV, S4, MIMO SSM, complex state, B/C normalization).
//
// Recurrent:
//
//   - [github.com/zerfoo/zerfoo/layers/recurrent] — Recurrent neural network layers.
//
// Higher-order / composite:
//
//   - [github.com/zerfoo/zerfoo/layers/components] — Reusable composite components
//     built from lower-level layers.
//   - [github.com/zerfoo/zerfoo/layers/hrm] — Hierarchical Reasoning Model layers.
//
// Registry:
//
//   - [github.com/zerfoo/zerfoo/layers/registry] — Central registration point that
//     maps layer names to builder functions via [registry.RegisterAll].
//
// # Layer registry
//
// The registry sub-package calls [model.RegisterLayer] for every built-in layer,
// associating a string name (e.g. "MatMul", "Softmax") with a generic builder
// function. Call [registry.RegisterAll] once at startup to make all layers
// available to the model builder.
//
// # Adding a new layer
//
// To add a new layer:
//
//  1. Create the layer in the appropriate sub-package (or a new sub-package if
//     no existing category fits). Implement the builder function with the
//     signature expected by [model.RegisterLayer].
//  2. Register the layer in [registry.RegisterAll] by adding a
//     [model.RegisterLayer] call with a unique name and the builder function.
//  3. Write tests in the same sub-package.
// Stability: stable
package layers
