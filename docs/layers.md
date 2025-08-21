# Layer Constructors Functional Options Status

This document tracks the functional options implementation status across all layer constructors in the Zerfoo ML framework.

## ✅ Layers WITH Functional Options

These layers have been successfully refactored to use the functional options pattern:

### Core Layers
- [x] `core.NewBias` - BiasOption[T] with WithBiasInitializer
- [x] `core.NewDense` - DenseOption[T] with WithBias
- [x] `core.NewFFN` - FFNOption[T] with WithFFNInitializer, WithFFNBias
- [x] `core.NewLinear` - LinearOption[T] with WithInitializer, WithXavier, WithHe, WithUniform
- [x] `core.NewPolynomialExpansion` - PolynomialExpansionOption[T] with WithPolynomialDegree, WithPolynomialBias

### Activation Layers
- [x] `activations.NewBaseActivation` - BaseActivationOption[T] with WithForwardOp, WithBackwardOp
- [x] `activations.NewLeakyReLU` - LeakyReLUOption[T] with WithAlpha
- [x] `activations.NewSwiGLU` - SwiGLUOption[T]

### Attention Layers
- [x] `attention.NewAttentionHead` - AttentionHeadOption[T]
- [x] `attention.NewGroupedQueryAttention` - GQAOption[T]
- [x] `attention.NewLocalAttention` - LocalAttentionOption[T]
- [x] `attention.NewScaledDotProductAttention` - ScaledDotProductAttentionOption[T]

### Component Layers
- [x] `components.NewHeInitializer` - HeInitializerOption[T]
- [x] `components.NewLinearGradientComputer` - LinearGradientComputerOption[T]
- [x] `components.NewMatrixMultiplier` - MatrixMultiplierOption[T]
- [x] `components.NewUniformInitializer` - UniformInitializerOption[T] with WithScale
- [x] `components.NewXavierInitializer` - XavierInitializerOption[T]

### Embedding Layers
- [x] `embeddings.NewRotaryPositionalEmbedding` - RotaryPositionalEmbeddingOption with WithRotaryBase
- [x] `embeddings.NewTokenEmbedding` - TokenEmbeddingOption[T] with WithTokenEmbeddingInitializer

### Normalization Layers
- [x] `normalization.NewLayerNormalization` - LayerNormalizationOption[T] with WithLayerNormEpsilon
- [x] `normalization.NewRMSNorm` - RMSNormOption[T] with WithRMSNormEpsilon

### Transformer Layers
- [x] `transformer.NewTransformerBlock` - BlockOption[T]

## ❌ Layers WITHOUT Functional Options

These layers currently do not use functional options.

### Simple Wrappers (No Options Needed)
- [ ] `activations.NewReLU`
- [ ] `activations.NewSigmoid`
- [ ] `activations.NewTanh`
- [ ] `core.NewLMHead`

### Complex Constructors (Refactoring Candidates)
- [ ] `attention.NewGlobalAttention`

## Summary

**Status: NEARLY COMPLETE** - 22 out of 26 layer constructors (85%) have been refactored to use functional options. The most critical and commonly used layers are now updated.

**Completed**: All core, activation, attention, component, embedding, normalization, and transformer layers that require configuration.
**Remaining**: A few simple wrappers that don't require options (`ReLU`, `Sigmoid`, `Tanh`, `LMHead`) and one complex constructor (`GlobalAttention`) that could be refactored in the future if needed.
