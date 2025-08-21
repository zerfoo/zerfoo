# Layer Constructors Functional Options Status

This document tracks the functional options implementation status across all layer constructors in the Zerfoo ML framework.

## ✅ Layers WITH Functional Options

These layers have been successfully refactored to use the functional options pattern:

### Core Layers
- [x] `core.NewBias` - BiasOption[T] with WithBiasInitializer
- [x] `core.NewDense` - DenseOption[T] with WithBias
- [x] `core.NewLinear` - LinearOption[T] with WithInitializer, WithXavier, WithHe, WithUniform

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

### Transformer Layers
- [x] `transformer.NewTransformerBlock` - BlockOption[T]

## ❌ Layers WITHOUT Functional Options

These layers currently do not use functional options and may benefit from refactoring:

### Activation Layers (Simple)
- [ ] `activations.NewReLU` - Simple wrapper, no options needed
- [ ] `activations.NewSigmoid` - Simple wrapper, no options needed  
- [ ] `activations.NewTanh` - Simple wrapper, no options needed

### Core Layers
- [x] `core.NewFFN` - FFNOption[T] with WithFFNInitializer, WithFFNBias
- [ ] `core.NewLMHead` - Simple wrapper around Linear
- [ ] `core.NewPolynomialExpansion` - Complex constructor with many parameters

### Embedding Layers
- [x] `embeddings.NewRotaryPositionalEmbedding` - RotaryPositionalEmbeddingOption[T] with WithRotaryBase
- [x] `embeddings.NewTokenEmbedding` - TokenEmbeddingOption[T] with WithTokenEmbeddingInitializer

### Normalization Layers
- [x] `normalization.NewLayerNormalization` - LayerNormalizationOption[T] with WithLayerNormEpsilon
- [x] `normalization.NewRMSNorm` - RMSNormOption[T] with WithRMSNormEpsilon

### Attention Layers
- [ ] `attention.NewGlobalAttention` - May need attention-specific options

## Summary

**Status: NEARLY COMPLETE** - 21 out of 25 layer constructors (84%) have been refactored to use functional options.

**Completed**: Core layers (including FFN), most activation layers, attention layers, components, transformer layers, embeddings, and normalization layers
**Remaining**: Simple activation wrappers, LMHead, PolynomialExpansion, and GlobalAttention

The framework has good functional options coverage for the most commonly used and configurable layers.