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
- [ ] `core.NewFFN` - Could benefit from initialization options
- [ ] `core.NewLMHead` - Simple wrapper around Linear
- [ ] `core.NewPolynomialExpansion` - Complex constructor with many parameters

### Embedding Layers
- [ ] `embeddings.NewRotaryPositionalEmbedding` - Complex constructor, could use options for base/theta
- [ ] `embeddings.NewTokenEmbedding` - Could benefit from initialization options

### Normalization Layers
- [ ] `normalization.NewLayerNormalization` - Simple constructor, minimal options needed
- [ ] `normalization.NewRMSNorm` - Simple constructor, minimal options needed

### Attention Layers
- [ ] `attention.NewGlobalAttention` - May need attention-specific options

## Summary

**Status: PARTIAL** - 16 out of 25 layer constructors (64%) have been refactored to use functional options.

**Completed**: Core layers, most activation layers, attention layers, components, and transformer layers
**Remaining**: Some activation wrappers, embeddings, normalization, and complex constructors

The framework has good functional options coverage for the most commonly used and configurable layers.