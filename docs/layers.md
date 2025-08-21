# Layer Constructors To Be Refactored to Use Functional Options

This document lists layer constructors that currently do not utilize the functional options pattern and should be refactored for consistency and extensibility.

**STATUS: COMPLETE** - All applicable layers have been refactored.

- [x] `activations.NewBaseActivation`
- [x] `activations.NewLeakyReLU`
- [x] `activations.NewReLU`
- [x] `activations.NewSigmoid`
- [x] `activations.NewSwiGLU`
- [x] `attention.NewAttentionHead`
- [x] `attention.NewScaledDotProductAttention`
- [x] `attention.NewGroupedQueryAttention`
- [x] `attention.NewLocalAttention`
- [x] `components.NewLinearGradientComputer`
- [x] `components.NewMatrixMultiplier`
- [x] `components.NewXavierInitializer`
- [x] `components.NewHeInitializer`
- [x] `components.NewUniformInitializer`
- [x] `core.NewBias`
- [x] `core.NewLinear`
- [x] `core.NewLMHead`
- [x] `core.NewPolynomialExpansion`
- [x] `embeddings.NewRotaryPositionalEmbedding`
- [x] `embeddings.NewTokenEmbedding`
- [x] `normalization.NewLayerNormalization`
- [x] `normalization.NewRMSNorm`
- [x] `transformer.NewTransformerBlock`