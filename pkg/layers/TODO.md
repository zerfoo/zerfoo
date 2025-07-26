# pkg/layers Package

This package provides reusable neural network layers (e.g., Dense, Convolutional, Activation layers). All layers adhere to a common `Layer` interface with `Forward` and `Backward` methods:contentReference[oaicite:24]{index=24}, enabling them to be composed into models. Each layer will also implement an `Update` method to adjust parameters using optimizers.
