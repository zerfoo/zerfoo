# pkg/optimizer Package

This package contains optimization algorithms for training, such as SGD, Adam, etc. Each optimizer implements a common `Optimizer` interface (with a `Step` method) to update model parameters using their gradients. For example, the `SGD` optimizer will adjust parameters by subtracting a fraction of the gradients each step.
