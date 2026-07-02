// Package gpuapi defines internal interfaces for GPU runtime operations. (Stability: beta)
//
// The GPU Runtime Abstraction Layer (GRAL) decouples GPUEngine and GPUStorage
// from vendor-specific APIs (CUDA, ROCm, OpenCL). Each vendor implements the
// Runtime, BLAS, DNN, and KernelRunner interfaces via adapter packages.
//
// These interfaces are internal and not exported to users. The public Engine[T]
// interface in compute/ is unchanged.
// Stability: beta
package gpuapi
