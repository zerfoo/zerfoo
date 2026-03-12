package compute

import (
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedNormAddProvider is implemented by engines that support fused
// RMSNorm + elementwise Add in a single GPU kernel launch.
// output = rmsnorm(input, weight, eps) + residual.
// This eliminates one kernel launch per fusion point.
type FusedNormAddProvider[T tensor.Numeric] interface {
	// GPUFusedNormAdd computes:
	//   normed = rmsnorm(input, weight, eps)
	//   output = normed + residual
	// All inputs are read-only. Returns (output, error).
	GPUFusedNormAdd(input, weight, residual *tensor.TensorNumeric[T], eps float32) (*tensor.TensorNumeric[T], error)
}
