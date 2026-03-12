package compute

import (
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedAddRMSNormProvider is implemented by engines that support fused
// residual-add + RMS normalization in a single GPU kernel launch.
// This eliminates one kernel launch per fusion point (2 per transformer layer).
type FusedAddRMSNormProvider[T tensor.Numeric] interface {
	// GPUFusedAddRMSNorm computes:
	//   residual = input + residual  (in-place update)
	//   output   = rmsnorm(residual, weight, eps)
	// Returns (normalized output, updated residual, scales, error).
	GPUFusedAddRMSNorm(input, residual, weight *tensor.TensorNumeric[T], eps float32) (
		normed *tensor.TensorNumeric[T],
		residualOut *tensor.TensorNumeric[T],
		scales *tensor.TensorNumeric[T],
		err error,
	)
}
