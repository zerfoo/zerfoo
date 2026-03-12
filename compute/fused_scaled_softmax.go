package compute

import (
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedScaledSoftmaxProvider is implemented by engines that support fused GPU scaled softmax.
// It computes output = softmax(input * scale) in a single kernel launch,
// eliminating the MulScalar + Softmax chain (saves 1 kernel launch per call).
type FusedScaledSoftmaxProvider[T tensor.Numeric] interface {
	GPUScaledSoftmax(input *tensor.TensorNumeric[T], scale float32, axis int) (*tensor.TensorNumeric[T], error)
}
