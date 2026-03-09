package compute

import (
	"github.com/zerfoo/zerfoo/internal/xblas"
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedRMSNorm computes x * rsqrt(mean(x^2) + eps) * weight in a single pass.
// This avoids materializing squared, mean, and rsqrt intermediate tensors.
// Input shape: [..., D] where D is the last dimension (hidden size).
// Weight shape: [D].
// Returns (output, scales) where output has same shape as input and scales
// has shape [..., 1] containing the per-row rsqrt(mean(x^2)+eps) values.
func FusedRMSNorm(input, weight *tensor.TensorNumeric[float32], epsilon float32) (output, scales *tensor.TensorNumeric[float32], err error) {
	shape := input.Shape()
	D := shape[len(shape)-1]
	total := input.Size()
	rows := total / D

	inData := input.Data()
	wData := weight.Data()
	outData := make([]float32, total)
	scaleData := make([]float32, rows)

	for row := range rows {
		off := row * D
		xblas.RMSNormF32(&outData[off], &inData[off], &wData[0], D, epsilon, &scaleData[row])
	}

	// Build scales shape: same as input but last dim = 1.
	scaleShape := make([]int, len(shape))
	copy(scaleShape, shape)
	scaleShape[len(scaleShape)-1] = 1

	output, err = tensor.New(shape, outData)
	if err != nil {
		return nil, nil, err
	}
	scales, err = tensor.New(scaleShape, scaleData)
	if err != nil {
		return nil, nil, err
	}
	return output, scales, nil
}
