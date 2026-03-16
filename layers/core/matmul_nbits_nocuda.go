//go:build !(cuda && cutlass)

package core

import "github.com/zerfoo/ztensor/tensor"

// tryQuantizedGemm is the fallback when CUTLASS quantized GEMM is not available.
// It always returns (nil, nil) to signal that the caller should use the
// CPU dequantize + MatMul path.
func tryQuantizedGemm[T tensor.Numeric](
	_ *tensor.TensorNumeric[T],
	_ *tensor.TensorNumeric[uint8],
	_ *tensor.TensorNumeric[T],
	_ *tensor.TensorNumeric[uint8],
	_ int,
) (*tensor.TensorNumeric[T], error) {
	return nil, nil
}
