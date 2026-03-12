//go:build !cuda && !rocm && !opencl

package tensor

import "fmt"

// ToGPU is a stub for non-GPU builds.
func ToGPU[T Numeric](t *TensorNumeric[T]) (*TensorNumeric[T], error) {
	return nil, fmt.Errorf("ToGPU: no GPU backend compiled (need -tags cuda, rocm, or opencl)")
}

// ToGPUDevice is a stub for non-GPU builds.
func ToGPUDevice[T Numeric](t *TensorNumeric[T], deviceID int) (*TensorNumeric[T], error) {
	return nil, fmt.Errorf("ToGPUDevice: no GPU backend compiled")
}

// ToCPU creates a CPU copy of a tensor.
func ToCPU[T Numeric](t *TensorNumeric[T]) *TensorNumeric[T] {
	data := t.Data()
	cpuData := make([]T, len(data))
	copy(cpuData, data)

	shape := make([]int, len(t.shape))
	copy(shape, t.shape)

	strides := make([]int, len(t.strides))
	copy(strides, t.strides)

	return &TensorNumeric[T]{
		shape:   shape,
		strides: strides,
		storage: NewCPUStorage(cpuData),
		isView:  false,
	}
}
