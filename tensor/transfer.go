//go:build cuda

package tensor

// ToGPU creates a new tensor with GPUStorage containing the same data as the
// source tensor. Shape and strides are preserved. The source tensor is not
// modified.
func ToGPU[T Numeric](t *TensorNumeric[T]) (*TensorNumeric[T], error) {
	gpuStorage, err := NewGPUStorageFromSlice(t.Data())
	if err != nil {
		return nil, err
	}

	shape := make([]int, len(t.shape))
	copy(shape, t.shape)

	strides := make([]int, len(t.strides))
	copy(strides, t.strides)

	return &TensorNumeric[T]{
		shape:   shape,
		strides: strides,
		storage: gpuStorage,
		isView:  false,
	}, nil
}

// ToCPU creates a new tensor with CPUStorage containing the same data as the
// source tensor. Shape and strides are preserved. The source tensor is not
// modified.
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
