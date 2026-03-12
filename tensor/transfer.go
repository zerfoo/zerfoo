package tensor

import (
	"fmt"

	"github.com/zerfoo/zerfoo/internal/gpuapi"
)

// ToGPU creates a new tensor with GPUStorage on device 0 containing the same
// data as the source tensor. Shape and strides are preserved. The source
// tensor is not modified.
func ToGPU[T Numeric](t *TensorNumeric[T]) (*TensorNumeric[T], error) {
	return ToGPUDevice(t, 0)
}

// ToGPUDevice creates a new tensor with GPUStorage on the specified device
// containing the same data as the source tensor. If the source tensor is
// already on a GPU, a peer-to-peer D2D copy is used when the devices differ;
// if on the same device, a D2D copy is performed. If the source is CPU-backed,
// an H2D copy targets the specified device.
func ToGPUDevice[T Numeric](t *TensorNumeric[T], deviceID int) (*TensorNumeric[T], error) {
	shape := make([]int, len(t.shape))
	copy(shape, t.shape)

	strides := make([]int, len(t.strides))
	copy(strides, t.strides)

	// Source is already on a GPU.
	if gs, ok := t.GetStorage().(*GPUStorage[T]); ok {
		srcDev := gs.DeviceID()
		rt := gs.runtime

		if err := rt.SetDevice(deviceID); err != nil {
			return nil, fmt.Errorf("ToGPUDevice: SetDevice(%d): %w", deviceID, err)
		}

		dst, err := NewGPUStorage[T](gs.Len(), deviceID)
		if err != nil {
			return nil, fmt.Errorf("ToGPUDevice: alloc on device %d: %w", deviceID, err)
		}

		if srcDev == deviceID {
			// Same device: use D2D copy.
			if err := rt.Memcpy(dst.Ptr(), gs.Ptr(), gs.byteSize, gpuapi.MemcpyDeviceToDevice); err != nil {
				_ = dst.Free()
				return nil, fmt.Errorf("ToGPUDevice: D2D copy: %w", err)
			}
		} else {
			// Cross-device: peer-to-peer copy.
			if err := rt.MemcpyPeer(dst.Ptr(), deviceID, gs.Ptr(), srcDev, gs.byteSize); err != nil {
				_ = dst.Free()
				return nil, fmt.Errorf("ToGPUDevice: peer copy %d->%d: %w", srcDev, deviceID, err)
			}
		}

		return &TensorNumeric[T]{
			shape:   shape,
			strides: strides,
			storage: dst,
			isView:  false,
		}, nil
	}

	// CPU source: H2D copy to target device.
	gpuStorage, err := NewGPUStorageFromSlice(t.Data(), deviceID)
	if err != nil {
		return nil, err
	}

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
