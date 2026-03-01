//go:build cuda

// Package cuda provides low-level CGO bindings for the CUDA runtime API.
package cuda

/*
#cgo LDFLAGS: -lcudart
#include <cuda_runtime.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// MemcpyKind specifies the direction of a memory copy.
type MemcpyKind int

const (
	// MemcpyHostToDevice copies from host to device.
	MemcpyHostToDevice MemcpyKind = C.cudaMemcpyHostToDevice
	// MemcpyDeviceToHost copies from device to host.
	MemcpyDeviceToHost MemcpyKind = C.cudaMemcpyDeviceToHost
	// MemcpyDeviceToDevice copies from device to device.
	MemcpyDeviceToDevice MemcpyKind = C.cudaMemcpyDeviceToDevice
)

// Malloc allocates size bytes on the CUDA device and returns a device pointer.
func Malloc(size int) (unsafe.Pointer, error) {
	var devPtr unsafe.Pointer

	err := C.cudaMalloc(&devPtr, C.size_t(size))
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("cudaMalloc failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return devPtr, nil
}

// Free releases device memory previously allocated with Malloc.
func Free(devPtr unsafe.Pointer) error {
	err := C.cudaFree(devPtr)
	if err != C.cudaSuccess {
		return fmt.Errorf("cudaFree failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return nil
}

// Memcpy copies count bytes between host and device memory.
func Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	err := C.cudaMemcpy(dst, src, C.size_t(count), uint32(kind))
	if err != C.cudaSuccess {
		return fmt.Errorf("cudaMemcpy failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return nil
}

// GetDeviceCount returns the number of CUDA-capable devices.
func GetDeviceCount() (int, error) {
	var count C.int

	err := C.cudaGetDeviceCount(&count)
	if err != C.cudaSuccess {
		return 0, fmt.Errorf("cudaGetDeviceCount failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return int(count), nil
}

// SetDevice sets the current CUDA device.
func SetDevice(deviceID int) error {
	err := C.cudaSetDevice(C.int(deviceID))
	if err != C.cudaSuccess {
		return fmt.Errorf("cudaSetDevice failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return nil
}
