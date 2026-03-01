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

// Stream wraps a cudaStream_t for asynchronous kernel execution and memory transfers.
type Stream struct {
	s C.cudaStream_t
}

// CreateStream creates a new CUDA stream.
func CreateStream() (*Stream, error) {
	var s C.cudaStream_t

	err := C.cudaStreamCreate(&s)
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("cudaStreamCreate failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return &Stream{s: s}, nil
}

// Synchronize blocks the calling CPU thread until all previously issued work
// on this stream has completed.
func (s *Stream) Synchronize() error {
	err := C.cudaStreamSynchronize(s.s)
	if err != C.cudaSuccess {
		return fmt.Errorf("cudaStreamSynchronize failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return nil
}

// Destroy releases the CUDA stream. The stream must not be used after Destroy.
func (s *Stream) Destroy() error {
	err := C.cudaStreamDestroy(s.s)
	if err != C.cudaSuccess {
		return fmt.Errorf("cudaStreamDestroy failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return nil
}

// Ptr returns the underlying cudaStream_t as an unsafe.Pointer.
// This is used to pass the stream to kernel launchers via CGO.
func (s *Stream) Ptr() unsafe.Pointer {
	return unsafe.Pointer(s.s)
}

// MemcpyAsync copies count bytes asynchronously on the given stream.
func MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, stream *Stream) error {
	var cs C.cudaStream_t
	if stream != nil {
		cs = stream.s
	}

	err := C.cudaMemcpyAsync(dst, src, C.size_t(count), uint32(kind), cs)
	if err != C.cudaSuccess {
		return fmt.Errorf("cudaMemcpyAsync failed: %s", C.GoString(C.cudaGetErrorString(err)))
	}

	return nil
}
