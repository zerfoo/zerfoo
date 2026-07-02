package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/ztensor/device"
)

// MemcpyKind specifies the direction of a memory copy operation.
type MemcpyKind int

const (
	// MemcpyHostToDevice copies from host (CPU) memory to device (GPU) memory.
	MemcpyHostToDevice MemcpyKind = iota
	// MemcpyDeviceToHost copies from device (GPU) memory to host (CPU) memory.
	MemcpyDeviceToHost
	// MemcpyDeviceToDevice copies between device (GPU) memory regions.
	MemcpyDeviceToDevice
)

// Stream represents an asynchronous command queue on a GPU device.
type Stream interface {
	// Synchronize blocks until all commands in the stream have completed.
	Synchronize() error
	// Destroy releases the stream resources.
	Destroy() error
	// Ptr returns the underlying vendor stream handle as an unsafe.Pointer.
	// For CUDA this is cudaStream_t, for ROCm this is hipStream_t.
	Ptr() unsafe.Pointer
}

// Runtime abstracts GPU device and memory management operations.
// Each vendor (CUDA, ROCm, OpenCL) provides an implementation.
type Runtime interface {
	// DeviceType returns the device type this runtime manages.
	DeviceType() device.Type

	// SetDevice sets the active GPU device for the calling goroutine.
	SetDevice(deviceID int) error
	// GetDeviceCount returns the number of available GPU devices.
	GetDeviceCount() (int, error)

	// Malloc allocates byteSize bytes of device memory.
	Malloc(byteSize int) (unsafe.Pointer, error)
	// Free releases device memory previously allocated by Malloc.
	Free(ptr unsafe.Pointer) error
	// Memcpy copies count bytes between host and device memory.
	Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error
	// MemcpyAsync copies count bytes asynchronously on the given stream.
	MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, stream Stream) error
	// MemcpyPeer copies count bytes between devices (peer-to-peer).
	MemcpyPeer(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, count int) error

	// CreateStream creates a new asynchronous command stream.
	CreateStream() (Stream, error)
}
