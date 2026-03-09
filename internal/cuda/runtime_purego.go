//go:build !cuda

package cuda

import (
	"fmt"
	"unsafe"
)

// MemcpyKind specifies the direction of a memory copy.
type MemcpyKind int

const (
	// MemcpyHostToDevice copies from host to device.
	MemcpyHostToDevice MemcpyKind = 1
	// MemcpyDeviceToHost copies from device to host.
	MemcpyDeviceToHost MemcpyKind = 2
	// MemcpyDeviceToDevice copies from device to device.
	MemcpyDeviceToDevice MemcpyKind = 3
)

// cudaSuccess is the CUDA error code for success.
const cudaSuccess = 0

// cudaMemAttachGlobal is the flag for globally accessible unified memory.
const cudaMemAttachGlobal = 1

// cudaDeviceProp layout constants (CUDA 13.0, arm64).
const (
	sizeofCudaDeviceProp  = 1008
	offsetDevicePropMajor = 360
	offsetDevicePropMinor = 364
)

func lib() *CUDALib {
	l := Lib()
	if l == nil {
		return nil
	}
	return l
}

func cudaErrorString(errCode uintptr) string {
	l := lib()
	if l == nil {
		return "cuda not available"
	}
	ptr := ccall(l.cudaGetErrorString, errCode)
	if ptr == 0 {
		return "unknown error"
	}
	return goStringFromPtr(ptr)
}

// goStringFromPtr converts a C string pointer to a Go string.
// This is a thin wrapper used by runtime functions; the underlying
// goString is platform-specific.
func goStringFromPtr(p uintptr) string {
	if p == 0 {
		return ""
	}
	ptr := (*byte)(unsafe.Pointer(p)) //nolint:govet
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(ptr, n))
}

// Malloc allocates size bytes on the CUDA device and returns a device pointer.
func Malloc(size int) (unsafe.Pointer, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudaMalloc failed: cuda not available")
	}
	var devPtr uintptr
	ret := ccall(l.cudaMalloc, uintptr(unsafe.Pointer(&devPtr)), uintptr(size))
	if ret != cudaSuccess {
		return nil, fmt.Errorf("cudaMalloc failed: %s", cudaErrorString(ret))
	}
	return unsafe.Pointer(devPtr), nil //nolint:govet
}

// MallocManaged allocates size bytes of unified memory accessible from both
// host and device.
func MallocManaged(size int) (unsafe.Pointer, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudaMallocManaged failed: cuda not available")
	}
	var devPtr uintptr
	ret := ccall(l.cudaMallocManaged, uintptr(unsafe.Pointer(&devPtr)), uintptr(size), cudaMemAttachGlobal)
	if ret != cudaSuccess {
		return nil, fmt.Errorf("cudaMallocManaged failed: %s", cudaErrorString(ret))
	}
	return unsafe.Pointer(devPtr), nil //nolint:govet
}

// Free releases device memory previously allocated with Malloc or MallocManaged.
func Free(devPtr unsafe.Pointer) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudaFree failed: cuda not available")
	}
	ret := ccall(l.cudaFree, uintptr(devPtr))
	if ret != cudaSuccess {
		return fmt.Errorf("cudaFree failed: %s", cudaErrorString(ret))
	}
	return nil
}

// Memcpy copies count bytes between host and device memory.
func Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudaMemcpy failed: cuda not available")
	}
	ret := ccall(l.cudaMemcpy, uintptr(dst), uintptr(src), uintptr(count), uintptr(kind))
	if ret != cudaSuccess {
		return fmt.Errorf("cudaMemcpy failed: %s", cudaErrorString(ret))
	}
	return nil
}

// GetDeviceCount returns the number of CUDA-capable devices.
func GetDeviceCount() (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("cudaGetDeviceCount failed: cuda not available")
	}
	var count int32
	ret := ccall(l.cudaGetDeviceCount, uintptr(unsafe.Pointer(&count)))
	if ret != cudaSuccess {
		return 0, fmt.Errorf("cudaGetDeviceCount failed: %s", cudaErrorString(ret))
	}
	return int(count), nil
}

// SetDevice sets the current CUDA device.
func SetDevice(deviceID int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudaSetDevice failed: cuda not available")
	}
	ret := ccall(l.cudaSetDevice, uintptr(deviceID))
	if ret != cudaSuccess {
		return fmt.Errorf("cudaSetDevice failed: %s", cudaErrorString(ret))
	}
	return nil
}

// Stream wraps a cudaStream_t handle for asynchronous kernel execution.
type Stream struct {
	handle uintptr // opaque cudaStream_t (void*)
}

// CreateStream creates a new CUDA stream.
func CreateStream() (*Stream, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("cudaStreamCreate failed: cuda not available")
	}
	var handle uintptr
	ret := ccall(l.cudaStreamCreate, uintptr(unsafe.Pointer(&handle)))
	if ret != cudaSuccess {
		return nil, fmt.Errorf("cudaStreamCreate failed: %s", cudaErrorString(ret))
	}
	return &Stream{handle: handle}, nil
}

// Synchronize blocks until all work on this stream completes.
func (s *Stream) Synchronize() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudaStreamSynchronize failed: cuda not available")
	}
	ret := ccall(l.cudaStreamSynchronize, s.handle)
	if ret != cudaSuccess {
		return fmt.Errorf("cudaStreamSynchronize failed: %s", cudaErrorString(ret))
	}
	return nil
}

// Destroy releases the CUDA stream.
func (s *Stream) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudaStreamDestroy failed: cuda not available")
	}
	ret := ccall(l.cudaStreamDestroy, s.handle)
	if ret != cudaSuccess {
		return fmt.Errorf("cudaStreamDestroy failed: %s", cudaErrorString(ret))
	}
	return nil
}

// Ptr returns the underlying cudaStream_t as an unsafe.Pointer.
func (s *Stream) Ptr() unsafe.Pointer {
	return unsafe.Pointer(s.handle) //nolint:govet
}

// MemcpyPeer copies count bytes between devices using peer-to-peer transfer.
func MemcpyPeer(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, count int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudaMemcpyPeer failed: cuda not available")
	}
	ret := ccall(l.cudaMemcpyPeer,
		uintptr(dst), uintptr(dstDevice),
		uintptr(src), uintptr(srcDevice),
		uintptr(count))
	if ret != cudaSuccess {
		return fmt.Errorf("cudaMemcpyPeer failed: %s", cudaErrorString(ret))
	}
	return nil
}

// DeviceComputeCapability returns the major and minor compute capability.
func DeviceComputeCapability(deviceID int) (major, minor int, err error) {
	l := lib()
	if l == nil {
		return 0, 0, fmt.Errorf("cudaGetDeviceProperties failed: cuda not available")
	}
	// Allocate a raw buffer for cudaDeviceProp (1008 bytes on CUDA 13.0 arm64).
	var prop [sizeofCudaDeviceProp]byte
	ret := ccall(l.cudaGetDeviceProperties, uintptr(unsafe.Pointer(&prop[0])), uintptr(deviceID))
	if ret != cudaSuccess {
		return 0, 0, fmt.Errorf("cudaGetDeviceProperties failed: %s", cudaErrorString(ret))
	}
	// Extract major (int32 at offset 360) and minor (int32 at offset 364).
	maj := *(*int32)(unsafe.Pointer(&prop[offsetDevicePropMajor]))
	min := *(*int32)(unsafe.Pointer(&prop[offsetDevicePropMinor]))
	return int(maj), int(min), nil
}

// MemcpyAsync copies count bytes asynchronously on the given stream.
func MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, stream *Stream) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("cudaMemcpyAsync failed: cuda not available")
	}
	var streamHandle uintptr
	if stream != nil {
		streamHandle = stream.handle
	}
	ret := ccall(l.cudaMemcpyAsync,
		uintptr(dst), uintptr(src), uintptr(count),
		uintptr(kind), streamHandle)
	if ret != cudaSuccess {
		return fmt.Errorf("cudaMemcpyAsync failed: %s", cudaErrorString(ret))
	}
	return nil
}
