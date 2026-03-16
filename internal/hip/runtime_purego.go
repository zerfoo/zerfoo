package hip

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
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

// hipSuccess is the HIP error code for success.
const hipSuccess = 0

func lib() *HIPLib {
	l := Lib()
	if l == nil {
		return nil
	}
	return l
}

func hipErrorString(errCode uintptr) string {
	l := lib()
	if l == nil {
		return "hip not available"
	}
	ptr := cuda.Ccall(l.hipGetErrorString, errCode)
	if ptr == 0 {
		return "unknown error"
	}
	return goStringFromPtr(ptr)
}

// ptrFromUintptr converts a uintptr to unsafe.Pointer without triggering
// go vet's unsafeptr analyzer. Required for purego FFI.
func ptrFromUintptr(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// goStringFromPtr converts a C string pointer to a Go string.
func goStringFromPtr(p uintptr) string {
	if p == 0 {
		return ""
	}
	ptr := (*byte)(ptrFromUintptr(p))
	var n int
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(ptr, n))
}

// Malloc allocates size bytes on the HIP device and returns a device pointer.
func Malloc(size int) (unsafe.Pointer, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("hipMalloc failed: hip not available")
	}
	var devPtr uintptr
	ret := cuda.Ccall(l.hipMalloc, uintptr(unsafe.Pointer(&devPtr)), uintptr(size))
	if ret != hipSuccess {
		return nil, fmt.Errorf("hipMalloc failed: %s", hipErrorString(ret))
	}
	return ptrFromUintptr(devPtr), nil
}

// Free releases device memory previously allocated with Malloc.
func Free(devPtr unsafe.Pointer) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("hipFree failed: hip not available")
	}
	ret := cuda.Ccall(l.hipFree, uintptr(devPtr))
	if ret != hipSuccess {
		return fmt.Errorf("hipFree failed: %s", hipErrorString(ret))
	}
	return nil
}

// Memcpy copies count bytes between host and device memory.
func Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("hipMemcpy failed: hip not available")
	}
	ret := cuda.Ccall(l.hipMemcpy, uintptr(dst), uintptr(src), uintptr(count), uintptr(kind))
	if ret != hipSuccess {
		return fmt.Errorf("hipMemcpy failed: %s", hipErrorString(ret))
	}
	return nil
}

// GetDeviceCount returns the number of HIP-capable devices.
func GetDeviceCount() (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("hipGetDeviceCount failed: hip not available")
	}
	var count int32
	ret := cuda.Ccall(l.hipGetDeviceCount, uintptr(unsafe.Pointer(&count)))
	if ret != hipSuccess {
		return 0, fmt.Errorf("hipGetDeviceCount failed: %s", hipErrorString(ret))
	}
	return int(count), nil
}

// SetDevice sets the current HIP device.
func SetDevice(deviceID int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("hipSetDevice failed: hip not available")
	}
	ret := cuda.Ccall(l.hipSetDevice, uintptr(deviceID))
	if ret != hipSuccess {
		return fmt.Errorf("hipSetDevice failed: %s", hipErrorString(ret))
	}
	return nil
}

// Stream wraps a hipStream_t handle for asynchronous kernel execution.
type Stream struct {
	handle uintptr // opaque hipStream_t (void*)
}

// StreamFromPtr wraps an existing hipStream_t handle as a Stream.
func StreamFromPtr(ptr unsafe.Pointer) *Stream {
	return &Stream{handle: uintptr(ptr)}
}

// CreateStream creates a new HIP stream.
func CreateStream() (*Stream, error) {
	l := lib()
	if l == nil {
		return nil, fmt.Errorf("hipStreamCreate failed: hip not available")
	}
	var handle uintptr
	ret := cuda.Ccall(l.hipStreamCreate, uintptr(unsafe.Pointer(&handle)))
	if ret != hipSuccess {
		return nil, fmt.Errorf("hipStreamCreate failed: %s", hipErrorString(ret))
	}
	return &Stream{handle: handle}, nil
}

// Synchronize blocks until all work on this stream completes.
func (s *Stream) Synchronize() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("hipStreamSynchronize failed: hip not available")
	}
	ret := cuda.Ccall(l.hipStreamSynchronize, s.handle)
	if ret != hipSuccess {
		return fmt.Errorf("hipStreamSynchronize failed: %s", hipErrorString(ret))
	}
	return nil
}

// Destroy releases the HIP stream.
func (s *Stream) Destroy() error {
	l := lib()
	if l == nil {
		return fmt.Errorf("hipStreamDestroy failed: hip not available")
	}
	ret := cuda.Ccall(l.hipStreamDestroy, s.handle)
	if ret != hipSuccess {
		return fmt.Errorf("hipStreamDestroy failed: %s", hipErrorString(ret))
	}
	return nil
}

// Ptr returns the underlying hipStream_t as an unsafe.Pointer.
func (s *Stream) Ptr() unsafe.Pointer {
	return ptrFromUintptr(s.handle)
}

// MemcpyPeer copies count bytes between devices using peer-to-peer transfer.
func MemcpyPeer(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, count int) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("hipMemcpyPeer failed: hip not available")
	}
	ret := cuda.Ccall(l.hipMemcpyPeer,
		uintptr(dst), uintptr(dstDevice),
		uintptr(src), uintptr(srcDevice),
		uintptr(count))
	if ret != hipSuccess {
		return fmt.Errorf("hipMemcpyPeer failed: %s", hipErrorString(ret))
	}
	return nil
}

// MemcpyAsync copies count bytes asynchronously on the given stream.
func MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, stream *Stream) error {
	l := lib()
	if l == nil {
		return fmt.Errorf("hipMemcpyAsync failed: hip not available")
	}
	var streamHandle uintptr
	if stream != nil {
		streamHandle = stream.handle
	}
	ret := cuda.Ccall(l.hipMemcpyAsync,
		uintptr(dst), uintptr(src), uintptr(count),
		uintptr(kind), streamHandle)
	if ret != hipSuccess {
		return fmt.Errorf("hipMemcpyAsync failed: %s", hipErrorString(ret))
	}
	return nil
}
