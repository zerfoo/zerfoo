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

// StreamFromPtr wraps an existing cudaStream_t handle as a Stream.
// The caller retains ownership of the handle; Destroy() must NOT be
// called on the returned Stream (it would destroy the engine's stream).
func StreamFromPtr(ptr unsafe.Pointer) *Stream {
	return &Stream{handle: uintptr(ptr)}
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

// CUDA device attribute constants.
const (
	// cudaDevAttrManagedMemory indicates the device supports managed memory.
	cudaDevAttrManagedMemory = 83
	// cudaDevAttrConcurrentManagedAccess indicates the device supports
	// concurrent access to managed memory from CPU and GPU without faulting.
	cudaDevAttrConcurrentManagedAccess = 89
)

// DeviceGetAttribute queries a device attribute.
func DeviceGetAttribute(attr, deviceID int) (int, error) {
	l := lib()
	if l == nil {
		return 0, fmt.Errorf("cudaDeviceGetAttribute failed: cuda not available")
	}
	var value int32
	ret := ccall(l.cudaDeviceGetAttribute, uintptr(unsafe.Pointer(&value)), uintptr(attr), uintptr(deviceID))
	if ret != cudaSuccess {
		return 0, fmt.Errorf("cudaDeviceGetAttribute(%d) failed: %s", attr, cudaErrorString(ret))
	}
	return int(value), nil
}

// ManagedMemorySupported returns true if the device supports unified
// (managed) memory with concurrent access from CPU and GPU. On GB10 with
// NVLink-C2C and shared LPDDR5x, this avoids all explicit H2D/D2H copies.
func ManagedMemorySupported(deviceID int) bool {
	managed, err := DeviceGetAttribute(cudaDevAttrManagedMemory, deviceID)
	if err != nil || managed == 0 {
		return false
	}
	concurrent, err := DeviceGetAttribute(cudaDevAttrConcurrentManagedAccess, deviceID)
	if err != nil || concurrent == 0 {
		return false
	}
	return true
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

// CUDA graph capture mode constants.
const (
	// cudaStreamCaptureModeGlobal captures all operations on any stream.
	// This mode blocks synchronous memcpy on the legacy stream during capture.
	cudaStreamCaptureModeGlobal = 0 //nolint:unused
	// cudaStreamCaptureModeRelaxed only captures operations on the
	// capturing stream. Other streams (including the legacy/default stream)
	// can execute normally. Operations on other streams are NOT captured.
	cudaStreamCaptureModeRelaxed = 2
)

// Graph wraps a cudaGraph_t handle.
type Graph struct {
	handle uintptr
}

// GraphExec wraps a cudaGraphExec_t handle for graph replay.
type GraphExec struct {
	handle uintptr
}

// StreamBeginCapture starts capturing GPU operations on the given stream.
// All kernel launches, memcpys, and other operations on the stream are
// recorded into a graph instead of being executed.
func StreamBeginCapture(s *Stream) error {
	l := lib()
	if l == nil || l.cudaStreamBeginCapture == 0 {
		return fmt.Errorf("cudaStreamBeginCapture: not available")
	}
	ret := ccall(l.cudaStreamBeginCapture, s.handle, cudaStreamCaptureModeRelaxed)
	if ret != cudaSuccess {
		return fmt.Errorf("cudaStreamBeginCapture failed: %s", cudaErrorString(ret))
	}
	return nil
}

// StreamEndCapture stops capturing on the stream and returns the captured graph.
func StreamEndCapture(s *Stream) (*Graph, error) {
	l := lib()
	if l == nil || l.cudaStreamEndCapture == 0 {
		return nil, fmt.Errorf("cudaStreamEndCapture: not available")
	}
	var graphHandle uintptr
	ret := ccall(l.cudaStreamEndCapture, s.handle, uintptr(unsafe.Pointer(&graphHandle)))
	if ret != cudaSuccess {
		return nil, fmt.Errorf("cudaStreamEndCapture failed: %s", cudaErrorString(ret))
	}
	return &Graph{handle: graphHandle}, nil
}

// GraphInstantiate creates an executable graph from a captured graph.
// The executable graph can be launched repeatedly without re-capturing.
func GraphInstantiate(g *Graph) (*GraphExec, error) {
	l := lib()
	if l == nil || l.cudaGraphInstantiate == 0 {
		return nil, fmt.Errorf("cudaGraphInstantiate: not available")
	}
	var execHandle uintptr
	// cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
	//                      unsigned long long flags)
	// flags=0 for default behavior.
	ret := ccall(l.cudaGraphInstantiate, uintptr(unsafe.Pointer(&execHandle)), g.handle, 0)
	if ret != cudaSuccess {
		return nil, fmt.Errorf("cudaGraphInstantiate failed: %s", cudaErrorString(ret))
	}
	return &GraphExec{handle: execHandle}, nil
}

// GraphLaunch launches an executable graph on the given stream.
// This replays the entire captured sequence of operations with minimal overhead.
func GraphLaunch(ge *GraphExec, s *Stream) error {
	l := lib()
	if l == nil || l.cudaGraphLaunch == 0 {
		return fmt.Errorf("cudaGraphLaunch: not available")
	}
	ret := ccall(l.cudaGraphLaunch, ge.handle, s.handle)
	if ret != cudaSuccess {
		return fmt.Errorf("cudaGraphLaunch failed: %s", cudaErrorString(ret))
	}
	return nil
}

// GraphDestroy releases a captured graph.
func GraphDestroy(g *Graph) error {
	l := lib()
	if l == nil || l.cudaGraphDestroy == 0 {
		return fmt.Errorf("cudaGraphDestroy: not available")
	}
	ret := ccall(l.cudaGraphDestroy, g.handle)
	if ret != cudaSuccess {
		return fmt.Errorf("cudaGraphDestroy failed: %s", cudaErrorString(ret))
	}
	return nil
}

// GraphExecDestroy releases an executable graph.
func GraphExecDestroy(ge *GraphExec) error {
	l := lib()
	if l == nil || l.cudaGraphExecDestroy == 0 {
		return fmt.Errorf("cudaGraphExecDestroy: not available")
	}
	ret := ccall(l.cudaGraphExecDestroy, ge.handle)
	if ret != cudaSuccess {
		return fmt.Errorf("cudaGraphExecDestroy failed: %s", cudaErrorString(ret))
	}
	return nil
}
