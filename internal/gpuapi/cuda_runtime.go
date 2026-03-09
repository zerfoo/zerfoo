package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/cuda"
)

// CUDARuntime implements the Runtime interface using the CUDA runtime API.
type CUDARuntime struct{}

// NewCUDARuntime returns a new CUDA runtime adapter.
func NewCUDARuntime() *CUDARuntime {
	return &CUDARuntime{}
}

func (r *CUDARuntime) DeviceType() device.Type { return device.CUDA }

func (r *CUDARuntime) SetDevice(deviceID int) error {
	return cuda.SetDevice(deviceID)
}

func (r *CUDARuntime) GetDeviceCount() (int, error) {
	return cuda.GetDeviceCount()
}

func (r *CUDARuntime) Malloc(byteSize int) (unsafe.Pointer, error) {
	return cuda.Malloc(byteSize)
}

func (r *CUDARuntime) Free(ptr unsafe.Pointer) error {
	return cuda.Free(ptr)
}

func (r *CUDARuntime) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	return cuda.Memcpy(dst, src, count, cudaMemcpyKind(kind))
}

func (r *CUDARuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, stream Stream) error {
	var cs *cuda.Stream
	if stream != nil {
		cs = stream.(*cudaStreamWrapper).inner
	}
	return cuda.MemcpyAsync(dst, src, count, cudaMemcpyKind(kind), cs)
}

func (r *CUDARuntime) MemcpyPeer(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, count int) error {
	return cuda.MemcpyPeer(dst, dstDevice, src, srcDevice, count)
}

func (r *CUDARuntime) CreateStream() (Stream, error) {
	s, err := cuda.CreateStream()
	if err != nil {
		return nil, err
	}
	return &cudaStreamWrapper{inner: s}, nil
}

// cudaStreamWrapper wraps *cuda.Stream to implement gpuapi.Stream.
type cudaStreamWrapper struct {
	inner *cuda.Stream
}

func (w *cudaStreamWrapper) Synchronize() error  { return w.inner.Synchronize() }
func (w *cudaStreamWrapper) Destroy() error       { return w.inner.Destroy() }
func (w *cudaStreamWrapper) Ptr() unsafe.Pointer  { return w.inner.Ptr() }

// cudaMemcpyKind converts gpuapi.MemcpyKind to cuda.MemcpyKind.
func cudaMemcpyKind(kind MemcpyKind) cuda.MemcpyKind {
	switch kind {
	case MemcpyHostToDevice:
		return cuda.MemcpyHostToDevice
	case MemcpyDeviceToHost:
		return cuda.MemcpyDeviceToHost
	case MemcpyDeviceToDevice:
		return cuda.MemcpyDeviceToDevice
	default:
		return cuda.MemcpyHostToDevice
	}
}

// Compile-time interface assertion.
var _ Runtime = (*CUDARuntime)(nil)
