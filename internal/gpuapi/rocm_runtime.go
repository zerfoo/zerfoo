//nolint:dupl // ROCm mirrors CUDA API shape intentionally
package gpuapi

import (
	"unsafe"

	"github.com/zerfoo/ztensor/device"
	"github.com/zerfoo/zerfoo/internal/hip"
)

// ROCmRuntime implements the Runtime interface using the AMD HIP runtime API.
type ROCmRuntime struct{}

// NewROCmRuntime returns a new ROCm runtime adapter.
func NewROCmRuntime() *ROCmRuntime {
	return &ROCmRuntime{}
}

func (r *ROCmRuntime) DeviceType() device.Type { return device.ROCm }

func (r *ROCmRuntime) SetDevice(deviceID int) error {
	return hip.SetDevice(deviceID)
}

func (r *ROCmRuntime) GetDeviceCount() (int, error) {
	return hip.GetDeviceCount()
}

func (r *ROCmRuntime) Malloc(byteSize int) (unsafe.Pointer, error) {
	return hip.Malloc(byteSize)
}

func (r *ROCmRuntime) Free(ptr unsafe.Pointer) error {
	return hip.Free(ptr)
}

func (r *ROCmRuntime) Memcpy(dst, src unsafe.Pointer, count int, kind MemcpyKind) error {
	return hip.Memcpy(dst, src, count, hipMemcpyKind(kind))
}

func (r *ROCmRuntime) MemcpyAsync(dst, src unsafe.Pointer, count int, kind MemcpyKind, stream Stream) error {
	var hs *hip.Stream
	if stream != nil {
		hs = stream.(*hipStreamWrapper).inner
	}
	return hip.MemcpyAsync(dst, src, count, hipMemcpyKind(kind), hs)
}

func (r *ROCmRuntime) MemcpyPeer(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, count int) error {
	return hip.MemcpyPeer(dst, dstDevice, src, srcDevice, count)
}

func (r *ROCmRuntime) CreateStream() (Stream, error) {
	s, err := hip.CreateStream()
	if err != nil {
		return nil, err
	}
	return &hipStreamWrapper{inner: s}, nil
}

// hipStreamWrapper wraps *hip.Stream to implement gpuapi.Stream.
type hipStreamWrapper struct {
	inner *hip.Stream
}

func (w *hipStreamWrapper) Synchronize() error { return w.inner.Synchronize() }
func (w *hipStreamWrapper) Destroy() error     { return w.inner.Destroy() }
func (w *hipStreamWrapper) Ptr() unsafe.Pointer { return w.inner.Ptr() }

// hipMemcpyKind converts gpuapi.MemcpyKind to hip.MemcpyKind.
func hipMemcpyKind(kind MemcpyKind) hip.MemcpyKind {
	switch kind {
	case MemcpyHostToDevice:
		return hip.MemcpyHostToDevice
	case MemcpyDeviceToHost:
		return hip.MemcpyDeviceToHost
	case MemcpyDeviceToDevice:
		return hip.MemcpyDeviceToDevice
	default:
		return hip.MemcpyHostToDevice
	}
}

// Compile-time interface assertion.
var _ Runtime = (*ROCmRuntime)(nil)
