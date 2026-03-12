package cuda

import (
	"testing"
	"unsafe"
)

// TestRuntimeParityAPI verifies that the runtime exports have the
// expected function signatures and that all functions fail gracefully
// when CUDA is not available.

// assignFunc is a generic helper that forces a compile-time type check
// on a function value without triggering QF1011.
func assignFunc[T any](fn T) T { return fn }

func TestRuntimeParitySignatures(t *testing.T) {
	_ = assignFunc[func(int) (unsafe.Pointer, error)](Malloc)
	_ = assignFunc[func(unsafe.Pointer) error](Free)
	_ = assignFunc[func(unsafe.Pointer, unsafe.Pointer, int, MemcpyKind) error](Memcpy)
	_ = assignFunc[func(unsafe.Pointer, unsafe.Pointer, int, MemcpyKind, *Stream) error](MemcpyAsync)
	_ = assignFunc[func() (int, error)](GetDeviceCount)
	_ = assignFunc[func(int) error](SetDevice)
	_ = assignFunc[func() (*Stream, error)](CreateStream)
	_ = assignFunc[func(int) (unsafe.Pointer, error)](MallocManaged)
	_ = assignFunc[func(unsafe.Pointer, int, unsafe.Pointer, int, int) error](MemcpyPeer)
	_ = assignFunc[func(int) (int, int, error)](DeviceComputeCapability)

	var s Stream
	_ = assignFunc[func() error](s.Synchronize)
	_ = assignFunc[func() error](s.Destroy)
	_ = assignFunc[func() unsafe.Pointer](s.Ptr)
}

func TestRuntimeParityMemcpyKindConstants(t *testing.T) {
	if MemcpyHostToDevice != 1 {
		t.Errorf("MemcpyHostToDevice = %d, want 1", MemcpyHostToDevice)
	}
	if MemcpyDeviceToHost != 2 {
		t.Errorf("MemcpyDeviceToHost = %d, want 2", MemcpyDeviceToHost)
	}
	if MemcpyDeviceToDevice != 3 {
		t.Errorf("MemcpyDeviceToDevice = %d, want 3", MemcpyDeviceToDevice)
	}
}

func TestRuntimeParityGracefulWithoutCUDA(t *testing.T) {
	if Available() {
		t.Skip("CUDA available, skipping graceful-failure tests")
	}

	_, err := Malloc(1024)
	if err == nil {
		t.Error("Malloc should fail without CUDA")
	}

	err = Free(nil)
	if err == nil {
		t.Error("Free should fail without CUDA")
	}

	_, err = GetDeviceCount()
	if err == nil {
		t.Error("GetDeviceCount should fail without CUDA")
	}

	err = SetDevice(0)
	if err == nil {
		t.Error("SetDevice should fail without CUDA")
	}

	_, err = CreateStream()
	if err == nil {
		t.Error("CreateStream should fail without CUDA")
	}

	_, _, err = DeviceComputeCapability(0)
	if err == nil {
		t.Error("DeviceComputeCapability should fail without CUDA")
	}

	_, err = MallocManaged(1024)
	if err == nil {
		t.Error("MallocManaged should fail without CUDA")
	}
}
