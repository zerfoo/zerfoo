//go:build cuda

package tensor

import (
	"testing"

	"github.com/zerfoo/zerfoo/device"
)

func TestGPUStorageInterfaceCompliance(t *testing.T) {
	var _ Storage[float32] = (*GPUStorage[float32])(nil)
	var _ Storage[float64] = (*GPUStorage[float64])(nil)
	var _ Storage[int] = (*GPUStorage[int])(nil)
}

func TestGPUStorageRoundTrip(t *testing.T) {
	src := []float32{1.0, 2.0, 3.0, 4.0, 5.0}

	s, err := NewGPUStorageFromSlice(src)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice failed: %v", err)
	}

	defer func() {
		if freeErr := s.Free(); freeErr != nil {
			t.Errorf("Free failed: %v", freeErr)
		}
	}()

	got := s.Slice()
	if len(got) != len(src) {
		t.Fatalf("Slice returned %d elements, want %d", len(got), len(src))
	}

	for i := range src {
		if src[i] != got[i] {
			t.Errorf("Slice()[%d] = %f, want %f", i, got[i], src[i])
		}
	}
}

func TestGPUStorageLen(t *testing.T) {
	s, err := NewGPUStorage[float32](10)
	if err != nil {
		t.Fatalf("NewGPUStorage failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	if s.Len() != 10 {
		t.Errorf("Len() = %d, want 10", s.Len())
	}
}

func TestGPUStorageSet(t *testing.T) {
	s, err := NewGPUStorageFromSlice([]float32{1.0, 2.0})
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	// Replace with different data
	newData := []float32{10.0, 20.0, 30.0}
	s.Set(newData)

	if s.Len() != 3 {
		t.Errorf("after Set, Len() = %d, want 3", s.Len())
	}

	got := s.Slice()
	for i := range newData {
		if newData[i] != got[i] {
			t.Errorf("after Set, Slice()[%d] = %f, want %f", i, got[i], newData[i])
		}
	}
}

func TestGPUStorageDeviceType(t *testing.T) {
	s, err := NewGPUStorage[float32](1)
	if err != nil {
		t.Fatalf("NewGPUStorage failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	if s.DeviceType() != device.CUDA {
		t.Errorf("DeviceType() = %d, want device.CUDA (%d)", s.DeviceType(), device.CUDA)
	}
}

func TestGPUStoragePtr(t *testing.T) {
	s, err := NewGPUStorage[float32](4)
	if err != nil {
		t.Fatalf("NewGPUStorage failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	if s.Ptr() == nil {
		t.Error("Ptr() returned nil for allocated storage")
	}
}

func TestGPUStorageFree(t *testing.T) {
	s, err := NewGPUStorage[float32](4)
	if err != nil {
		t.Fatalf("NewGPUStorage failed: %v", err)
	}

	err = s.Free()
	if err != nil {
		t.Errorf("Free failed: %v", err)
	}

	if s.Ptr() != nil {
		t.Error("after Free, Ptr() should return nil")
	}

	if s.Len() != 0 {
		t.Errorf("after Free, Len() = %d, want 0", s.Len())
	}

	// Double free should be safe
	err = s.Free()
	if err != nil {
		t.Errorf("double Free should not error, got: %v", err)
	}
}

func TestGPUStorageEmptySlice(t *testing.T) {
	s, err := NewGPUStorage[float32](0)
	if err != nil {
		t.Fatalf("NewGPUStorage(0) failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	got := s.Slice()
	if len(got) != 0 {
		t.Errorf("Slice() for empty storage returned %d elements", len(got))
	}
}
