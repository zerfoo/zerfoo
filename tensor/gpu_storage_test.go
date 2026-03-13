package tensor

import (
	"testing"

	"github.com/zerfoo/zerfoo/device"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/gpuapi"
)

func skipIfNoCUDA(t *testing.T) {
	t.Helper()
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
}

func TestGPUStorageInterfaceCompliance(t *testing.T) {
	var _ Storage[float32] = (*GPUStorage[float32])(nil)
	var _ Storage[float64] = (*GPUStorage[float64])(nil)
	var _ Storage[int] = (*GPUStorage[int])(nil)
}

func TestGPUStorageRoundTrip(t *testing.T) {
	skipIfNoCUDA(t)
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
	skipIfNoCUDA(t)
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
	skipIfNoCUDA(t)
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
	skipIfNoCUDA(t)
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
	skipIfNoCUDA(t)
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
	skipIfNoCUDA(t)
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
	skipIfNoCUDA(t)
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

func TestGPUStorageTrySlice(t *testing.T) {
	skipIfNoCUDA(t)
	src := []float32{1.0, 2.0, 3.0}

	s, err := NewGPUStorageFromSlice(src)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	got, err := s.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice failed: %v", err)
	}

	if len(got) != len(src) {
		t.Fatalf("TrySlice returned %d elements, want %d", len(got), len(src))
	}

	for i := range src {
		if src[i] != got[i] {
			t.Errorf("TrySlice()[%d] = %f, want %f", i, got[i], src[i])
		}
	}
}

func TestGPUStorageTrySliceEmpty(t *testing.T) {
	skipIfNoCUDA(t)
	s, err := NewGPUStorage[float32](0)
	if err != nil {
		t.Fatalf("NewGPUStorage(0) failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	got, err := s.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice failed: %v", err)
	}

	if len(got) != 0 {
		t.Errorf("TrySlice() for empty storage returned %d elements", len(got))
	}
}

func TestGPUStorageTrySet(t *testing.T) {
	skipIfNoCUDA(t)
	s, err := NewGPUStorageFromSlice([]float32{1.0, 2.0})
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	newData := []float32{10.0, 20.0, 30.0}
	if err := s.TrySet(newData); err != nil {
		t.Fatalf("TrySet failed: %v", err)
	}

	if s.Len() != 3 {
		t.Errorf("after TrySet, Len() = %d, want 3", s.Len())
	}

	got, err := s.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice after TrySet failed: %v", err)
	}

	for i := range newData {
		if newData[i] != got[i] {
			t.Errorf("after TrySet, TrySlice()[%d] = %f, want %f", i, got[i], newData[i])
		}
	}
}

func TestGPUStorageTrySetSameLength(t *testing.T) {
	skipIfNoCUDA(t)
	s, err := NewGPUStorageFromSlice([]float32{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	newData := []float32{4.0, 5.0, 6.0}
	if err := s.TrySet(newData); err != nil {
		t.Fatalf("TrySet same length failed: %v", err)
	}

	got, err := s.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice failed: %v", err)
	}

	for i := range newData {
		if newData[i] != got[i] {
			t.Errorf("TrySlice()[%d] = %f, want %f", i, got[i], newData[i])
		}
	}
}

func TestManagedGPUStorageRoundTrip(t *testing.T) {
	skipIfNoCUDA(t)
	pool := newTestPool(t)

	s, err := NewManagedGPUStorage[float32](pool, 5)
	if err != nil {
		t.Fatalf("NewManagedGPUStorage failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	if !s.Managed() {
		t.Error("Managed() = false, want true")
	}

	if s.Len() != 5 {
		t.Errorf("Len() = %d, want 5", s.Len())
	}

	// Write data directly via TrySet (no Memcpy for managed)
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	if err := s.TrySet(data); err != nil {
		t.Fatalf("TrySet failed: %v", err)
	}

	// Read data back via TrySlice (no Memcpy for managed)
	got, err := s.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice failed: %v", err)
	}

	for i := range data {
		if data[i] != got[i] {
			t.Errorf("TrySlice()[%d] = %f, want %f", i, got[i], data[i])
		}
	}
}

func TestManagedGPUStorageTrySetResize(t *testing.T) {
	skipIfNoCUDA(t)
	pool := newTestPool(t)

	s, err := NewManagedGPUStorage[float32](pool, 2)
	if err != nil {
		t.Fatalf("NewManagedGPUStorage failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	// Resize from 2 to 4 elements
	data := []float32{10.0, 20.0, 30.0, 40.0}
	if err := s.TrySet(data); err != nil {
		t.Fatalf("TrySet resize failed: %v", err)
	}

	if s.Len() != 4 {
		t.Errorf("after TrySet resize, Len() = %d, want 4", s.Len())
	}

	got, err := s.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice failed: %v", err)
	}

	for i := range data {
		if data[i] != got[i] {
			t.Errorf("TrySlice()[%d] = %f, want %f", i, got[i], data[i])
		}
	}
}

func TestManagedGPUStorageFree(t *testing.T) {
	skipIfNoCUDA(t)
	pool := newTestPool(t)

	s, err := NewManagedGPUStorage[float32](pool, 4)
	if err != nil {
		t.Fatalf("NewManagedGPUStorage failed: %v", err)
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

func TestManagedGPUStorageNotManagedByDefault(t *testing.T) {
	skipIfNoCUDA(t)
	s, err := NewGPUStorage[float32](4)
	if err != nil {
		t.Fatalf("NewGPUStorage failed: %v", err)
	}

	defer func() { _ = s.Free() }()

	if s.Managed() {
		t.Error("regular GPUStorage should not be managed")
	}
}

func TestGPUStorageSubSlice(t *testing.T) {
	skipIfNoCUDA(t)

	src := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	parent, err := NewGPUStorageFromSlice(src)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice: %v", err)
	}
	defer func() { _ = parent.Free() }()

	tests := []struct {
		name   string
		offset int
		length int
		want   []float32
	}{
		{"first half", 0, 4, []float32{10, 20, 30, 40}},
		{"second half", 4, 4, []float32{50, 60, 70, 80}},
		{"middle", 2, 3, []float32{30, 40, 50}},
		{"single element", 5, 1, []float32{60}},
		{"full range", 0, 8, src},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sub := parent.SubSlice(tt.offset, tt.length)

			if sub.Len() != tt.length {
				t.Fatalf("Len() = %d, want %d", sub.Len(), tt.length)
			}

			if !sub.view {
				t.Error("SubSlice should create a non-owning view")
			}

			got, err := sub.TrySlice()
			if err != nil {
				t.Fatalf("TrySlice: %v", err)
			}

			if len(got) != len(tt.want) {
				t.Fatalf("TrySlice returned %d elements, want %d", len(got), len(tt.want))
			}

			for i, w := range tt.want {
				if got[i] != w {
					t.Errorf("TrySlice()[%d] = %f, want %f", i, got[i], w)
				}
			}
		})
	}
}

func TestGPUStorageSubSliceNoD2HForView(t *testing.T) {
	skipIfNoCUDA(t)

	// Verify that SubSlice itself does not trigger any data transfer.
	// It should only perform pointer arithmetic on the device pointer.
	src := []float32{1, 2, 3, 4, 5, 6}
	parent, err := NewGPUStorageFromSlice(src)
	if err != nil {
		t.Fatalf("NewGPUStorageFromSlice: %v", err)
	}
	defer func() { _ = parent.Free() }()

	sub := parent.SubSlice(2, 3)

	// The view's device pointer should be offset from the parent's.
	if sub.Ptr() == parent.Ptr() {
		t.Error("SubSlice(2, 3) should have an offset device pointer")
	}
	if sub.Ptr() == nil {
		t.Error("SubSlice device pointer should not be nil")
	}
	if sub.DeviceID() != parent.DeviceID() {
		t.Errorf("DeviceID = %d, want %d", sub.DeviceID(), parent.DeviceID())
	}

	// Verify data is correct when we do read it.
	got, err := sub.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice: %v", err)
	}
	want := []float32{3, 4, 5}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("got[%d] = %f, want %f", i, got[i], w)
		}
	}
}

func TestGPUStorageSubSliceManagedPreservesFlag(t *testing.T) {
	skipIfNoCUDA(t)
	pool := newTestPool(t)

	parent, err := NewManagedGPUStorage[float32](pool, 6)
	if err != nil {
		t.Fatalf("NewManagedGPUStorage: %v", err)
	}
	defer func() { _ = parent.Free() }()

	if err := parent.TrySet([]float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("TrySet: %v", err)
	}

	sub := parent.SubSlice(1, 3)

	if !sub.managed {
		t.Error("SubSlice of managed storage should preserve managed flag")
	}

	got, err := sub.TrySlice()
	if err != nil {
		t.Fatalf("TrySlice: %v", err)
	}
	want := []float32{2, 3, 4}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("got[%d] = %f, want %f", i, got[i], w)
		}
	}
}

// newTestPool returns a CUDAMemPool for testing.
func newTestPool(t *testing.T) *gpuapi.CUDAMemPool {
	t.Helper()
	return gpuapi.NewCUDAMemPool()
}
