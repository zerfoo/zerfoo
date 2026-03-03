package tensor

import (
	"testing"

	"github.com/zerfoo/zerfoo/device"
)

// TestStorageInterfaceCompliance verifies that CPUStorage satisfies the
// Storage interface at compile time.
func TestStorageInterfaceCompliance(t *testing.T) {
	// Compile-time assertion: CPUStorage[float32] must implement Storage[float32].
	var _ Storage[float32] = (*CPUStorage[float32])(nil)
	var _ Storage[float64] = (*CPUStorage[float64])(nil)
	var _ Storage[int] = (*CPUStorage[int])(nil)
}

func TestCPUStorageLen(t *testing.T) {
	tests := []struct {
		name string
		data []float32
		want int
	}{
		{name: "empty", data: []float32{}, want: 0},
		{name: "single", data: []float32{1.0}, want: 1},
		{name: "multiple", data: []float32{1.0, 2.0, 3.0}, want: 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewCPUStorage(tt.data)
			if got := s.Len(); got != tt.want {
				t.Errorf("Len() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestCPUStorageSliceIdentity(t *testing.T) {
	data := []float32{1.0, 2.0, 3.0}
	s := NewCPUStorage(data)
	got := s.Slice()

	// Slice() must return the same underlying array (zero copy).
	if &got[0] != &data[0] {
		t.Error("Slice() returned a copy; expected zero-copy identity")
	}
}

func TestCPUStorageSet(t *testing.T) {
	s := NewCPUStorage([]float32{1.0, 2.0})
	newData := []float32{10.0, 20.0, 30.0}
	s.Set(newData)

	got := s.Slice()
	if len(got) != 3 {
		t.Fatalf("Slice() len = %d after Set, want 3", len(got))
	}
	for i, v := range newData {
		if got[i] != v {
			t.Errorf("Slice()[%d] = %v, want %v", i, got[i], v)
		}
	}
}

func TestCPUStorageDeviceType(t *testing.T) {
	s := NewCPUStorage([]float32{1.0})
	if got := s.DeviceType(); got != device.CPU {
		t.Errorf("DeviceType() = %v, want %v", got, device.CPU)
	}
}
