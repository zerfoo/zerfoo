package tensor

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/device"
)

func TestBFloat16StorageInterfaceCompliance(t *testing.T) {
	var _ Storage[float32] = (*BFloat16Storage)(nil)
}

func TestBFloat16StorageRoundTrip(t *testing.T) {
	tests := []struct {
		name      string
		input     []float32
		tolerance float32
	}{
		{
			name:      "positive values",
			input:     []float32{1.0, 2.0, 3.0, 4.0},
			tolerance: 0.02,
		},
		{
			name:      "negative values",
			input:     []float32{-1.0, -0.5, -0.25, -3.14},
			tolerance: 0.02,
		},
		{
			name:      "mixed",
			input:     []float32{0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0},
			tolerance: 1.0, // BF16 has ~7-bit mantissa, large values lose precision
		},
		{
			name:      "small values",
			input:     []float32{0.001, 0.002, 0.003},
			tolerance: 0.001,
		},
		{
			name:      "single element",
			input:     []float32{42.0},
			tolerance: 0.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewBFloat16StorageFromFloat32(tt.input)

			if got := s.Len(); got != len(tt.input) {
				t.Fatalf("Len() = %d, want %d", got, len(tt.input))
			}

			got := s.Slice()
			if len(got) != len(tt.input) {
				t.Fatalf("Slice() len = %d, want %d", len(got), len(tt.input))
			}

			for i, want := range tt.input {
				diff := float32(math.Abs(float64(got[i] - want)))
				if diff > tt.tolerance {
					t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want, diff)
				}
			}
		})
	}
}

func TestBFloat16StorageFromRaw(t *testing.T) {
	// Create from float32, get raw bytes, then reconstruct from raw.
	input := []float32{1.0, -2.0, 3.5, 0.0}
	s1 := NewBFloat16StorageFromFloat32(input)
	slice1 := s1.Slice()

	s2, err := NewBFloat16StorageFromRaw(s1.raw, s1.Len())
	if err != nil {
		t.Fatalf("NewBFloat16StorageFromRaw: %v", err)
	}

	slice2 := s2.Slice()
	for i := range slice1 {
		if slice1[i] != slice2[i] {
			t.Errorf("index %d: raw round-trip mismatch: %v != %v", i, slice1[i], slice2[i])
		}
	}
}

func TestBFloat16StorageFromRawErrors(t *testing.T) {
	tests := []struct {
		name        string
		raw         []byte
		numElements int
	}{
		{name: "zero elements", raw: []byte{0, 0}, numElements: 0},
		{name: "negative elements", raw: []byte{0, 0}, numElements: -1},
		{name: "too short", raw: []byte{0, 0}, numElements: 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewBFloat16StorageFromRaw(tt.raw, tt.numElements)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestBFloat16StorageSet(t *testing.T) {
	s := NewBFloat16StorageFromFloat32([]float32{1.0, 2.0})
	s.Set([]float32{10.0, 20.0, 30.0})

	if got := s.Len(); got != 3 {
		t.Fatalf("Len() = %d after Set, want 3", got)
	}

	got := s.Slice()
	want := []float32{10.0, 20.0, 30.0}
	for i, w := range want {
		diff := float32(math.Abs(float64(got[i] - w)))
		if diff > 0.5 {
			t.Errorf("index %d: got %v, want %v", i, got[i], w)
		}
	}
}

func TestBFloat16StorageSliceCached(t *testing.T) {
	s := NewBFloat16StorageFromFloat32([]float32{1.0, 2.0, 3.0})
	s1 := s.Slice()
	s2 := s.Slice()
	if &s1[0] != &s2[0] {
		t.Error("Slice() should return cached slice on second call")
	}
}

func TestBFloat16StorageSetInvalidatesCache(t *testing.T) {
	s := NewBFloat16StorageFromFloat32([]float32{1.0, 2.0})
	_ = s.Slice() // populate cache
	s.Set([]float32{10.0, 20.0})
	got := s.Slice()
	if diff := float32(math.Abs(float64(got[0] - 10.0))); diff > 0.5 {
		t.Errorf("Set did not invalidate cache: got %v, want ~10.0", got[0])
	}
}

func TestBFloat16StorageDeviceType(t *testing.T) {
	s := NewBFloat16StorageFromFloat32([]float32{1.0})
	if got := s.DeviceType(); got != device.CPU {
		t.Errorf("DeviceType() = %v, want %v", got, device.CPU)
	}
}

func TestBFloat16StorageByteSize(t *testing.T) {
	s := NewBFloat16StorageFromFloat32([]float32{1.0, 2.0, 3.0})
	if got := s.ByteSize(); got != 6 {
		t.Errorf("ByteSize() = %d, want 6", got)
	}
}

func TestBFloat16StorageMemoryHalved(t *testing.T) {
	n := 1024
	f32Data := make([]float32, n)
	for i := range f32Data {
		f32Data[i] = float32(i)
	}

	bf16 := NewBFloat16StorageFromFloat32(f32Data)
	f32Bytes := n * 4
	bf16Bytes := bf16.ByteSize()

	if bf16Bytes != f32Bytes/2 {
		t.Errorf("BF16 byte size %d is not half of F32 byte size %d", bf16Bytes, f32Bytes)
	}
}
