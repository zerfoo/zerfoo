package tensor

import (
	"math"
	"testing"

	"github.com/zerfoo/float16"
)

func TestBFloat16Storage_RoundTrip(t *testing.T) {
	tests := []struct {
		name string
		src  []float32
	}{
		{"zeros", []float32{0, 0, 0, 0}},
		{"positive", []float32{1.0, 2.0, 3.0, 4.0}},
		{"negative", []float32{-1.0, -0.5, -0.25, -0.125}},
		{"mixed", []float32{1.5, -2.5, 0.0, 100.0}},
		{"small", []float32{0.001, 0.002, 0.003, 0.004}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewBFloat16Storage(tt.src)
			if s.Len() != len(tt.src) {
				t.Fatalf("Len() = %d, want %d", s.Len(), len(tt.src))
			}

			got := s.Slice()
			if len(got) != len(tt.src) {
				t.Fatalf("Slice() len = %d, want %d", len(got), len(tt.src))
			}

			for i, want := range tt.src {
				// BF16 has limited precision; compare with BF16 round-trip.
				expected := float16.BFloat16FromFloat32(want).ToFloat32()
				if got[i] != expected {
					t.Errorf("Slice()[%d] = %v, want %v", i, got[i], expected)
				}
			}
		})
	}
}

func TestBFloat16Storage_RawBytes(t *testing.T) {
	src := []float32{1.0, 2.0, 3.0, 4.0}
	s := NewBFloat16Storage(src)

	raw := s.RawBytes()
	if len(raw) != len(src)*2 {
		t.Fatalf("RawBytes() len = %d, want %d", len(raw), len(src)*2)
	}
}

func TestBFloat16Storage_Set(t *testing.T) {
	s := NewBFloat16Storage([]float32{1.0, 2.0})
	s.Set([]float32{3.0, 4.0, 5.0})

	if s.Len() != 3 {
		t.Fatalf("after Set, Len() = %d, want 3", s.Len())
	}

	got := s.Slice()
	expected := []float32{3.0, 4.0, 5.0}
	for i, want := range expected {
		bf := float16.BFloat16FromFloat32(want).ToFloat32()
		if got[i] != bf {
			t.Errorf("Slice()[%d] = %v, want %v", i, got[i], bf)
		}
	}
}

func TestBFloat16Storage_GPUPtr(t *testing.T) {
	s := NewBFloat16Storage([]float32{1.0})

	ptr, size, dev := s.GPUPtr()
	if ptr != nil || size != 0 || dev != 0 {
		t.Fatalf("initial GPUPtr should be nil/0/0")
	}
}

func TestBFloat16Storage_FromRaw(t *testing.T) {
	// Encode manually then create from raw.
	src := []float32{1.0, -2.0}
	encoded := make([]uint16, len(src))
	for i, v := range src {
		encoded[i] = uint16(float16.BFloat16FromFloat32(v))
	}

	s := NewBFloat16StorageFromRaw(encoded)
	got := s.Slice()
	for i, want := range src {
		expected := float16.BFloat16FromFloat32(want).ToFloat32()
		if got[i] != expected {
			t.Errorf("Slice()[%d] = %v, want %v", i, got[i], expected)
		}
	}
}

func TestBFloat16Storage_Precision(t *testing.T) {
	// Verify max relative error vs float32 is bounded.
	src := make([]float32, 256)
	for i := range src {
		src[i] = float32(i-128) * 0.1
	}

	s := NewBFloat16Storage(src)
	got := s.Slice()

	var maxRelErr float64
	for i, want := range src {
		if want == 0 {
			continue
		}
		rel := math.Abs(float64(got[i]-want)) / math.Abs(float64(want))
		if rel > maxRelErr {
			maxRelErr = rel
		}
	}

	// BF16 has ~0.8% relative precision (7-bit mantissa).
	if maxRelErr > 1e-2 {
		t.Errorf("max relative error = %e, want < 1e-2", maxRelErr)
	}
}

func TestBFloat16Storage_Empty(t *testing.T) {
	s := NewBFloat16Storage(nil)
	if s.Len() != 0 {
		t.Fatalf("Len() = %d, want 0", s.Len())
	}

	got := s.Slice()
	if len(got) != 0 {
		t.Fatalf("Slice() len = %d, want 0", len(got))
	}

	raw := s.RawBytes()
	if raw != nil {
		t.Fatalf("RawBytes() should be nil for empty storage")
	}
}
