package tensor

import (
	"math"
	"testing"
	"unsafe"
)

func TestFloat16Storage_CreateFromF32(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		maxRel float64
	}{
		{
			name:   "positive values",
			input:  []float32{1.0, 2.0, 3.0, 4.0},
			maxRel: 0.001,
		},
		{
			name:   "negative values",
			input:  []float32{-1.0, -2.0, -3.0, -4.0},
			maxRel: 0.001,
		},
		{
			name:   "mixed sign",
			input:  []float32{-4.0, -2.0, 0.0, 2.0, 4.0},
			maxRel: 0.001,
		},
		{
			name:   "powers of two",
			input:  []float32{0.5, 1.0, 2.0, 4.0, 8.0, 16.0},
			maxRel: 0.0,
		},
		{
			name:   "zeros",
			input:  []float32{0.0, 0.0, 0.0},
			maxRel: 0.0,
		},
		{
			name:   "single element",
			input:  []float32{42.0},
			maxRel: 0.001,
		},
		{
			name:   "empty",
			input:  nil,
			maxRel: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewFloat16StorageFromF32(tt.input)
			if s.Len() != len(tt.input) {
				t.Fatalf("Len() = %d, want %d", s.Len(), len(tt.input))
			}

			out := s.Slice()
			if len(out) != len(tt.input) {
				t.Fatalf("Slice() len = %d, want %d", len(out), len(tt.input))
			}

			for i, want := range tt.input {
				got := out[i]
				if want == 0 {
					if got != 0 {
						t.Errorf("[%d] got %g, want 0", i, got)
					}
					continue
				}
				rel := math.Abs(float64(got-want)) / math.Abs(float64(want))
				if rel > tt.maxRel {
					t.Errorf("[%d] got %g, want %g, rel error %g > %g", i, got, want, rel, tt.maxRel)
				}
			}
		})
	}
}

func TestFloat16Storage_Len(t *testing.T) {
	tests := []struct {
		name string
		data []float32
		want int
	}{
		{"empty", nil, 0},
		{"one", []float32{1.0}, 1},
		{"five", []float32{1, 2, 3, 4, 5}, 5},
		{"hundred", make([]float32, 100), 100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewFloat16StorageFromF32(tt.data)
			if got := s.Len(); got != tt.want {
				t.Errorf("Len() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestFloat16Storage_SubSlice(t *testing.T) {
	src := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	parent := NewFloat16StorageFromF32(src)

	tests := []struct {
		name   string
		offset int
		length int
		want   []float32
	}{
		{"first half", 0, 4, []float32{1, 2, 3, 4}},
		{"second half", 4, 4, []float32{5, 6, 7, 8}},
		{"middle", 2, 3, []float32{3, 4, 5}},
		{"single element", 5, 1, []float32{6}},
		{"full range", 0, 8, src},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sub := parent.SubSlice(tt.offset, tt.length)

			if sub.Len() != tt.length {
				t.Fatalf("Len() = %d, want %d", sub.Len(), tt.length)
			}

			// Verify zero-copy: sub.data should share the same backing array as parent.data.
			parentStart := &parent.data[tt.offset*2]
			subStart := &sub.data[0]
			if parentStart != subStart {
				t.Error("SubSlice should create a zero-copy view (backing array differs)")
			}

			got := sub.Slice()
			if len(got) != len(tt.want) {
				t.Fatalf("Slice() len = %d, want %d", len(got), len(tt.want))
			}

			for i, w := range tt.want {
				if got[i] != w {
					t.Errorf("Slice()[%d] = %g, want %g", i, got[i], w)
				}
			}
		})
	}
}

func TestFloat16Storage_GPUPtr(t *testing.T) {
	// Use real heap-allocated objects so go vet does not flag unsafe.Pointer misuse.
	sentinel0 := new(byte)
	sentinel1 := new(byte)

	tests := []struct {
		name     string
		ptr      unsafe.Pointer
		byteSize int
		deviceID int
	}{
		{"nil pointer", nil, 0, 0},
		{"device 0", unsafe.Pointer(sentinel0), 1024, 0},
		{"device 1", unsafe.Pointer(sentinel1), 2048, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewFloat16StorageFromF32([]float32{1.0, 2.0})
			s.SetGPUPtr(tt.ptr, tt.byteSize, tt.deviceID)

			gotPtr, gotSize, gotDev := s.GPUPtr()
			if gotPtr != tt.ptr {
				t.Errorf("GPUPtr() ptr = %v, want %v", gotPtr, tt.ptr)
			}
			if gotSize != tt.byteSize {
				t.Errorf("GPUPtr() byteSize = %d, want %d", gotSize, tt.byteSize)
			}
			if gotDev != tt.deviceID {
				t.Errorf("GPUPtr() deviceID = %d, want %d", gotDev, tt.deviceID)
			}
		})
	}
}

func TestFloat16Storage_RawBytes(t *testing.T) {
	tests := []struct {
		name     string
		input    []float32
		wantSize int
	}{
		{"empty", nil, 0},
		{"one element", []float32{1.0}, 2},
		{"four elements", []float32{1, 2, 3, 4}, 8},
		{"ten elements", make([]float32, 10), 20},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewFloat16StorageFromF32(tt.input)
			raw := s.RawBytes()
			if len(raw) != tt.wantSize {
				t.Errorf("RawBytes() len = %d, want %d (2 * Len=%d)", len(raw), tt.wantSize, s.Len())
			}
		})
	}
}

func TestFloat16Storage_DeviceType(t *testing.T) {
	s := NewFloat16StorageFromF32([]float32{1.0})
	if dt := s.DeviceType(); dt != 0 { // device.CPU = 0
		t.Errorf("DeviceType() = %v, want CPU (0)", dt)
	}
}

func TestFloat16Storage_Set(t *testing.T) {
	s := NewFloat16StorageFromF32([]float32{1.0, 2.0})
	s.Set([]float32{4.0, 8.0, 16.0})
	if s.Len() != 3 {
		t.Fatalf("after Set, Len() = %d, want 3", s.Len())
	}
	out := s.Slice()
	for i, want := range []float32{4.0, 8.0, 16.0} {
		if out[i] != want {
			t.Errorf("[%d] got %g, want %g", i, out[i], want)
		}
	}
}

func TestFloat16Storage_SetGPUByteSize(t *testing.T) {
	s := NewFloat16StorageFromF32([]float32{1.0, 2.0})
	sentinel := new(byte)
	s.SetGPUPtr(unsafe.Pointer(sentinel), 4, 0)
	s.SetGPUByteSize(256)

	_, gotSize, _ := s.GPUPtr()
	if gotSize != 256 {
		t.Errorf("after SetGPUByteSize(256), GPUPtr() byteSize = %d, want 256", gotSize)
	}
}

func TestFloat16Storage_InterfaceCompliance(t *testing.T) {
	var _ Storage[float32] = (*Float16Storage)(nil)
}
