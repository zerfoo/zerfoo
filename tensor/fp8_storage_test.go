package tensor

import (
	"math"
	"testing"
)

func TestFP8E4M3Storage_RoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		maxRel float64 // max relative error allowed
	}{
		{
			name:   "uniform positive values",
			input:  []float32{1.0, 2.0, 3.0, 4.0},
			maxRel: 0.1,
		},
		{
			name:   "uniform negative values",
			input:  []float32{-1.0, -2.0, -3.0, -4.0},
			maxRel: 0.1,
		},
		{
			name:   "mixed sign same magnitude",
			input:  []float32{-4.0, -2.0, 0.0, 2.0, 4.0},
			maxRel: 0.1,
		},
		{
			name:   "powers of two",
			input:  []float32{1.0, 2.0, 4.0, 8.0, 16.0},
			maxRel: 0.1,
		},
		{
			name:   "zeros",
			input:  []float32{0.0, 0.0, 0.0},
			maxRel: 0.0,
		},
		{
			name:   "single element",
			input:  []float32{42.0},
			maxRel: 0.1,
		},
		{
			name:   "identical values",
			input:  []float32{5.0, 5.0, 5.0},
			maxRel: 0.1,
		},
		{
			name:   "large positive only",
			input:  []float32{100.0, 200.0, 300.0},
			maxRel: 0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewFP8E4M3Storage(tt.input)
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

func TestFP8E4M3Storage_Scale(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	s := NewFP8E4M3Storage(input)
	scale := s.Scale()
	// Scale should be absmax / 448 = 4.0 / 448
	expected := float32(4.0 / 448.0)
	if math.Abs(float64(scale-expected)) > 1e-6 {
		t.Errorf("Scale() = %g, want %g", scale, expected)
	}
}

func TestFP8E4M3Storage_DeviceType(t *testing.T) {
	s := NewFP8E4M3Storage([]float32{1.0})
	if dt := s.DeviceType(); dt != 0 { // device.CPU = 0
		t.Errorf("DeviceType() = %v, want CPU (0)", dt)
	}
}

func TestFP8E4M3Storage_Set(t *testing.T) {
	s := NewFP8E4M3Storage([]float32{1.0, 2.0})
	s.Set([]float32{4.0, 8.0, 16.0})
	if s.Len() != 3 {
		t.Fatalf("after Set, Len() = %d, want 3", s.Len())
	}
	out := s.Slice()
	for i, want := range []float32{4.0, 8.0, 16.0} {
		rel := math.Abs(float64(out[i]-want)) / math.Abs(float64(want))
		if rel > 0.1 {
			t.Errorf("[%d] got %g, want %g, rel error %g", i, out[i], want, rel)
		}
	}
}

func TestFP8E4M3Storage_Empty(t *testing.T) {
	s := NewFP8E4M3Storage(nil)
	if s.Len() != 0 {
		t.Errorf("Len() = %d, want 0", s.Len())
	}
	if out := s.Slice(); len(out) != 0 {
		t.Errorf("Slice() len = %d, want 0", len(out))
	}
}

func TestFP8E5M2Storage_RoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		input  []float32
		maxRel float64
	}{
		{
			name:   "uniform positive values",
			input:  []float32{1.0, 2.0, 4.0, 8.0},
			maxRel: 0.1,
		},
		{
			name:   "negative values",
			input:  []float32{-1.0, -2.0, -4.0, -8.0},
			maxRel: 0.1,
		},
		{
			name:   "mixed sign same magnitude",
			input:  []float32{-8.0, -4.0, 0.0, 4.0, 8.0},
			maxRel: 0.1,
		},
		{
			name:   "powers of two",
			input:  []float32{1.0, 2.0, 4.0, 8.0, 16.0},
			maxRel: 0.1,
		},
		{
			name:   "zeros",
			input:  []float32{0.0, 0.0, 0.0},
			maxRel: 0.0,
		},
		{
			name:   "single element",
			input:  []float32{42.0},
			maxRel: 0.1,
		},
		{
			name:   "identical values",
			input:  []float32{5.0, 5.0, 5.0},
			maxRel: 0.1,
		},
		{
			name:   "large positive only",
			input:  []float32{100.0, 200.0, 300.0},
			maxRel: 0.1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewFP8E5M2Storage(tt.input)
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

func TestFP8E5M2Storage_Scale(t *testing.T) {
	input := []float32{1.0, 2.0, 3.0, 4.0}
	s := NewFP8E5M2Storage(input)
	scale := s.Scale()
	expected := float32(4.0 / 57344.0)
	if math.Abs(float64(scale-expected)) > 1e-10 {
		t.Errorf("Scale() = %g, want %g", scale, expected)
	}
}

func TestFP8E5M2Storage_DeviceType(t *testing.T) {
	s := NewFP8E5M2Storage([]float32{1.0})
	if dt := s.DeviceType(); dt != 0 {
		t.Errorf("DeviceType() = %v, want CPU (0)", dt)
	}
}

func TestFP8E5M2Storage_Set(t *testing.T) {
	s := NewFP8E5M2Storage([]float32{1.0, 2.0})
	s.Set([]float32{10.0, 20.0, 30.0})
	if s.Len() != 3 {
		t.Fatalf("after Set, Len() = %d, want 3", s.Len())
	}
	out := s.Slice()
	for i, want := range []float32{10.0, 20.0, 30.0} {
		rel := math.Abs(float64(out[i]-want)) / math.Abs(float64(want))
		if rel > 0.1 {
			t.Errorf("[%d] got %g, want %g, rel error %g", i, out[i], want, rel)
		}
	}
}

func TestFP8E5M2Storage_Empty(t *testing.T) {
	s := NewFP8E5M2Storage(nil)
	if s.Len() != 0 {
		t.Errorf("Len() = %d, want 0", s.Len())
	}
	if out := s.Slice(); len(out) != 0 {
		t.Errorf("Slice() len = %d, want 0", len(out))
	}
}

func TestEncodeDecodeE5M2(t *testing.T) {
	tests := []struct {
		name  string
		input float32
	}{
		{"zero", 0.0},
		{"one", 1.0},
		{"negative one", -1.0},
		{"two", 2.0},
		{"half", 0.5},
		{"quarter", 0.25},
		{"large", 1024.0},
		{"negative large", -1024.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := encodeE5M2(tt.input)
			decoded := decodeE5M2(encoded)

			if tt.input == 0 {
				if decoded != 0 {
					t.Errorf("round-trip of %g: got %g", tt.input, decoded)
				}
				return
			}

			rel := math.Abs(float64(decoded-tt.input)) / math.Abs(float64(tt.input))
			if rel > 0.1 {
				t.Errorf("round-trip of %g: got %g, rel error %g", tt.input, decoded, rel)
			}
		})
	}
}
