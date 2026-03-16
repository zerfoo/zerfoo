package xblas

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

func TestQ4DotBlock(t *testing.T) {
	tests := []struct {
		name string
		// Setup: pack nibbles and define activation values.
		nibbles [32]int // 32 quantized int4 values (0-15, zero-point 8)
		scale   float32
		x       [32]float32 // 32 activation values
	}{
		{
			name:    "zero block",
			nibbles: [32]int{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8},
			scale:   1.0,
			x:       [32]float32{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:    "all same nibble 15",
			nibbles: [32]int{15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
			scale:   0.5,
			x:       [32]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:    "all same nibble 0",
			nibbles: [32]int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			scale:   0.5,
			x:       [32]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:    "max scale",
			nibbles: [32]int{15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0},
			scale:   100.0,
			x:       [32]float32{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		},
		{
			name:    "min scale",
			nibbles: [32]int{15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0, 15, 0},
			scale:   0.001,
			x:       [32]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:    "random pattern",
			nibbles: [32]int{3, 12, 7, 1, 14, 5, 9, 0, 15, 8, 2, 11, 6, 13, 4, 10, 3, 12, 7, 1, 14, 5, 9, 0, 15, 8, 2, 11, 6, 13, 4, 10},
			scale:   0.25,
			x:       [32]float32{0.5, -0.3, 1.2, -0.7, 0.1, 0.9, -1.1, 0.4, -0.2, 0.8, 0.6, -0.5, 1.0, -0.8, 0.3, 0.7, 0.5, -0.3, 1.2, -0.7, 0.1, 0.9, -1.1, 0.4, -0.2, 0.8, 0.6, -0.5, 1.0, -0.8, 0.3, 0.7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Pack nibbles: GGML split format (low nibbles=first half, high nibbles=second half).
			var packed [16]byte
			for p := range 16 {
				lo := tt.nibbles[p] & 0x0F
				hi := tt.nibbles[p+16] & 0x0F
				packed[p] = byte(lo | (hi << 4))
			}

			got := q4DotBlock(&packed[0], tt.scale, &tt.x[0], 32)

			// Reference: manual dequant + dot.
			var want float32
			for i := range 32 {
				dequantized := float32(tt.nibbles[i]-8) * tt.scale
				want += dequantized * tt.x[i]
			}

			// Use relative tolerance for large-scale values.
			tol := float32(1e-5)
			if math.Abs(float64(want)) > 1 {
				tol = float32(math.Abs(float64(want))) * 1e-6
			}
			if diff := float32(math.Abs(float64(got - want))); diff > tol {
				t.Errorf("got %f, want %f (diff=%f)", got, want, diff)
			}
		})
	}
}

func TestQ4DotBlock_MatchesDequantPath(t *testing.T) {
	// Compare q4DotBlock against dequantQ4Block + manual dot for real Q4 data.
	m, k := 1, 256
	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	aQ4 := tensor.QuantizeQ4(aF32)

	x := make([]float32, k)
	for i := range x {
		x[i] = float32(i%5-2) * 0.1
	}

	blocksPerRow := k / 32

	// Compute via q4DotBlock.
	var gotSum float32
	for bi := range blocksPerRow {
		scale := aQ4.BlockScaleF32(bi)
		gotSum += q4DotBlock(aQ4.BlockData(bi), scale, &x[bi*32], 32)
	}

	// Compute via dequantQ4Block + dot.
	var wantSum float32
	var buf [32]float32
	for bi := range blocksPerRow {
		scale := aQ4.BlockScaleF32(bi)
		dequantQ4Block(aQ4.BlockData(bi), scale, &buf)
		for p := range 32 {
			wantSum += buf[p] * x[bi*32+p]
		}
	}

	if diff := float32(math.Abs(float64(gotSum - wantSum))); diff > 1e-4 {
		t.Errorf("q4DotBlock sum=%f, dequant+dot sum=%f (diff=%f)", gotSum, wantSum, diff)
	}
}

func TestQ4DotBlock_ZeroScale(t *testing.T) {
	var packed [16]byte
	for i := range packed {
		packed[i] = 0xFF // all nibbles 15
	}
	var x [32]float32
	for i := range x {
		x[i] = 1.0
	}
	got := q4DotBlock(&packed[0], 0.0, &x[0], 32)
	if got != 0 {
		t.Errorf("zero scale: got %f, want 0", got)
	}
}

func BenchmarkQ4DotBlock(b *testing.B) {
	k := 4096
	aF32 := make([]float32, k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	aQ4 := tensor.QuantizeQ4(aF32)

	x := make([]float32, k)
	for i := range x {
		x[i] = float32(i%5-2) * 0.1
	}

	blocksPerRow := k / 32

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		var sum float32
		for bi := range blocksPerRow {
			sum += q4DotBlock(aQ4.BlockData(bi), aQ4.BlockScaleF32(bi), &x[bi*32], 32)
		}
		_ = sum
	}
}

func BenchmarkQ4DotBlock_VsDequant(b *testing.B) {
	k := 4096
	aF32 := make([]float32, k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	aQ4 := tensor.QuantizeQ4(aF32)

	x := make([]float32, k)
	for i := range x {
		x[i] = float32(i%5-2) * 0.1
	}

	blocksPerRow := k / 32

	b.Run("q4DotBlock", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			var sum float32
			for bi := range blocksPerRow {
				sum += q4DotBlock(aQ4.BlockData(bi), aQ4.BlockScaleF32(bi), &x[bi*32], 32)
			}
			_ = sum
		}
	})

	b.Run("dequant+dot", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			var sum float32
			var buf [32]float32
			for bi := range blocksPerRow {
				dequantQ4Block(aQ4.BlockData(bi), aQ4.BlockScaleF32(bi), &buf)
				for p := range 32 {
					sum += buf[p] * x[bi*32+p]
				}
			}
			_ = sum
		}
	})

	b.Run("dequant+sgemmAccRow", func(b *testing.B) {
		b.ReportAllocs()
		n := 1
		c := make([]float32, n)
		for range b.N {
			c[0] = 0
			var buf [32]float32
			for bi := range blocksPerRow {
				dequantQ4Block(aQ4.BlockData(bi), aQ4.BlockScaleF32(bi), &buf)
				for p := range 32 {
					if aVal := buf[p]; aVal != 0 {
						sgemmAccRow(unsafe.Pointer(&c[0]), unsafe.Pointer(&x[bi*32+p]), aVal, n)
					}
				}
			}
		}
	})
}
