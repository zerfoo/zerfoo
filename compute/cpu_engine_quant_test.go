package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestCPUEngine_MatMul_QuantizedStorage(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	m, k, n := 2, 32, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		stor   tensor.Storage[float32]
		maxErr float32
	}{
		{"Q4_0", tensor.QuantizeQ4(aF32), 0.15},
		{"Q8_0", tensor.QuantizeQ8(aF32), 0.02},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tensor.NewWithStorage([]int{m, k}, tt.stor)
			if err != nil {
				t.Fatal(err)
			}

			result, err := engine.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul failed: %v", err)
			}

			if result.Shape()[0] != m || result.Shape()[1] != n {
				t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
			}

			// Reference: dequantize then float32 GEMM.
			refA, _ := tensor.New[float32]([]int{m, k}, tt.stor.Slice())
			refResult, err := engine.MatMul(ctx, refA, b)
			if err != nil {
				t.Fatal(err)
			}

			got := result.Data()
			want := refResult.Data()
			for i := range got {
				diff := float32(math.Abs(float64(got[i] - want[i])))
				if diff > tt.maxErr {
					t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
				}
			}
		})
	}
}

func TestCPUEngine_MatMul_Q4KStorage(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// K must be a multiple of 256 for Q4_K super-blocks.
	m, k, n := 2, 256, 3

	aF32 := make([]float32, m*k)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}
	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	// Quantize aF32 to Q4_K using the quantizer.
	q4k := quantizeQ4K(aF32)
	a, err := tensor.NewWithStorage([]int{m, k}, q4k)
	if err != nil {
		t.Fatal(err)
	}

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul with Q4_K failed: %v", err)
	}

	if result.Shape()[0] != m || result.Shape()[1] != n {
		t.Errorf("shape = %v, want [%d %d]", result.Shape(), m, n)
	}

	// Reference: dequantize Q4_K then float32 GEMM.
	refA, _ := tensor.New[float32]([]int{m, k}, q4k.Slice())
	refResult, err := engine.MatMul(ctx, refA, b)
	if err != nil {
		t.Fatal(err)
	}

	got := result.Data()
	want := refResult.Data()
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > 0.3 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestCPUEngine_MatMul_Q4KStorage_BWeight_GEMV(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	// GEMV: m=1, K must be multiple of 256.
	k, n := 256, 4

	// A is [1, K] float32 activation.
	aData := make([]float32, k)
	for i := range aData {
		aData[i] = float32(i%17-8) * 0.05
	}
	a, err := tensor.New[float32]([]int{1, k}, aData)
	if err != nil {
		t.Fatal(err)
	}

	// B is Q4_K weight. The Q4KStorage holds data in [N, K] layout.
	// The tensor shape is [K, N] (virtual transpose for MatMul).
	// So MatMul(A[1,K], B[K,N]) = C[1,N], but internally B_q4k is [N,K].
	bF32 := make([]float32, n*k)
	for i := range bF32 {
		bF32[i] = float32(math.Sin(float64(i)*0.03)) * 1.5
	}
	q4kB := quantizeQ4K(bF32)
	b, err := tensor.NewWithStorage([]int{k, n}, q4kB)
	if err != nil {
		t.Fatal(err)
	}

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul with Q4_K B-weight failed: %v", err)
	}

	if result.Shape()[0] != 1 || result.Shape()[1] != n {
		t.Errorf("shape = %v, want [1 %d]", result.Shape(), n)
	}

	// Reference: manually compute y[j] = sum_i( dequant(B_q4k[j,i]) * x[i] )
	// where B_q4k is in [N, K] layout.
	dequantB := q4kB.Slice() // [N*K] flat in [N,K] order
	want := make([]float32, n)
	for j := range n {
		var sum float32
		for i := range k {
			sum += dequantB[j*k+i] * aData[i]
		}
		want[j] = sum
	}

	got := result.Data()
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		if diff > 0.5 {
			t.Errorf("index %d: got %v, want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

// quantizeQ4K quantizes float32 values into Q4_K format using a simple
// quantizer that matches the Q4_K super-block layout (256 values, 8 sub-blocks).
func quantizeQ4K(data []float32) *tensor.Q4KStorage {
	const superBlockSize = 256
	const blockBytes = 144
	const numSubBlocks = 8
	const subBlockSize = 32

	nBlocks := (len(data) + superBlockSize - 1) / superBlockSize
	raw := make([]byte, nBlocks*blockBytes)

	for bi := range nBlocks {
		off := bi * superBlockSize
		end := off + superBlockSize
		if end > len(data) {
			end = len(data)
		}
		var values [superBlockSize]float32
		copy(values[:], data[off:end])

		var subScales, subMins [numSubBlocks]float32
		for sb := range numSubBlocks {
			sOff := sb * subBlockSize
			minVal := values[sOff]
			maxVal := values[sOff]
			for j := 1; j < subBlockSize; j++ {
				v := values[sOff+j]
				if v < minVal {
					minVal = v
				}
				if v > maxVal {
					maxVal = v
				}
			}
			if minVal > 0 {
				minVal = 0
			}
			subScales[sb] = (maxVal - minVal) / 15.0
			subMins[sb] = -minVal
		}

		var maxScale, maxMin float32
		for sb := range numSubBlocks {
			if subScales[sb] > maxScale {
				maxScale = subScales[sb]
			}
			if subMins[sb] > maxMin {
				maxMin = subMins[sb]
			}
		}

		d := maxScale / 63.0
		dmin := maxMin / 63.0

		var scalesQ, minsQ [numSubBlocks]uint8
		for sb := range numSubBlocks {
			if d > 0 {
				scalesQ[sb] = uint8(math.Round(float64(subScales[sb] / d)))
				if scalesQ[sb] > 63 {
					scalesQ[sb] = 63
				}
			}
			if dmin > 0 {
				minsQ[sb] = uint8(math.Round(float64(subMins[sb] / dmin)))
				if minsQ[sb] > 63 {
					minsQ[sb] = 63
				}
			}
		}

		blk := raw[bi*blockBytes : (bi+1)*blockBytes]

		// fp16 d and dmin (approximate using float32->float16 truncation).
		dBits := float32ToFloat16Bits(d)
		dminBits := float32ToFloat16Bits(dmin)
		blk[0] = byte(dBits)
		blk[1] = byte(dBits >> 8)
		blk[2] = byte(dminBits)
		blk[3] = byte(dminBits >> 8)

		// Pack 6-bit scales and mins.
		for i := range 4 {
			blk[4+i] = (scalesQ[i] & 63) | ((scalesQ[4+i] >> 4) << 6)
			blk[8+i] = (minsQ[i] & 63) | ((minsQ[4+i] >> 4) << 6)
		}
		for i := range 4 {
			blk[12+i] = (scalesQ[4+i] & 0xF) | ((minsQ[4+i] & 0xF) << 4)
		}

		// Quantize values to 4-bit.
		for group := range 4 {
			sb0 := group * 2
			sb1 := group*2 + 1

			sc0 := float16BitsToFloat32(float32ToFloat16Bits(d)) * float32(scalesQ[sb0])
			mn0 := float16BitsToFloat32(float32ToFloat16Bits(dmin)) * float32(minsQ[sb0])
			sc1 := float16BitsToFloat32(float32ToFloat16Bits(d)) * float32(scalesQ[sb1])
			mn1 := float16BitsToFloat32(float32ToFloat16Bits(dmin)) * float32(minsQ[sb1])

			var invScale0, invScale1 float32
			if sc0 > 0 {
				invScale0 = 1.0 / sc0
			}
			if sc1 > 0 {
				invScale1 = 1.0 / sc1
			}

			baseOut := group * 64
			baseQ := group * 32
			for l := range 32 {
				v0 := values[baseOut+l]
				v1 := values[baseOut+l+32]
				q0 := clampInt(int(math.Round(float64((v0+mn0)*invScale0))), 0, 15)
				q1 := clampInt(int(math.Round(float64((v1+mn1)*invScale1))), 0, 15)
				blk[16+baseQ+l] = byte(q0) | (byte(q1) << 4)
			}
		}
	}

	q4k, err := tensor.NewQ4KStorageFromRaw(raw, len(data))
	if err != nil {
		panic(err)
	}
	return q4k
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 1
	exp := int((bits>>23)&0xFF) - 127
	frac := bits & 0x7FFFFF

	if exp < -24 {
		return uint16(sign << 15)
	}
	if exp < -14 {
		frac = (frac | 0x800000) >> uint((-14-exp)+13)
		return uint16(sign<<15) | uint16(frac)
	}
	if exp > 15 {
		return uint16(sign<<15) | 0x7C00
	}
	return uint16(sign<<15) | uint16((exp+15)<<10) | uint16(frac>>13)
}

func float16BitsToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1F)
	frac := uint32(h & 0x3FF)

	if exp == 0 {
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for frac&0x400 == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
		exp++
	} else if exp == 31 {
		return math.Float32frombits(sign<<31 | 0x7F800000 | frac<<13)
	}

	return math.Float32frombits(sign<<31 | (exp+127-15)<<23 | frac<<13)
}

func TestCPUEngine_MatMul_Q4Storage_Batched(t *testing.T) {
	engine := NewCPUEngine(numeric.Float32Ops{})
	ctx := context.Background()

	batch, m, k, n := 2, 1, 32, 4

	aF32 := make([]float32, batch*m*k)
	for i := range aF32 {
		aF32[i] = float32(i%9-4) * 0.05
	}
	q4 := tensor.QuantizeQ4(aF32)
	a, err := tensor.NewWithStorage([]int{batch, m, k}, q4)
	if err != nil {
		t.Fatal(err)
	}

	bData := make([]float32, k*n)
	for i := range bData {
		bData[i] = float32(i%5-2) * 0.1
	}
	b, err := tensor.New[float32]([]int{k, n}, bData)
	if err != nil {
		t.Fatal(err)
	}

	result, err := engine.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("Batched MatMul with Q4 failed: %v", err)
	}

	shape := result.Shape()
	if shape[0] != batch || shape[1] != m || shape[2] != n {
		t.Errorf("output shape = %v, want [%d %d %d]", shape, batch, m, n)
	}
}
