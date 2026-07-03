package kernels

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// dequantizeQ4K dequantizes one Q4_K super-block (144 bytes) into 256 float32 values.
// Inlined here to avoid an import cycle with the tensor package.
func dequantizeQ4K(raw []byte, dst []float32) {
	d := float16BitsToFloat32(binary.LittleEndian.Uint16(raw[0:2]))
	dmin := float16BitsToFloat32(binary.LittleEndian.Uint16(raw[2:4]))

	sc := raw[4:16]
	var scales, mins [8]uint8
	for i := range 4 {
		scales[i] = sc[i] & 63
		mins[i] = sc[4+i] & 63
	}
	for i := range 4 {
		scales[4+i] = (sc[8+i] & 0xF) | ((sc[i] >> 6) << 4)
		mins[4+i] = (sc[8+i] >> 4) | ((sc[4+i] >> 6) << 4)
	}

	qdata := raw[16:]
	for group := range 4 {
		sb0 := group * 2
		sb1 := group*2 + 1
		sc0 := d * float32(scales[sb0])
		mn0 := dmin * float32(mins[sb0])
		sc1 := d * float32(scales[sb1])
		mn1 := dmin * float32(mins[sb1])

		baseOut := group * 64
		baseQ := group * 32
		for l := range 32 {
			q := qdata[baseQ+l]
			dst[baseOut+l] = sc0*float32(q&0xF) - mn0
			dst[baseOut+l+32] = sc1*float32(q>>4) - mn1
		}
	}
}

// buildQ4KTestData constructs M rows of Q4_K super-blocks and computes the
// reference GEMV output using dequantizeQ4K for the dequantization.
// K must be a multiple of 256.
func buildQ4KTestData(M, K int) (raw []byte, x []float32, ref []float32) {
	const superBlockSize = 256
	const blockBytes = 144

	blocksPerRow := K / superBlockSize
	raw = make([]byte, M*blocksPerRow*blockBytes)
	x = make([]float32, K)
	ref = make([]float32, M)

	for i := range x {
		x[i] = float32(i%17-8) * 0.05
	}

	for row := range M {
		rowValues := make([]float32, K)
		for i := range K {
			rowValues[i] = float32(math.Sin(float64(row*K+i)*0.03)) * 1.5
		}

		for bi := range blocksPerRow {
			blkValues := rowValues[bi*superBlockSize : (bi+1)*superBlockSize]
			blkRaw := buildQ4KBlockFromValues(blkValues)
			off := (row*blocksPerRow + bi) * blockBytes
			copy(raw[off:off+blockBytes], blkRaw)
		}

		dequant := make([]float32, K)
		for bi := range blocksPerRow {
			off := (row*blocksPerRow + bi) * blockBytes
			dequantizeQ4K(raw[off:off+blockBytes], dequant[bi*superBlockSize:(bi+1)*superBlockSize])
		}
		var sum float32
		for k := range K {
			sum += dequant[k] * x[k]
		}
		ref[row] = sum
	}

	return raw, x, ref
}

// buildQ4KBlockFromValues quantizes 256 float32 values into a Q4_K super-block.
func buildQ4KBlockFromValues(values []float32) []byte {
	const numSubBlocks = 8
	const subBlockSize = 32

	subScales := make([]float32, numSubBlocks)
	subMins := make([]float32, numSubBlocks)

	for sb := range numSubBlocks {
		off := sb * subBlockSize
		minVal := values[off]
		maxVal := values[off]
		for j := 1; j < subBlockSize; j++ {
			v := values[off+j]
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

	scalesQ := make([]uint8, numSubBlocks)
	minsQ := make([]uint8, numSubBlocks)
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

	raw := make([]byte, 144)

	binary.LittleEndian.PutUint16(raw[0:2], float32ToFloat16Bits(d))
	binary.LittleEndian.PutUint16(raw[2:4], float32ToFloat16Bits(dmin))

	for i := range 4 {
		raw[4+i] = (scalesQ[i] & 63) | ((scalesQ[4+i] >> 4) << 6)
		raw[8+i] = (minsQ[i] & 63) | ((minsQ[4+i] >> 4) << 6)
	}
	for i := range 4 {
		raw[12+i] = (scalesQ[4+i] & 0xF) | ((minsQ[4+i] & 0xF) << 4)
	}

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
			raw[16+baseQ+l] = byte(q0) | (byte(q1) << 4)
		}
	}

	return raw
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

func TestGemvQ4KF32_Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, K := 64, 256
	raw, x, ref := buildQ4KTestData(M, K)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devW, err := cuda.Malloc(len(raw))
	if err != nil {
		t.Fatalf("cuda.Malloc W: %v", err)
	}
	defer func() { _ = cuda.Free(devW) }()

	devX, err := cuda.Malloc(K * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc x: %v", err)
	}
	defer func() { _ = cuda.Free(devX) }()

	devY, err := cuda.Malloc(M * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc y: %v", err)
	}
	defer func() { _ = cuda.Free(devY) }()

	if err := cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy W: %v", err)
	}
	if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), K*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy x: %v", err)
	}

	if err := GemvQ4KF32(devW, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("GemvQ4KF32: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, M*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy y: %v", err)
	}

	maxRelErr := 0.0
	for i := range got {
		absRef := math.Abs(float64(ref[i]))
		diff := math.Abs(float64(got[i] - ref[i]))
		var relErr float64
		if absRef > 1e-6 {
			relErr = diff / absRef
		} else {
			relErr = diff
		}
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > gemvReductionRelTol {
			t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
			if t.Failed() {
				break
			}
		}
	}
	t.Logf("max relative error: %e", maxRelErr)
}

func TestGemvQ4KF32_LargerMatrix(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, K := 512, 1024
	raw, x, ref := buildQ4KTestData(M, K)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devW, err := cuda.Malloc(len(raw))
	if err != nil {
		t.Fatalf("cuda.Malloc W: %v", err)
	}
	defer func() { _ = cuda.Free(devW) }()

	devX, err := cuda.Malloc(K * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc x: %v", err)
	}
	defer func() { _ = cuda.Free(devX) }()

	devY, err := cuda.Malloc(M * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc y: %v", err)
	}
	defer func() { _ = cuda.Free(devY) }()

	if err := cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy W: %v", err)
	}
	if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), K*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy x: %v", err)
	}

	if err := GemvQ4KF32(devW, devX, devY, M, K, stream.Ptr()); err != nil {
		t.Fatalf("GemvQ4KF32: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, M*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy y: %v", err)
	}

	maxRelErr := 0.0
	for i := range got {
		absRef := math.Abs(float64(ref[i]))
		diff := math.Abs(float64(got[i] - ref[i]))
		var relErr float64
		if absRef > 1e-6 {
			relErr = diff / absRef
		} else {
			relErr = diff
		}
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		if relErr > gemvReductionRelTol {
			t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
			if t.Failed() {
				break
			}
		}
	}
	t.Logf("max relative error: %e", maxRelErr)
}

func TestGemvQ4KF32_MultipleSizes(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cases := []struct {
		name string
		M, K int
	}{
		{"small_32x256", 32, 256},
		{"medium_64x512", 64, 512},
		{"square_256x256", 256, 256},
		{"wide_128x1024", 128, 1024},
		{"tall_1024x256", 1024, 256},
		{"large_512x2048", 512, 2048},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, x, ref := buildQ4KTestData(tc.M, tc.K)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devW, err := cuda.Malloc(len(raw))
			if err != nil {
				t.Fatalf("cuda.Malloc W: %v", err)
			}
			defer func() { _ = cuda.Free(devW) }()

			devX, err := cuda.Malloc(tc.K * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy W: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), tc.K*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := GemvQ4KF32(devW, devX, devY, tc.M, tc.K, stream.Ptr()); err != nil {
				t.Fatalf("GemvQ4KF32: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			got := make([]float32, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, tc.M*4, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			maxRelErr := 0.0
			for i := range got {
				absRef := math.Abs(float64(ref[i]))
				diff := math.Abs(float64(got[i] - ref[i]))
				var relErr float64
				if absRef > 1e-6 {
					relErr = diff / absRef
				} else {
					relErr = diff
				}
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
				if relErr > gemvReductionRelTol {
					t.Errorf("y[%d] = %f, want %f (rel err %e)", i, got[i], ref[i], relErr)
					if t.Failed() {
						break
					}
				}
			}
			t.Logf("max relative error: %e", maxRelErr)
		})
	}
}

func BenchmarkGemvQ4KF32_4096(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}

	M, K := 4096, 4096
	raw, x, _ := buildQ4KTestData(M, K)

	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devW, _ := cuda.Malloc(len(raw))
	defer func() { _ = cuda.Free(devW) }()
	devX, _ := cuda.Malloc(K * 4)
	defer func() { _ = cuda.Free(devX) }()
	devY, _ := cuda.Malloc(M * 4)
	defer func() { _ = cuda.Free(devY) }()

	_ = cuda.Memcpy(devW, unsafe.Pointer(&raw[0]), len(raw), cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devX, unsafe.Pointer(&x[0]), K*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = GemvQ4KF32(devW, devX, devY, M, K, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(K) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}
