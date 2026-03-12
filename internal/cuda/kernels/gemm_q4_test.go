package kernels

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// quantizeQ4 performs Q4_0 quantization inline (avoids tensor import cycle).
// Returns packed bytes (18 bytes per block of 32 values) and dequantized reference.
func quantizeQ4(src []float32) (packed []byte, dequant []float32) {
	const blockSize = 32
	n := len(src)
	nBlocks := (n + blockSize - 1) / blockSize
	packed = make([]byte, nBlocks*18)
	dequant = make([]float32, n)

	for bi := range nBlocks {
		offset := bi * blockSize

		// Find absmax.
		var absMax float32
		for j := range blockSize {
			idx := offset + j
			if idx < n {
				if av := float32(math.Abs(float64(src[idx]))); av > absMax {
					absMax = av
				}
			}
		}

		// Compute scale.
		var scale float32
		if absMax > 0 {
			scale = absMax / 7.0
		}

		// Convert scale to float16 bits (IEEE 754 half precision).
		scaleBits := float32ToFloat16Bits(scale)

		blkOff := bi * 18
		binary.LittleEndian.PutUint16(packed[blkOff:blkOff+2], scaleBits)

		// Quantize and pack.
		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		for j := 0; j < blockSize; j += 2 {
			var v0, v1 float32
			if offset+j < n {
				v0 = src[offset+j]
			}
			if offset+j+1 < n {
				v1 = src[offset+j+1]
			}

			q0 := clampQ4(int(math.Round(float64(v0 * invScale))))
			q1 := clampQ4(int(math.Round(float64(v1 * invScale))))

			packed[blkOff+2+j/2] = byte(q0+8) | (byte(q1+8) << 4)
		}

		// Dequantize for reference.
		f16Scale := float16BitsToFloat32(scaleBits)
		for j := 0; j < blockSize; j += 2 {
			p := packed[blkOff+2+j/2]
			d0 := float32(int(p&0x0F)-8) * f16Scale
			d1 := float32(int(p>>4)-8) * f16Scale
			if offset+j < n {
				dequant[offset+j] = d0
			}
			if offset+j+1 < n {
				dequant[offset+j+1] = d1
			}
		}
	}
	return packed, dequant
}

func clampQ4(v int) int {
	if v < -8 {
		return -8
	}
	if v > 7 {
		return 7
	}
	return v
}

// float32ToFloat16Bits converts a float32 to IEEE 754 half-precision bits.
func float32ToFloat16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := (b >> 16) & 0x8000
	exp := int((b>>23)&0xFF) - 127 + 15
	frac := b & 0x7FFFFF

	if exp <= 0 {
		return uint16(sign) // flush to zero
	}
	if exp >= 31 {
		return uint16(sign | 0x7C00) // infinity
	}
	return uint16(sign | uint32(exp)<<10 | (frac >> 13))
}

// float16BitsToFloat32 converts IEEE 754 half-precision bits to float32.
func float16BitsToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	frac := uint32(bits) & 0x3FF

	if exp == 0 {
		return 0
	}
	if exp == 31 {
		return float32(math.Inf(1))
	}

	f32Exp := exp - 15 + 127
	f32Bits := (sign << 31) | (f32Exp << 23) | (frac << 13)
	return math.Float32frombits(f32Bits)
}

func TestGemmQ4F32_Correctness(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	// Small matrix: M=2, K=32, N=4.
	// K=32 so each row of A is exactly 1 Q4 block.
	M, K, N := 2, 32, 4

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	// Create float32 source data for A.
	aF32 := make([]float32, M*K)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}

	// Quantize A to Q4.
	aBytes, aDequant := quantizeQ4(aF32)

	// Create B matrix.
	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.1
	}

	// Compute reference: dequant(A) * B on CPU.
	ref := make([]float32, M*N)
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				sum += aDequant[i*K+k] * bF32[k*N+j]
			}
			ref[i*N+j] = sum
		}
	}

	// Allocate device memory.
	devA, err := cuda.Malloc(len(aBytes))
	if err != nil {
		t.Fatalf("cuda.Malloc A: %v", err)
	}
	defer func() { _ = cuda.Free(devA) }()

	devB, err := cuda.Malloc(K * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc B: %v", err)
	}
	defer func() { _ = cuda.Free(devB) }()

	devC, err := cuda.Malloc(M * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc C: %v", err)
	}
	defer func() { _ = cuda.Free(devC) }()

	// Copy H2D.
	if err := cuda.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	// Run kernel.
	if err := GemmQ4F32(devA, devB, devC, M, K, N, stream.Ptr()); err != nil {
		t.Fatalf("GemmQ4F32: %v", err)
	}

	// Sync.
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	// Copy D2H.
	got := make([]float32, M*N)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devC, M*N*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C: %v", err)
	}

	// Compare with Q4 tolerance.
	const tol = 0.15
	for i := range got {
		if diff := math.Abs(float64(got[i] - ref[i])); diff > tol {
			t.Errorf("C[%d] = %f, want %f (diff %f > tol %f)", i, got[i], ref[i], diff, tol)
		}
	}
}

func TestGemmQ4F32_LargerMatrix(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}
	M, K, N := 64, 128, 64

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	aF32 := make([]float32, M*K)
	for i := range aF32 {
		aF32[i] = float32(i%11-5) * 0.05
	}
	aBytes, aDequant := quantizeQ4(aF32)

	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%9-4) * 0.05
	}

	// Reference.
	ref := make([]float32, M*N)
	for i := range M {
		for j := range N {
			var sum float32
			for k := range K {
				sum += aDequant[i*K+k] * bF32[k*N+j]
			}
			ref[i*N+j] = sum
		}
	}

	devA, err := cuda.Malloc(len(aBytes))
	if err != nil {
		t.Fatalf("cuda.Malloc A: %v", err)
	}
	defer func() { _ = cuda.Free(devA) }()

	devB, err := cuda.Malloc(K * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc B: %v", err)
	}
	defer func() { _ = cuda.Free(devB) }()

	devC, err := cuda.Malloc(M * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc C: %v", err)
	}
	defer func() { _ = cuda.Free(devC) }()

	if err := cuda.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	if err := GemmQ4F32(devA, devB, devC, M, K, N, stream.Ptr()); err != nil {
		t.Fatalf("GemmQ4F32: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M*N)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devC, M*N*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C: %v", err)
	}

	const tol = 0.2
	maxDiff := 0.0
	for i := range got {
		diff := math.Abs(float64(got[i] - ref[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > tol {
			t.Errorf("C[%d] = %f, want %f (diff %f)", i, got[i], ref[i], diff)
			if t.Failed() {
				break // Don't flood output.
			}
		}
	}
	t.Logf("max diff: %f", maxDiff)
}

func BenchmarkGemmQ4F32_1024(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}
	M, K, N := 1024, 1024, 1024

	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	aF32 := make([]float32, M*K)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.01
	}
	aBytes, _ := quantizeQ4(aF32)

	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.01
	}

	devA, _ := cuda.Malloc(len(aBytes))
	defer func() { _ = cuda.Free(devA) }()
	devB, _ := cuda.Malloc(K * N * 4)
	defer func() { _ = cuda.Free(devB) }()
	devC, _ := cuda.Malloc(M * N * 4)
	defer func() { _ = cuda.Free(devC) }()

	_ = cuda.Memcpy(devA, unsafe.Pointer(&aBytes[0]), len(aBytes), cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = GemmQ4F32(devA, devB, devC, M, K, N, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	// Q4 GEMM effective FLOPS: 2*M*K*N per iteration.
	flops := 2.0 * float64(M) * float64(K) * float64(N) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}
