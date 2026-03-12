package kernels

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// quantizeQ4 performs Q4_0 quantization inline (avoids tensor import cycle).
// Returns packed bytes in GGUF interleaved format and dequantized reference.
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

		// Quantize and pack (GGML Q4_0 split format).
		var invScale float32
		if scale > 0 {
			invScale = 1.0 / scale
		}
		const halfBlock = blockSize / 2
		for j := 0; j < halfBlock; j++ {
			var v0, v1 float32
			if offset+j < n {
				v0 = src[offset+j]
			}
			if offset+j+halfBlock < n {
				v1 = src[offset+j+halfBlock]
			}

			q0 := clampQ4(int(math.Round(float64(v0 * invScale))))
			q1 := clampQ4(int(math.Round(float64(v1 * invScale))))

			packed[blkOff+2+j] = byte(q0+8) | (byte(q1+8) << 4)
		}

		// Dequantize for reference.
		f16Scale := float16BitsToFloat32(scaleBits)
		for j := 0; j < halfBlock; j++ {
			p := packed[blkOff+2+j]
			d0 := float32(int(p&0x0F)-8) * f16Scale
			d1 := float32(int(p>>4)-8) * f16Scale
			if offset+j < n {
				dequant[offset+j] = d0
			}
			if offset+j+halfBlock < n {
				dequant[offset+j+halfBlock] = d1
			}
		}
	}
	return packed, dequant
}

// repackQ4ForGPU converts GGUF interleaved Q4 bytes to the GPU separated layout:
// [all_scales: nBlocks*2 bytes] [pad to 16B] [all_data: nBlocks*16 bytes]
func repackQ4ForGPU(packed []byte, nBlocks int) (gpuBytes []byte, dataOffset int) {
	scaleBytes := nBlocks * 2
	paddedScaleBytes := (scaleBytes + 15) &^ 15
	dataBytes := nBlocks * 16
	gpuBytes = make([]byte, paddedScaleBytes+dataBytes)
	dataOffset = paddedScaleBytes

	for i := 0; i < nBlocks; i++ {
		blkOff := i * 18
		// Copy scale (2 bytes)
		copy(gpuBytes[i*2:i*2+2], packed[blkOff:blkOff+2])
		// Copy packed data (16 bytes)
		copy(gpuBytes[paddedScaleBytes+i*16:paddedScaleBytes+i*16+16], packed[blkOff+2:blkOff+18])
	}
	return gpuBytes, dataOffset
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
	M, K, N := 2, 32, 4

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	aF32 := make([]float32, M*K)
	for i := range aF32 {
		aF32[i] = float32(i%7-3) * 0.1
	}

	aBytes, aDequant := quantizeQ4(aF32)
	nBlocks := M * (K / 32)
	gpuBytes, dataOffset := repackQ4ForGPU(aBytes, nBlocks)

	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.1
	}

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

	devA, err := cuda.Malloc(len(gpuBytes))
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

	if err := cuda.Memcpy(devA, unsafe.Pointer(&gpuBytes[0]), len(gpuBytes), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	if err := GemmQ4F32(devA, devB, devC, M, K, N, dataOffset, stream.Ptr()); err != nil {
		t.Fatalf("GemmQ4F32: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M*N)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devC, M*N*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C: %v", err)
	}

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
	nBlocks := M * (K / 32)
	gpuBytes, dataOffset := repackQ4ForGPU(aBytes, nBlocks)

	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%9-4) * 0.05
	}

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

	devA, err := cuda.Malloc(len(gpuBytes))
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

	if err := cuda.Memcpy(devA, unsafe.Pointer(&gpuBytes[0]), len(gpuBytes), cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	if err := GemmQ4F32(devA, devB, devC, M, K, N, dataOffset, stream.Ptr()); err != nil {
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
				break
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
	nBlocks := M * (K / 32)
	gpuBytes, dataOffset := repackQ4ForGPU(aBytes, nBlocks)

	bF32 := make([]float32, K*N)
	for i := range bF32 {
		bF32[i] = float32(i%5-2) * 0.01
	}

	devA, _ := cuda.Malloc(len(gpuBytes))
	defer func() { _ = cuda.Free(devA) }()
	devB, _ := cuda.Malloc(K * N * 4)
	defer func() { _ = cuda.Free(devB) }()
	devC, _ := cuda.Malloc(M * N * 4)
	defer func() { _ = cuda.Free(devC) }()

	_ = cuda.Memcpy(devA, unsafe.Pointer(&gpuBytes[0]), len(gpuBytes), cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devB, unsafe.Pointer(&bF32[0]), K*N*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = GemmQ4F32(devA, devB, devC, M, K, N, dataOffset, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(K) * float64(N) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}
