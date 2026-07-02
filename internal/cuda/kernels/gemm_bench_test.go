//go:build cuda && cutlass

package kernels

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// toDeviceB allocates device memory and copies host data (benchmark helper).
func toDeviceB(b *testing.B, data []float32) unsafe.Pointer {
	b.Helper()

	byteSize := len(data) * int(unsafe.Sizeof(data[0]))
	devPtr, err := cuda.Malloc(byteSize)

	if err != nil {
		b.Fatalf("Malloc: %v", err)
	}

	if err := cuda.Memcpy(devPtr, unsafe.Pointer(&data[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		b.Fatalf("Memcpy H2D: %v", err)
	}

	return devPtr
}

// toDeviceBytesB allocates device memory and copies a byte slice.
func toDeviceBytesB(b *testing.B, data []byte) unsafe.Pointer {
	b.Helper()

	devPtr, err := cuda.Malloc(len(data))
	if err != nil {
		b.Fatalf("Malloc: %v", err)
	}

	if err := cuda.Memcpy(devPtr, unsafe.Pointer(&data[0]), len(data), cuda.MemcpyHostToDevice); err != nil {
		b.Fatalf("Memcpy H2D: %v", err)
	}

	return devPtr
}

func benchInt4GEMM(b *testing.B, M, K, N int) {
	groupSize := 32

	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}

	defer stream.Destroy()

	// Packed INT4 weights: M * K/2 bytes.
	packedA := make([]byte, M*K/2)
	for i := range packedA {
		packedA[i] = 0x53 // two INT4 values: 3 and 5
	}

	devA := toDeviceBytesB(b, packedA)
	defer cuda.Free(devA)

	// FP32 activations: K x N.
	activations := make([]float32, K*N)
	for i := range activations {
		activations[i] = 0.01
	}

	devB := toDeviceB(b, activations)
	defer cuda.Free(devB)

	// Output: M x N.
	outBytes := M * N * 4
	devOut, err := cuda.Malloc(outBytes)
	if err != nil {
		b.Fatalf("Malloc output: %v", err)
	}

	defer cuda.Free(devOut)

	// Scales: M * (K / groupSize).
	numGroups := K / groupSize
	scales := make([]float32, M*numGroups)
	for i := range scales {
		scales[i] = 0.1
	}

	devScales := toDeviceB(b, scales)
	defer cuda.Free(devScales)

	// Zeros: M * (K / groupSize) bytes.
	zeros := make([]byte, M*numGroups)
	devZeros := toDeviceBytesB(b, zeros)
	defer cuda.Free(devZeros)

	b.ResetTimer()

	for range b.N {
		if err := GemmInt4F32(devA, devB, devOut, devScales, devZeros, M, K, N, groupSize, stream.Ptr()); err != nil {
			b.Fatalf("GemmInt4F32: %v", err)
		}
	}

	if err := stream.Synchronize(); err != nil {
		b.Fatalf("Synchronize: %v", err)
	}
}

func benchInt8GEMM(b *testing.B, M, K, N int) {
	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}

	defer stream.Destroy()

	// INT8 weights: M * K bytes.
	weightsI8 := make([]byte, M*K)
	for i := range weightsI8 {
		weightsI8[i] = 42
	}

	devA := toDeviceBytesB(b, weightsI8)
	defer cuda.Free(devA)

	// FP32 activations: K x N.
	activations := make([]float32, K*N)
	for i := range activations {
		activations[i] = 0.01
	}

	devB := toDeviceB(b, activations)
	defer cuda.Free(devB)

	// Output: M x N.
	outBytes := M * N * 4
	devOut, err := cuda.Malloc(outBytes)
	if err != nil {
		b.Fatalf("Malloc output: %v", err)
	}

	defer cuda.Free(devOut)

	b.ResetTimer()

	for range b.N {
		if err := GemmInt8F32(devA, devB, devOut, M, K, N, stream.Ptr()); err != nil {
			b.Fatalf("GemmInt8F32: %v", err)
		}
	}

	if err := stream.Synchronize(); err != nil {
		b.Fatalf("Synchronize: %v", err)
	}
}

func BenchmarkGemmInt4_1024(b *testing.B) { benchInt4GEMM(b, 1024, 1024, 1024) }
func BenchmarkGemmInt4_2048(b *testing.B) { benchInt4GEMM(b, 2048, 2048, 2048) }
func BenchmarkGemmInt4_4096(b *testing.B) { benchInt4GEMM(b, 4096, 4096, 4096) }

func BenchmarkGemmInt8_1024(b *testing.B) { benchInt8GEMM(b, 1024, 1024, 1024) }
func BenchmarkGemmInt8_2048(b *testing.B) { benchInt8GEMM(b, 2048, 2048, 2048) }
func BenchmarkGemmInt8_4096(b *testing.B) { benchInt8GEMM(b, 4096, 4096, 4096) }
