package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cpuSgemv computes y = A*x on the CPU for reference.
func cpuSgemv(A []float32, x []float32, M, N int) []float32 {
	y := make([]float32, M)
	for i := range M {
		var sum float32
		for j := range N {
			sum += A[i*N+j] * x[j]
		}
		y[i] = sum
	}
	return y
}

// buildSgemvTestData constructs a known A[M x N] matrix and x[N] vector.
func buildSgemvTestData(M, N int) (A, x []float32) {
	A = make([]float32, M*N)
	x = make([]float32, N)
	for i := range N {
		x[i] = float32(i%17-8) * 0.05
	}
	for i := range M {
		for j := range N {
			A[i*N+j] = float32(math.Sin(float64(i*N+j)*0.03)) * 1.5
		}
	}
	return A, x
}

func TestSgemvM1_Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	M, N := 64, 256
	A, x := buildSgemvTestData(M, N)
	ref := cpuSgemv(A, x, M, N)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devA, err := cuda.Malloc(M * N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc A: %v", err)
	}
	defer func() { _ = cuda.Free(devA) }()

	devX, err := cuda.Malloc(N * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc x: %v", err)
	}
	defer func() { _ = cuda.Free(devX) }()

	devY, err := cuda.Malloc(M * 4)
	if err != nil {
		t.Fatalf("cuda.Malloc y: %v", err)
	}
	defer func() { _ = cuda.Free(devY) }()

	if err := cuda.Memcpy(devA, unsafe.Pointer(&A[0]), M*N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), N*4, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy x: %v", err)
	}

	if err := SgemvM1(devY, devA, devX, M, N, stream.Ptr()); err != nil {
		t.Fatalf("SgemvM1: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, M)
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, M*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy y: %v", err)
	}

	checkGemvRelError(t, got, ref, gemvReductionAbsTol, gemvReductionRelTol)
}

func TestSgemvM1_MultipleSizes(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	cases := []struct {
		name string
		M, N int
	}{
		{"small_32x64", 32, 64},
		{"medium_128x512", 128, 512},
		{"gemma3_1b_1536x1536", 1536, 1536},
		{"gemma3_1b_6144x1536", 6144, 1536},
		{"odd_N_127x255", 127, 255},
		{"large_4096x4096", 4096, 4096},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			A, x := buildSgemvTestData(tc.M, tc.N)
			ref := cpuSgemv(A, x, tc.M, tc.N)

			stream, err := cuda.CreateStream()
			if err != nil {
				t.Fatalf("CreateStream: %v", err)
			}
			defer func() { _ = stream.Destroy() }()

			devA, err := cuda.Malloc(tc.M * tc.N * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc A: %v", err)
			}
			defer func() { _ = cuda.Free(devA) }()

			devX, err := cuda.Malloc(tc.N * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc x: %v", err)
			}
			defer func() { _ = cuda.Free(devX) }()

			devY, err := cuda.Malloc(tc.M * 4)
			if err != nil {
				t.Fatalf("cuda.Malloc y: %v", err)
			}
			defer func() { _ = cuda.Free(devY) }()

			if err := cuda.Memcpy(devA, unsafe.Pointer(&A[0]), tc.M*tc.N*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy A: %v", err)
			}
			if err := cuda.Memcpy(devX, unsafe.Pointer(&x[0]), tc.N*4, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy x: %v", err)
			}

			if err := SgemvM1(devY, devA, devX, tc.M, tc.N, stream.Ptr()); err != nil {
				t.Fatalf("SgemvM1: %v", err)
			}

			if err := stream.Synchronize(); err != nil {
				t.Fatalf("Synchronize: %v", err)
			}

			got := make([]float32, tc.M)
			if err := cuda.Memcpy(unsafe.Pointer(&got[0]), devY, tc.M*4, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy y: %v", err)
			}

			checkGemvRelError(t, got, ref, gemvReductionAbsTol, gemvReductionRelTol)
		})
	}
}

func BenchmarkSgemvM1_4096(b *testing.B) {
	if !cuda.Available() {
		b.Skip("CUDA not available")
	}

	M, N := 4096, 4096
	A, x := buildSgemvTestData(M, N)

	stream, err := cuda.CreateStream()
	if err != nil {
		b.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	devA, _ := cuda.Malloc(M * N * 4)
	defer func() { _ = cuda.Free(devA) }()
	devX, _ := cuda.Malloc(N * 4)
	defer func() { _ = cuda.Free(devX) }()
	devY, _ := cuda.Malloc(M * 4)
	defer func() { _ = cuda.Free(devY) }()

	_ = cuda.Memcpy(devA, unsafe.Pointer(&A[0]), M*N*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devX, unsafe.Pointer(&x[0]), N*4, cuda.MemcpyHostToDevice)

	b.ResetTimer()
	for b.Loop() {
		_ = SgemvM1(devY, devA, devX, M, N, stream.Ptr())
	}
	_ = stream.Synchronize()

	elapsed := b.Elapsed()
	flops := 2.0 * float64(M) * float64(N) * float64(b.N)
	gflops := flops / elapsed.Seconds() / 1e9
	b.ReportMetric(gflops, "GFLOPS")
}
