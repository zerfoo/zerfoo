package cublas

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// cpuMatMul computes C = A * B on CPU for reference.
// A is [m, k], B is [k, n], C is [m, n], all row-major.
func cpuMatMul(a, b []float32, m, n, k int) []float32 {
	c := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for p := 0; p < k; p++ {
				sum += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			c[i*n+j] = float32(sum)
		}
	}
	return c
}

// TestSgemmParityCPU verifies that the purego cuBLAS Sgemm path produces
// the same result as a CPU reference matrix multiply. Max rel error < 1e-5.
func TestSgemmParityCPU(t *testing.T) {
	if !Available() {
		t.Skip("cuBLAS not available (no GPU)")
	}

	m, n, k := 16, 8, 12
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%7-3) * 0.1
	}
	for i := range b {
		b[i] = float32(i%11-5) * 0.05
	}

	expected := cpuMatMul(a, b, m, n, k)

	aBytes := len(a) * 4
	bBytes := len(b) * 4
	cBytes := m * n * 4

	devA, err := cuda.Malloc(aBytes)
	if err != nil {
		t.Fatalf("Malloc A: %v", err)
	}
	defer func() { _ = cuda.Free(devA) }()

	devB, err := cuda.Malloc(bBytes)
	if err != nil {
		t.Fatalf("Malloc B: %v", err)
	}
	defer func() { _ = cuda.Free(devB) }()

	devC, err := cuda.Malloc(cBytes)
	if err != nil {
		t.Fatalf("Malloc C: %v", err)
	}
	defer func() { _ = cuda.Free(devC) }()

	if err := cuda.Memcpy(devA, unsafe.Pointer(&a[0]), aBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&b[0]), bBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer func() { _ = h.Destroy() }()

	if err := Sgemm(h, m, n, k, 1.0, devA, devB, 0.0, devC); err != nil {
		t.Fatalf("Sgemm: %v", err)
	}

	result := make([]float32, m*n)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devC, cBytes, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	mismatches := 0
	for i := range expected {
		rel := float64(0)
		if expected[i] != 0 {
			rel = math.Abs(float64(result[i]-expected[i])) / math.Abs(float64(expected[i]))
		} else {
			rel = math.Abs(float64(result[i]))
		}
		if rel > 1e-5 {
			if mismatches < 5 {
				t.Errorf("C[%d] = %f, want %f (rel err %e)", i, result[i], expected[i], rel)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, m*n)
	}
}

// TestGemmExParityCPU verifies that the purego cuBLAS GemmEx path (FP32)
// produces the same result as a CPU reference. Max rel error < 1e-5.
func TestGemmExParityCPU(t *testing.T) {
	if !Available() {
		t.Skip("cuBLAS not available (no GPU)")
	}

	m, n, k := 8, 10, 6
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%9-4) * 0.08
	}
	for i := range b {
		b[i] = float32(i%13-6) * 0.04
	}

	expected := cpuMatMul(a, b, m, n, k)

	aBytes := len(a) * 4
	bBytes := len(b) * 4
	cBytes := m * n * 4

	devA, err := cuda.Malloc(aBytes)
	if err != nil {
		t.Fatalf("Malloc A: %v", err)
	}
	defer func() { _ = cuda.Free(devA) }()

	devB, err := cuda.Malloc(bBytes)
	if err != nil {
		t.Fatalf("Malloc B: %v", err)
	}
	defer func() { _ = cuda.Free(devB) }()

	devC, err := cuda.Malloc(cBytes)
	if err != nil {
		t.Fatalf("Malloc C: %v", err)
	}
	defer func() { _ = cuda.Free(devC) }()

	if err := cuda.Memcpy(devA, unsafe.Pointer(&a[0]), aBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}
	if err := cuda.Memcpy(devB, unsafe.Pointer(&b[0]), bBytes, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer func() { _ = h.Destroy() }()

	if err := GemmEx(h, m, n, k, 1.0,
		devA, CudaR32F,
		devB, CudaR32F,
		0.0,
		devC, CudaR32F,
		CublasCompute32F,
	); err != nil {
		t.Fatalf("GemmEx: %v", err)
	}

	result := make([]float32, m*n)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devC, cBytes, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	mismatches := 0
	for i := range expected {
		rel := float64(0)
		if expected[i] != 0 {
			rel = math.Abs(float64(result[i]-expected[i])) / math.Abs(float64(expected[i]))
		} else {
			rel = math.Abs(float64(result[i]))
		}
		if rel > 1e-5 {
			if mismatches < 5 {
				t.Errorf("C[%d] = %f, want %f (rel err %e)", i, result[i], expected[i], rel)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, m*n)
	}
}

func TestGemmExCallable(t *testing.T) {
	tests := []struct {
		name        string
		aType       CudaDataType
		bType       CudaDataType
		cType       CudaDataType
		computeType CublasComputeType
	}{
		{
			name:        "FP32",
			aType:       CudaR32F,
			bType:       CudaR32F,
			cType:       CudaR32F,
			computeType: CublasCompute32F,
		},
		{
			name:        "BF16",
			aType:       CudaR16BF,
			bType:       CudaR16BF,
			cType:       CudaR16BF,
			computeType: CublasCompute32F,
		},
		{
			name:        "FP16",
			aType:       CudaR16F,
			bType:       CudaR16F,
			cType:       CudaR16F,
			computeType: CublasCompute32F,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// GemmEx should return an error since cuBLAS library is not available
			// in this test environment (no GPU). This verifies the wrapper compiles
			// and is callable with the correct signature.
			h := &Handle{}
			err := GemmEx(h, 2, 2, 2, 1.0,
				nil, tc.aType,
				nil, tc.bType,
				0.0,
				nil, tc.cType,
				tc.computeType,
			)
			if err == nil {
				t.Fatal("expected error from GemmEx without cuBLAS library, got nil")
			}
		})
	}
}

func TestCublasGemmDefaultValue(t *testing.T) {
	// CUBLAS_GEMM_DEFAULT is -1 in C, which is 0xFFFFFFFF as uint32.
	if cublasGemmDefault != 0xFFFFFFFF {
		t.Errorf("cublasGemmDefault = %#x, want 0xFFFFFFFF", cublasGemmDefault)
	}
}
