package cublas

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestCreateAndDestroyHandle(t *testing.T) {
	if !Available() {
		t.Skip("cuBLAS not available")
	}
	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle failed: %v", err)
	}

	err = h.Destroy()
	if err != nil {
		t.Fatalf("Destroy failed: %v", err)
	}
}

func TestSgemm(t *testing.T) {
	if !Available() {
		t.Skip("cuBLAS not available")
	}
	// A = [[1, 2], [3, 4]]  (2x2)
	// B = [[5, 6], [7, 8]]  (2x2)
	// C = A * B = [[19, 22], [43, 50]]
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	expected := []float32{19, 22, 43, 50}

	m, n, k := 2, 2, 2
	elemSize := int(unsafe.Sizeof(float32(0)))
	byteSize := 4 * elemSize

	// Allocate device memory
	devA, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc A: %v", err)
	}

	defer func() { _ = cuda.Free(devA) }()

	devB, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc B: %v", err)
	}

	defer func() { _ = cuda.Free(devB) }()

	devC, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc C: %v", err)
	}

	defer func() { _ = cuda.Free(devC) }()

	// Copy to device
	if err := cuda.Memcpy(devA, unsafe.Pointer(&a[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A H2D: %v", err)
	}

	if err := cuda.Memcpy(devB, unsafe.Pointer(&b[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B H2D: %v", err)
	}

	// Create handle and run Sgemm
	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}

	defer func() { _ = h.Destroy() }()

	err = Sgemm(h, m, n, k, 1.0, devA, devB, 0.0, devC)
	if err != nil {
		t.Fatalf("Sgemm: %v", err)
	}

	// Copy result back
	result := make([]float32, 4)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devC, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C D2H: %v", err)
	}

	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("C[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestGemmEx(t *testing.T) {
	if !Available() {
		t.Skip("cuBLAS not available")
	}
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		m, n, k  int
		aType    CudaDataType
		bType    CudaDataType
		cType    CudaDataType
		expected []float32
		tol      float32
	}{
		{
			name:     "FP32_2x2",
			a:        []float32{1, 2, 3, 4},
			b:        []float32{5, 6, 7, 8},
			m:        2, n: 2, k: 2,
			aType:    CudaR32F,
			bType:    CudaR32F,
			cType:    CudaR32F,
			expected: []float32{19, 22, 43, 50},
			tol:      0,
		},
		{
			name:     "FP32_identity",
			a:        []float32{1, 0, 0, 1},
			b:        []float32{3, 7, 11, 13},
			m:        2, n: 2, k: 2,
			aType:    CudaR32F,
			bType:    CudaR32F,
			cType:    CudaR32F,
			expected: []float32{3, 7, 11, 13},
			tol:      0,
		},
	}

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}
	defer func() { _ = h.Destroy() }()

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			elemSize := 4 // float32
			aBytes := len(tc.a) * elemSize
			bBytes := len(tc.b) * elemSize
			cBytes := tc.m * tc.n * elemSize

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

			if err := cuda.Memcpy(devA, unsafe.Pointer(&tc.a[0]), aBytes, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy A: %v", err)
			}
			if err := cuda.Memcpy(devB, unsafe.Pointer(&tc.b[0]), bBytes, cuda.MemcpyHostToDevice); err != nil {
				t.Fatalf("Memcpy B: %v", err)
			}

			err = GemmEx(h, tc.m, tc.n, tc.k, 1.0,
				devA, tc.aType,
				devB, tc.bType,
				0.0,
				devC, tc.cType,
				CublasCompute32F,
			)
			if err != nil {
				t.Fatalf("GemmEx: %v", err)
			}

			result := make([]float32, tc.m*tc.n)
			if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devC, cBytes, cuda.MemcpyDeviceToHost); err != nil {
				t.Fatalf("Memcpy C D2H: %v", err)
			}

			for i, want := range tc.expected {
				diff := result[i] - want
				if diff < 0 {
					diff = -diff
				}
				if diff > tc.tol {
					t.Errorf("C[%d] = %f, want %f (tol %f)", i, result[i], want, tc.tol)
				}
			}
		})
	}
}

func TestSgemmNonSquare(t *testing.T) {
	if !Available() {
		t.Skip("cuBLAS not available")
	}
	// A = [[1, 2, 3]]    (1x3)
	// B = [[4], [5], [6]] (3x1)
	// C = A * B = [[32]]  (1x1)
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}

	m, n, k := 1, 1, 3
	elemSize := int(unsafe.Sizeof(float32(0)))

	devA, err := cuda.Malloc(3 * elemSize)
	if err != nil {
		t.Fatalf("Malloc A: %v", err)
	}

	defer func() { _ = cuda.Free(devA) }()

	devB, err := cuda.Malloc(3 * elemSize)
	if err != nil {
		t.Fatalf("Malloc B: %v", err)
	}

	defer func() { _ = cuda.Free(devB) }()

	devC, err := cuda.Malloc(1 * elemSize)
	if err != nil {
		t.Fatalf("Malloc C: %v", err)
	}

	defer func() { _ = cuda.Free(devC) }()

	if err := cuda.Memcpy(devA, unsafe.Pointer(&a[0]), 3*elemSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy A: %v", err)
	}

	if err := cuda.Memcpy(devB, unsafe.Pointer(&b[0]), 3*elemSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy B: %v", err)
	}

	h, err := CreateHandle()
	if err != nil {
		t.Fatalf("CreateHandle: %v", err)
	}

	defer func() { _ = h.Destroy() }()

	err = Sgemm(h, m, n, k, 1.0, devA, devB, 0.0, devC)
	if err != nil {
		t.Fatalf("Sgemm: %v", err)
	}

	result := make([]float32, 1)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devC, 1*elemSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy C: %v", err)
	}

	if result[0] != 32.0 {
		t.Errorf("C[0] = %f, want 32.0", result[0])
	}
}
