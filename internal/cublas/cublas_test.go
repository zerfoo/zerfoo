//go:build cuda

package cublas

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

func TestCreateAndDestroyHandle(t *testing.T) {
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

func TestSgemmNonSquare(t *testing.T) {
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
