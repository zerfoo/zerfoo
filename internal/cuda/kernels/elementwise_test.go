//go:build cuda

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// helper allocates device memory, copies host data, and returns a device pointer.
func toDevice(t *testing.T, data []float32) unsafe.Pointer {
	t.Helper()

	byteSize := len(data) * int(unsafe.Sizeof(data[0]))
	devPtr, err := cuda.Malloc(byteSize)

	if err != nil {
		t.Fatalf("Malloc: %v", err)
	}

	if err := cuda.Memcpy(devPtr, unsafe.Pointer(&data[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy H2D: %v", err)
	}

	return devPtr
}

// fromDevice copies device memory to a host slice.
func fromDevice(t *testing.T, devPtr unsafe.Pointer, n int) []float32 {
	t.Helper()

	result := make([]float32, n)
	byteSize := n * int(unsafe.Sizeof(float32(0)))

	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devPtr, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	return result
}

func TestKernelAdd(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	n := len(a)

	devA := toDevice(t, a)
	defer func() { _ = cuda.Free(devA) }()

	devB := toDevice(t, b)
	defer func() { _ = cuda.Free(devB) }()

	devC, _ := cuda.Malloc(n * 4)
	defer func() { _ = cuda.Free(devC) }()

	if err := Add(devA, devB, devC, n); err != nil {
		t.Fatalf("Add: %v", err)
	}

	result := fromDevice(t, devC, n)
	expected := []float32{6, 8, 10, 12}

	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestKernelMulScalar(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	n := len(a)

	devA := toDevice(t, a)
	defer func() { _ = cuda.Free(devA) }()

	devC, _ := cuda.Malloc(n * 4)
	defer func() { _ = cuda.Free(devC) }()

	if err := MulScalar(devA, 3.0, devC, n); err != nil {
		t.Fatalf("MulScalar: %v", err)
	}

	result := fromDevice(t, devC, n)
	expected := []float32{3, 6, 9, 12}

	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestKernelExp(t *testing.T) {
	a := []float32{0, 1, 2}
	n := len(a)

	devA := toDevice(t, a)
	defer func() { _ = cuda.Free(devA) }()

	devC, _ := cuda.Malloc(n * 4)
	defer func() { _ = cuda.Free(devC) }()

	if err := Exp(devA, devC, n); err != nil {
		t.Fatalf("Exp: %v", err)
	}

	result := fromDevice(t, devC, n)

	for i, v := range a {
		want := float32(math.Exp(float64(v)))
		if math.Abs(float64(result[i]-want)) > 1e-5 {
			t.Errorf("[%d] = %f, want %f", i, result[i], want)
		}
	}
}

func TestKernelTanh(t *testing.T) {
	a := []float32{-1, 0, 1, 2}
	n := len(a)

	devA := toDevice(t, a)
	defer func() { _ = cuda.Free(devA) }()

	devC, _ := cuda.Malloc(n * 4)
	defer func() { _ = cuda.Free(devC) }()

	if err := Tanh(devA, devC, n); err != nil {
		t.Fatalf("Tanh: %v", err)
	}

	result := fromDevice(t, devC, n)

	for i, v := range a {
		want := float32(math.Tanh(float64(v)))
		if math.Abs(float64(result[i]-want)) > 1e-5 {
			t.Errorf("[%d] = %f, want %f", i, result[i], want)
		}
	}
}

func TestKernelFill(t *testing.T) {
	n := 8
	devPtr, _ := cuda.Malloc(n * 4)

	defer func() { _ = cuda.Free(devPtr) }()

	if err := Fill(devPtr, 42.0, n); err != nil {
		t.Fatalf("Fill: %v", err)
	}

	result := fromDevice(t, devPtr, n)

	for i, v := range result {
		if v != 42.0 {
			t.Errorf("[%d] = %f, want 42.0", i, v)
		}
	}
}
