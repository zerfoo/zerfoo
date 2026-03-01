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

	if err := Add(devA, devB, devC, n, nil); err != nil {
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

	if err := MulScalar(devA, 3.0, devC, n, nil); err != nil {
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

	if err := Exp(devA, devC, n, nil); err != nil {
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

	if err := Tanh(devA, devC, n, nil); err != nil {
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

func TestKernelSumAxis(t *testing.T) {
	// shape [2,3], axis=1 => outer=2, inner=1, axisSize=3
	// Row sums: [1+2+3, 4+5+6] = [6, 15]
	input := []float32{1, 2, 3, 4, 5, 6}

	devIn := toDevice(t, input)
	defer func() { _ = cuda.Free(devIn) }()

	outN := 2 // outer * inner
	devOut, _ := cuda.Malloc(outN * 4)
	defer func() { _ = cuda.Free(devOut) }()

	if err := SumAxis(devIn, devOut, 2, 1, 3, nil); err != nil {
		t.Fatalf("SumAxis: %v", err)
	}

	result := fromDevice(t, devOut, outN)
	expected := []float32{6, 15}

	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-5 {
			t.Errorf("[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestKernelSumAxisAxis0(t *testing.T) {
	// shape [2,3], axis=0 => outer=1, inner=3, axisSize=2
	// Column sums: [1+4, 2+5, 3+6] = [5, 7, 9]
	input := []float32{1, 2, 3, 4, 5, 6}

	devIn := toDevice(t, input)
	defer func() { _ = cuda.Free(devIn) }()

	outN := 3 // outer * inner
	devOut, _ := cuda.Malloc(outN * 4)
	defer func() { _ = cuda.Free(devOut) }()

	if err := SumAxis(devIn, devOut, 1, 3, 2, nil); err != nil {
		t.Fatalf("SumAxis axis0: %v", err)
	}

	result := fromDevice(t, devOut, outN)
	expected := []float32{5, 7, 9}

	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-5 {
			t.Errorf("[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestKernelSoftmax(t *testing.T) {
	// 2D softmax: shape [2,3], axis=1 (last axis)
	// outer=2, inner=1, axisSize=3
	input := []float32{1, 2, 3, 1, 1, 1}
	n := len(input)

	devIn := toDevice(t, input)
	defer func() { _ = cuda.Free(devIn) }()

	devOut, _ := cuda.Malloc(n * 4)
	defer func() { _ = cuda.Free(devOut) }()

	if err := Softmax(devIn, devOut, 2, 1, 3, nil); err != nil {
		t.Fatalf("Softmax: %v", err)
	}

	result := fromDevice(t, devOut, n)

	// Row 0: softmax([1,2,3])
	e1 := float32(math.Exp(1))
	e2 := float32(math.Exp(2))
	e3 := float32(math.Exp(3))
	sum0 := e1 + e2 + e3
	expected := []float32{e1 / sum0, e2 / sum0, e3 / sum0}

	// Row 1: softmax([1,1,1]) = [1/3, 1/3, 1/3]
	expected = append(expected, 1.0/3.0, 1.0/3.0, 1.0/3.0)

	for i := range expected {
		if math.Abs(float64(result[i]-expected[i])) > 1e-5 {
			t.Errorf("[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestKernelSoftmaxAxis0(t *testing.T) {
	// 2D softmax: shape [2,3], axis=0
	// outer=1, inner=3, axisSize=2
	input := []float32{1, 2, 3, 4, 5, 6}
	n := len(input)

	devIn := toDevice(t, input)
	defer func() { _ = cuda.Free(devIn) }()

	devOut, _ := cuda.Malloc(n * 4)
	defer func() { _ = cuda.Free(devOut) }()

	if err := Softmax(devIn, devOut, 1, 3, 2, nil); err != nil {
		t.Fatalf("Softmax axis0: %v", err)
	}

	result := fromDevice(t, devOut, n)

	for col := 0; col < 3; col++ {
		v0 := input[col]
		v1 := input[3+col]
		e0 := float32(math.Exp(float64(v0)))
		e1 := float32(math.Exp(float64(v1)))
		s := e0 + e1
		want0 := e0 / s
		want1 := e1 / s

		if math.Abs(float64(result[col]-want0)) > 1e-5 {
			t.Errorf("col %d row 0: got %f, want %f", col, result[col], want0)
		}

		if math.Abs(float64(result[3+col]-want1)) > 1e-5 {
			t.Errorf("col %d row 1: got %f, want %f", col, result[3+col], want1)
		}
	}
}

func TestKernelFill(t *testing.T) {
	n := 8
	devPtr, _ := cuda.Malloc(n * 4)

	defer func() { _ = cuda.Free(devPtr) }()

	if err := Fill(devPtr, 42.0, n, nil); err != nil {
		t.Fatalf("Fill: %v", err)
	}

	result := fromDevice(t, devPtr, n)

	for i, v := range result {
		if v != 42.0 {
			t.Errorf("[%d] = %f, want 42.0", i, v)
		}
	}
}

func TestKernelAddOnStream(t *testing.T) {
	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}

	defer func() { _ = stream.Destroy() }()

	a := []float32{1, 2, 3, 4}
	b := []float32{10, 20, 30, 40}
	n := len(a)

	devA := toDevice(t, a)
	defer func() { _ = cuda.Free(devA) }()

	devB := toDevice(t, b)
	defer func() { _ = cuda.Free(devB) }()

	devC, _ := cuda.Malloc(n * 4)
	defer func() { _ = cuda.Free(devC) }()

	if err := Add(devA, devB, devC, n, stream.Ptr()); err != nil {
		t.Fatalf("Add on stream: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	result := fromDevice(t, devC, n)
	expected := []float32{11, 22, 33, 44}

	for i := range expected {
		if result[i] != expected[i] {
			t.Errorf("[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}
