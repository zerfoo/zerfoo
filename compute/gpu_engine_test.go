//go:build cuda

package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func newTestGPUEngine(t *testing.T) *GPUEngine[float32] {
	t.Helper()

	eng, err := NewGPUEngine[float32](numeric.Float32Arithmetic{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}

	t.Cleanup(func() {
		if err := eng.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})

	return eng
}

func TestGPUEngine_InterfaceCompliance(t *testing.T) {
	var _ Engine[float32] = (*GPUEngine[float32])(nil)
}

func TestGPUEngine_MatMul2D(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	// A = [[1, 2], [3, 4]] (2x2)
	// B = [[5, 6], [7, 8]] (2x2)
	// C = [[19, 22], [43, 50]]
	a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})

	c, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	expected := []float32{19, 22, 43, 50}
	data := c.Data()

	for i, want := range expected {
		if data[i] != want {
			t.Errorf("C[%d] = %f, want %f", i, data[i], want)
		}
	}
}

func TestGPUEngine_MatMulNonSquare(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	// A (2x3) * B (3x2) = C (2x2)
	a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := tensor.New[float32]([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12})

	c, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	// Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
	// Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
	expected := []float32{58, 64, 139, 154}
	data := c.Data()

	for i, want := range expected {
		if data[i] != want {
			t.Errorf("C[%d] = %f, want %f", i, data[i], want)
		}
	}
}

func TestGPUEngine_MatMulBatched(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	// Batch of 2: A (2x2x3) * B (2x3x1) = C (2x2x1)
	a, _ := tensor.New[float32]([]int{2, 2, 3}, []float32{
		1, 2, 3, 4, 5, 6, // batch 0
		7, 8, 9, 10, 11, 12, // batch 1
	})
	b, _ := tensor.New[float32]([]int{2, 3, 1}, []float32{
		1, 1, 1, // batch 0
		2, 2, 2, // batch 1
	})

	c, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	// batch 0: [[1+2+3], [4+5+6]] = [[6], [15]]
	// batch 1: [[7*2+8*2+9*2], [10*2+11*2+12*2]] = [[48], [66]]
	expected := []float32{6, 15, 48, 66}
	data := c.Data()

	for i, want := range expected {
		if data[i] != want {
			t.Errorf("C[%d] = %f, want %f", i, data[i], want)
		}
	}
}

func TestGPUEngine_MatMulParityWithCPU(t *testing.T) {
	gpuEng := newTestGPUEngine(t)
	cpuEng := NewCPUEngine[float32](numeric.Float32Arithmetic{})
	ctx := context.Background()

	// 4x8 * 8x4 = 4x4
	m, k, n := 4, 8, 4
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)

	for i := range aData {
		aData[i] = float32(i+1) * 0.1
	}

	for i := range bData {
		bData[i] = float32(i+1) * 0.01
	}

	a, _ := tensor.New[float32]([]int{m, k}, aData)
	b, _ := tensor.New[float32]([]int{k, n}, bData)

	gpuResult, err := gpuEng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("GPU MatMul: %v", err)
	}

	cpuResult, err := cpuEng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("CPU MatMul: %v", err)
	}

	gpuData := gpuResult.Data()
	cpuData := cpuResult.Data()

	for i := range gpuData {
		diff := float64(gpuData[i] - cpuData[i])
		if math.Abs(diff) > 1e-5 {
			t.Errorf("parity mismatch at [%d]: GPU=%f, CPU=%f, diff=%e", i, gpuData[i], cpuData[i], diff)
		}
	}
}

func TestGPUEngine_MatMulErrors(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{2, 3}, nil)
	b, _ := tensor.New[float32]([]int{4, 2}, nil) // incompatible: 3 != 4

	_, err := eng.MatMul(ctx, a, b)
	if err == nil {
		t.Error("expected error for incompatible shapes, got nil")
	}

	// Nil inputs
	_, err = eng.MatMul(ctx, nil, b)
	if err == nil {
		t.Error("expected error for nil input, got nil")
	}
}

func TestGPUEngine_FallbackAdd(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{3}, []float32{1, 2, 3})
	b, _ := tensor.New[float32]([]int{3}, []float32{4, 5, 6})

	c, err := eng.Add(ctx, a, b)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	expected := []float32{5, 7, 9}
	data := c.Data()

	for i, want := range expected {
		if data[i] != want {
			t.Errorf("Add[%d] = %f, want %f", i, data[i], want)
		}
	}
}

func TestGPUEngine_FallbackSoftmax(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	a, _ := tensor.New[float32]([]int{1, 3}, []float32{1, 2, 3})

	c, err := eng.Softmax(ctx, a, 1)
	if err != nil {
		t.Fatalf("Softmax: %v", err)
	}

	data := c.Data()
	sum := float32(0)

	for _, v := range data {
		sum += v
	}

	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("Softmax sum = %f, want 1.0", sum)
	}
}
