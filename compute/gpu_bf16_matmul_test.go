package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestGPUEngine_MatMulBF16BWeight(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()

	tests := []struct {
		name string
		m, k, n int
	}{
		{"2x2", 2, 2, 2},
		{"2x3x2", 2, 3, 2},
		{"4x4", 4, 4, 4},
		{"1x4x1", 1, 4, 1},
		{"8x16x8", 8, 16, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aData := make([]float32, tt.m*tt.k)
			bData := make([]float32, tt.k*tt.n)
			for i := range aData {
				aData[i] = float32(i%7+1) * 0.1
			}
			for i := range bData {
				bData[i] = float32(i%5+1) * 0.1
			}

			// Compute expected result using FP32 CPU engine.
			cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
			cpuA, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			cpuB, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			// Create A as FP32, B with BFloat16Storage.
			a, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			bf16B := tensor.NewBFloat16Storage(bData)
			b, _ := tensor.NewWithStorage[float32]([]int{tt.k, tt.n}, bf16B)

			got, err := eng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul BF16 B: %v", err)
			}

			gotData := got.Data()
			expData := expected.Data()
			if len(gotData) != len(expData) {
				t.Fatalf("output size mismatch: got %d, want %d", len(gotData), len(expData))
			}

			var maxRelErr float64
			for i := range gotData {
				if expData[i] == 0 {
					continue
				}
				rel := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
				if rel > maxRelErr {
					maxRelErr = rel
				}
			}

			if maxRelErr > 1e-3 {
				t.Errorf("max relative error = %e, want < 1e-3", maxRelErr)
			}
		})
	}
}

func TestGPUEngine_MatMulBF16AWeight(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()

	tests := []struct {
		name string
		m, k, n int
	}{
		{"2x2", 2, 2, 2},
		{"4x4", 4, 4, 4},
		{"8x16x8", 8, 16, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aData := make([]float32, tt.m*tt.k)
			bData := make([]float32, tt.k*tt.n)
			for i := range aData {
				aData[i] = float32(i%7+1) * 0.1
			}
			for i := range bData {
				bData[i] = float32(i%5+1) * 0.1
			}

			// Compute expected result using FP32 CPU engine.
			cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
			cpuA, _ := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			cpuB, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			// Create A with BFloat16Storage, B as FP32.
			bf16A := tensor.NewBFloat16Storage(aData)
			a, _ := tensor.NewWithStorage[float32]([]int{tt.m, tt.k}, bf16A)
			b, _ := tensor.New[float32]([]int{tt.k, tt.n}, bData)

			got, err := eng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("MatMul BF16 A: %v", err)
			}

			gotData := got.Data()
			expData := expected.Data()
			if len(gotData) != len(expData) {
				t.Fatalf("output size mismatch: got %d, want %d", len(gotData), len(expData))
			}

			var maxRelErr float64
			for i := range gotData {
				if expData[i] == 0 {
					continue
				}
				rel := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
				if rel > maxRelErr {
					maxRelErr = rel
				}
			}

			if maxRelErr > 1e-3 {
				t.Errorf("max relative error = %e, want < 1e-3", maxRelErr)
			}
		})
	}
}

func TestGPUEngine_UploadWeightsBF16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	// Create a tensor with BFloat16Storage.
	src := []float32{1.0, 2.0, 3.0, 4.0}
	bs := tensor.NewBFloat16Storage(src)
	tns, err := tensor.NewWithStorage[float32]([]int{2, 2}, bs)
	if err != nil {
		t.Fatalf("NewWithStorage: %v", err)
	}

	// Upload should cache the GPU pointer.
	if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{tns}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	ptr, byteSize, _ := bs.GPUPtr()
	if ptr == nil {
		t.Fatal("GPUPtr should be non-nil after UploadWeights")
	}
	if byteSize != len(src)*2 {
		t.Errorf("GPUPtr byteSize = %d, want %d", byteSize, len(src)*2)
	}

	// Second upload should be a no-op (already cached).
	if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{tns}); err != nil {
		t.Fatalf("second UploadWeights: %v", err)
	}

	ptr2, _, _ := bs.GPUPtr()
	if ptr2 != ptr {
		t.Error("GPUPtr changed after second UploadWeights, expected cache hit")
	}
}

func TestGPUEngine_MatMulBF16BWeight_AfterUpload(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	m, k, n := 4, 4, 4

	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	for i := range aData {
		aData[i] = float32(i+1) * 0.1
	}
	for i := range bData {
		bData[i] = float32(i+1) * 0.05
	}

	// Pre-upload BF16 weights.
	bs := tensor.NewBFloat16Storage(bData)
	bTensor, _ := tensor.NewWithStorage[float32]([]int{k, n}, bs)
	if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{bTensor}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	// MatMul should use the pre-uploaded GPU pointer.
	a, _ := tensor.New[float32]([]int{m, k}, aData)
	got, err := eng.MatMul(ctx, a, bTensor)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	// Compare with CPU FP32 reference.
	cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
	cpuA, _ := tensor.New[float32]([]int{m, k}, aData)
	cpuB, _ := tensor.New[float32]([]int{k, n}, bData)
	expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
	if err != nil {
		t.Fatalf("CPU MatMul: %v", err)
	}

	gotData := got.Data()
	expData := expected.Data()

	var maxRelErr float64
	for i := range gotData {
		if expData[i] == 0 {
			continue
		}
		rel := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
		if rel > maxRelErr {
			maxRelErr = rel
		}
	}

	if maxRelErr > 1e-3 {
		t.Errorf("max relative error = %e, want < 1e-3", maxRelErr)
	}
}
