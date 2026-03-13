package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestFP16MatMul_BatchDimensions(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()
	eng.dtype = DTypeFP16

	ctx := context.Background()
	cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})

	tests := []struct {
		name   string
		aShape []int
		bShape []int
	}{
		{"2D_2x3_3x2", []int{2, 3}, []int{3, 2}},
		{"3D_batch4_2x3_3x2", []int{4, 2, 3}, []int{4, 3, 2}},
		{"3D_batch1_2x3_3x2", []int{1, 2, 3}, []int{1, 3, 2}},
		{"3D_broadcast_B", []int{4, 2, 3}, []int{1, 3, 2}},
		{"4D_batch2x3_2x2_2x2", []int{2, 3, 2, 2}, []int{2, 3, 2, 2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aSize := 1
			for _, d := range tt.aShape {
				aSize *= d
			}
			bSize := 1
			for _, d := range tt.bShape {
				bSize *= d
			}

			aData := make([]float32, aSize)
			bData := make([]float32, bSize)
			for i := range aData {
				aData[i] = float32(i%7+1) * 0.1
			}
			for i := range bData {
				bData[i] = float32(i%5+1) * 0.1
			}

			cpuA, _ := tensor.New[float32](tt.aShape, aData)
			cpuB, _ := tensor.New[float32](tt.bShape, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			a, _ := tensor.New[float32](tt.aShape, aData)
			b, _ := tensor.New[float32](tt.bShape, bData)

			got, err := eng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("FP16 MatMul: %v", err)
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
				relErr := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
			}

			// FP16 has limited precision; allow up to 5% relative error.
			if maxRelErr > 0.05 {
				t.Errorf("max relative error %.4f exceeds threshold 0.05", maxRelErr)
			}
		})
	}
}

func TestFP16MatMulNative_BothFloat16Storage(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})

	tests := []struct {
		name   string
		aShape []int
		bShape []int
	}{
		{"2D_2x3_3x2", []int{2, 3}, []int{3, 2}},
		{"3D_batch4_2x3_3x2", []int{4, 2, 3}, []int{4, 3, 2}},
		{"3D_broadcast_B", []int{4, 2, 3}, []int{1, 3, 2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aSize := 1
			for _, d := range tt.aShape {
				aSize *= d
			}
			bSize := 1
			for _, d := range tt.bShape {
				bSize *= d
			}

			aData := make([]float32, aSize)
			bData := make([]float32, bSize)
			for i := range aData {
				aData[i] = float32(i%7+1) * 0.1
			}
			for i := range bData {
				bData[i] = float32(i%5+1) * 0.1
			}

			// CPU reference.
			cpuA, _ := tensor.New[float32](tt.aShape, aData)
			cpuB, _ := tensor.New[float32](tt.bShape, bData)
			expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
			if err != nil {
				t.Fatalf("CPU MatMul: %v", err)
			}

			// Both A and B as Float16Storage.
			fp16A := tensor.NewFloat16StorageFromF32(aData)
			a, _ := tensor.NewWithStorage[float32](tt.aShape, fp16A)
			fp16B := tensor.NewFloat16StorageFromF32(bData)
			b, _ := tensor.NewWithStorage[float32](tt.bShape, fp16B)

			got, err := eng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("FP16 native MatMul: %v", err)
			}

			// Verify output is Float16Storage.
			if _, ok := got.GetStorage().(*tensor.Float16Storage); !ok {
				t.Fatalf("expected Float16Storage output, got %T", got.GetStorage())
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
				relErr := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
				if relErr > maxRelErr {
					maxRelErr = relErr
				}
			}

			if maxRelErr > 1e-2 {
				t.Errorf("max relative error %.4f exceeds threshold 1e-2", maxRelErr)
			}
		})
	}
}

func TestFP16MatMulNative_MixedInputs(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})

	aData := []float32{1, 2, 3, 4, 5, 6}
	bData := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}

	// CPU reference.
	cpuA, _ := tensor.New[float32]([]int{2, 3}, aData)
	cpuB, _ := tensor.New[float32]([]int{3, 2}, bData)
	expected, err := cpuEng.MatMul(ctx, cpuA, cpuB)
	if err != nil {
		t.Fatalf("CPU MatMul: %v", err)
	}

	t.Run("A_fp16_B_f32", func(t *testing.T) {
		fp16A := tensor.NewFloat16StorageFromF32(aData)
		a, _ := tensor.NewWithStorage[float32]([]int{2, 3}, fp16A)
		b, _ := tensor.New[float32]([]int{3, 2}, bData)

		got, err := eng.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMul: %v", err)
		}

		if _, ok := got.GetStorage().(*tensor.Float16Storage); !ok {
			t.Fatalf("expected Float16Storage output, got %T", got.GetStorage())
		}

		gotData := got.Data()
		expData := expected.Data()
		var maxRelErr float64
		for i := range gotData {
			if expData[i] == 0 {
				continue
			}
			relErr := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
		if maxRelErr > 1e-2 {
			t.Errorf("max relative error %.4f exceeds threshold 1e-2", maxRelErr)
		}
	})

	t.Run("A_f32_B_fp16", func(t *testing.T) {
		a, _ := tensor.New[float32]([]int{2, 3}, aData)
		fp16B := tensor.NewFloat16StorageFromF32(bData)
		b, _ := tensor.NewWithStorage[float32]([]int{3, 2}, fp16B)

		got, err := eng.MatMul(ctx, a, b)
		if err != nil {
			t.Fatalf("MatMul: %v", err)
		}

		if _, ok := got.GetStorage().(*tensor.Float16Storage); !ok {
			t.Fatalf("expected Float16Storage output, got %T", got.GetStorage())
		}

		gotData := got.Data()
		expData := expected.Data()
		var maxRelErr float64
		for i := range gotData {
			if expData[i] == 0 {
				continue
			}
			relErr := math.Abs(float64(gotData[i]-expData[i])) / math.Abs(float64(expData[i]))
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
		if maxRelErr > 1e-2 {
			t.Errorf("max relative error %.4f exceeds threshold 1e-2", maxRelErr)
		}
	})
}
