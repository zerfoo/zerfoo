package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGPUEngine_Gather_FP16Output verifies that when dtype is DTypeFP16,
// Gather converts its F32 output to Float16Storage on GPU. This is the
// single F32->FP16 conversion point for the entire FP16 forward pass.
func TestGPUEngine_Gather_FP16Output(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng, err := NewGPUEngine[float32](numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = eng.Close() }()
	eng.SetDType(DTypeFP16)

	// Embedding table: 4 rows x 3 dims, uploaded as FP16 weights.
	embedData := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	embedTensor, err := tensor.New[float32]([]int{4, 3}, embedData)
	if err != nil {
		t.Fatalf("tensor.New embed: %v", err)
	}
	if err := eng.UploadWeights([]*tensor.TensorNumeric[float32]{embedTensor}); err != nil {
		t.Fatalf("UploadWeights: %v", err)
	}

	indices, err := tensor.New[int]([]int{2}, []int{1, 3})
	if err != nil {
		t.Fatalf("tensor.New indices: %v", err)
	}

	output, err := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	if err != nil {
		t.Fatalf("tensor.New output: %v", err)
	}

	if err := eng.Gather(context.Background(), embedTensor, indices, output); err != nil {
		t.Fatalf("Gather: %v", err)
	}

	// Output must be Float16Storage when dtype=FP16.
	fs, ok := output.GetStorage().(*tensor.Float16Storage)
	if !ok {
		t.Fatalf("expected Float16Storage after FP16 Gather, got %T", output.GetStorage())
	}
	if fs.Len() != 6 {
		t.Errorf("Float16Storage.Len() = %d, want 6", fs.Len())
	}

	// Verify correctness: rows 1 and 3 (values 4,5,6 and 10,11,12).
	data := output.Data()
	want := []float32{4, 5, 6, 10, 11, 12}
	for i, w := range want {
		if diff := data[i] - w; diff > 0.01 || diff < -0.01 {
			t.Errorf("output[%d] = %f, want %f", i, data[i], w)
		}
	}
}

// TestGPUEngine_Gather_F32OutputWhenNotFP16 verifies that when dtype is
// DTypeF32 (default), Gather produces normal GPUStorage (no FP16 conversion).
func TestGPUEngine_Gather_F32OutputWhenNotFP16(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng := newTestGPUEngine(t)
	ctx := context.Background()

	embedData := []float32{1, 2, 3, 4, 5, 6}
	embedTensor, err := tensor.New[float32]([]int{2, 3}, embedData)
	if err != nil {
		t.Fatalf("tensor.New embed: %v", err)
	}
	embedGPU, err := tensor.ToGPU(embedTensor)
	if err != nil {
		t.Fatalf("ToGPU: %v", err)
	}

	indices, err := tensor.New[int]([]int{1}, []int{0})
	if err != nil {
		t.Fatalf("tensor.New indices: %v", err)
	}

	output, err := tensor.New[float32]([]int{1, 3}, make([]float32, 3))
	if err != nil {
		t.Fatalf("tensor.New output: %v", err)
	}

	if err := eng.Gather(ctx, embedGPU, indices, output); err != nil {
		t.Fatalf("Gather: %v", err)
	}

	// Output must be GPUStorage[float32], NOT Float16Storage.
	if _, ok := output.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		t.Fatalf("expected GPUStorage[float32] for F32 Gather, got %T", output.GetStorage())
	}

	data := output.Data()
	want := []float32{1, 2, 3}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("output[%d] = %f, want %f", i, data[i], w)
		}
	}
}
