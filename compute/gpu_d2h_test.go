package compute

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGPUEngine_Gather_NoD2H verifies that GPUEngine.Gather on GPU-resident
// params produces a GPU-resident output tensor without triggering a D2H copy.
// The output should have GPUStorage, confirming data stays on the GPU.
func TestGPUEngine_Gather_NoD2H(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng := newTestGPUEngine(t)
	ctx := context.Background()

	// Create GPU-resident embedding table: 4 rows x 3 dims.
	embedData := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	}
	embedCPU, err := tensor.New[float32]([]int{4, 3}, embedData)
	if err != nil {
		t.Fatalf("tensor.New: %v", err)
	}
	embedGPU, err := tensor.ToGPU(embedCPU)
	if err != nil {
		t.Fatalf("ToGPU: %v", err)
	}

	// Indices (CPU-resident is fine; Gather uploads them).
	indices, err := tensor.New[int]([]int{2}, []int{1, 3})
	if err != nil {
		t.Fatalf("tensor.New indices: %v", err)
	}

	// Output tensor (will be overwritten by Gather).
	output, err := tensor.New[float32]([]int{2, 3}, make([]float32, 6))
	if err != nil {
		t.Fatalf("tensor.New output: %v", err)
	}

	if err := eng.Gather(ctx, embedGPU, indices, output); err != nil {
		t.Fatalf("Gather: %v", err)
	}

	// Verify output is GPU-resident (no premature D2H).
	gs, ok := output.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		t.Fatal("Gather output should have GPUStorage, but storage is CPU-backed; " +
			"this suggests a D2H copy occurred in the Gather path")
	}
	if gs.Len() != 6 {
		t.Errorf("GPUStorage.Len() = %d, want 6", gs.Len())
	}

	// Verify correctness: rows 1 and 3 of the embedding table.
	data := output.Data() // intentional D2H here for verification only
	want := []float32{4, 5, 6, 10, 11, 12}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("output[%d] = %f, want %f", i, data[i], w)
		}
	}
}

// TestGPUEngine_Gather_IndicesNotD2H verifies that the GPU Gather path does
// not call .Data() on GPU-resident index tensors (which would trigger D2H).
// Instead, indices should be uploaded via H2D memcpy from their CPU data.
func TestGPUEngine_Gather_IndicesNotD2H(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	eng := newTestGPUEngine(t)
	ctx := context.Background()

	// GPU-resident params.
	params, err := tensor.New[float32]([]int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("tensor.New params: %v", err)
	}
	paramsGPU, err := tensor.ToGPU(params)
	if err != nil {
		t.Fatalf("ToGPU params: %v", err)
	}

	// CPU indices (the normal case in decode — embedding lookup).
	indices, err := tensor.New[int]([]int{1}, []int{2})
	if err != nil {
		t.Fatalf("tensor.New indices: %v", err)
	}

	output, err := tensor.New[float32]([]int{1, 2}, make([]float32, 2))
	if err != nil {
		t.Fatalf("tensor.New output: %v", err)
	}

	if err := eng.Gather(ctx, paramsGPU, indices, output); err != nil {
		t.Fatalf("Gather: %v", err)
	}

	// Output should be GPU-resident.
	if _, ok := output.GetStorage().(*tensor.GPUStorage[float32]); !ok {
		t.Fatal("expected GPU-resident output after Gather with GPU params")
	}

	data := output.Data()
	want := []float32{5, 6}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("output[%d] = %f, want %f", i, data[i], w)
		}
	}
}
