package inference

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestTransposeWeight2D_F32(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	// 2x3 matrix: [[1,2,3],[4,5,6]]
	w, err := tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatal(err)
	}

	got, err := transposeWeight2D(engine, false, "test.weight", w)
	if err != nil {
		t.Fatal(err)
	}

	shape := got.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Fatalf("expected shape [3,2], got %v", shape)
	}

	// Expected: [[1,4],[2,5],[3,6]]
	data := got.Data()
	expected := []float32{1, 4, 2, 5, 3, 6}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("data[%d] = %f, want %f", i, data[i], v)
		}
	}
}

func TestTransposeWeight2D_Q4VirtualTranspose(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Create Q4 quantized storage and verify virtual transpose.
	data := make([]float32, 64) // minimum Q4 block size
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	q4 := tensor.QuantizeQ4(data)
	w, err := tensor.NewWithStorage[float32]([]int{2, 32}, q4)
	if err != nil {
		t.Fatal(err)
	}

	got, err := transposeWeight2D(engine, false, "test.q4", w)
	if err != nil {
		t.Fatal(err)
	}

	shape := got.Shape()
	if shape[0] != 32 || shape[1] != 2 {
		t.Fatalf("expected shape [32,2], got %v", shape)
	}

	// Storage should be the same Q4Storage instance (virtual transpose).
	if _, ok := got.GetStorage().(*tensor.Q4Storage); !ok {
		t.Fatal("expected Q4Storage to be preserved after virtual transpose")
	}
}

func TestTransposeWeight2D_Q4GPU(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i) * 0.1
	}
	q4 := tensor.QuantizeQ4(data)
	w, err := tensor.NewWithStorage[float32]([]int{2, 32}, q4)
	if err != nil {
		t.Fatal(err)
	}

	// Simulate GPU path (isGPUEngine=true).
	got, err := transposeWeight2D(engine, true, "test.q4.gpu", w)
	if err != nil {
		t.Fatal(err)
	}

	shape := got.Shape()
	if shape[0] != 32 || shape[1] != 2 {
		t.Fatalf("expected shape [32,2], got %v", shape)
	}

	if _, ok := got.GetStorage().(*tensor.Q4Storage); !ok {
		t.Fatal("expected Q4Storage to be preserved on GPU path")
	}
}

func TestTransposeWeight2D_Float16(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	f32 := []float32{1, 2, 3, 4, 5, 6}
	fp16 := tensor.NewFloat16StorageFromF32(f32)
	w, err := tensor.NewWithStorage[float32]([]int{2, 3}, fp16)
	if err != nil {
		t.Fatal(err)
	}

	got, err := transposeWeight2D(engine, false, "test.fp16", w)
	if err != nil {
		t.Fatal(err)
	}

	shape := got.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Fatalf("expected shape [3,2], got %v", shape)
	}

	// Should preserve Float16Storage.
	gotFP16, ok := got.GetStorage().(*tensor.Float16Storage)
	if !ok {
		t.Fatal("expected Float16Storage to be preserved after transpose")
	}

	// Verify transposed values: [[1,4],[2,5],[3,6]].
	result := gotFP16.Slice()
	expected := []float32{1, 4, 2, 5, 3, 6}
	for i, v := range expected {
		if diff := result[i] - v; diff > 0.01 || diff < -0.01 {
			t.Errorf("result[%d] = %f, want %f", i, result[i], v)
		}
	}
}

func TestTransposeWeight2D_Q8CPU(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Q8 on CPU path should do virtual transpose (shape swap).
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	q8 := tensor.QuantizeQ8(data)
	w, err := tensor.NewWithStorage[float32]([]int{2, 32}, q8)
	if err != nil {
		t.Fatal(err)
	}

	got, err := transposeWeight2D(engine, false, "test.q8.cpu", w)
	if err != nil {
		t.Fatal(err)
	}

	shape := got.Shape()
	if shape[0] != 32 || shape[1] != 2 {
		t.Fatalf("expected shape [32,2], got %v", shape)
	}

	// CPU path: Q8 uses virtual transpose.
	if _, ok := got.GetStorage().(*tensor.Q8Storage); !ok {
		t.Fatal("expected Q8Storage to be preserved on CPU path")
	}
}

func TestTransposeWeight2D_Q8GPU(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Q8 on GPU path should dequantize to F32.
	data := make([]float32, 64)
	for i := range data {
		data[i] = float32(i) * 0.01
	}
	q8 := tensor.QuantizeQ8(data)
	w, err := tensor.NewWithStorage[float32]([]int{2, 32}, q8)
	if err != nil {
		t.Fatal(err)
	}

	got, err := transposeWeight2D(engine, true, "test.q8.gpu", w)
	if err != nil {
		t.Fatal(err)
	}

	shape := got.Shape()
	if shape[0] != 32 || shape[1] != 2 {
		t.Fatalf("expected shape [32,2], got %v", shape)
	}

	// GPU path: Q8 is dequantized to F32 — storage should NOT be Q8Storage.
	if _, ok := got.GetStorage().(*tensor.Q8Storage); ok {
		t.Fatal("expected Q8Storage to be dequantized on GPU path")
	}
}

func TestTransposeWeight2D_FP8(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	f32 := make([]float32, 6)
	for i := range f32 {
		f32[i] = float32(i+1) * 0.5
	}
	fp8 := tensor.NewFP8E4M3Storage(f32)
	w, err := tensor.NewWithStorage[float32]([]int{2, 3}, fp8)
	if err != nil {
		t.Fatal(err)
	}

	for _, gpu := range []bool{false, true} {
		label := "cpu"
		if gpu {
			label = "gpu"
		}
		t.Run(label, func(t *testing.T) {
			got, err := transposeWeight2D(engine, gpu, "test.fp8", w)
			if err != nil {
				t.Fatal(err)
			}
			shape := got.Shape()
			if shape[0] != 3 || shape[1] != 2 {
				t.Fatalf("expected shape [3,2], got %v", shape)
			}
			if _, ok := got.GetStorage().(*tensor.FP8E4M3Storage); !ok {
				t.Fatal("expected FP8E4M3Storage to be preserved")
			}
		})
	}
}
