//go:build cuda

package tensor

import (
	"testing"

	"github.com/zerfoo/zerfoo/device"
)

func TestToGPURoundTrip(t *testing.T) {
	src, err := New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	gpu, err := ToGPU(src)
	if err != nil {
		t.Fatalf("ToGPU failed: %v", err)
	}

	// Verify device type
	if gpu.GetStorage().DeviceType() != device.CUDA {
		t.Errorf("expected CUDA storage, got %d", gpu.GetStorage().DeviceType())
	}

	// Transfer back to CPU
	cpu := ToCPU(gpu)

	if cpu.GetStorage().DeviceType() != device.CPU {
		t.Errorf("expected CPU storage, got %d", cpu.GetStorage().DeviceType())
	}

	// Verify data integrity
	srcData := src.Data()
	cpuData := cpu.Data()

	if len(srcData) != len(cpuData) {
		t.Fatalf("data length mismatch: %d vs %d", len(srcData), len(cpuData))
	}

	for i := range srcData {
		if srcData[i] != cpuData[i] {
			t.Errorf("data[%d] = %f, want %f", i, cpuData[i], srcData[i])
		}
	}
}

func TestToGPUPreservesShape(t *testing.T) {
	src, err := New[float32]([]int{3, 4, 2}, nil)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	gpu, err := ToGPU(src)
	if err != nil {
		t.Fatalf("ToGPU failed: %v", err)
	}

	srcShape := src.Shape()
	gpuShape := gpu.Shape()

	if len(srcShape) != len(gpuShape) {
		t.Fatalf("shape length mismatch: %v vs %v", srcShape, gpuShape)
	}

	for i := range srcShape {
		if srcShape[i] != gpuShape[i] {
			t.Errorf("shape[%d] = %d, want %d", i, gpuShape[i], srcShape[i])
		}
	}

	srcStrides := src.Strides()
	gpuStrides := gpu.Strides()

	for i := range srcStrides {
		if srcStrides[i] != gpuStrides[i] {
			t.Errorf("strides[%d] = %d, want %d", i, gpuStrides[i], srcStrides[i])
		}
	}
}

func TestToCPUPreservesShape(t *testing.T) {
	src, err := New[float32]([]int{2, 5}, nil)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	gpu, err := ToGPU(src)
	if err != nil {
		t.Fatalf("ToGPU failed: %v", err)
	}

	cpu := ToCPU(gpu)

	srcShape := src.Shape()
	cpuShape := cpu.Shape()

	if len(srcShape) != len(cpuShape) {
		t.Fatalf("shape length mismatch: %v vs %v", srcShape, cpuShape)
	}

	for i := range srcShape {
		if srcShape[i] != cpuShape[i] {
			t.Errorf("shape[%d] = %d, want %d", i, cpuShape[i], srcShape[i])
		}
	}
}

func TestToGPUDoesNotModifySource(t *testing.T) {
	src, err := New[float32]([]int{3}, []float32{10, 20, 30})
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if src.GetStorage().DeviceType() != device.CPU {
		t.Fatal("source should be CPU storage")
	}

	_, err = ToGPU(src)
	if err != nil {
		t.Fatalf("ToGPU failed: %v", err)
	}

	// Source should still be CPU
	if src.GetStorage().DeviceType() != device.CPU {
		t.Error("ToGPU should not modify source storage type")
	}

	// Source data should be unchanged
	data := src.Data()
	expected := []float32{10, 20, 30}

	for i := range expected {
		if data[i] != expected[i] {
			t.Errorf("source data[%d] = %f, want %f", i, data[i], expected[i])
		}
	}
}
