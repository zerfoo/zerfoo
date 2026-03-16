package core

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestScatterND_Forward(t *testing.T) {
	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	tests := []struct {
		name         string
		dataShape    []int
		data         []float32
		indicesShape []int
		indices      []float32
		updatesShape []int
		updates      []float32
		want         []float32
	}{
		{
			name:         "1D scatter single element",
			dataShape:    []int{5},
			data:         []float32{1, 2, 3, 4, 5},
			indicesShape: []int{1, 1},
			indices:      []float32{2},
			updatesShape: []int{1},
			updates:      []float32{99},
			want:         []float32{1, 2, 99, 4, 5},
		},
		{
			name:         "1D scatter multiple elements",
			dataShape:    []int{5},
			data:         []float32{1, 2, 3, 4, 5},
			indicesShape: []int{3, 1},
			indices:      []float32{0, 2, 4},
			updatesShape: []int{3},
			updates:      []float32{10, 30, 50},
			want:         []float32{10, 2, 30, 4, 50},
		},
		{
			name:         "2D scatter rows",
			dataShape:    []int{3, 4},
			data:         []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			indicesShape: []int{2, 1},
			indices:      []float32{0, 2},
			updatesShape: []int{2, 4},
			updates:      []float32{91, 92, 93, 94, 95, 96, 97, 98},
			want:         []float32{91, 92, 93, 94, 5, 6, 7, 8, 95, 96, 97, 98},
		},
		{
			name:         "2D scatter specific cells",
			dataShape:    []int{3, 4},
			data:         []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			indicesShape: []int{2, 2},
			indices:      []float32{0, 1, 2, 3},
			updatesShape: []int{2},
			updates:      []float32{77, 88},
			want:         []float32{1, 77, 3, 4, 5, 6, 7, 8, 9, 10, 11, 88},
		},
		{
			name:         "3D scatter slice",
			dataShape:    []int{2, 3, 2},
			data:         []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			indicesShape: []int{1, 1},
			indices:      []float32{1},
			updatesShape: []int{1, 3, 2},
			updates:      []float32{21, 22, 23, 24, 25, 26},
			want:         []float32{1, 2, 3, 4, 5, 6, 21, 22, 23, 24, 25, 26},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &ScatterND[float32]{engine: eng}

			dataTensor, err := tensor.New[float32](tt.dataShape, tt.data)
			if err != nil {
				t.Fatal(err)
			}
			idxTensor, err := tensor.New[float32](tt.indicesShape, tt.indices)
			if err != nil {
				t.Fatal(err)
			}
			updTensor, err := tensor.New[float32](tt.updatesShape, tt.updates)
			if err != nil {
				t.Fatal(err)
			}

			got, err := node.Forward(context.Background(), dataTensor, idxTensor, updTensor)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			result := got.Data()
			if len(result) != len(tt.want) {
				t.Fatalf("got len %d, want %d", len(result), len(tt.want))
			}
			for i, v := range result {
				if v != tt.want[i] {
					t.Errorf("index %d: got %v, want %v", i, v, tt.want[i])
				}
			}
		})
	}
}

func TestScatterND_GPUParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	gpuEng, err := compute.NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	cpuEng := compute.NewCPUEngine[float32](ops)

	tests := []struct {
		name         string
		dataShape    []int
		data         []float32
		indicesShape []int
		indices      []float32
		updatesShape []int
		updates      []float32
	}{
		{
			name:         "1D single scatter",
			dataShape:    []int{8},
			data:         []float32{1, 2, 3, 4, 5, 6, 7, 8},
			indicesShape: []int{1, 1},
			indices:      []float32{3},
			updatesShape: []int{1},
			updates:      []float32{99},
		},
		{
			name:         "2D row scatter",
			dataShape:    []int{4, 3},
			data:         []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			indicesShape: []int{2, 1},
			indices:      []float32{1, 3},
			updatesShape: []int{2, 3},
			updates:      []float32{51, 52, 53, 54, 55, 56},
		},
		{
			name:         "2D cell scatter",
			dataShape:    []int{3, 4},
			data:         []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			indicesShape: []int{3, 2},
			indices:      []float32{0, 0, 1, 2, 2, 3},
			updatesShape: []int{3},
			updates:      []float32{77, 88, 99},
		},
		{
			name:         "3D KV cache style",
			dataShape:    []int{2, 4, 3},
			data:         make([]float32, 24),
			indicesShape: []int{1, 1},
			indices:      []float32{0},
			updatesShape: []int{1, 4, 3},
			updates:      []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// CPU result.
			cpuNode := &ScatterND[float32]{engine: cpuEng}

			cpuData, _ := tensor.New[float32](tt.dataShape, tt.data)
			cpuIdx, _ := tensor.New[float32](tt.indicesShape, tt.indices)
			cpuUpd, _ := tensor.New[float32](tt.updatesShape, tt.updates)

			cpuOut, err := cpuNode.Forward(context.Background(), cpuData, cpuIdx, cpuUpd)
			if err != nil {
				t.Fatalf("CPU Forward: %v", err)
			}
			cpuResult := cpuOut.Data()

			// GPU result: data and updates on GPU, indices on CPU.
			gpuNode := &ScatterND[float32]{engine: gpuEng}

			gpuData, err := tensor.ToGPU(cpuData)
			if err != nil {
				t.Fatalf("ToGPU data: %v", err)
			}
			// Indices stay on CPU (ScatterND reads them on CPU).
			gpuIdx, _ := tensor.New[float32](tt.indicesShape, tt.indices)
			gpuUpd, err := tensor.ToGPU(cpuUpd)
			if err != nil {
				t.Fatalf("ToGPU updates: %v", err)
			}

			gpuOut, err := gpuNode.Forward(context.Background(), gpuData, gpuIdx, gpuUpd)
			if err != nil {
				t.Fatalf("GPU Forward: %v", err)
			}

			// Verify output is GPU-resident.
			if _, ok := gpuOut.GetStorage().(*tensor.GPUStorage[float32]); !ok {
				t.Fatalf("expected GPU-resident output, got %T", gpuOut.GetStorage())
			}

			gpuResult := gpuOut.Data()

			if len(gpuResult) != len(cpuResult) {
				t.Fatalf("length mismatch: GPU=%d CPU=%d", len(gpuResult), len(cpuResult))
			}
			for i := range cpuResult {
				if gpuResult[i] != cpuResult[i] {
					t.Errorf("index %d: GPU=%v CPU=%v", i, gpuResult[i], cpuResult[i])
				}
			}
		})
	}
}
