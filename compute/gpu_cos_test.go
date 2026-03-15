package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestGPUEngine_CosParity verifies GPU Cos matches CPU Cos across various shapes.
func TestGPUEngine_CosParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available")
	}

	ops := numeric.Float32Ops{}
	gpuEng, err := NewGPUEngine[float32](ops)
	if err != nil {
		t.Fatalf("NewGPUEngine: %v", err)
	}
	defer func() { _ = gpuEng.Close() }()

	cpuEng := NewCPUEngine[float32](ops)
	ctx := context.Background()

	tests := []struct {
		name  string
		shape []int
	}{
		{"1D_4", []int{4}},
		{"2D_2x3", []int{2, 3}},
		{"3D_1x1x2048", []int{1, 1, 2048}},
		{"3D_3x4x5", []int{3, 4, 5}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := 1
			for _, d := range tc.shape {
				n *= d
			}
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i) * 0.1
			}

			a, err := tensor.New[float32](tc.shape, data)
			if err != nil {
				t.Fatalf("tensor.New: %v", err)
			}

			gpuOut, err := gpuEng.Cos(ctx, a)
			if err != nil {
				t.Fatalf("GPU Cos: %v", err)
			}
			cpuOut, err := cpuEng.Cos(ctx, a)
			if err != nil {
				t.Fatalf("CPU Cos: %v", err)
			}

			gData := gpuOut.Data()
			cData := cpuOut.Data()
			if len(gData) != len(cData) {
				t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gData), len(cData))
			}

			for i := range gData {
				diff := math.Abs(float64(gData[i] - cData[i]))
				if diff > 1e-6 {
					t.Errorf("[%d] GPU=%v, CPU=%v, diff=%e", i, gData[i], cData[i], diff)
				}
			}

			// Verify shape is preserved.
			gs := gpuOut.Shape()
			if len(gs) != len(tc.shape) {
				t.Fatalf("shape rank mismatch: got %d, want %d", len(gs), len(tc.shape))
			}
			for i := range gs {
				if gs[i] != tc.shape[i] {
					t.Errorf("shape[%d]: got %d, want %d", i, gs[i], tc.shape[i])
				}
			}

			// Spot-check: cos(0) = 1.
			if data[0] == 0 && math.Abs(float64(gData[0]-1.0)) > 1e-6 {
				t.Errorf("cos(0) = %v, want 1.0", gData[0])
			}
		})
	}
}
