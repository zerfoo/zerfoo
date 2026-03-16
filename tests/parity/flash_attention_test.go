//go:build cuda && cutlass

package parity

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TestFlashAttentionParityGQA verifies that GQA with flash attention on GPU
// produces the same output as GQA with naive attention on CPU within tolerance.
func TestFlashAttentionParityGQA(t *testing.T) {
	batch, seqLen, modelDim := 1, 16, 32
	numQueryHeads, numKVHeads := 4, 4
	headDim := modelDim / numQueryHeads

	cpuOps := numeric.Float32Ops{}
	cpuEngine := compute.NewCPUEngine[float32](cpuOps)
	gpuEngine, err := compute.NewGPUEngine[float32](cpuOps, 0)
	if err != nil {
		t.Skipf("GPU engine not available: %v", err)
	}

	// Build CPU GQA.
	cpuGQA, err := attention.NewGroupedQueryAttention(cpuEngine, cpuOps, modelDim, numQueryHeads, numKVHeads)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention (CPU): %v", err)
	}

	// Build GPU GQA with same parameters.
	gpuGQA, err := attention.NewGroupedQueryAttention(gpuEngine, cpuOps, modelDim, numQueryHeads, numKVHeads)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention (GPU): %v", err)
	}

	// Copy weights from CPU GQA to GPU GQA.
	cpuParams := cpuGQA.Parameters()
	gpuParams := gpuGQA.Parameters()
	if len(cpuParams) != len(gpuParams) {
		t.Fatalf("param count mismatch: CPU=%d GPU=%d", len(cpuParams), len(gpuParams))
	}
	for i, cp := range cpuParams {
		gpuParams[i].Value = cpuTensorToGPU(t, cp.Value)
	}

	// Create input.
	n := batch * seqLen * modelDim
	inputData := make([]float32, n)
	for i := range inputData {
		inputData[i] = float32(i%13-6) * 0.02
	}

	cpuInput, err := tensor.New([]int{batch, seqLen, modelDim}, inputData)
	if err != nil {
		t.Fatalf("tensor.New (CPU): %v", err)
	}
	gpuInput := cpuTensorToGPU(t, cpuInput)

	// Forward pass.
	ctx := context.Background()
	cpuOut, err := cpuGQA.Forward(ctx, cpuInput)
	if err != nil {
		t.Fatalf("CPU Forward: %v", err)
	}

	gpuOut, err := gpuGQA.Forward(ctx, gpuInput)
	if err != nil {
		t.Fatalf("GPU Forward: %v", err)
	}

	// Compare.
	cpuData := cpuOut.Data()
	gpuData := gpuOut.Data()
	if len(cpuData) != len(gpuData) {
		t.Fatalf("output size mismatch: CPU=%d GPU=%d", len(cpuData), len(gpuData))
	}

	_ = headDim
	tol := float32(1e-3)
	mismatches := 0
	for i := range cpuData {
		diff := float32(math.Abs(float64(cpuData[i] - gpuData[i])))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d]: cpu=%f gpu=%f diff=%f", i, cpuData[i], gpuData[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, len(cpuData))
	}
}

// BenchmarkFlashAttention benchmarks flash attention at various sequence lengths.
func BenchmarkFlashAttention(b *testing.B) {
	seqLens := []int{128, 512, 1024, 2048}
	headDim := 64
	batchHeads := 4

	for _, seqLen := range seqLens {
		b.Run(seqLenName(seqLen), func(b *testing.B) {
			n := batchHeads * seqLen * headDim
			data := make([]float32, n)
			for i := range data {
				data[i] = float32(i%7-3) * 0.1
			}

			gpuQ := cpuSliceToGPU(b, data, []int{batchHeads, seqLen, headDim})
			gpuK := cpuSliceToGPU(b, data, []int{batchHeads, seqLen, headDim})
			gpuV := cpuSliceToGPU(b, data, []int{batchHeads, seqLen, headDim})

			ops := numeric.Float32Ops{}
			gpuEngine, err := compute.NewGPUEngine[float32](ops, 0)
			if err != nil {
				b.Skipf("GPU engine not available: %v", err)
			}

			sdpa := attention.NewScaledDotProductAttention[float32](gpuEngine, headDim)

			// Warm up.
			ctx := context.Background()
			if _, err := sdpa.Forward(ctx, gpuQ, gpuK, gpuV, nil); err != nil {
				b.Fatalf("warm-up: %v", err)
			}
			// Flash attention internally synchronizes via stream.

			b.ResetTimer()
			for range b.N {
				if _, err := sdpa.Forward(ctx, gpuQ, gpuK, gpuV, nil); err != nil {
					b.Fatalf("Forward: %v", err)
				}
			}
			// Flash attention internally synchronizes via stream.
		})
	}
}

func seqLenName(seqLen int) string {
	switch {
	case seqLen >= 1024:
		return string(rune('0'+seqLen/1024)) + "k"
	default:
		return string(rune('0'+seqLen/100)) + "xx"
	}
}

func cpuTensorToGPU[T tensor.Numeric](t testing.TB, src *tensor.TensorNumeric[T]) *tensor.TensorNumeric[T] {
	t.Helper()
	dst, err := tensor.ToGPU(src)
	if err != nil {
		t.Fatalf("ToGPU: %v", err)
	}
	return dst
}

func cpuSliceToGPU(b testing.TB, data []float32, shape []int) *tensor.TensorNumeric[float32] {
	b.Helper()
	t, err := tensor.New(shape, data)
	if err != nil {
		b.Fatalf("tensor.New: %v", err)
	}
	gpu, err := tensor.ToGPU(t)
	if err != nil {
		b.Fatalf("ToGPU: %v", err)
	}
	return gpu
}
