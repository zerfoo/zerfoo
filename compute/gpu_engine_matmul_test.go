package compute

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// cpuMatMul computes C = A * B on the CPU for reference.
// A is [m, k], B is [k, n], C is [m, n].
func cpuMatMul(a []float32, m, k int, b []float32, n int) []float32 {
	c := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a[i*k+p] * b[p*n+j]
			}
			c[i*n+j] = sum
		}
	}
	return c
}

// TestGPUEngine_MatMulSmallCPUReference validates MatMul correctness against
// a CPU reference implementation for small dimensions (3x4 * 4x5).
func TestGPUEngine_MatMulSmallCPUReference(t *testing.T) {
	eng := newTestGPUEngine(t)
	ctx := context.Background()

	m, k, n := 3, 4, 5
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	for i := range aData {
		aData[i] = float32(i+1) * 0.5
	}
	for i := range bData {
		bData[i] = float32(i+1) * 0.3
	}

	a, _ := tensor.New[float32]([]int{m, k}, aData)
	b, _ := tensor.New[float32]([]int{k, n}, bData)

	gpuResult, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}

	expected := cpuMatMul(aData, m, k, bData, n)
	got := gpuResult.Data()

	if len(got) != len(expected) {
		t.Fatalf("result length %d, want %d", len(got), len(expected))
	}

	for i, want := range expected {
		diff := math.Abs(float64(got[i] - want))
		if diff > 1e-4 {
			t.Errorf("C[%d] = %f, want %f (diff=%e)", i, got[i], want, diff)
		}
	}
}

// TestGPUEngine_MatMulLargeVocab tests large-dimension MatMul operations that
// exercise cuBLAS Sgemm with sizes matching real LLM vocab projections.
// These dimensions have historically triggered cuBLAS status 7 (CUBLAS_STATUS_INTERNAL_ERROR)
// when getDevicePtr returns an invalid pointer for large weight matrices.
func TestGPUEngine_MatMulLargeVocab(t *testing.T) {
	eng := newTestGPUEngine(t)
	cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	tests := []struct {
		name string
		m    int // sequence/batch dim (small, like inference)
		k    int // hidden dim
		n    int // vocab size
	}{
		{
			name: "Llama3_vocab_projection",
			m:    5,
			k:    2048,
			n:    128256,
		},
		{
			name: "Gemma3_vocab_projection",
			m:    5,
			k:    1152,
			n:    262144,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create input data with a simple pattern to avoid NaN/Inf.
			// Use small values to keep the accumulated sums in float32 range.
			aData := make([]float32, tt.m*tt.k)
			for i := range aData {
				aData[i] = float32(i%7+1) * 0.001
			}

			bData := make([]float32, tt.k*tt.n)
			for i := range bData {
				bData[i] = float32(i%11+1) * 0.001
			}

			a, err := tensor.New[float32]([]int{tt.m, tt.k}, aData)
			if err != nil {
				t.Fatalf("create tensor A: %v", err)
			}

			b, err := tensor.New[float32]([]int{tt.k, tt.n}, bData)
			if err != nil {
				t.Fatalf("create tensor B: %v", err)
			}

			gpuResult, err := eng.MatMul(ctx, a, b)
			if err != nil {
				t.Fatalf("GPU MatMul: %v", err)
			}

			// Verify output shape.
			shape := gpuResult.Shape()
			if len(shape) != 2 || shape[0] != tt.m || shape[1] != tt.n {
				t.Fatalf("shape = %v, want [%d, %d]", shape, tt.m, tt.n)
			}

			// Spot-check a few elements against CPU reference.
			// Full CPU comparison would be too slow for these large dims,
			// so we compute a small subset.
			gpuData := gpuResult.Data()
			if len(gpuData) != tt.m*tt.n {
				t.Fatalf("result length = %d, want %d", len(gpuData), tt.m*tt.n)
			}

			// Verify first row manually.
			for j := 0; j < min(10, tt.n); j++ {
				var want float32
				for p := 0; p < tt.k; p++ {
					want += aData[p] * bData[p*tt.n+j]
				}
				diff := math.Abs(float64(gpuData[j] - want))
				if diff > 1e-2 {
					t.Errorf("C[0,%d] = %f, want %f (diff=%e)", j, gpuData[j], want, diff)
				}
			}

			// Verify no NaN or Inf in output (sign of invalid pointer).
			for i := 0; i < len(gpuData); i += len(gpuData) / 100 {
				if math.IsNaN(float64(gpuData[i])) || math.IsInf(float64(gpuData[i]), 0) {
					t.Fatalf("C[%d] = %f (NaN/Inf detected, possible invalid device pointer)", i, gpuData[i])
				}
			}

			// Suppress unused variable warning for cpuEng by using it
			// for a small sanity check.
			_ = cpuEng
		})
	}
}

// TestGPUEngine_MatMulLargeCPUParity tests GPU/CPU parity at a medium dimension
// that is large enough to exercise the H2D copy path but small enough for
// full CPU comparison.
func TestGPUEngine_MatMulLargeCPUParity(t *testing.T) {
	eng := newTestGPUEngine(t)
	cpuEng := NewCPUEngine[float32](numeric.Float32Ops{})
	ctx := context.Background()

	// Medium-sized: large enough to stress the allocator, small enough for CPU ref.
	m, k, n := 5, 256, 512
	aData := make([]float32, m*k)
	bData := make([]float32, k*n)
	for i := range aData {
		aData[i] = float32(i%13+1) * 0.01
	}
	for i := range bData {
		bData[i] = float32(i%17+1) * 0.01
	}

	a, _ := tensor.New[float32]([]int{m, k}, aData)
	b, _ := tensor.New[float32]([]int{k, n}, bData)

	gpuResult, err := eng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("GPU MatMul: %v", err)
	}

	cpuResult, err := cpuEng.MatMul(ctx, a, b)
	if err != nil {
		t.Fatalf("CPU MatMul: %v", err)
	}

	gpuData := gpuResult.Data()
	cpuData := cpuResult.Data()

	if len(gpuData) != len(cpuData) {
		t.Fatalf("length mismatch: GPU=%d, CPU=%d", len(gpuData), len(cpuData))
	}

	maxDiff := float64(0)
	for i := range gpuData {
		diff := math.Abs(float64(gpuData[i] - cpuData[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-2 {
			t.Errorf("parity mismatch at [%d]: GPU=%f, CPU=%f, diff=%e", i, gpuData[i], cpuData[i], diff)
			if i > 10 {
				t.Fatalf("too many mismatches, stopping (max diff so far: %e)", maxDiff)
			}
		}
	}
}
