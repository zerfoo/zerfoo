package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestTryFlashForwardFallback verifies that without CUTLASS, tryFlashForward
// always returns (nil, nil) and the SDPA falls through to naive attention.
func TestTryFlashForwardFallback(t *testing.T) {
	q, _ := tensor.New([]int{2, 4, 8}, make([]float32, 2*4*8))
	k, _ := tensor.New([]int{2, 4, 8}, make([]float32, 2*4*8))
	v, _ := tensor.New([]int{2, 4, 8}, make([]float32, 2*4*8))

	result, err := tryFlashForward(q, k, v, 8, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Fatal("expected nil result from fallback tryFlashForward")
	}
}

// TestSDPAFlashFallbackParity verifies that SDPA produces correct output via
// the naive path when flash attention falls back (non-CUDA build).
func TestSDPAFlashFallbackParity(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	headDim := 8
	sdpa := NewScaledDotProductAttention[float32](engine, headDim)

	batchHeads := 2
	seqLen := 4

	// Small deterministic values.
	qData := make([]float32, batchHeads*seqLen*headDim)
	kData := make([]float32, batchHeads*seqLen*headDim)
	vData := make([]float32, batchHeads*seqLen*headDim)
	for i := range qData {
		qData[i] = float32(i%7-3) * 0.1
		kData[i] = float32(i%5-2) * 0.1
		vData[i] = float32(i%11-5) * 0.1
	}

	q, err := tensor.New([]int{batchHeads, seqLen, headDim}, qData)
	if err != nil {
		t.Fatalf("tensor Q: %v", err)
	}
	k, err := tensor.New([]int{batchHeads, seqLen, headDim}, kData)
	if err != nil {
		t.Fatalf("tensor K: %v", err)
	}
	v, err := tensor.New([]int{batchHeads, seqLen, headDim}, vData)
	if err != nil {
		t.Fatalf("tensor V: %v", err)
	}

	result, err := sdpa.Forward(context.Background(), q, k, v, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Compute expected via CPU reference.
	expected := naiveSoftmaxAttention(qData, kData, vData, batchHeads, seqLen, headDim)

	resultData := result.Data()
	tol := float32(1e-5)
	mismatches := 0
	for i := range resultData {
		diff := float32(math.Abs(float64(resultData[i] - expected[i])))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %f)", i, resultData[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, len(resultData))
	}
}

// naiveSoftmaxAttention computes softmax(Q*K^T / sqrt(d)) * V on CPU.
// Q, K, V are [batchHeads, seqLen, headDim] in row-major order.
func naiveSoftmaxAttention(qIn, kIn, vIn []float32, batchHeads, seqLen, headDim int) []float32 {
	out := make([]float32, batchHeads*seqLen*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))

	for b := range batchHeads {
		base := b * seqLen * headDim
		for i := range seqLen {
			scores := make([]float64, seqLen)
			maxScore := -math.MaxFloat64

			for j := range seqLen {
				dot := 0.0
				for d := range headDim {
					dot += float64(qIn[base+i*headDim+d]) * float64(kIn[base+j*headDim+d])
				}
				scores[j] = dot * scale
				if scores[j] > maxScore {
					maxScore = scores[j]
				}
			}

			sum := 0.0
			for j := range seqLen {
				scores[j] = math.Exp(scores[j] - maxScore)
				sum += scores[j]
			}
			for j := range seqLen {
				scores[j] /= sum
			}

			for d := range headDim {
				acc := 0.0
				for j := range seqLen {
					acc += scores[j] * float64(vIn[base+j*headDim+d])
				}
				out[base+i*headDim+d] = float32(acc)
			}
		}
	}
	return out
}
