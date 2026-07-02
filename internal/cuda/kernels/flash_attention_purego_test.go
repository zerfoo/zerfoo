//go:build !cuda

package kernels

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// naiveAttention computes softmax(Q*K^T / sqrt(d)) * V on CPU for reference.
func naiveAttention(Q, K, V []float32, batch, heads, seqLen, headDim int, causal bool) []float32 {
	O := make([]float32, batch*heads*seqLen*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))

	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			bh := b*heads + h
			base := bh * seqLen * headDim

			for i := 0; i < seqLen; i++ {
				scores := make([]float64, seqLen)
				maxScore := -math.MaxFloat64

				for j := 0; j < seqLen; j++ {
					if causal && j > i {
						scores[j] = -math.MaxFloat64
						continue
					}
					dot := 0.0
					for d := 0; d < headDim; d++ {
						dot += float64(Q[base+i*headDim+d]) * float64(K[base+j*headDim+d])
					}
					scores[j] = dot * scale
					if scores[j] > maxScore {
						maxScore = scores[j]
					}
				}

				sum := 0.0
				for j := 0; j < seqLen; j++ {
					scores[j] = math.Exp(scores[j] - maxScore)
					sum += scores[j]
				}
				for j := 0; j < seqLen; j++ {
					scores[j] /= sum
				}

				for d := 0; d < headDim; d++ {
					acc := 0.0
					for j := 0; j < seqLen; j++ {
						acc += scores[j] * float64(V[base+j*headDim+d])
					}
					O[base+i*headDim+d] = float32(acc)
				}
			}
		}
	}
	return O
}

// naiveDecodeAttention computes the CPU reference for decode attention with GQA.
// K/V layout: [batch, kvLen, numKVHeads*headDim] (heads packed in dim).
// Q layout:   [batch*numQHeads, headDim] (one row per query head).
// O layout:   [batch*numQHeads, headDim].
func naiveDecodeAttention(Q, K, V []float32, batch, numQHeads, numKVHeads, kvLen, headDim int) []float32 {
	O := make([]float32, batch*numQHeads*headDim)
	scale := 1.0 / math.Sqrt(float64(headDim))
	headRatio := numQHeads / numKVHeads
	kvDim := numKVHeads * headDim

	for b := 0; b < batch; b++ {
		for qh := 0; qh < numQHeads; qh++ {
			bh := b*numQHeads + qh
			kvHead := qh / headRatio

			scores := make([]float64, kvLen)
			maxScore := -math.MaxFloat64
			for j := 0; j < kvLen; j++ {
				dot := 0.0
				for d := 0; d < headDim; d++ {
					dot += float64(Q[bh*headDim+d]) * float64(K[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
				}
				scores[j] = dot * scale
				if scores[j] > maxScore {
					maxScore = scores[j]
				}
			}

			sum := 0.0
			for j := 0; j < kvLen; j++ {
				scores[j] = math.Exp(scores[j] - maxScore)
				sum += scores[j]
			}
			for j := 0; j < kvLen; j++ {
				scores[j] /= sum
			}

			for d := 0; d < headDim; d++ {
				acc := 0.0
				for j := 0; j < kvLen; j++ {
					acc += scores[j] * float64(V[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
				}
				O[bh*headDim+d] = float32(acc)
			}
		}
	}
	return O
}

func TestFlashAttentionPuregoDecodeNonGQA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	batch, numQHeads, numKVHeads, headDim, kvLen := 1, 4, 4, 64, 32
	numBH := batch * numQHeads

	qSize := numBH * headDim
	kvSize := batch * numKVHeads * kvLen * headDim

	Q := make([]float32, qSize)
	K := make([]float32, kvSize)
	V := make([]float32, kvSize)

	for i := range Q {
		Q[i] = float32(i%7-3) * 0.1
	}
	for i := range K {
		K[i] = float32(i%5-2) * 0.1
		V[i] = float32(i%11-5) * 0.1
	}

	expected := naiveDecodeAttention(Q, K, V, batch, numQHeads, numKVHeads, kvLen, headDim)

	devQ, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()

	devK, err := cuda.Malloc(kvSize * 4)
	if err != nil {
		t.Fatalf("Malloc K: %v", err)
	}
	defer func() { _ = cuda.Free(devK) }()

	devV, err := cuda.Malloc(kvSize * 4)
	if err != nil {
		t.Fatalf("Malloc V: %v", err)
	}
	defer func() { _ = cuda.Free(devV) }()

	devO, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), qSize*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), kvSize*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), kvSize*4, cuda.MemcpyHostToDevice)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := FlashAttentionDecode(devQ, devK, devV, devO, numBH, kvLen, headDim, kvLen, nil, numQHeads, numKVHeads, stream.Ptr()); err != nil {
		t.Fatalf("FlashAttentionDecode: %v", err)
	}
	_ = stream.Synchronize()

	result := make([]float32, qSize)
	_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

	tol := 1e-4
	mismatches := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, result[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, qSize)
	}
}

func TestFlashAttentionPuregoDecodeGQA(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	batch, numQHeads, numKVHeads, headDim, kvLen := 1, 8, 4, 256, 32
	numBH := batch * numQHeads

	qSize := numBH * headDim
	kvSize := batch * numKVHeads * kvLen * headDim

	Q := make([]float32, qSize)
	K := make([]float32, kvSize)
	V := make([]float32, kvSize)

	for i := range Q {
		Q[i] = float32(i%7-3) * 0.1
	}
	for i := range K {
		K[i] = float32(i%5-2) * 0.1
		V[i] = float32(i%11-5) * 0.1
	}

	expected := naiveDecodeAttention(Q, K, V, batch, numQHeads, numKVHeads, kvLen, headDim)

	devQ, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()

	devK, err := cuda.Malloc(kvSize * 4)
	if err != nil {
		t.Fatalf("Malloc K: %v", err)
	}
	defer func() { _ = cuda.Free(devK) }()

	devV, err := cuda.Malloc(kvSize * 4)
	if err != nil {
		t.Fatalf("Malloc V: %v", err)
	}
	defer func() { _ = cuda.Free(devV) }()

	devO, err := cuda.Malloc(qSize * 4)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	_ = cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), qSize*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devK, unsafe.Pointer(&K[0]), kvSize*4, cuda.MemcpyHostToDevice)
	_ = cuda.Memcpy(devV, unsafe.Pointer(&V[0]), kvSize*4, cuda.MemcpyHostToDevice)

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := FlashAttentionDecode(devQ, devK, devV, devO, numBH, kvLen, headDim, kvLen, nil, numQHeads, numKVHeads, stream.Ptr()); err != nil {
		t.Fatalf("FlashAttentionDecode GQA: %v", err)
	}
	_ = stream.Synchronize()

	result := make([]float32, qSize)
	_ = cuda.Memcpy(unsafe.Pointer(&result[0]), devO, qSize*4, cuda.MemcpyDeviceToHost)

	tol := 1e-4
	mismatches := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, result[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d (GQA)", mismatches, qSize)
	}
}

// TestFlashAttentionPuregoParityNonCausal verifies the purego flash attention
// forward path matches naive CPU attention. Max rel error < 1e-4.
func TestFlashAttentionPuregoParityNonCausal(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	batch, heads, seqLen, headDim := 2, 2, 32, 16
	n := batch * heads * seqLen * headDim

	Q := make([]float32, n)
	K := make([]float32, n)
	V := make([]float32, n)
	for i := range Q {
		Q[i] = float32(i%7-3) * 0.1
		K[i] = float32(i%5-2) * 0.1
		V[i] = float32(i%11-5) * 0.1
	}

	expected := naiveAttention(Q, K, V, batch, heads, seqLen, headDim, false)

	byteSize := n * 4
	devQ, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()

	devK, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc K: %v", err)
	}
	defer func() { _ = cuda.Free(devK) }()

	devV, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc V: %v", err)
	}
	defer func() { _ = cuda.Free(devV) }()

	devO, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	if err := cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy Q: %v", err)
	}
	if err := cuda.Memcpy(devK, unsafe.Pointer(&K[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy K: %v", err)
	}
	if err := cuda.Memcpy(devV, unsafe.Pointer(&V[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy V: %v", err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := FlashAttentionForward(devQ, devK, devV, devO, batch, heads, seqLen, headDim, false, stream.Ptr()); err != nil {
		t.Fatalf("FlashAttentionForward: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	result := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devO, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	tol := 1e-4
	mismatches := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, result[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, n)
	}
}

// TestFlashAttentionPuregoParityCausal verifies the purego flash attention
// forward path with causal masking matches naive CPU attention. Max rel error < 1e-4.
func TestFlashAttentionPuregoParityCausal(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	batch, heads, seqLen, headDim := 1, 2, 32, 16
	n := batch * heads * seqLen * headDim

	Q := make([]float32, n)
	K := make([]float32, n)
	V := make([]float32, n)
	for i := range Q {
		Q[i] = float32(i%13-6) * 0.05
		K[i] = float32(i%9-4) * 0.05
		V[i] = float32(i%7-3) * 0.05
	}

	expected := naiveAttention(Q, K, V, batch, heads, seqLen, headDim, true)

	byteSize := n * 4
	devQ, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()

	devK, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc K: %v", err)
	}
	defer func() { _ = cuda.Free(devK) }()

	devV, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc V: %v", err)
	}
	defer func() { _ = cuda.Free(devV) }()

	devO, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	if err := cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy Q: %v", err)
	}
	if err := cuda.Memcpy(devK, unsafe.Pointer(&K[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy K: %v", err)
	}
	if err := cuda.Memcpy(devV, unsafe.Pointer(&V[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy V: %v", err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := FlashAttentionForward(devQ, devK, devV, devO, batch, heads, seqLen, headDim, true, stream.Ptr()); err != nil {
		t.Fatalf("FlashAttentionForward (causal): %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	result := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devO, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	tol := 1e-4
	mismatches := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, result[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches (causal): %d / %d", mismatches, n)
	}
}

// TestFlashAttentionPuregoParityWolfShape verifies the prefill kernel at the
// shape Wolf CrossAsset attention uses through FusedSDPA: 4 (batch*heads)
// independent slices, a short 12-token sequence, head_dim=64, non-causal.
// Short sequences (seq_len << BLOCK_SIZE) leave most threads of the single
// query tile idle and exercise the q_count/k_count edge handling.
func TestFlashAttentionPuregoParityWolfShape(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	batch, heads, seqLen, headDim := 4, 1, 12, 64
	n := batch * heads * seqLen * headDim

	Q := make([]float32, n)
	K := make([]float32, n)
	V := make([]float32, n)
	for i := range Q {
		Q[i] = float32(i%13-6) * 0.07
		K[i] = float32(i%9-4) * 0.06
		V[i] = float32(i%7-3) * 0.09
	}

	expected := naiveAttention(Q, K, V, batch, heads, seqLen, headDim, false)

	byteSize := n * 4
	devQ, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc Q: %v", err)
	}
	defer func() { _ = cuda.Free(devQ) }()
	devK, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc K: %v", err)
	}
	defer func() { _ = cuda.Free(devK) }()
	devV, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc V: %v", err)
	}
	defer func() { _ = cuda.Free(devV) }()
	devO, err := cuda.Malloc(byteSize)
	if err != nil {
		t.Fatalf("Malloc O: %v", err)
	}
	defer func() { _ = cuda.Free(devO) }()

	if err := cuda.Memcpy(devQ, unsafe.Pointer(&Q[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy Q: %v", err)
	}
	if err := cuda.Memcpy(devK, unsafe.Pointer(&K[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy K: %v", err)
	}
	if err := cuda.Memcpy(devV, unsafe.Pointer(&V[0]), byteSize, cuda.MemcpyHostToDevice); err != nil {
		t.Fatalf("Memcpy V: %v", err)
	}

	stream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	if err := FlashAttentionForward(devQ, devK, devV, devO, batch, heads, seqLen, headDim, false, stream.Ptr()); err != nil {
		t.Fatalf("FlashAttentionForward: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	result := make([]float32, n)
	if err := cuda.Memcpy(unsafe.Pointer(&result[0]), devO, byteSize, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	tol := 1e-4
	mismatches := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - expected[i]))
		if diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, result[i], expected[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, n)
	}
}
