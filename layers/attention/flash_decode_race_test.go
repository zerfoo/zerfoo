//go:build !(rocm && cutlass)

package attention

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/ztensor/tensor"
)

// naiveDecodeRef computes the CPU reference for single-query decode attention
// with GQA. Q layout [batch*numQHeads, headDim]; K/V layout
// [batch, kvLen, numKVHeads*headDim] (mirrors the kernel-package reference).
func naiveDecodeRef(q, k, v []float32, batch, numQHeads, numKVHeads, kvLen, headDim int) []float32 {
	o := make([]float32, batch*numQHeads*headDim)
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
					dot += float64(q[bh*headDim+d]) * float64(k[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
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
			for d := 0; d < headDim; d++ {
				acc := 0.0
				for j := 0; j < kvLen; j++ {
					acc += (scores[j] / sum) * float64(v[b*kvLen*kvDim+j*kvDim+kvHead*headDim+d])
				}
				o[bh*headDim+d] = float32(acc)
			}
		}
	}
	return o
}

// TestTryFlashDecodeEngineStreamParity exercises the #865 fix: Q/K/V are
// produced by asynchronous device-to-device copies enqueued on the engine
// stream, then decode is invoked WITH that engine stream. Launching the
// split-KV kernel on the engine stream orders it after those producers, so the
// output matches the CPU reference across the multi-split combine (kvLen >
// chunkSize) and the GQA head mapping.
//
// This is a correctness / wiring guard on the engine-stream path, NOT a
// red-on-pre-fix race reproduction. The cross-stream race the fix prevents
// cannot be surfaced by a self-contained unit test: tryFlashDecode allocates
// its output and split-KV scratch with cudaMalloc (tensor.NewGPUStorage), which
// device-synchronizes and drains every pending engine-stream producer before
// the kernel launches -- so an unordered private-stream launch still reads
// fully-written Q/K/V here. Verified empirically on the GB10 gate: forcing the
// pre-fix private-stream launch still passes even behind 8 GiB of queued async
// producer work. The race only manifests with a pooled engine (no per-call
// cudaMalloc), i.e. under real training -- the same reason the sibling
// flash-forward fix (#866) validated via a training A/B, not a unit test. See
// docs/devlog.md 2026-07-02 and docs/lore.md L-0006.
//
// GPU-only: without CUDA the decode wrapper bails to the non-fused path.
func TestTryFlashDecodeEngineStreamParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("CUDA not available (no GPU)")
	}

	const (
		batch      = 1
		numQHeads  = 4
		numKVHeads = 4
		headDim    = 64
		kvLen      = 512 // > chunkSize (256): exercises the multi-split combine.
	)
	numBH := batch * numQHeads
	kvDim := numKVHeads * headDim

	qSize := numBH * headDim
	kvSize := batch * kvLen * kvDim

	hostQ := make([]float32, qSize)
	hostK := make([]float32, kvSize)
	hostV := make([]float32, kvSize)
	for i := range hostQ {
		hostQ[i] = float32(i%7-3) * 0.1
	}
	for i := range hostK {
		hostK[i] = float32(i%5-2) * 0.1
		hostV[i] = float32(i%11-5) * 0.1
	}
	want := naiveDecodeRef(hostQ, hostK, hostV, batch, numQHeads, numKVHeads, kvLen, headDim)

	// Engine stream: the producing stream the decode kernel must order behind.
	engStream, err := cuda.CreateStream()
	if err != nil {
		t.Fatalf("CreateStream: %v", err)
	}
	defer func() { _ = engStream.Destroy() }()

	qStore, err := tensor.NewGPUStorage[float32](qSize)
	if err != nil {
		t.Fatalf("NewGPUStorage Q: %v", err)
	}
	kStore, err := tensor.NewGPUStorage[float32](kvSize)
	if err != nil {
		t.Fatalf("NewGPUStorage K: %v", err)
	}
	vStore, err := tensor.NewGPUStorage[float32](kvSize)
	if err != nil {
		t.Fatalf("NewGPUStorage V: %v", err)
	}

	// Stage the correct Q/K/V into device buffers (synchronous), then have the
	// engine stream copy them into the real targets device-to-device. A D2D
	// MemcpyAsync is genuinely asynchronous (an H2D copy from pageable Go memory
	// would run inline), so the producers land on the engine stream itself.
	stage := func(host []float32) unsafe.Pointer {
		t.Helper()
		p, err := cuda.Malloc(len(host) * 4)
		if err != nil {
			t.Fatalf("Malloc stage: %v", err)
		}
		if err := cuda.Memcpy(p, unsafe.Pointer(&host[0]), len(host)*4, cuda.MemcpyHostToDevice); err != nil {
			t.Fatalf("Memcpy stage: %v", err)
		}
		return p
	}
	stageQ := stage(hostQ)
	defer func() { _ = cuda.Free(stageQ) }()
	stageK := stage(hostK)
	defer func() { _ = cuda.Free(stageK) }()
	stageV := stage(hostV)
	defer func() { _ = cuda.Free(stageV) }()

	if err := cuda.MemcpyAsync(qStore.Ptr(), stageQ, qSize*4, cuda.MemcpyDeviceToDevice, engStream); err != nil {
		t.Fatalf("MemcpyAsync Q: %v", err)
	}
	if err := cuda.MemcpyAsync(kStore.Ptr(), stageK, kvSize*4, cuda.MemcpyDeviceToDevice, engStream); err != nil {
		t.Fatalf("MemcpyAsync K: %v", err)
	}
	if err := cuda.MemcpyAsync(vStore.Ptr(), stageV, kvSize*4, cuda.MemcpyDeviceToDevice, engStream); err != nil {
		t.Fatalf("MemcpyAsync V: %v", err)
	}

	q, err := tensor.NewWithStorage[float32]([]int{numBH, 1, headDim}, qStore)
	if err != nil {
		t.Fatalf("NewWithStorage Q: %v", err)
	}
	k, err := tensor.NewWithStorage[float32]([]int{batch, kvLen, kvDim}, kStore)
	if err != nil {
		t.Fatalf("NewWithStorage K: %v", err)
	}
	v, err := tensor.NewWithStorage[float32]([]int{batch, kvLen, kvDim}, vStore)
	if err != nil {
		t.Fatalf("NewWithStorage V: %v", err)
	}

	out, err := tryFlashDecode(q, k, v, headDim, numQHeads, numKVHeads, engStream.Ptr())
	if err != nil {
		t.Fatalf("tryFlashDecode: %v", err)
	}
	if out == nil {
		t.Fatal("tryFlashDecode bailed unexpectedly on a GPU decode shape")
	}

	got := make([]float32, qSize)
	outStore, ok := out.GetStorage().(*tensor.GPUStorage[float32])
	if !ok {
		t.Fatalf("output storage is %T, want *GPUStorage[float32]", out.GetStorage())
	}
	if err := cuda.Memcpy(unsafe.Pointer(&got[0]), outStore.Ptr(), qSize*4, cuda.MemcpyDeviceToHost); err != nil {
		t.Fatalf("Memcpy D2H: %v", err)
	}

	const tol = 1e-3
	mismatches := 0
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
			if mismatches < 5 {
				t.Errorf("output[%d] = %f, want %f (diff %e)", i, got[i], want[i], diff)
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Errorf("total mismatches: %d / %d", mismatches, qSize)
	}
}
