package generate

import (
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// TestPagedKV_MemoryEfficiency compares memory usage between pre-allocated
// KVCache and PagedKVCache for 8 concurrent sequences of varying lengths.
// Paged allocation should use <= 50% of pre-allocated memory for mixed lengths.
func TestPagedKV_MemoryEfficiency(t *testing.T) {
	const (
		numLayers = 32
		headDim   = 128
		maxSeqLen = 2048
		blockSize = 16
		numKVH    = 4 // GQA KV heads
	)
	poolHD := numKVH * headDim // per-position storage

	seqLens := []int{128, 256, 512, 512, 1024, 1024, 2048, 2048}

	// Pre-allocated: each sequence gets numLayers * maxSeqLen * poolHD * 2 (K+V).
	elemSize := int(unsafe.Sizeof(float32(0)))
	preAllocPerSeq := numLayers * maxSeqLen * poolHD * 2 * elemSize
	preAllocTotal := len(seqLens) * preAllocPerSeq

	// Paged: blocks of blockSize tokens. Each sequence uses ceil(seqLen/blockSize) blocks.
	blockElems := numLayers * blockSize * poolHD
	blockBytes := blockElems * 2 * elemSize // K + V
	totalBlocks := 0
	for _, sl := range seqLens {
		blocks := (sl + blockSize - 1) / blockSize
		totalBlocks += blocks
	}
	pagedTotal := totalBlocks * blockBytes

	ratio := float64(pagedTotal) / float64(preAllocTotal)

	t.Logf("Pre-allocated total: %d MB", preAllocTotal/(1024*1024))
	t.Logf("Paged total:         %d MB", pagedTotal/(1024*1024))
	t.Logf("Ratio (paged/pre):   %.2f", ratio)

	if ratio > 0.50 {
		t.Errorf("paged/pre-allocated ratio = %.2f, want <= 0.50", ratio)
	}

	// Functional verification: actually allocate and use paged caches.
	maxMB := pagedTotal/(1024*1024) + 1
	pool, err := NewBlockPool[float32](numLayers, blockSize, poolHD, maxMB)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	caches := make([]*PagedKVCache[float32], len(seqLens))
	for i, sl := range seqLens {
		caches[i] = NewPagedKVCache[float32](pool, numLayers)

		// Append sl tokens (one at a time for layer 0 only to verify allocation).
		kData := make([]float32, numKVH*headDim)
		vData := make([]float32, numKVH*headDim)
		k, _ := tensor.New([]int{numKVH, 1, headDim}, kData)
		v, _ := tensor.New([]int{numKVH, 1, headDim}, vData)

		for tok := range sl {
			for layer := range numLayers {
				if err := caches[i].Append(layer, k, v); err != nil {
					t.Fatalf("seq %d, layer %d, tok %d: %v", i, layer, tok, err)
				}
			}
			_ = tok
		}

		if got := caches[i].SeqLen(); got != sl {
			t.Errorf("seq %d: SeqLen = %d, want %d", i, got, sl)
		}
	}

	// Report blocks used.
	blocksUsed := pool.Cap() - pool.Available()
	t.Logf("Blocks used: %d / %d", blocksUsed, pool.Cap())

	// Clean up.
	for _, c := range caches {
		c.Free()
	}
	if pool.Available() != pool.Cap() {
		t.Errorf("not all blocks returned: available=%d, cap=%d", pool.Available(), pool.Cap())
	}
}

// BenchmarkPagedKVCache_GQAAppend benchmarks appending with multi-channel
// tensors (GQA-like: numKVHeads channels).
func BenchmarkPagedKVCache_GQAAppend(b *testing.B) {
	const (
		layers    = 32
		blockSize = 16
		numKVH    = 4
		headDim   = 128
		poolHD    = numKVH * headDim
	)
	pool, err := NewBlockPool[float32](layers, blockSize, poolHD, 256)
	if err != nil {
		b.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, layers)

	kData := make([]float32, numKVH*headDim)
	vData := make([]float32, numKVH*headDim)
	k, _ := tensor.New([]int{numKVH, 1, headDim}, kData)
	v, _ := tensor.New([]int{numKVH, 1, headDim}, vData)

	b.ResetTimer()
	b.ReportAllocs()
	for i := range b.N {
		if i%(blockSize*50) == 0 && i > 0 {
			cache.Free()
			cache = NewPagedKVCache[float32](pool, layers)
		}
		for layer := range layers {
			if err := cache.Append(layer, k, v); err != nil {
				b.Fatal(err)
			}
		}
	}
	_ = cache.SeqLen() // prevent dead code elimination
}
