package generate

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BenchmarkDecodeToken measures the cost of a single KV cache update
// and retrieval cycle that mirrors the GQA decode hot path. This
// provides a CI-checkable baseline: if someone adds an O(maxSeqLen)
// operation to the decode path, the benchmark will show a clear
// regression in ns/op.
func BenchmarkDecodeToken(b *testing.B) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})

	modelDim := 64
	headDim := 8
	numKVHeads := 4
	numQueryHeads := 16
	maxSeqLen := 2048
	kvDim := numKVHeads * headDim

	cache := NewKVCache[float32](1, maxSeqLen)

	// Pre-fill with 128 tokens to simulate a warm cache.
	for i := range 128 {
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		for j := range kData {
			kData[j] = float32(i*kvDim+j) * 0.001
			vData[j] = float32(i*kvDim+j) * 0.002
		}
		k, err := tensor.New([]int{numKVHeads, 1, headDim}, kData)
		if err != nil {
			b.Fatal(err)
		}
		v, err := tensor.New([]int{numKVHeads, 1, headDim}, vData)
		if err != nil {
			b.Fatal(err)
		}
		if err := cache.Update(0, k, v); err != nil {
			b.Fatal(err)
		}
	}

	ctx := context.Background()
	replicationFactor := numQueryHeads / numKVHeads
	_ = replicationFactor
	_ = modelDim

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Simulate decode: update cache with 1 new token.
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		k, err := tensor.New([]int{numKVHeads, 1, headDim}, kData)
		if err != nil {
			b.Fatal(err)
		}
		v, err := tensor.New([]int{numKVHeads, 1, headDim}, vData)
		if err != nil {
			b.Fatal(err)
		}

		if err := cache.Update(0, k, v); err != nil {
			// Cache is full, reset and re-prefill.
			cache.Reset()
			for j := range 128 {
				kPre := make([]float32, kvDim)
				vPre := make([]float32, kvDim)
				for idx := range kPre {
					kPre[idx] = float32(j*kvDim+idx) * 0.001
					vPre[idx] = float32(j*kvDim+idx) * 0.002
				}
				kt, tErr := tensor.New([]int{numKVHeads, 1, headDim}, kPre)
				if tErr != nil {
					b.Fatal(tErr)
				}
				vt, tErr := tensor.New([]int{numKVHeads, 1, headDim}, vPre)
				if tErr != nil {
					b.Fatal(tErr)
				}
				if err := cache.Update(0, kt, vt); err != nil {
					b.Fatal(err)
				}
			}
			continue
		}

		// Get cached KV and simulate head expansion (the hot path).
		lkv, ok := cache.Get(0)
		if !ok {
			b.Fatal("cache.Get(0) returned false")
		}

		// Repeat K/V for GQA head expansion (the operation that regressed).
		if numQueryHeads != numKVHeads && numKVHeads > 1 {
			_, err := engine.Repeat(ctx, lkv.Key, 1, replicationFactor)
			if err != nil {
				b.Fatal(err)
			}
			_, err = engine.Repeat(ctx, lkv.Value, 1, replicationFactor)
			if err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkDecodeTokenMHA benchmarks the MHA (non-GQA) decode path
// where no Repeat is needed. This serves as the baseline against
// which GQA overhead can be measured.
func BenchmarkDecodeTokenMHA(b *testing.B) {
	modelDim := 64
	headDim := 8
	numKVHeads := 8
	maxSeqLen := 2048
	kvDim := numKVHeads * headDim

	cache := NewKVCache[float32](1, maxSeqLen)

	for i := range 128 {
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		for j := range kData {
			kData[j] = float32(i*kvDim+j) * 0.001
			vData[j] = float32(i*kvDim+j) * 0.002
		}
		k, err := tensor.New([]int{numKVHeads, 1, headDim}, kData)
		if err != nil {
			b.Fatal(err)
		}
		v, err := tensor.New([]int{numKVHeads, 1, headDim}, vData)
		if err != nil {
			b.Fatal(err)
		}
		if err := cache.Update(0, k, v); err != nil {
			b.Fatal(err)
		}
	}

	_ = modelDim

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		k, err := tensor.New([]int{numKVHeads, 1, headDim}, kData)
		if err != nil {
			b.Fatal(err)
		}
		v, err := tensor.New([]int{numKVHeads, 1, headDim}, vData)
		if err != nil {
			b.Fatal(err)
		}

		if err := cache.Update(0, k, v); err != nil {
			cache.Reset()
			for j := range 128 {
				kPre := make([]float32, kvDim)
				vPre := make([]float32, kvDim)
				for idx := range kPre {
					kPre[idx] = float32(j*kvDim+idx) * 0.001
					vPre[idx] = float32(j*kvDim+idx) * 0.002
				}
				kt, tErr := tensor.New([]int{numKVHeads, 1, headDim}, kPre)
				if tErr != nil {
					b.Fatal(tErr)
				}
				vt, tErr := tensor.New([]int{numKVHeads, 1, headDim}, vPre)
				if tErr != nil {
					b.Fatal(tErr)
				}
				if err := cache.Update(0, kt, vt); err != nil {
					b.Fatal(err)
				}
			}
			continue
		}

		// No Repeat needed for MHA.
		_, ok := cache.Get(0)
		if !ok {
			b.Fatal("cache.Get(0) returned false")
		}
	}
}
