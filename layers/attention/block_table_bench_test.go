// Benchmark results (Intel i7-6660U, 2.40GHz, darwin/amd64):
//
// BenchmarkGQA_GatherCopy/seq128-4    2101   2396345 ns/op   302297 B/op   379 allocs/op
// BenchmarkGQA_GatherCopy/seq512-4     471   4890086 ns/op  1036580 B/op   382 allocs/op
// BenchmarkGQA_GatherCopy/seq1024-4    266  10938976 ns/op  1981026 B/op   384 allocs/op
// BenchmarkGQA_BlockTable/seq128-4     883   2887880 ns/op   302320 B/op   380 allocs/op
// BenchmarkGQA_BlockTable/seq512-4     373   7681644 ns/op  1036601 B/op   383 allocs/op
// BenchmarkGQA_BlockTable/seq1024-4    177  20486216 ns/op  1981047 B/op   385 allocs/op
//
// Note: BlockTable is currently slower because contextCacheReader.ReadKV
// calls PagedKVCache.GetKV which still gathers blocks into a contiguous
// tensor (same work as the gather-copy path). A true zero-copy block-table
// reader that passes block pointers directly to SDPA would eliminate this
// overhead.
package attention

import (
	"context"
	"fmt"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// benchGQASetup creates a GQA layer with small dimensions, pre-fills a cache
// with seqLen tokens, and returns the GQA, context, single-token input, and
// the prefill length for truncation between iterations.
func benchGQASetup(b *testing.B, seqLen int, paged bool) (
	gqa *GroupedQueryAttention[float32],
	ctx context.Context,
	tokenInput *tensor.TensorNumeric[float32],
	prefillLen int,
) {
	b.Helper()

	const (
		modelDim = 64
		numQ     = 4
		numKV    = 2
		headDim  = modelDim / numQ // 16
	)

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	var err error
	gqa, err = NewGroupedQueryAttention[float32](
		engine, numeric.Float32Ops{}, modelDim, numQ, numKV,
		WithMaxSeqLen[float32](seqLen+128),
	)
	if err != nil {
		b.Fatalf("construct GQA: %v", err)
	}
	gqa.LayerIndex = 0

	if paged {
		pool, poolErr := generate.NewBlockPool[float32](1, 16, numKV*headDim, 64)
		if poolErr != nil {
			b.Fatalf("NewBlockPool: %v", poolErr)
		}
		cache := generate.NewPagedKVCache[float32](pool, 1)
		ctx = generate.WithCache(context.Background(), generate.CacheProvider[float32](cache))
	} else {
		cache := generate.NewKVCache[float32](1, seqLen+128)
		ctx = generate.WithKVCache(context.Background(), cache)
	}

	// Pre-fill the cache with seqLen tokens.
	data := make([]float32, modelDim)
	for i := range data {
		data[i] = float32(i%11) / 7.0
	}
	for tok := range seqLen {
		inp, tensorErr := tensor.New([]int{1, 1, modelDim}, data)
		if tensorErr != nil {
			b.Fatalf("tensor token %d: %v", tok, tensorErr)
		}
		if _, fwdErr := gqa.Forward(ctx, inp); fwdErr != nil {
			b.Fatalf("prefill token %d: %v", tok, fwdErr)
		}
	}

	// Build the single-token input for the benchmark loop.
	tokenInput, err = tensor.New([]int{1, 1, modelDim}, data)
	if err != nil {
		b.Fatalf("tensor: %v", err)
	}

	return gqa, ctx, tokenInput, seqLen
}

func BenchmarkGQA_GatherCopy(b *testing.B) {
	for _, seqLen := range []int{128, 512, 1024} {
		b.Run(fmt.Sprintf("seq%d", seqLen), func(b *testing.B) {
			gqa, ctx, inp, prefill := benchGQASetup(b, seqLen, false)
			cache, _ := generate.GetCache[float32](ctx)
			b.ResetTimer()
			b.ReportAllocs()
			for range b.N {
				if _, err := gqa.Forward(ctx, inp); err != nil {
					b.Fatal(err)
				}
				cache.Truncate(prefill)
			}
		})
	}
}

func BenchmarkGQA_BlockTable(b *testing.B) {
	for _, seqLen := range []int{128, 512, 1024} {
		b.Run(fmt.Sprintf("seq%d", seqLen), func(b *testing.B) {
			gqa, ctx, inp, prefill := benchGQASetup(b, seqLen, true)
			cache, ok := generate.GetCache[float32](ctx)
			if !ok {
				b.Fatal("expected cache in context")
			}
			gqa.SetBlockTableReader(&contextCacheReader[float32]{cache: cache})

			b.ResetTimer()
			b.ReportAllocs()
			for range b.N {
				if _, err := gqa.Forward(ctx, inp); err != nil {
					b.Fatal(err)
				}
				cache.Truncate(prefill)
			}
		})
	}
}

// contextCacheReader implements BlockTableReader by delegating to a
// CacheProvider (works with both KVCache and PagedKVCache).
type contextCacheReader[T tensor.Numeric] struct {
	cache generate.CacheProvider[T]
}

func (r *contextCacheReader[T]) ReadKV(layer int) (k, v *tensor.TensorNumeric[T], ok bool) {
	lkv, found := r.cache.Get(layer)
	if !found {
		return nil, nil, false
	}
	return lkv.Key, lkv.Value, true
}
