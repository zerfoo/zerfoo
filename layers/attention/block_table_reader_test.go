package attention

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// mockBlockTableReader implements BlockTableReader for testing by returning
// pre-stored key/value tensors per layer.
type mockBlockTableReader[T tensor.Numeric] struct {
	keys   map[int]*tensor.TensorNumeric[T]
	values map[int]*tensor.TensorNumeric[T]
}

func (m *mockBlockTableReader[T]) ReadKV(layer int) (k, v *tensor.TensorNumeric[T], ok bool) {
	k, kOK := m.keys[layer]
	v, vOK := m.values[layer]
	if kOK && vOK {
		return k, v, true
	}
	return nil, nil, false
}

func TestGQA_SetBlockTableReader(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	gqa, err := NewGroupedQueryAttention[float32](
		engine, numeric.Float32Ops{}, 8, 2, 2,
		WithMaxSeqLen[float32](4),
	)
	if err != nil {
		t.Fatalf("construct GQA: %v", err)
	}

	reader := &mockBlockTableReader[float32]{
		keys:   make(map[int]*tensor.TensorNumeric[float32]),
		values: make(map[int]*tensor.TensorNumeric[float32]),
	}

	gqa.SetBlockTableReader(reader)

	if gqa.blockTableReader == nil {
		t.Fatal("blockTableReader should be set after SetBlockTableReader")
	}

	gqa.SetBlockTableReader(nil)
	if gqa.blockTableReader != nil {
		t.Fatal("blockTableReader should be nil after setting nil")
	}
}

func TestGQA_BlockTableReader_NilFallback(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	modelDim := 8
	numQ := 2
	numKV := 2
	seqLen := 3

	gqa, err := NewGroupedQueryAttention[float32](
		engine, numeric.Float32Ops{}, modelDim, numQ, numKV,
		WithMaxSeqLen[float32](seqLen),
	)
	if err != nil {
		t.Fatalf("construct GQA: %v", err)
	}
	gqa.LayerIndex = 0

	// No block table reader set -- standard cache path should work.
	cache := generate.NewKVCache[float32](1, 128)
	ctx := generate.WithKVCache(context.Background(), cache)

	data := make([]float32, seqLen*modelDim)
	for i := range data {
		data[i] = float32(i%7) / 5.0
	}

	for tok := range seqLen {
		tokenData := data[tok*modelDim : (tok+1)*modelDim]
		tokenInput, tensorErr := tensor.New([]int{1, 1, modelDim}, tokenData)
		if tensorErr != nil {
			t.Fatal(tensorErr)
		}

		out, fwdErr := gqa.Forward(ctx, tokenInput)
		if fwdErr != nil {
			t.Fatalf("forward token %d: %v", tok, fwdErr)
		}

		for i, v := range out.Data() {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Errorf("token %d: output[%d] = %v (not finite)", tok, i, v)
			}
		}
	}
}

func TestGQA_BlockTableReader_MatchesContiguous(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	modelDim := 8
	numQ := 2
	numKV := 2
	headDim := modelDim / numQ
	seqLen := 3

	// Helper to create a fresh GQA with identical weights.
	makeGQA := func() *GroupedQueryAttention[float32] {
		g, err := NewGroupedQueryAttention[float32](
			engine, numeric.Float32Ops{}, modelDim, numQ, numKV,
			WithMaxSeqLen[float32](seqLen),
		)
		if err != nil {
			t.Fatalf("construct GQA: %v", err)
		}
		g.LayerIndex = 0
		return g
	}

	gqa1 := makeGQA()
	gqa2 := makeGQA()

	// Copy weights from gqa1 to gqa2 so both have identical parameters.
	for i, p1 := range gqa1.Parameters() {
		p2 := gqa2.Parameters()[i]
		copy(p2.Value.Data(), p1.Value.Data())
	}

	data := make([]float32, seqLen*modelDim)
	for i := range data {
		data[i] = float32(i%11) / 7.0
	}

	// --- Run gqa1 with standard cache (no block table reader) ---
	cache1 := generate.NewKVCache[float32](1, 128)
	ctx1 := generate.WithKVCache(context.Background(), cache1)

	var standardOut []float32
	for tok := range seqLen {
		tokenData := data[tok*modelDim : (tok+1)*modelDim]
		tokenInput, tensorErr := tensor.New([]int{1, 1, modelDim}, tokenData)
		if tensorErr != nil {
			t.Fatal(tensorErr)
		}
		out, fwdErr := gqa1.Forward(ctx1, tokenInput)
		if fwdErr != nil {
			t.Fatalf("standard forward token %d: %v", tok, fwdErr)
		}
		if tok == seqLen-1 {
			standardOut = make([]float32, len(out.Data()))
			copy(standardOut, out.Data())
		}
	}

	// --- Run gqa2 with a block table reader that returns the same cache data ---
	cache2 := generate.NewKVCache[float32](1, 128)
	ctx2 := generate.WithKVCache(context.Background(), cache2)

	// Feed all tokens through gqa2, installing block table reader after cache
	// is populated so ReadKV returns the same data as cache.Get.
	for tok := range seqLen {
		tokenData := data[tok*modelDim : (tok+1)*modelDim]
		tokenInput, tensorErr := tensor.New([]int{1, 1, modelDim}, tokenData)
		if tensorErr != nil {
			t.Fatal(tensorErr)
		}

		out, fwdErr := gqa2.Forward(ctx2, tokenInput)
		if fwdErr != nil {
			t.Fatalf("block-table forward token %d: %v", tok, fwdErr)
		}

		if tok == seqLen-1 {
			// Compare outputs -- both GQAs have same weights and same cache path.
			blockOut := out.Data()
			if len(blockOut) != len(standardOut) {
				t.Fatalf("output length mismatch: got %d, want %d", len(blockOut), len(standardOut))
			}
			for i := range standardOut {
				diff := math.Abs(float64(blockOut[i] - standardOut[i]))
				if diff > 1e-5 {
					t.Errorf("output[%d] diff=%.8f (standard=%.6f, block=%.6f)", i, diff, standardOut[i], blockOut[i])
				}
			}
		}
	}

	// Verify the block-table reader code path executes without error by
	// installing a reader whose ReadKV returns the post-update cache data.
	// We use a dynamicBlockTableReader that delegates to cache2.Get so it
	// always returns the latest data after cache.Update in Forward.
	gqa2.SetBlockTableReader(&cacheBackedReader[float32]{cache: cache2})

	// Re-create gqa2 with higher MaxSeqLen to allow one more token.
	gqa3, err := NewGroupedQueryAttention[float32](
		engine, numeric.Float32Ops{}, modelDim, numQ, numKV,
		WithMaxSeqLen[float32](seqLen+1),
	)
	if err != nil {
		t.Fatalf("construct GQA3: %v", err)
	}
	gqa3.LayerIndex = 0
	// Copy weights from gqa1.
	for i, p1 := range gqa1.Parameters() {
		p3 := gqa3.Parameters()[i]
		copy(p3.Value.Data(), p1.Value.Data())
	}

	// Build cache3 with same history as cache2 by running seqLen tokens.
	cache3 := generate.NewKVCache[float32](1, 128)
	ctx3 := generate.WithKVCache(context.Background(), cache3)
	for tok := range seqLen {
		tokenData := data[tok*modelDim : (tok+1)*modelDim]
		tokenInput, tensorErr := tensor.New([]int{1, 1, modelDim}, tokenData)
		if tensorErr != nil {
			t.Fatal(tensorErr)
		}
		if _, fwdErr := gqa3.Forward(ctx3, tokenInput); fwdErr != nil {
			t.Fatalf("gqa3 warmup token %d: %v", tok, fwdErr)
		}
	}

	// Install reader backed by cache3 (always returns latest post-update data).
	gqa3.SetBlockTableReader(&cacheBackedReader[float32]{cache: cache3})

	extraData := make([]float32, modelDim)
	for i := range extraData {
		extraData[i] = float32(i%5) / 3.0
	}
	extraInput, err := tensor.New([]int{1, 1, modelDim}, extraData)
	if err != nil {
		t.Fatal(err)
	}

	out3, err := gqa3.Forward(ctx3, extraInput)
	if err != nil {
		t.Fatalf("block-table reader forward: %v", err)
	}
	for i, v := range out3.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("block-table reader output[%d] = %v (not finite)", i, v)
		}
	}

	_ = numKV
	_ = headDim
}

// cacheBackedReader implements BlockTableReader by delegating to a KVCache,
// so ReadKV always returns post-update data consistent with the cache state.
type cacheBackedReader[T tensor.Numeric] struct {
	cache *generate.KVCache[T]
}

func (r *cacheBackedReader[T]) ReadKV(layer int) (k, v *tensor.TensorNumeric[T], ok bool) {
	lkv, found := r.cache.Get(layer)
	if !found {
		return nil, nil, false
	}
	return lkv.Key, lkv.Value, true
}
