package generate

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func newTestEngine() compute.Engine[float32] {
	return compute.NewCPUEngine(numeric.Float32Ops{})
}

func TestCompressedKVCache_NewAndNumLayers(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 4, 8, 64, 32)
	if got := cache.NumLayers(); got != 4 {
		t.Errorf("NumLayers() = %d, want 4", got)
	}
}

func TestCompressedKVCache_GetEmpty(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 2, 8, 64, 32)
	_, ok := cache.Get(0)
	if ok {
		t.Error("Get(0) on empty cache should return false")
	}
}

func TestCompressedKVCache_GetOutOfRange(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 2, 8, 64, 32)
	_, ok := cache.Get(5)
	if ok {
		t.Error("Get(5) with 2 layers should return false")
	}
	_, ok = cache.Get(-1)
	if ok {
		t.Error("Get(-1) should return false")
	}
}

func TestCompressedKVCache_UpdateWithinChunk(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 1, 1, 4, 64)

	k := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update error: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true after update")
	}

	// Still within chunk, no compression. Shape: [1, 1, 4].
	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != 4 {
		t.Errorf("Key shape = %v, want [1 1 4]", shape)
	}

	data := lkv.Key.Data()
	want := []float32{1, 2, 3, 4}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, data[i], w)
		}
	}
}

func TestCompressedKVCache_ChunkCompression(t *testing.T) {
	engine := newTestEngine()
	const dim = 2
	const chunkSize = 2
	cache := NewCompressedKVCache[float32](engine, 1, 1, dim, chunkSize)

	// Insert 2 tokens to fill a chunk.
	k1 := makeTensor(t, []int{1, 1, dim}, []float32{2, 4})
	v1 := makeTensor(t, []int{1, 1, dim}, []float32{6, 8})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}

	k2 := makeTensor(t, []int{1, 1, dim}, []float32{4, 6})
	v2 := makeTensor(t, []int{1, 1, dim}, []float32{10, 12})
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatal(err)
	}

	// Chunk is full and compressed. Get should return [1, 1, 2] (one compressed chunk).
	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [1 1 %d]", shape, dim)
	}

	// Mean of [2,4] and [4,6] = [3, 5].
	kData := lkv.Key.Data()
	wantK := []float32{3, 5}
	for i, w := range wantK {
		if kData[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, kData[i], w)
		}
	}

	// Mean of [6,8] and [10,12] = [8, 10].
	vData := lkv.Value.Data()
	wantV := []float32{8, 10}
	for i, w := range wantV {
		if vData[i] != w {
			t.Errorf("Value data[%d] = %v, want %v", i, vData[i], w)
		}
	}
}

func TestCompressedKVCache_ChunkPlusRecent(t *testing.T) {
	engine := newTestEngine()
	const dim = 2
	const chunkSize = 2
	cache := NewCompressedKVCache[float32](engine, 1, 1, dim, chunkSize)

	// Fill one chunk (2 tokens) + 1 recent token.
	tokens := [][]float32{
		{2, 4}, // chunk 0, token 0
		{4, 6}, // chunk 0, token 1 → triggers compression
		{10, 20}, // recent token
	}
	for i, tok := range tokens {
		k := makeTensor(t, []int{1, 1, dim}, tok)
		v := makeTensor(t, []int{1, 1, dim}, tok)
		if err := cache.Update(0, k, v); err != nil {
			t.Fatalf("Update(%d): %v", i, err)
		}
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	// Shape: [1, 2, 2] — 1 compressed chunk + 1 recent token.
	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [1 2 %d]", shape, dim)
	}

	// First position: mean of [2,4] and [4,6] = [3, 5].
	// Second position: recent [10, 20].
	data := lkv.Key.Data()
	want := []float32{3, 5, 10, 20}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, data[i], w)
		}
	}
}

func TestCompressedKVCache_128Tokens_ChunkSize64(t *testing.T) {
	engine := newTestEngine()
	const (
		batch     = 1
		dim       = 4
		chunkSize = 64
		numTokens = 128
	)
	cache := NewCompressedKVCache[float32](engine, 1, 1, dim, chunkSize)

	// Insert 128 tokens.
	for i := range numTokens {
		kData := make([]float32, dim)
		vData := make([]float32, dim)
		for d := range dim {
			kData[d] = float32(i*dim + d)
			vData[d] = float32(i*dim + d + 1000)
		}
		k := makeTensor(t, []int{batch, 1, dim}, kData)
		v := makeTensor(t, []int{batch, 1, dim}, vData)
		if err := cache.Update(0, k, v); err != nil {
			t.Fatalf("Update(%d): %v", i, err)
		}
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	// 128 tokens / 64 chunkSize = 2 compressed chunks, 0 recent.
	// Shape should be [batch, 2, dim].
	shape := lkv.Key.Shape()
	if shape[0] != batch || shape[1] != 2 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [%d 2 %d]", shape, batch, dim)
	}

	vShape := lkv.Value.Shape()
	if vShape[0] != batch || vShape[1] != 2 || vShape[2] != dim {
		t.Errorf("Value shape = %v, want [%d 2 %d]", vShape, batch, dim)
	}

	// Verify chunk 0 mean: tokens 0..63, each token d-th element = i*dim+d.
	// Mean of d-th element across tokens 0..63: mean(d, d+dim, d+2*dim, ..., d+63*dim)
	//   = d + dim * mean(0..63) = d + dim * 31.5
	kData := lkv.Key.Data()
	for d := range dim {
		expected := float32(d) + float32(dim)*31.5
		if kData[d] != expected {
			t.Errorf("chunk0 Key[%d] = %v, want %v", d, kData[d], expected)
		}
	}

	// Verify chunk 1 mean: tokens 64..127.
	// d-th element of token i = i*dim+d.
	// Mean = d + dim * mean(64..127) = d + dim * 95.5
	for d := range dim {
		expected := float32(d) + float32(dim)*95.5
		got := kData[dim+d]
		if got != expected {
			t.Errorf("chunk1 Key[%d] = %v, want %v", d, got, expected)
		}
	}
}

func TestCompressedKVCache_SeqLen(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 1, 1, 2, 4)

	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() on empty = %d, want 0", got)
	}

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if got := cache.SeqLen(); got != 1 {
		t.Errorf("SeqLen() after 1 token = %d, want 1", got)
	}

	// Fill chunk (3 more tokens to reach chunkSize=4).
	for range 3 {
		if err := cache.Update(0, k, v); err != nil {
			t.Fatal(err)
		}
	}
	// 4 tokens stored, compressed into 1 chunk.
	if got := cache.SeqLen(); got != 4 {
		t.Errorf("SeqLen() after 4 tokens = %d, want 4", got)
	}

	// Add one more.
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if got := cache.SeqLen(); got != 5 {
		t.Errorf("SeqLen() after 5 tokens = %d, want 5", got)
	}
}

func TestCompressedKVCache_Reset(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 1, 1, 2, 2)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	cache.Reset()

	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() after Reset = %d, want 0", got)
	}
	_, ok := cache.Get(0)
	if ok {
		t.Error("Get(0) after Reset should return false")
	}
}

func TestCompressedKVCache_UpdateOutOfRange(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 1, 1, 2, 4)
	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Update(5, k, v); err == nil {
		t.Error("Update with out-of-range layer should return error")
	}
	if err := cache.Update(-1, k, v); err == nil {
		t.Error("Update with negative layer should return error")
	}
}

func TestCompressedKVCache_DimensionMismatch(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 1, 1, 4, 4)

	k1 := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}

	k2 := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v2 := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k2, v2); err == nil {
		t.Error("Update with mismatched dimensions should return error")
	}
}

func TestCompressedKVCache_Non3DTensor(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 1, 1, 2, 4)
	k, _ := tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	v, _ := tensor.New([]int{2, 3}, []float32{7, 8, 9, 10, 11, 12})
	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected error for 2D tensors")
	}
}

func TestCompressedKVCache_MultiLayer(t *testing.T) {
	engine := newTestEngine()
	cache := NewCompressedKVCache[float32](engine, 3, 1, 2, 4)

	for layer := range 3 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer), 0})
		v := makeTensor(t, []int{1, 1, 2}, []float32{0, float32(layer)})
		if err := cache.Update(layer, k, v); err != nil {
			t.Fatalf("Update layer %d: %v", layer, err)
		}
	}

	for layer := range 3 {
		lkv, ok := cache.Get(layer)
		if !ok {
			t.Errorf("Get(%d) should return true", layer)
			continue
		}
		data := lkv.Key.Data()
		if data[0] != float32(layer) {
			t.Errorf("Layer %d Key[0] = %v, want %v", layer, data[0], float32(layer))
		}
	}
}

func TestCompressedKVCache_BatchUpdate(t *testing.T) {
	engine := newTestEngine()
	const dim = 2
	const chunkSize = 2
	cache := NewCompressedKVCache[float32](engine, 1, 1, dim, chunkSize)

	// Batch=2, insert 2 tokens to fill chunk.
	k1 := makeTensor(t, []int{2, 1, dim}, []float32{2, 4, 6, 8})
	v1 := makeTensor(t, []int{2, 1, dim}, []float32{1, 1, 1, 1})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}

	k2 := makeTensor(t, []int{2, 1, dim}, []float32{4, 6, 10, 12})
	v2 := makeTensor(t, []int{2, 1, dim}, []float32{1, 1, 1, 1})
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatal(err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 2 || shape[1] != 1 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [2 1 %d]", shape, dim)
	}

	// Batch 0: mean([2,4], [4,6]) = [3, 5].
	// Batch 1: mean([6,8], [10,12]) = [8, 10].
	data := lkv.Key.Data()
	want := []float32{3, 5, 8, 10}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, data[i], w)
		}
	}
}

func TestCompressedKVCache_BulkInsert(t *testing.T) {
	engine := newTestEngine()
	const dim = 2
	const chunkSize = 2
	cache := NewCompressedKVCache[float32](engine, 1, 1, dim, chunkSize)

	// Insert 2 tokens at once (seq_len=2), should fill and compress a chunk.
	k := makeTensor(t, []int{1, 2, dim}, []float32{2, 4, 4, 6})
	v := makeTensor(t, []int{1, 2, dim}, []float32{10, 20, 30, 40})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 1 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [1 1 %d]", shape, dim)
	}

	// Mean of [2,4] and [4,6] = [3, 5].
	data := lkv.Key.Data()
	if data[0] != 3 || data[1] != 5 {
		t.Errorf("Key data = %v, want [3 5]", data)
	}
}

func TestCompressedKVCache_Truncate(t *testing.T) {
	engine := newTestEngine()
	const dim = 2
	const chunkSize = 2
	cache := NewCompressedKVCache[float32](engine, 1, 1, dim, chunkSize)

	// Insert 5 tokens: 2 chunks compressed + 1 recent.
	for i := range 5 {
		k := makeTensor(t, []int{1, 1, dim}, []float32{float32(i), float32(i)})
		v := makeTensor(t, []int{1, 1, dim}, []float32{float32(i), float32(i)})
		if err := cache.Update(0, k, v); err != nil {
			t.Fatal(err)
		}
	}
	if got := cache.SeqLen(); got != 5 {
		t.Fatalf("SeqLen() = %d, want 5", got)
	}

	// Truncate to 3 tokens: 1 full chunk (2 tokens) + 1 recent.
	cache.Truncate(3)
	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen() after Truncate(3) = %d, want 3", got)
	}

	// Truncate to 0 clears everything.
	cache.Truncate(0)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() after Truncate(0) = %d, want 0", got)
	}
}

func TestCompressedKVCache_CacheProviderCompliance(t *testing.T) {
	// Compile-time interface assertion.
	var _ CacheProvider[float32] = (*CompressedKVCache[float32])(nil)

	// Exercise all CacheProvider methods through the interface.
	engine := newTestEngine()
	var cache CacheProvider[float32] = NewCompressedKVCache[float32](engine, 2, 1, 2, 2)

	if cache.SeqLen() != 0 {
		t.Fatalf("SeqLen() = %d, want 0", cache.SeqLen())
	}

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}
	if s := lkv.Key.Shape(); s[0] != 1 || s[1] != 1 || s[2] != 2 {
		t.Errorf("Key shape = %v, want [1 1 2]", s)
	}

	if cache.SeqLen() != 1 {
		t.Errorf("SeqLen() = %d, want 1", cache.SeqLen())
	}

	cache.Truncate(0)
	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen() after Truncate(0) = %d, want 0", cache.SeqLen())
	}

	// Re-populate and reset.
	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update after truncate: %v", err)
	}
	cache.Reset()
	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen() after Reset = %d, want 0", cache.SeqLen())
	}
}

func TestCompressedKVCache_ChunkSizeGreaterThanSeqLen(t *testing.T) {
	engine := newTestEngine()
	const dim = 2
	const chunkSize = 100 // larger than the number of tokens we'll insert
	cache := NewCompressedKVCache[float32](engine, 1, 1, dim, chunkSize)

	// Insert 3 tokens, all within a single chunk (no compression).
	tokens := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	for i, tok := range tokens {
		k := makeTensor(t, []int{1, 1, dim}, tok)
		v := makeTensor(t, []int{1, 1, dim}, tok)
		if err := cache.Update(0, k, v); err != nil {
			t.Fatalf("Update(%d): %v", i, err)
		}
	}

	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen() = %d, want 3", got)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	// No compression happened — shape [1, 3, 2], raw tokens returned.
	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 3 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [1 3 %d]", shape, dim)
	}

	data := lkv.Key.Data()
	want := []float32{1, 2, 3, 4, 5, 6}
	for i, w := range want {
		if data[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, data[i], w)
		}
	}
}

func TestWithCompressedKV_CreatesCompressedKVCache(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  100,
		MaxSeqLen:  512,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	eng := newTestEngine()
	gen := NewGenerator[float32](nil, nil, eng, cfg, WithCompressedKV(64))

	if gen.compressedKVChunkSize != 64 {
		t.Errorf("compressedKVChunkSize = %d, want 64", gen.compressedKVChunkSize)
	}
}

func TestWithCompressedKV_DefaultChunkSize(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  100,
		MaxSeqLen:  512,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	eng := newTestEngine()

	// chunkSize <= 0 defaults to 64.
	gen := NewGenerator[float32](nil, nil, eng, cfg, WithCompressedKV(0))
	if gen.compressedKVChunkSize != 64 {
		t.Errorf("compressedKVChunkSize = %d, want 64 (default)", gen.compressedKVChunkSize)
	}

	gen = NewGenerator[float32](nil, nil, eng, cfg, WithCompressedKV(-10))
	if gen.compressedKVChunkSize != 64 {
		t.Errorf("compressedKVChunkSize = %d, want 64 (default for negative)", gen.compressedKVChunkSize)
	}
}

func TestWithCompressedKV_DefaultCacheWithoutOption(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  100,
		MaxSeqLen:  512,
		EOSTokenID: 2,
		NumLayers:  4,
	}
	eng := newTestEngine()
	gen := NewGenerator[float32](nil, nil, eng, cfg)

	if gen.compressedKVChunkSize != 0 {
		t.Errorf("compressedKVChunkSize = %d, want 0 (disabled)", gen.compressedKVChunkSize)
	}
}
