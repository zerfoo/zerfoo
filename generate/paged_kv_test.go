package generate

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestPagedKVCache_NewAndEmpty(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 4, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	cache := NewPagedKVCache[float32](pool, 2)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() on empty cache = %d, want 0", got)
	}
}

func TestPagedKVCache_GetEmpty(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 4, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 2)

	_, ok := cache.GetKV(0)
	if ok {
		t.Error("GetKV(0) on empty cache should return false")
	}
}

func TestPagedKVCache_AppendSingleToken(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 16, 4, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 2)

	// Append one token to layer 0: [1, 1, 4]
	k := makeTestTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeTestTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Append(0, k, v); err != nil {
		t.Fatalf("Append: %v", err)
	}

	if got := cache.SeqLen(); got != 1 {
		t.Errorf("SeqLen() = %d, want 1", got)
	}

	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true after append")
	}

	if got := lkv.Key.Shape(); got[0] != 1 || got[1] != 1 || got[2] != 4 {
		t.Errorf("Key shape = %v, want [1 1 4]", got)
	}

	kd := lkv.Key.Data()
	want := []float32{1, 2, 3, 4}
	for i, w := range want {
		if kd[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, kd[i], w)
		}
	}

	vd := lkv.Value.Data()
	wantV := []float32{5, 6, 7, 8}
	for i, w := range wantV {
		if vd[i] != w {
			t.Errorf("Value data[%d] = %v, want %v", i, vd[i], w)
		}
	}
}

func TestPagedKVCache_AppendMultipleTokens(t *testing.T) {
	pool, err := NewBlockPool[float32](1, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	// Append 3 tokens one at a time.
	for i := range 3 {
		k := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(i * 2), float32(i*2 + 1)})
		v := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(i*2 + 10), float32(i*2 + 11)})
		if err := cache.Append(0, k, v); err != nil {
			t.Fatalf("Append(%d): %v", i, err)
		}
	}

	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen() = %d, want 3", got)
	}

	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 3 || shape[2] != 2 {
		t.Errorf("Key shape = %v, want [1 3 2]", shape)
	}

	kd := lkv.Key.Data()
	wantK := []float32{0, 1, 2, 3, 4, 5}
	for i, w := range wantK {
		if kd[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, kd[i], w)
		}
	}
}

func TestPagedKVCache_BlockBoundary(t *testing.T) {
	// blockSize=4, fill exactly 4 tokens (fills one block), then add a 5th.
	pool, err := NewBlockPool[float32](1, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	for i := range 5 {
		k := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(i), 0})
		v := makeTestTensor(t, []int{1, 1, 2}, []float32{0, float32(i)})
		if err := cache.Append(0, k, v); err != nil {
			t.Fatalf("Append(%d): %v", i, err)
		}
	}

	if got := cache.SeqLen(); got != 5 {
		t.Errorf("SeqLen() = %d, want 5", got)
	}

	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true")
	}

	shape := lkv.Key.Shape()
	if shape[1] != 5 {
		t.Errorf("Key seq_len = %d, want 5", shape[1])
	}

	// Verify data integrity across block boundary.
	kd := lkv.Key.Data()
	for i := range 5 {
		if kd[i*2] != float32(i) {
			t.Errorf("Key[%d][0] = %v, want %v", i, kd[i*2], float32(i))
		}
	}
}

func TestPagedKVCache_MultiLayer(t *testing.T) {
	pool, err := NewBlockPool[float32](3, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 3)

	// Append one token per layer.
	for layer := range 3 {
		k := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(layer), 0})
		v := makeTestTensor(t, []int{1, 1, 2}, []float32{0, float32(layer)})
		if err := cache.Append(layer, k, v); err != nil {
			t.Fatalf("Append layer %d: %v", layer, err)
		}
	}

	for layer := range 3 {
		lkv, ok := cache.GetKV(layer)
		if !ok {
			t.Errorf("GetKV(%d) should return true", layer)
			continue
		}
		kd := lkv.Key.Data()
		if kd[0] != float32(layer) {
			t.Errorf("Layer %d Key[0] = %v, want %v", layer, kd[0], float32(layer))
		}
	}
}

func TestPagedKVCache_Free(t *testing.T) {
	pool, err := NewBlockPool[float32](1, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}

	availBefore := pool.Available()

	cache := NewPagedKVCache[float32](pool, 1)

	// Append 5 tokens (will use 2 blocks: 4 + 1).
	for i := range 5 {
		k := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(i), 0})
		v := makeTestTensor(t, []int{1, 1, 2}, []float32{0, float32(i)})
		if err := cache.Append(0, k, v); err != nil {
			t.Fatalf("Append(%d): %v", i, err)
		}
	}

	blocksUsed := availBefore - pool.Available()
	if blocksUsed != 2 {
		t.Errorf("blocks used = %d, want 2", blocksUsed)
	}

	cache.Free()

	if got := pool.Available(); got != availBefore {
		t.Errorf("Available() after Free = %d, want %d", got, availBefore)
	}

	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() after Free = %d, want 0", got)
	}
}

func TestPagedKVCache_ManyTokens(t *testing.T) {
	const n = 100
	pool, err := NewBlockPool[float32](2, 16, 4, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 2)

	for i := range n {
		for layer := range 2 {
			k := makeTestTensor(t, []int{1, 1, 4}, []float32{float32(i), float32(layer), 0, 0})
			v := makeTestTensor(t, []int{1, 1, 4}, []float32{0, 0, float32(i), float32(layer)})
			if err := cache.Append(layer, k, v); err != nil {
				t.Fatalf("Append(layer=%d, token=%d): %v", layer, i, err)
			}
		}
	}

	if got := cache.SeqLen(); got != n {
		t.Errorf("SeqLen() = %d, want %d", got, n)
	}

	// Verify first and last entries for layer 0.
	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true")
	}
	shape := lkv.Key.Shape()
	if shape[1] != n {
		t.Errorf("Key seq_len = %d, want %d", shape[1], n)
	}

	kd := lkv.Key.Data()
	if kd[0] != 0 { // first token, first elem
		t.Errorf("Key[0][0] = %v, want 0", kd[0])
	}
	last := (n - 1) * 4
	if kd[last] != float32(n-1) {
		t.Errorf("Key[%d][0] = %v, want %v", n-1, kd[last], float32(n-1))
	}
}

func TestPagedKVCache_AppendChannelsDimMismatch(t *testing.T) {
	// Pool headDim=2, tensor [2, 1, 2] has channels*dim=4 != 2.
	pool, err := NewBlockPool[float32](1, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	k := makeTestTensor(t, []int{2, 1, 2}, []float32{1, 2, 3, 4})
	v := makeTestTensor(t, []int{2, 1, 2}, []float32{5, 6, 7, 8})
	if err := cache.Append(0, k, v); err == nil {
		t.Error("Append with channels*dim != headDim should return error")
	}
}

func TestPagedKVCache_MultiChannel(t *testing.T) {
	// Simulate GQA with 2 KV heads, headDim=3: pool headDim = 2*3 = 6.
	const (
		numKVHeads = 2
		headDim    = 3
		poolHD     = numKVHeads * headDim // 6
	)
	pool, err := NewBlockPool[float32](1, 4, poolHD, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	// Prefill 2 tokens: shape [2, 2, 3] (channels=2, seqLen=2, dim=3).
	// Layout: [ch0_pos0(3), ch0_pos1(3), ch1_pos0(3), ch1_pos1(3)]
	kData := []float32{
		1, 2, 3, // ch0, pos0
		4, 5, 6, // ch0, pos1
		7, 8, 9, // ch1, pos0
		10, 11, 12, // ch1, pos1
	}
	vData := []float32{
		21, 22, 23,
		24, 25, 26,
		27, 28, 29,
		30, 31, 32,
	}
	k := makeTestTensor(t, []int{2, 2, 3}, kData)
	v := makeTestTensor(t, []int{2, 2, 3}, vData)

	if err := cache.Append(0, k, v); err != nil {
		t.Fatalf("Append prefill: %v", err)
	}
	if got := cache.SeqLen(); got != 2 {
		t.Errorf("SeqLen() = %d, want 2", got)
	}

	// Decode 1 more token: shape [2, 1, 3].
	kDec := makeTestTensor(t, []int{2, 1, 3}, []float32{13, 14, 15, 16, 17, 18})
	vDec := makeTestTensor(t, []int{2, 1, 3}, []float32{33, 34, 35, 36, 37, 38})

	if err := cache.Append(0, kDec, vDec); err != nil {
		t.Fatalf("Append decode: %v", err)
	}
	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen() = %d, want 3", got)
	}

	// GetKV should return [2, 3, 3].
	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true")
	}
	shape := lkv.Key.Shape()
	if shape[0] != 2 || shape[1] != 3 || shape[2] != 3 {
		t.Fatalf("Key shape = %v, want [2 3 3]", shape)
	}

	// Verify K data: [ch0: pos0,pos1,pos2 | ch1: pos0,pos1,pos2]
	gotK := lkv.Key.Data()
	wantK := []float32{
		1, 2, 3, 4, 5, 6, 13, 14, 15, // ch0
		7, 8, 9, 10, 11, 12, 16, 17, 18, // ch1
	}
	for i, w := range wantK {
		if gotK[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, gotK[i], w)
		}
	}

	gotV := lkv.Value.Data()
	wantV := []float32{
		21, 22, 23, 24, 25, 26, 33, 34, 35,
		27, 28, 29, 30, 31, 32, 36, 37, 38,
	}
	for i, w := range wantV {
		if gotV[i] != w {
			t.Errorf("Value data[%d] = %v, want %v", i, gotV[i], w)
		}
	}
}

func TestPagedKVCache_MultiChannelLayoutChange(t *testing.T) {
	// After first Append with channels=2, a second Append with channels=1 should error.
	pool, err := NewBlockPool[float32](1, 4, 4, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	k1 := makeTestTensor(t, []int{2, 1, 2}, []float32{1, 2, 3, 4})
	v1 := makeTestTensor(t, []int{2, 1, 2}, []float32{5, 6, 7, 8})
	if err := cache.Append(0, k1, v1); err != nil {
		t.Fatalf("First Append: %v", err)
	}

	// channels=1, dim=4 => channels*dim=4 matches headDim, but layout differs.
	k2 := makeTestTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v2 := makeTestTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})
	if err := cache.Append(0, k2, v2); err == nil {
		t.Error("Append with changed channel layout should return error")
	}
}

func TestPagedKVCache_AppendLayerOutOfRange(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 2)

	k := makeTestTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTestTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Append(5, k, v); err == nil {
		t.Error("Append with out-of-range layer should return error")
	}
	if err := cache.Append(-1, k, v); err == nil {
		t.Error("Append with negative layer should return error")
	}
}

func TestPagedKVCache_GetOutOfRange(t *testing.T) {
	pool, err := NewBlockPool[float32](2, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 2)

	_, ok := cache.GetKV(5)
	if ok {
		t.Error("GetKV(5) with 2 layers should return false")
	}
	_, ok = cache.GetKV(-1)
	if ok {
		t.Error("GetKV(-1) should return false")
	}
}

func TestPagedKVCache_AppendMultipleSeqLen(t *testing.T) {
	// Append a tensor with seqLen > 1 (e.g., prefill with 3 tokens at once).
	pool, err := NewBlockPool[float32](1, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	k := makeTestTensor(t, []int{1, 3, 2}, []float32{1, 2, 3, 4, 5, 6})
	v := makeTestTensor(t, []int{1, 3, 2}, []float32{7, 8, 9, 10, 11, 12})
	if err := cache.Append(0, k, v); err != nil {
		t.Fatalf("Append: %v", err)
	}

	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen() = %d, want 3", got)
	}

	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true")
	}

	kd := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4, 5, 6}
	for i, w := range wantK {
		if kd[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, kd[i], w)
		}
	}
}

func TestPagedKVCache_PrefillThenDecode(t *testing.T) {
	// Simulate prefill (3 tokens at once) then decode (1 token at a time).
	pool, err := NewBlockPool[float32](1, 4, 2, 1)
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, 1)

	// Prefill 3 tokens.
	k := makeTestTensor(t, []int{1, 3, 2}, []float32{1, 2, 3, 4, 5, 6})
	v := makeTestTensor(t, []int{1, 3, 2}, []float32{10, 20, 30, 40, 50, 60})
	if err := cache.Append(0, k, v); err != nil {
		t.Fatalf("Append prefill: %v", err)
	}

	// Decode 2 more tokens.
	for i := range 2 {
		k := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(7 + i*2), float32(8 + i*2)})
		v := makeTestTensor(t, []int{1, 1, 2}, []float32{float32(70 + i*10), float32(80 + i*10)})
		if err := cache.Append(0, k, v); err != nil {
			t.Fatalf("Append decode %d: %v", i, err)
		}
	}

	if got := cache.SeqLen(); got != 5 {
		t.Errorf("SeqLen() = %d, want 5", got)
	}

	lkv, ok := cache.GetKV(0)
	if !ok {
		t.Fatal("GetKV(0) should return true")
	}
	shape := lkv.Key.Shape()
	if shape[1] != 5 {
		t.Errorf("Key seq_len = %d, want 5", shape[1])
	}

	kd := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	for i, w := range wantK {
		if kd[i] != w {
			t.Errorf("Key data[%d] = %v, want %v", i, kd[i], w)
		}
	}
}

func BenchmarkPagedKVCache_Append(b *testing.B) {
	const (
		layers    = 32
		blockSize = 16
		headDim   = 128
	)
	pool, err := NewBlockPool[float32](layers, blockSize, headDim, 256)
	if err != nil {
		b.Fatalf("NewBlockPool: %v", err)
	}
	cache := NewPagedKVCache[float32](pool, layers)

	kData := make([]float32, headDim)
	vData := make([]float32, headDim)
	for i := range kData {
		kData[i] = float32(i)
		vData[i] = float32(i)
	}
	k, _ := tensor.New([]int{1, 1, headDim}, kData)
	v, _ := tensor.New([]int{1, 1, headDim}, vData)

	b.ResetTimer()
	b.ReportAllocs()
	for i := range b.N {
		// Reset cache periodically to avoid pool exhaustion.
		if i%(blockSize*100) == 0 && i > 0 {
			cache.Free()
			cache = NewPagedKVCache[float32](pool, layers)
		}
		for layer := range layers {
			if err := cache.Append(layer, k, v); err != nil {
				b.Fatal(err)
			}
		}
	}
}

func makeTestTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatalf("tensor.New(%v): %v", shape, err)
	}
	return tn
}
