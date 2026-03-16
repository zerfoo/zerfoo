package generate

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func makeTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatalf("tensor.New(%v) error: %v", shape, err)
	}
	return tn
}

func TestKVCache_NewAndNumLayers(t *testing.T) {
	cache := NewKVCache[float32](4, 128)
	if got := cache.NumLayers(); got != 4 {
		t.Errorf("NumLayers() = %d, want 4", got)
	}
}

func TestKVCache_GetEmpty(t *testing.T) {
	cache := NewKVCache[float32](2, 128)
	_, ok := cache.Get(0)
	if ok {
		t.Error("Get(0) on empty cache should return false")
	}
}

func TestKVCache_GetOutOfRange(t *testing.T) {
	cache := NewKVCache[float32](2, 128)
	_, ok := cache.Get(5)
	if ok {
		t.Error("Get(5) with 2 layers should return false")
	}
	_, ok = cache.Get(-1)
	if ok {
		t.Error("Get(-1) should return false")
	}
}

func TestKVCache_UpdateAndGet(t *testing.T) {
	cache := NewKVCache[float32](2, 128)

	// First update: [batch=1, seq=1, dim=4]
	k1 := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update(0) error: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true after update")
	}
	if got := lkv.Key.Shape(); got[1] != 1 {
		t.Errorf("Key seq_len = %d, want 1", got[1])
	}
}

func TestKVCache_UpdateConcat(t *testing.T) {
	cache := NewKVCache[float32](1, 128)

	// First token: [batch=1, seq=1, dim=2]
	k1 := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v1 := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update error: %v", err)
	}

	// Second token: [batch=1, seq=1, dim=2]
	k2 := makeTensor(t, []int{1, 1, 2}, []float32{5, 6})
	v2 := makeTensor(t, []int{1, 1, 2}, []float32{7, 8})
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Update error: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	// Key should now be [1, 2, 2]
	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != 2 {
		t.Errorf("Key shape = %v, want [1 2 2]", shape)
	}

	// Verify data: [1,2] then [5,6]
	data := lkv.Key.Data()
	want := []float32{1, 2, 5, 6}
	for i, v := range want {
		if data[i] != v {
			t.Errorf("Key data[%d] = %v, want %v", i, data[i], v)
		}
	}

	// Value should also be [1, 2, 2]
	vshape := lkv.Value.Shape()
	if vshape[0] != 1 || vshape[1] != 2 || vshape[2] != 2 {
		t.Errorf("Value shape = %v, want [1 2 2]", vshape)
	}
}

func TestKVCache_UpdateThreeTokens(t *testing.T) {
	cache := NewKVCache[float32](1, 128)

	for i := range 3 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(i*2 + 1), float32(i*2 + 2)})
		v := makeTensor(t, []int{1, 1, 2}, []float32{float32(i*2 + 10), float32(i*2 + 11)})
		if err := cache.Update(0, k, v); err != nil {
			t.Fatalf("Update(%d) error: %v", i, err)
		}
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}
	shape := lkv.Key.Shape()
	if shape[1] != 3 {
		t.Errorf("Key seq_len = %d, want 3", shape[1])
	}
}

func TestKVCache_MultiLayer(t *testing.T) {
	cache := NewKVCache[float32](3, 128)

	for layer := range 3 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer), 0})
		v := makeTensor(t, []int{1, 1, 2}, []float32{0, float32(layer)})
		if err := cache.Update(layer, k, v); err != nil {
			t.Fatalf("Update layer %d error: %v", layer, err)
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

func TestKVCache_SeqLen(t *testing.T) {
	cache := NewKVCache[float32](1, 128)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() on empty cache = %d, want 0", got)
	}

	k1 := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}
	if got := cache.SeqLen(); got != 1 {
		t.Errorf("SeqLen() after 1 token = %d, want 1", got)
	}

	k2 := makeTensor(t, []int{1, 1, 4}, []float32{9, 10, 11, 12})
	v2 := makeTensor(t, []int{1, 1, 4}, []float32{13, 14, 15, 16})
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatal(err)
	}
	if got := cache.SeqLen(); got != 2 {
		t.Errorf("SeqLen() after 2 tokens = %d, want 2", got)
	}
}

func TestKVCache_Reset(t *testing.T) {
	cache := NewKVCache[float32](2, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if err := cache.Update(1, k, v); err != nil {
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

func TestKVCache_UpdateOutOfRange(t *testing.T) {
	cache := NewKVCache[float32](1, 128)
	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Update(5, k, v); err == nil {
		t.Error("Update with out-of-range layer should return error")
	}
	if err := cache.Update(-1, k, v); err == nil {
		t.Error("Update with negative layer should return error")
	}
}

func TestKVCache_DimensionMismatch(t *testing.T) {
	cache := NewKVCache[float32](1, 128)

	k1 := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}

	// Different feature dimension.
	k2 := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v2 := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k2, v2); err == nil {
		t.Error("Update with mismatched dimensions should return error")
	}
}

func TestKVCache_SeqLenEmpty(t *testing.T) {
	cache := NewKVCache[float32](0, 128)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() with 0 layers = %d, want 0", got)
	}
}

func TestKVCache_BatchedUpdate(t *testing.T) {
	cache := NewKVCache[float32](1, 128)

	// Batch of 2: [batch=2, seq=1, dim=2]
	k1 := makeTensor(t, []int{2, 1, 2}, []float32{1, 2, 3, 4})
	v1 := makeTensor(t, []int{2, 1, 2}, []float32{5, 6, 7, 8})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}

	k2 := makeTensor(t, []int{2, 1, 2}, []float32{9, 10, 11, 12})
	v2 := makeTensor(t, []int{2, 1, 2}, []float32{13, 14, 15, 16})
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatal(err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 2 || shape[1] != 2 || shape[2] != 2 {
		t.Errorf("Key shape = %v, want [2 2 2]", shape)
	}

	// Batch 0: [1,2] then [9,10]. Batch 1: [3,4] then [11,12].
	// Layout in buffer: batch 0 occupies [0..maxSeqLen*dim), batch 1 occupies [maxSeqLen*dim..2*maxSeqLen*dim)
	// Get returns a sub-slice [0:batch*cursor*dim] but with shape [2, 2, 2],
	// so the data returned is [batch0_seq0, batch0_seq1, batch1_seq0, batch1_seq1]
	// Wait - the buffer layout is [batch, maxSeqLen, dim], but the sub-slice
	// returned by Get is [0:batch*cursor*dim]. That only works if maxSeqLen==cursor
	// for contiguous access. Otherwise we need to compact.
	// Actually, Get creates a tensor from keyBuf[:size] where size=batch*cursor*dim.
	// But the buffer is laid out as [batch, maxSeqLen, dim], so batch 0 data is at
	// [0..maxSeqLen*dim) and batch 1 is at [maxSeqLen*dim..2*maxSeqLen*dim).
	// If cursor < maxSeqLen, the sub-slice won't be contiguous per batch.
	// This is a bug - let me verify the test.
	data := lkv.Key.Data()
	want := []float32{1, 2, 9, 10, 3, 4, 11, 12}
	for i, v := range want {
		if data[i] != v {
			t.Errorf("Key data[%d] = %v, want %v", i, data[i], v)
		}
	}
}

func TestKVCache_Overflow(t *testing.T) {
	cache := NewKVCache[float32](1, 2)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	// Third update should overflow.
	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected overflow error")
	}
}

func TestKVCache_Non3DTensor(t *testing.T) {
	cache := NewKVCache[float32](1, 128)
	k, _ := tensor.New([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	v, _ := tensor.New([]int{2, 3}, []float32{7, 8, 9, 10, 11, 12})
	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected error for 2D tensors")
	}
}

func TestKVCache_BatchMismatch(t *testing.T) {
	cache := NewKVCache[float32](1, 128)

	k1 := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v1 := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}

	k2 := makeTensor(t, []int{2, 1, 2}, []float32{1, 2, 3, 4})
	v2 := makeTensor(t, []int{2, 1, 2}, []float32{5, 6, 7, 8})
	if err := cache.Update(0, k2, v2); err == nil {
		t.Error("expected error for batch mismatch")
	}
}

func TestKVCache_ResetRetainsBuffers(t *testing.T) {
	cache := NewKVCache[float32](1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	cache.Reset()

	// After reset, should be able to update again without new allocation.
	k2 := makeTensor(t, []int{1, 1, 2}, []float32{5, 6})
	v2 := makeTensor(t, []int{1, 1, 2}, []float32{7, 8})
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Update after Reset: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get should return true after update post-reset")
	}
	data := lkv.Key.Data()
	if data[0] != 5 || data[1] != 6 {
		t.Errorf("data = %v, want [5 6]", data[:2])
	}
}

func TestKVCache_ManyUpdates(t *testing.T) {
	const n = 100
	cache := NewKVCache[float32](1, n)

	for i := range n {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(i), float32(i + 1)})
		v := makeTensor(t, []int{1, 1, 2}, []float32{float32(i + 10), float32(i + 11)})
		if err := cache.Update(0, k, v); err != nil {
			t.Fatalf("Update(%d): %v", i, err)
		}
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get should return true")
	}
	shape := lkv.Key.Shape()
	if shape[1] != n {
		t.Errorf("seq_len = %d, want %d", shape[1], n)
	}

	// Verify first and last entries.
	data := lkv.Key.Data()
	if data[0] != 0 || data[1] != 1 {
		t.Errorf("first entry = [%v %v], want [0 1]", data[0], data[1])
	}
	last := (n - 1) * 2
	if data[last] != float32(n-1) || data[last+1] != float32(n) {
		t.Errorf("last entry = [%v %v], want [%d %d]", data[last], data[last+1], n-1, n)
	}
}

func BenchmarkKVCache_Update(b *testing.B) {
	const (
		maxSeq = 2048
		batch  = 2
		dim    = 64
	)
	cache := NewKVCache[float32](1, maxSeq)

	k := makeBenchTensor(b, []int{batch, 1, dim})
	v := makeBenchTensor(b, []int{batch, 1, dim})

	// Warm up: fill to establish buffer.
	if err := cache.Update(0, k, v); err != nil {
		b.Fatal(err)
	}
	cache.Reset()

	b.ResetTimer()
	b.ReportAllocs()
	for i := range b.N {
		pos := i % maxSeq
		if pos == 0 {
			cache.Reset()
		}
		if err := cache.Update(0, k, v); err != nil {
			b.Fatal(err)
		}
	}
}

func makeBenchTensor(b *testing.B, shape []int) *tensor.TensorNumeric[float32] {
	b.Helper()
	size := 1
	for _, s := range shape {
		size *= s
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i)
	}
	t, err := tensor.New(shape, data)
	if err != nil {
		b.Fatal(err)
	}
	return t
}
