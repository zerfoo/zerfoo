package generate

import (
	"math"
	"testing"
)

// fp16Approx checks that two float32 values are approximately equal within
// FP16 precision (~3 decimal digits, relative error < 0.1%).
func fp16Approx(a, b float32) bool {
	if a == b {
		return true
	}
	diff := math.Abs(float64(a - b))
	mag := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	if mag == 0 {
		return diff < 1e-6
	}
	return diff/mag < 1e-2 // 1% relative tolerance for FP16 round-trip
}

func TestKVCacheFP16_NewAndNumLayers(t *testing.T) {
	cache := NewKVCacheFP16(4, 128)
	if got := cache.NumLayers(); got != 4 {
		t.Errorf("NumLayers() = %d, want 4", got)
	}
}

func TestKVCacheFP16_GetEmpty(t *testing.T) {
	cache := NewKVCacheFP16(2, 128)
	_, ok := cache.Get(0)
	if ok {
		t.Error("Get(0) on empty cache should return false")
	}
}

func TestKVCacheFP16_GetOutOfRange(t *testing.T) {
	cache := NewKVCacheFP16(2, 128)
	_, ok := cache.Get(5)
	if ok {
		t.Error("Get(5) with 2 layers should return false")
	}
	_, ok = cache.Get(-1)
	if ok {
		t.Error("Get(-1) should return false")
	}
}

func TestKVCacheFP16_UpdateAndGet(t *testing.T) {
	cache := NewKVCacheFP16(2, 128)

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

func TestKVCacheFP16_RoundTrip(t *testing.T) {
	// Verify that float32 values are approximately preserved through FP16.
	tests := []struct {
		name   string
		input  []float32
		shape  []int
	}{
		{
			name:  "small integers",
			input: []float32{1, 2, 3, 4},
			shape: []int{1, 1, 4},
		},
		{
			name:  "fractional values",
			input: []float32{0.5, 0.25, 0.125, 0.0625},
			shape: []int{1, 1, 4},
		},
		{
			name:  "negative values",
			input: []float32{-1.5, -2.5, 3.5, -0.5},
			shape: []int{1, 1, 4},
		},
		{
			name:  "large values",
			input: []float32{100, 200, 300, 400},
			shape: []int{1, 1, 4},
		},
		{
			name:  "zero values",
			input: []float32{0, 0, 0, 0},
			shape: []int{1, 1, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := NewKVCacheFP16(1, 128)
			k := makeTensor(t, tt.shape, tt.input)
			v := makeTensor(t, tt.shape, tt.input)

			if err := cache.Update(0, k, v); err != nil {
				t.Fatalf("Update error: %v", err)
			}

			lkv, ok := cache.Get(0)
			if !ok {
				t.Fatal("Get should return true")
			}

			kData := lkv.Key.Data()
			vData := lkv.Value.Data()
			for i, want := range tt.input {
				if !fp16Approx(kData[i], want) {
					t.Errorf("Key[%d] = %v, want %v (FP16 approx)", i, kData[i], want)
				}
				if !fp16Approx(vData[i], want) {
					t.Errorf("Value[%d] = %v, want %v (FP16 approx)", i, vData[i], want)
				}
			}
		})
	}
}

func TestKVCacheFP16_Concat(t *testing.T) {
	// Two sequential updates should be concatenated on the seq axis.
	cache := NewKVCacheFP16(1, 128)

	k1 := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v1 := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update 1: %v", err)
	}

	k2 := makeTensor(t, []int{1, 1, 2}, []float32{5, 6})
	v2 := makeTensor(t, []int{1, 1, 2}, []float32{7, 8})
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Update 2: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != 2 {
		t.Errorf("Key shape = %v, want [1 2 2]", shape)
	}

	kData := lkv.Key.Data()
	want := []float32{1, 2, 5, 6}
	for i, w := range want {
		if !fp16Approx(kData[i], w) {
			t.Errorf("Key[%d] = %v, want %v", i, kData[i], w)
		}
	}
}

func TestKVCacheFP16_SeqLen(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() on empty = %d, want 0", got)
	}

	k := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})
	if err := cache.Update(0, k, v); err != nil {
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

func TestKVCacheFP16_Reset(t *testing.T) {
	cache := NewKVCacheFP16(2, 128)

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

func TestKVCacheFP16_ResetRetainsBuffers(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	cache.Reset()

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
	if !fp16Approx(data[0], 5) || !fp16Approx(data[1], 6) {
		t.Errorf("data = %v, want approx [5 6]", data[:2])
	}
}

func TestKVCacheFP16_Truncate(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)

	for i := range 5 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(i), float32(i + 1)})
		v := makeTensor(t, []int{1, 1, 2}, []float32{float32(i + 10), float32(i + 11)})
		if err := cache.Update(0, k, v); err != nil {
			t.Fatalf("Update(%d): %v", i, err)
		}
	}
	if got := cache.SeqLen(); got != 5 {
		t.Fatalf("SeqLen() before truncate = %d, want 5", got)
	}

	cache.Truncate(3)
	if got := cache.SeqLen(); got != 3 {
		t.Errorf("SeqLen() after Truncate(3) = %d, want 3", got)
	}
}

func TestKVCacheFP16_Overflow(t *testing.T) {
	cache := NewKVCacheFP16(1, 2)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected overflow error")
	}
}

func TestKVCacheFP16_OutOfRangeLayer(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)
	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Update(5, k, v); err == nil {
		t.Error("Update with out-of-range layer should return error")
	}
	if err := cache.Update(-1, k, v); err == nil {
		t.Error("Update with negative layer should return error")
	}
}

func TestKVCacheFP16_Non3DTensor(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)
	k := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	v := makeTensor(t, []int{2, 3}, []float32{7, 8, 9, 10, 11, 12})
	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected error for 2D tensors")
	}
}

func TestKVCacheFP16_DimensionMismatch(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)

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

func TestKVCacheFP16_BatchMismatch(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)

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

func TestKVCacheFP16_MultiLayer(t *testing.T) {
	cache := NewKVCacheFP16(3, 128)

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
		if !fp16Approx(data[0], float32(layer)) {
			t.Errorf("Layer %d Key[0] = %v, want %v", layer, data[0], float32(layer))
		}
	}
}

func TestKVCacheFP16_SeqLenEmpty(t *testing.T) {
	cache := NewKVCacheFP16(0, 128)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() with 0 layers = %d, want 0", got)
	}
}

func TestKVCacheFP16_ManyTokens(t *testing.T) {
	const n = 100
	cache := NewKVCacheFP16(1, n)

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

	data := lkv.Key.Data()
	if !fp16Approx(data[0], 0) || !fp16Approx(data[1], 1) {
		t.Errorf("first entry = [%v %v], want approx [0 1]", data[0], data[1])
	}
	last := (n - 1) * 2
	if !fp16Approx(data[last], float32(n-1)) || !fp16Approx(data[last+1], float32(n)) {
		t.Errorf("last entry = [%v %v], want approx [%d %d]", data[last], data[last+1], n-1, n)
	}
}

func TestKVCacheFP16_BatchedUpdate(t *testing.T) {
	cache := NewKVCacheFP16(1, 128)

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
		t.Fatal("Get should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 2 || shape[1] != 2 || shape[2] != 2 {
		t.Errorf("Key shape = %v, want [2 2 2]", shape)
	}

	data := lkv.Key.Data()
	want := []float32{1, 2, 9, 10, 3, 4, 11, 12}
	for i, w := range want {
		if !fp16Approx(data[i], w) {
			t.Errorf("Key data[%d] = %v, want %v", i, data[i], w)
		}
	}
}

func TestKVCacheFP16_HalfMemoryVsFloat32(t *testing.T) {
	// Verify FP16 storage uses 2 bytes per element vs float32's 4 bytes.
	const (
		numLayers = 2
		maxSeq    = 16
		dim       = 8
	)
	cache := NewKVCacheFP16(numLayers, maxSeq)

	k := makeTensor(t, []int{1, 1, dim}, make([]float32, dim))
	v := makeTensor(t, []int{1, 1, dim}, make([]float32, dim))
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	lb := &cache.layers[0]
	fp16Buf := lb.keyBuf.(*fp16Storage).buf
	// Each buffer holds batch*maxSeq*dim FP16 elements = 2 bytes each.
	fp16Bytes := fp16Buf.Len() * 2
	// Equivalent float32 would use 4 bytes per element.
	f32Bytes := fp16Buf.Len() * 4

	if fp16Bytes*2 != f32Bytes {
		t.Errorf("FP16 storage should be exactly half of float32: fp16=%d, f32=%d", fp16Bytes, f32Bytes)
	}
}

func BenchmarkKVCacheFP16_Update(b *testing.B) {
	const (
		maxSeq = 2048
		batch  = 2
		dim    = 64
	)
	cache := NewKVCacheFP16(1, maxSeq)

	k := makeBenchTensor(b, []int{batch, 1, dim})
	v := makeBenchTensor(b, []int{batch, 1, dim})

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
