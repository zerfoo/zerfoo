package generate

import (
	"math"
	"testing"
)

// q4Approx checks that two float32 values are approximately equal within
// Q4 precision. Q4 with group_size=128 and symmetric [-7,7] range has
// limited precision; the round-trip error is bounded by scale/2 where
// scale = absmax/7 for the group. We use 15% relative tolerance (similar
// to FP8) to account for the coarse quantization grid.
func q4Approx(a, b float32) bool {
	if a == b {
		return true
	}
	diff := math.Abs(float64(a - b))
	mag := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	if mag == 0 {
		return diff < 1e-3
	}
	return diff/mag < 0.16 // 16% relative tolerance for Q4 round-trip
}

func TestKVCacheQ4_NewAndNumLayers(t *testing.T) {
	cache := NewKVCacheQ4(4, 128)
	if got := cache.NumLayers(); got != 4 {
		t.Errorf("NumLayers() = %d, want 4", got)
	}
}

func TestKVCacheQ4_GetEmpty(t *testing.T) {
	cache := NewKVCacheQ4(2, 128)
	_, ok := cache.Get(0)
	if ok {
		t.Error("Get(0) on empty cache should return false")
	}
}

func TestKVCacheQ4_GetOutOfRange(t *testing.T) {
	cache := NewKVCacheQ4(2, 128)
	_, ok := cache.Get(5)
	if ok {
		t.Error("Get(5) with 2 layers should return false")
	}
	_, ok = cache.Get(-1)
	if ok {
		t.Error("Get(-1) should return false")
	}
}

func TestKVCacheQ4_UpdateAndGet(t *testing.T) {
	cache := NewKVCacheQ4(2, 128)

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

// TestKVCacheQ4_256Tokens is the primary acceptance test: store 256 tokens,
// verify dequantized values are within 0.05 of originals.
func TestKVCacheQ4_256Tokens(t *testing.T) {
	const (
		numTokens = 256
		dim       = 128 // matches group size for clean groups
	)

	cache := NewKVCacheQ4(1, numTokens)

	// Build input data with varied values in [-1, 1] range.
	allK := make([]float32, numTokens*dim)
	allV := make([]float32, numTokens*dim)
	for i := range allK {
		// Use a deterministic pattern that exercises positive, negative, and zero.
		allK[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
		allV[i] = float32(math.Cos(float64(i)*0.1)) * 0.5
	}

	// Insert one token at a time.
	for tok := range numTokens {
		off := tok * dim
		k := makeTensor(t, []int{1, 1, dim}, allK[off:off+dim])
		v := makeTensor(t, []int{1, 1, dim}, allV[off:off+dim])
		if err := cache.Update(0, k, v); err != nil {
			t.Fatalf("Update token %d: %v", tok, err)
		}
	}

	if got := cache.SeqLen(); got != numTokens {
		t.Fatalf("SeqLen() = %d, want %d", got, numTokens)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != numTokens || shape[2] != dim {
		t.Fatalf("Key shape = %v, want [1 %d %d]", shape, numTokens, dim)
	}

	kData := lkv.Key.Data()
	vData := lkv.Value.Data()
	var maxKeyErr, maxValErr float64
	for i := range allK {
		kErr := math.Abs(float64(kData[i] - allK[i]))
		vErr := math.Abs(float64(vData[i] - allV[i]))
		if kErr > maxKeyErr {
			maxKeyErr = kErr
		}
		if vErr > maxValErr {
			maxValErr = vErr
		}
		if kErr > 0.05 {
			t.Errorf("Key[%d] = %v, want %v (err=%v > 0.05)", i, kData[i], allK[i], kErr)
		}
		if vErr > 0.05 {
			t.Errorf("Value[%d] = %v, want %v (err=%v > 0.05)", i, vData[i], allV[i], vErr)
		}
		// Stop flooding output on widespread failure.
		if t.Failed() {
			t.Fatalf("stopping early; maxKeyErr=%v, maxValErr=%v", maxKeyErr, maxValErr)
		}
	}
	t.Logf("max key error: %v, max value error: %v", maxKeyErr, maxValErr)
}

// TestKVCacheQ4_MemoryReduction verifies Q4 storage uses approximately 0.5 bytes
// per element (4-bit), which is ~8x smaller than float32.
func TestKVCacheQ4_MemoryReduction(t *testing.T) {
	const (
		maxSeq = 16
		dim    = 128
	)
	cache := NewKVCacheQ4(1, maxSeq)

	k := makeTensor(t, []int{1, 1, dim}, make([]float32, dim))
	v := makeTensor(t, []int{1, 1, dim}, make([]float32, dim))
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	lb := &cache.layers[0]
	q4Buf := lb.keyBuf.(*q4Storage)
	numElements := q4Buf.n
	packedBytes := q4Buf.rawBytes()
	f32Bytes := numElements * 4

	// Q4 packs 2 elements per byte.
	expectedPacked := (numElements + 1) / 2
	if packedBytes != expectedPacked {
		t.Errorf("Q4 packed bytes = %d, want %d (0.5 byte/element)", packedBytes, expectedPacked)
	}

	// Total Q4 bytes (packed + scales) should be much less than float32.
	totalQ4 := q4Buf.totalBytes()
	ratio := float64(f32Bytes) / float64(totalQ4)
	if ratio < 4.0 {
		t.Errorf("Q4 compression ratio = %.1fx, want >= 4x (f32=%d, q4=%d)",
			ratio, f32Bytes, totalQ4)
	}
	t.Logf("compression ratio: %.1fx (f32=%d bytes, q4=%d bytes)", ratio, f32Bytes, totalQ4)
}

func TestKVCacheQ4_Accuracy(t *testing.T) {
	tests := []struct {
		name  string
		input []float32
		shape []int
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
			name:  "zero values",
			input: []float32{0, 0, 0, 0},
			shape: []int{1, 1, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache := NewKVCacheQ4(1, 128)
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
				if !q4Approx(kData[i], want) {
					t.Errorf("Key[%d] = %v, want %v (Q4 approx)", i, kData[i], want)
				}
				if !q4Approx(vData[i], want) {
					t.Errorf("Value[%d] = %v, want %v (Q4 approx)", i, vData[i], want)
				}
			}
		})
	}
}

func TestKVCacheQ4_Concat(t *testing.T) {
	// Use dim=128 to match q4GroupSize for clean group boundaries.
	// Use sin/cos values so all elements have similar magnitude.
	const dim = 128
	cache := NewKVCacheQ4(1, 16)

	k1 := make([]float32, dim)
	v1 := make([]float32, dim)
	for i := range dim {
		k1[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
		v1[i] = float32(math.Cos(float64(i)*0.1)) * 0.5
	}
	if err := cache.Update(0, makeTensor(t, []int{1, 1, dim}, k1), makeTensor(t, []int{1, 1, dim}, v1)); err != nil {
		t.Fatalf("Update 1: %v", err)
	}

	k2 := make([]float32, dim)
	v2 := make([]float32, dim)
	for i := range dim {
		k2[i] = float32(math.Sin(float64(i+dim)*0.1)) * 0.5
		v2[i] = float32(math.Cos(float64(i+dim)*0.1)) * 0.5
	}
	if err := cache.Update(0, makeTensor(t, []int{1, 1, dim}, k2), makeTensor(t, []int{1, 1, dim}, v2)); err != nil {
		t.Fatalf("Update 2: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 1 || shape[1] != 2 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [1 2 %d]", shape, dim)
	}

	kData := lkv.Key.Data()
	for i := range dim {
		diff := math.Abs(float64(kData[i] - k1[i]))
		if diff > 0.05 {
			t.Errorf("Token0 Key[%d] = %v, want %v (err=%v)", i, kData[i], k1[i], diff)
			break
		}
	}
	for i := range dim {
		diff := math.Abs(float64(kData[dim+i] - k2[i]))
		if diff > 0.05 {
			t.Errorf("Token1 Key[%d] = %v, want %v (err=%v)", i, kData[dim+i], k2[i], diff)
			break
		}
	}
}

func TestKVCacheQ4_SeqLen(t *testing.T) {
	cache := NewKVCacheQ4(1, 128)
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

func TestKVCacheQ4_Reset(t *testing.T) {
	cache := NewKVCacheQ4(2, 128)

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

func TestKVCacheQ4_ResetRetainsBuffers(t *testing.T) {
	cache := NewKVCacheQ4(1, 128)

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
	if !q4Approx(data[0], 5) || !q4Approx(data[1], 6) {
		t.Errorf("data = %v, want approx [5 6]", data[:2])
	}
}

func TestKVCacheQ4_Truncate(t *testing.T) {
	cache := NewKVCacheQ4(1, 128)

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

func TestKVCacheQ4_Overflow(t *testing.T) {
	cache := NewKVCacheQ4(1, 2)

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

func TestKVCacheQ4_OutOfRangeLayer(t *testing.T) {
	cache := NewKVCacheQ4(1, 128)
	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Update(5, k, v); err == nil {
		t.Error("Update with out-of-range layer should return error")
	}
	if err := cache.Update(-1, k, v); err == nil {
		t.Error("Update with negative layer should return error")
	}
}

func TestKVCacheQ4_Non3DTensor(t *testing.T) {
	cache := NewKVCacheQ4(1, 128)
	k := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	v := makeTensor(t, []int{2, 3}, []float32{7, 8, 9, 10, 11, 12})
	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected error for 2D tensors")
	}
}

func TestKVCacheQ4_DimensionMismatch(t *testing.T) {
	cache := NewKVCacheQ4(1, 128)

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

func TestKVCacheQ4_BatchMismatch(t *testing.T) {
	cache := NewKVCacheQ4(1, 128)

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

func TestKVCacheQ4_MultiLayer(t *testing.T) {
	cache := NewKVCacheQ4(3, 128)

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
		if !q4Approx(data[0], float32(layer)) {
			t.Errorf("Layer %d Key[0] = %v, want %v", layer, data[0], float32(layer))
		}
	}
}

func TestKVCacheQ4_SeqLenEmpty(t *testing.T) {
	cache := NewKVCacheQ4(0, 128)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() with 0 layers = %d, want 0", got)
	}
}

func TestKVCacheQ4_BatchedUpdate(t *testing.T) {
	// Use dim=128 to match q4GroupSize so each token's values form
	// a complete group — this avoids cross-token scale interference.
	const dim = 128
	cache := NewKVCacheQ4(1, 16)

	makeSinVals := func(offset int, amp float32) []float32 {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(math.Sin(float64(i+offset)*0.1)) * amp
		}
		return v
	}

	// batch=2, seq=1, dim=128.
	b0k1 := makeSinVals(0, 0.5)
	b1k1 := makeSinVals(dim, 0.5)
	k1Data := append(b0k1, b1k1...)
	v1Data := make([]float32, 2*dim)
	for i := range v1Data {
		v1Data[i] = float32(math.Cos(float64(i)*0.1)) * 0.3
	}
	k1 := makeTensor(t, []int{2, 1, dim}, k1Data)
	v1 := makeTensor(t, []int{2, 1, dim}, v1Data)
	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}

	b0k2 := makeSinVals(2*dim, 0.5)
	b1k2 := makeSinVals(3*dim, 0.5)
	k2Data := append(b0k2, b1k2...)
	v2Data := make([]float32, 2*dim)
	for i := range v2Data {
		v2Data[i] = float32(math.Cos(float64(i+2*dim)*0.1)) * 0.3
	}
	k2 := makeTensor(t, []int{2, 1, dim}, k2Data)
	v2 := makeTensor(t, []int{2, 1, dim}, v2Data)
	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatal(err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get should return true")
	}

	shape := lkv.Key.Shape()
	if shape[0] != 2 || shape[1] != 2 || shape[2] != dim {
		t.Errorf("Key shape = %v, want [2 2 %d]", shape, dim)
	}

	// Verify dequantized values are within Q4 tolerance.
	data := lkv.Key.Data()
	for i := range dim {
		diff := math.Abs(float64(data[i] - b0k1[i]))
		if diff > 0.05 {
			t.Errorf("Batch0 Token0 Key[%d] = %v, want %v (err=%v)", i, data[i], b0k1[i], diff)
			break
		}
	}
	for i := range dim {
		diff := math.Abs(float64(data[dim+i] - b0k2[i]))
		if diff > 0.05 {
			t.Errorf("Batch0 Token1 Key[%d] = %v, want %v (err=%v)", i, data[dim+i], b0k2[i], diff)
			break
		}
	}
}

func BenchmarkKVCacheQ4_Update(b *testing.B) {
	const (
		maxSeq = 2048
		batch  = 2
		dim    = 64
	)
	cache := NewKVCacheQ4(1, maxSeq)

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
