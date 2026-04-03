package generate

import (
	"math"
	"testing"
)

// q3Approx checks that two float32 values are approximately equal within
// Q3 precision. Q3 with a non-uniform codebook of 8 centroids per group
// has higher quantization error than Q4. We use 20% relative tolerance.
func q3Approx(a, b float32) bool {
	if a == b {
		return true
	}
	diff := math.Abs(float64(a - b))
	mag := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	if mag == 0 {
		return diff < 1e-3
	}
	return diff/mag < 0.20
}

func TestKVCacheQ3_NewAndNumLayers(t *testing.T) {
	cache := NewKVCacheQ3(4, 128)
	if got := cache.NumLayers(); got != 4 {
		t.Errorf("NumLayers() = %d, want 4", got)
	}
}

func TestKVCacheQ3_GetEmpty(t *testing.T) {
	cache := NewKVCacheQ3(2, 128)
	_, ok := cache.Get(0)
	if ok {
		t.Error("Get(0) on empty cache should return false")
	}
}

func TestKVCacheQ3_GetOutOfRange(t *testing.T) {
	cache := NewKVCacheQ3(2, 128)
	_, ok := cache.Get(5)
	if ok {
		t.Error("Get(5) with 2 layers should return false")
	}
	_, ok = cache.Get(-1)
	if ok {
		t.Error("Get(-1) should return false")
	}
}

func TestKVCacheQ3_UpdateAndGet(t *testing.T) {
	cache := NewKVCacheQ3(2, 128)

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

// TestKVCacheQ3_256Tokens is the primary acceptance test: store 256 tokens,
// verify dequantized values are within Q3 tolerance.
func TestKVCacheQ3_256Tokens(t *testing.T) {
	const (
		numTokens = 256
		dim       = 128 // matches group size for clean groups
	)

	cache := NewKVCacheQ3(1, numTokens)

	// Build input data with varied values in [-1, 1] range.
	allK := make([]float32, numTokens*dim)
	allV := make([]float32, numTokens*dim)
	for i := range allK {
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
		// Q3 tolerance: 0.12 (higher than Q4's 0.05 due to only 8 centroids
		// per group of 128 elements)
		if kErr > 0.12 {
			t.Errorf("Key[%d] = %v, want %v (err=%v > 0.12)", i, kData[i], allK[i], kErr)
		}
		if vErr > 0.12 {
			t.Errorf("Value[%d] = %v, want %v (err=%v > 0.12)", i, vData[i], allV[i], vErr)
		}
		if t.Failed() {
			t.Fatalf("stopping early; maxKeyErr=%v, maxValErr=%v", maxKeyErr, maxValErr)
		}
	}
	t.Logf("max key error: %v, max value error: %v", maxKeyErr, maxValErr)
}

// TestKVCacheQ3_MemoryReduction verifies Q3 storage uses approximately
// 0.375 bytes per element (3-bit), which is ~5-6x smaller than float32
// when including codebook overhead.
func TestKVCacheQ3_MemoryReduction(t *testing.T) {
	const (
		maxSeq = 16
		dim    = 128
	)
	cache := NewKVCacheQ3(1, maxSeq)

	k := makeTensor(t, []int{1, 1, dim}, make([]float32, dim))
	v := makeTensor(t, []int{1, 1, dim}, make([]float32, dim))
	if err := cache.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	lb := &cache.layers[0]
	q3Buf := lb.keyBuf.(*q3Storage)
	numElements := q3Buf.n
	f32Bytes := numElements * 4

	// Q3 packs 3 bits per element: ceil(n*3/8) bytes.
	packedBytes := q3Buf.rawBytes()
	expectedPacked := (numElements*3 + 7) / 8
	if packedBytes != expectedPacked {
		t.Errorf("Q3 packed bytes = %d, want %d (0.375 byte/element)", packedBytes, expectedPacked)
	}

	// Total Q3 bytes (packed + codebook) should be much less than float32.
	totalQ3 := q3Buf.totalBytes()
	ratio := float64(f32Bytes) / float64(totalQ3)
	if ratio < 3.0 {
		t.Errorf("Q3 compression ratio = %.1fx, want >= 3x (f32=%d, q3=%d)",
			ratio, f32Bytes, totalQ3)
	}
	t.Logf("compression ratio: %.1fx (f32=%d bytes, q3=%d bytes)", ratio, f32Bytes, totalQ3)
}

func TestKVCacheQ3_Accuracy(t *testing.T) {
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
			cache := NewKVCacheQ3(1, 128)
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
				if !q3Approx(kData[i], want) {
					t.Errorf("Key[%d] = %v, want %v (Q3 approx)", i, kData[i], want)
				}
				if !q3Approx(vData[i], want) {
					t.Errorf("Value[%d] = %v, want %v (Q3 approx)", i, vData[i], want)
				}
			}
		})
	}
}

func TestKVCacheQ3_Concat(t *testing.T) {
	const dim = 128
	cache := NewKVCacheQ3(1, 16)

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
		if diff > 0.15 {
			t.Errorf("Token0 Key[%d] = %v, want %v (err=%v)", i, kData[i], k1[i], diff)
			break
		}
	}
	for i := range dim {
		diff := math.Abs(float64(kData[dim+i] - k2[i]))
		if diff > 0.15 {
			t.Errorf("Token1 Key[%d] = %v, want %v (err=%v)", i, kData[dim+i], k2[i], diff)
			break
		}
	}
}

func TestKVCacheQ3_SeqLen(t *testing.T) {
	cache := NewKVCacheQ3(1, 128)
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

func TestKVCacheQ3_Reset(t *testing.T) {
	cache := NewKVCacheQ3(2, 128)

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

func TestKVCacheQ3_ResetRetainsBuffers(t *testing.T) {
	cache := NewKVCacheQ3(1, 128)

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
	if !q3Approx(data[0], 5) || !q3Approx(data[1], 6) {
		t.Errorf("data = %v, want approx [5 6]", data[:2])
	}
}

func TestKVCacheQ3_Truncate(t *testing.T) {
	cache := NewKVCacheQ3(1, 128)

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

func TestKVCacheQ3_Overflow(t *testing.T) {
	cache := NewKVCacheQ3(1, 2)

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

func TestKVCacheQ3_OutOfRangeLayer(t *testing.T) {
	cache := NewKVCacheQ3(1, 128)
	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := cache.Update(5, k, v); err == nil {
		t.Error("Update with out-of-range layer should return error")
	}
	if err := cache.Update(-1, k, v); err == nil {
		t.Error("Update with negative layer should return error")
	}
}

func TestKVCacheQ3_Non3DTensor(t *testing.T) {
	cache := NewKVCacheQ3(1, 128)
	k := makeTensor(t, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	v := makeTensor(t, []int{2, 3}, []float32{7, 8, 9, 10, 11, 12})
	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected error for 2D tensors")
	}
}

func TestKVCacheQ3_DimensionMismatch(t *testing.T) {
	cache := NewKVCacheQ3(1, 128)

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

func TestKVCacheQ3_BatchMismatch(t *testing.T) {
	cache := NewKVCacheQ3(1, 128)

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

func TestKVCacheQ3_MultiLayer(t *testing.T) {
	cache := NewKVCacheQ3(3, 128)

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
		if !q3Approx(data[0], float32(layer)) {
			t.Errorf("Layer %d Key[0] = %v, want %v", layer, data[0], float32(layer))
		}
	}
}

func TestKVCacheQ3_SeqLenEmpty(t *testing.T) {
	cache := NewKVCacheQ3(0, 128)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() with 0 layers = %d, want 0", got)
	}
}

func TestKVCacheQ3_BatchedUpdate(t *testing.T) {
	const dim = 128
	cache := NewKVCacheQ3(1, 16)

	makeSinVals := func(offset int, amp float32) []float32 {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(math.Sin(float64(i+offset)*0.1)) * amp
		}
		return v
	}

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

	data := lkv.Key.Data()
	for i := range dim {
		diff := math.Abs(float64(data[i] - b0k1[i]))
		if diff > 0.15 {
			t.Errorf("Batch0 Token0 Key[%d] = %v, want %v (err=%v)", i, data[i], b0k1[i], diff)
			break
		}
	}
	for i := range dim {
		diff := math.Abs(float64(data[dim+i] - b0k2[i]))
		if diff > 0.15 {
			t.Errorf("Batch0 Token1 Key[%d] = %v, want %v (err=%v)", i, data[dim+i], b0k2[i], diff)
			break
		}
	}
}

// TestQ3Storage_3BitPackUnpack verifies the 3-bit packing round-trips correctly.
func TestQ3Storage_3BitPackUnpack(t *testing.T) {
	s := newQ3Storage(16)

	// Write values 0-7 twice.
	for i := range 16 {
		s.set3Bit(i, byte(i%8))
	}

	for i := range 16 {
		got := s.get3Bit(i)
		want := byte(i % 8)
		if got != want {
			t.Errorf("get3Bit(%d) = %d, want %d", i, got, want)
		}
	}
}

// TestQ3Storage_CodebookQuality verifies the k-means codebook produces
// reasonable centroids for a known distribution.
func TestQ3Storage_CodebookQuality(t *testing.T) {
	s := newQ3Storage(q3GroupSize)

	// Fill with values from a known distribution: sin wave in [-0.5, 0.5].
	for i := range q3GroupSize {
		s.vals[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	s.quantizeGroup(0)

	// Decode and check round-trip error.
	out := s.decode()
	var maxErr float64
	for i := range q3GroupSize {
		err := math.Abs(float64(out[i] - s.vals[i]))
		if err > maxErr {
			maxErr = err
		}
	}

	// With 8 centroids over a [-0.5, 0.5] range, max error should be
	// well under 0.1.
	if maxErr > 0.1 {
		t.Errorf("max round-trip error = %v, want < 0.1", maxErr)
	}
	t.Logf("codebook max error: %v", maxErr)
}

func BenchmarkKVCacheQ3_Update(b *testing.B) {
	const (
		maxSeq = 2048
		batch  = 2
		dim    = 64
	)
	cache := NewKVCacheQ3(1, maxSeq)

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
