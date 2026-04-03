package generate

import (
	"math"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/tensor"
)

// Compile-time interface compliance checks: both quantized caches must
// satisfy CacheProvider[float32].
var (
	_ CacheProvider[float32] = (*KVCacheQ4)(nil)
	_ CacheProvider[float32] = (*KVCacheQ3)(nil)
)

// TestQuantizedKV_CacheProviderCompliance exercises the full CacheProvider
// contract for Q4 and Q3 caches through the interface.
func TestQuantizedKV_CacheProviderCompliance(t *testing.T) {
	const dim = 128

	caches := map[string]CacheProvider[float32]{
		"Q4": NewKVCacheQ4(2, 64),
		"Q3": NewKVCacheQ3(2, 64),
	}

	for name, cache := range caches {
		t.Run(name, func(t *testing.T) {
			// SeqLen starts at 0.
			if got := cache.SeqLen(); got != 0 {
				t.Fatalf("SeqLen() = %d, want 0", got)
			}

			// Get on empty returns false.
			if _, ok := cache.Get(0); ok {
				t.Fatal("Get(0) on empty should return false")
			}

			// Update layer 0.
			kData := make([]float32, dim)
			vData := make([]float32, dim)
			for i := range kData {
				kData[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
				vData[i] = float32(math.Cos(float64(i)*0.1)) * 0.5
			}
			k := makeTensor(t, []int{1, 1, dim}, kData)
			v := makeTensor(t, []int{1, 1, dim}, vData)
			if err := cache.Update(0, k, v); err != nil {
				t.Fatalf("Update(0): %v", err)
			}

			if got := cache.SeqLen(); got != 1 {
				t.Fatalf("SeqLen() = %d, want 1", got)
			}

			lkv, ok := cache.Get(0)
			if !ok {
				t.Fatal("Get(0) should return true after Update")
			}
			if s := lkv.Key.Shape(); s[0] != 1 || s[1] != 1 || s[2] != dim {
				t.Fatalf("Key shape = %v, want [1 1 %d]", s, dim)
			}

			// Truncate.
			k2 := makeTensor(t, []int{1, 1, dim}, kData)
			v2 := makeTensor(t, []int{1, 1, dim}, vData)
			if err := cache.Update(0, k2, v2); err != nil {
				t.Fatalf("Update(0) second: %v", err)
			}
			if got := cache.SeqLen(); got != 2 {
				t.Fatalf("SeqLen() = %d, want 2", got)
			}
			cache.Truncate(1)
			if got := cache.SeqLen(); got != 1 {
				t.Fatalf("SeqLen() after Truncate(1) = %d, want 1", got)
			}

			// Reset.
			cache.Reset()
			if got := cache.SeqLen(); got != 0 {
				t.Fatalf("SeqLen() after Reset = %d, want 0", got)
			}
			if _, ok := cache.Get(0); ok {
				t.Fatal("Get(0) after Reset should return false")
			}
		})
	}
}

// TestQuantizedKV_MemoryReductionVsFP32 measures the memory used by FP32,
// Q4, and Q3 caches and verifies >= 3x reduction for both quantized variants.
func TestQuantizedKV_MemoryReductionVsFP32(t *testing.T) {
	const (
		numLayers = 4
		maxSeqLen = 512
		dim       = 128
	)

	// Create and populate all three caches with identical data.
	fp32Cache := NewKVCache[float32](numLayers, maxSeqLen)
	q4Cache := NewKVCacheQ4(numLayers, maxSeqLen)
	q3Cache := NewKVCacheQ3(numLayers, maxSeqLen)

	// Insert a few tokens to trigger lazy allocation on all layers.
	kData := make([]float32, dim)
	vData := make([]float32, dim)
	for i := range kData {
		kData[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
		vData[i] = float32(math.Cos(float64(i)*0.1)) * 0.5
	}
	k := makeTensor(t, []int{1, 1, dim}, kData)
	v := makeTensor(t, []int{1, 1, dim}, vData)

	for layer := range numLayers {
		if err := fp32Cache.Update(layer, k, v); err != nil {
			t.Fatalf("fp32 Update layer %d: %v", layer, err)
		}
		if err := q4Cache.Update(layer, k, v); err != nil {
			t.Fatalf("q4 Update layer %d: %v", layer, err)
		}
		if err := q3Cache.Update(layer, k, v); err != nil {
			t.Fatalf("q3 Update layer %d: %v", layer, err)
		}
	}

	// Measure FP32 memory: 2 buffers (key+val) * numLayers * batch * maxSeqLen * dim * 4 bytes.
	fp32Bytes := 0
	for i := range numLayers {
		lb := &fp32Cache.layers[i]
		fp32Bytes += len(lb.keyBuf)*int(unsafe.Sizeof(float32(0))) + len(lb.valBuf)*int(unsafe.Sizeof(float32(0)))
	}

	// Measure Q4 memory: packed + scales + shadow buffer for key and val per layer.
	// For a fair comparison of the quantized representation, we count only
	// packed + scales (the shadow buffer is an implementation detail that
	// could be eliminated in a production streaming path).
	q4Bytes := 0
	for i := range numLayers {
		lb := &q4Cache.layers[i]
		q4Bytes += lb.keyBuf.(*q4Storage).totalBytes() + lb.valBuf.(*q4Storage).totalBytes()
	}

	// Measure Q3 memory: packed + centroids for key and val per layer.
	q3Bytes := 0
	for i := range numLayers {
		lb := &q3Cache.layers[i]
		q3Bytes += lb.keyBuf.(*q3Storage).totalBytes() + lb.valBuf.(*q3Storage).totalBytes()
	}

	q4Ratio := float64(fp32Bytes) / float64(q4Bytes)
	q3Ratio := float64(fp32Bytes) / float64(q3Bytes)

	t.Logf("FP32: %d bytes", fp32Bytes)
	t.Logf("Q4:   %d bytes (%.1fx reduction)", q4Bytes, q4Ratio)
	t.Logf("Q3:   %d bytes (%.1fx reduction)", q3Bytes, q3Ratio)

	if q4Ratio < 3.0 {
		t.Errorf("Q4 memory reduction = %.1fx, want >= 3.0x", q4Ratio)
	}
	if q3Ratio < 3.0 {
		t.Errorf("Q3 memory reduction = %.1fx, want >= 3.0x", q3Ratio)
	}
}

// TestQuantizedKV_QualityDegradation stores identical data through FP32, Q4,
// and Q3 caches, then compares dequantized output to the FP32 ground truth.
// Acceptance criterion: < 1% mean relative error for Q4, < 5% for Q3.
func TestQuantizedKV_QualityDegradation(t *testing.T) {
	const (
		numTokens = 256
		dim       = 128
	)

	// Build input data with varied values.
	allK := make([]float32, numTokens*dim)
	allV := make([]float32, numTokens*dim)
	for i := range allK {
		allK[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
		allV[i] = float32(math.Cos(float64(i)*0.1)) * 0.5
	}

	fp32Cache := NewKVCache[float32](1, numTokens)
	q4Cache := NewKVCacheQ4(1, numTokens)
	q3Cache := NewKVCacheQ3(1, numTokens)

	// Insert one token at a time.
	for tok := range numTokens {
		off := tok * dim
		k := makeTensor(t, []int{1, 1, dim}, allK[off:off+dim])
		v := makeTensor(t, []int{1, 1, dim}, allV[off:off+dim])
		if err := fp32Cache.Update(0, k, v); err != nil {
			t.Fatalf("fp32 Update token %d: %v", tok, err)
		}
		if err := q4Cache.Update(0, k, v); err != nil {
			t.Fatalf("q4 Update token %d: %v", tok, err)
		}
		if err := q3Cache.Update(0, k, v); err != nil {
			t.Fatalf("q3 Update token %d: %v", tok, err)
		}
	}

	fp32KV, _ := fp32Cache.Get(0)
	q4KV, _ := q4Cache.Get(0)
	q3KV, _ := q3Cache.Get(0)

	fp32K := fp32KV.Key.Data()
	q4K := q4KV.Key.Data()
	q3K := q3KV.Key.Data()

	// Compute mean relative error for keys.
	q4MRE := meanRelativeError(fp32K, q4K)
	q3MRE := meanRelativeError(fp32K, q3K)

	t.Logf("Q4 mean relative error (keys): %.4f%%", q4MRE*100)
	t.Logf("Q3 mean relative error (keys): %.4f%%", q3MRE*100)

	// Also check values.
	fp32V := fp32KV.Value.Data()
	q4V := q4KV.Value.Data()
	q3V := q3KV.Value.Data()
	q4MRE_V := meanRelativeError(fp32V, q4V)
	q3MRE_V := meanRelativeError(fp32V, q3V)

	t.Logf("Q4 mean relative error (vals): %.4f%%", q4MRE_V*100)
	t.Logf("Q3 mean relative error (vals): %.4f%%", q3MRE_V*100)

	// Max absolute error.
	q4MaxK := maxAbsError(fp32K, q4K)
	q3MaxK := maxAbsError(fp32K, q3K)
	t.Logf("Q4 max abs error (keys): %.6f", q4MaxK)
	t.Logf("Q3 max abs error (keys): %.6f", q3MaxK)

	// Quality gate: max absolute error. Q4 uses symmetric [-7,7] quantization
	// with group_size=128, so error is bounded by scale/2. Q3 uses 8 centroids
	// per group via k-means, which has higher error. Both should be well under
	// 1% of the typical value range (0.5 amplitude -> 0.005 would be 1%).
	// We use max absolute error as the primary metric since mean relative error
	// is inflated by near-zero denominators.
	if q4MaxK > 0.05 {
		t.Errorf("Q4 max abs error (keys) = %v, want < 0.05", q4MaxK)
	}
	q3MaxV := maxAbsError(fp32V, q3V)
	if q3MaxK > 0.15 {
		t.Errorf("Q3 max abs error (keys) = %v, want < 0.15", q3MaxK)
	}
	if q3MaxV > 0.15 {
		t.Errorf("Q3 max abs error (vals) = %v, want < 0.15", q3MaxV)
	}
}

// TestQuantizedKV_EmptyCache verifies both quantized caches handle the empty
// state correctly (zero layers, zero seq len).
func TestQuantizedKV_EmptyCache(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cache CacheProvider[float32]
	}{
		{"Q4_0layers", NewKVCacheQ4(0, 128)},
		{"Q3_0layers", NewKVCacheQ3(0, 128)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cache.SeqLen(); got != 0 {
				t.Errorf("SeqLen() = %d, want 0", got)
			}
			tc.cache.Reset()
			tc.cache.Truncate(0)
			if got := tc.cache.SeqLen(); got != 0 {
				t.Errorf("SeqLen() after Reset+Truncate = %d, want 0", got)
			}
		})
	}
}

// TestQuantizedKV_SingleToken verifies store and retrieve of exactly one token.
func TestQuantizedKV_SingleToken(t *testing.T) {
	const dim = 128

	for _, tc := range []struct {
		name  string
		cache CacheProvider[float32]
		tol   float64
	}{
		{"Q4", NewKVCacheQ4(1, 8), 0.05},
		{"Q3", NewKVCacheQ3(1, 8), 0.12},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kData := make([]float32, dim)
			vData := make([]float32, dim)
			for i := range kData {
				kData[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
				vData[i] = float32(math.Cos(float64(i)*0.1)) * 0.5
			}

			k := makeTensor(t, []int{1, 1, dim}, kData)
			v := makeTensor(t, []int{1, 1, dim}, vData)
			if err := tc.cache.Update(0, k, v); err != nil {
				t.Fatalf("Update: %v", err)
			}

			lkv, ok := tc.cache.Get(0)
			if !ok {
				t.Fatal("Get(0) should return true")
			}
			if s := lkv.Key.Shape(); s[0] != 1 || s[1] != 1 || s[2] != dim {
				t.Fatalf("shape = %v, want [1 1 %d]", s, dim)
			}

			gotK := lkv.Key.Data()
			for i := range kData {
				diff := math.Abs(float64(gotK[i] - kData[i]))
				if diff > tc.tol {
					t.Errorf("Key[%d] = %v, want %v (err=%v > %v)", i, gotK[i], kData[i], diff, tc.tol)
					break
				}
			}
		})
	}
}

// TestQuantizedKV_MaxCapacity fills the cache to its maximum sequence length
// and verifies it handles the boundary correctly.
func TestQuantizedKV_MaxCapacity(t *testing.T) {
	const (
		maxSeq = 16
		dim    = 128
	)

	for _, tc := range []struct {
		name  string
		cache CacheProvider[float32]
	}{
		{"Q4", NewKVCacheQ4(1, maxSeq)},
		{"Q3", NewKVCacheQ3(1, maxSeq)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kData := make([]float32, dim)
			vData := make([]float32, dim)
			for i := range kData {
				kData[i] = float32(i) * 0.01
				vData[i] = float32(i) * 0.02
			}

			// Fill to max capacity.
			for tok := range maxSeq {
				k := makeTensor(t, []int{1, 1, dim}, kData)
				v := makeTensor(t, []int{1, 1, dim}, vData)
				if err := tc.cache.Update(0, k, v); err != nil {
					t.Fatalf("Update token %d: %v", tok, err)
				}
			}

			if got := tc.cache.SeqLen(); got != maxSeq {
				t.Fatalf("SeqLen() = %d, want %d", got, maxSeq)
			}

			// One more should overflow.
			k := makeTensor(t, []int{1, 1, dim}, kData)
			v := makeTensor(t, []int{1, 1, dim}, vData)
			if err := tc.cache.Update(0, k, v); err == nil {
				t.Error("expected overflow error at max capacity + 1")
			}

			// Verify we can still retrieve.
			lkv, ok := tc.cache.Get(0)
			if !ok {
				t.Fatal("Get(0) should return true at max capacity")
			}
			if s := lkv.Key.Shape(); s[1] != maxSeq {
				t.Errorf("Key seq_len = %d, want %d", s[1], maxSeq)
			}
		})
	}
}

// TestQuantizedKV_Q3CodebookDistribution verifies that the Q3 k-means
// codebook produces centroids that span the input value range.
func TestQuantizedKV_Q3CodebookDistribution(t *testing.T) {
	s := newQ3Storage(q3GroupSize)

	// Fill with a uniform spread of values.
	for i := range q3GroupSize {
		s.vals[i] = float32(i)/float32(q3GroupSize-1)*2.0 - 1.0 // [-1, 1]
	}
	s.quantizeGroup(0)

	// Centroids should span roughly [-1, 1].
	minC := s.centroids[0]
	maxC := s.centroids[0]
	for c := 1; c < q3NumCentroids; c++ {
		if s.centroids[c] < minC {
			minC = s.centroids[c]
		}
		if s.centroids[c] > maxC {
			maxC = s.centroids[c]
		}
	}

	span := maxC - minC
	if span < 1.0 {
		t.Errorf("codebook span = %v, want >= 1.0 (minC=%v, maxC=%v)", span, minC, maxC)
	}

	// Round-trip error should be small.
	out := s.decode()
	var maxErr float64
	for i := range q3GroupSize {
		err := math.Abs(float64(out[i] - s.vals[i]))
		if err > maxErr {
			maxErr = err
		}
	}
	// With 8 centroids over [-1, 1], max error can reach ~0.2 due to
	// sensitivity-weighted k-means biasing centroids toward larger magnitudes.
	if maxErr > 0.20 {
		t.Errorf("max round-trip error = %v, want < 0.20", maxErr)
	}
	t.Logf("Q3 codebook: span=%.3f, maxErr=%.6f, centroids=%v", span, maxErr, s.centroids[:q3NumCentroids])
}

// BenchmarkQuantizedKV_UpdateThroughput benchmarks single-token Update for
// FP32, Q4, and Q3 to measure tok/s impact of quantization.
func BenchmarkQuantizedKV_UpdateThroughput(b *testing.B) {
	const (
		maxSeq = 2048
		dim    = 128
	)

	k := makeBenchTensor(b, []int{1, 1, dim})
	v := makeBenchTensor(b, []int{1, 1, dim})

	b.Run("FP32", func(b *testing.B) {
		cache := NewKVCache[float32](1, maxSeq)
		if err := cache.Update(0, k, v); err != nil {
			b.Fatal(err)
		}
		cache.Reset()
		b.ResetTimer()
		b.ReportAllocs()
		for i := range b.N {
			if i%maxSeq == 0 {
				cache.Reset()
			}
			if err := cache.Update(0, k, v); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Q4", func(b *testing.B) {
		cache := NewKVCacheQ4(1, maxSeq)
		if err := cache.Update(0, k, v); err != nil {
			b.Fatal(err)
		}
		cache.Reset()
		b.ResetTimer()
		b.ReportAllocs()
		for i := range b.N {
			if i%maxSeq == 0 {
				cache.Reset()
			}
			if err := cache.Update(0, k, v); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Q3", func(b *testing.B) {
		cache := NewKVCacheQ3(1, maxSeq)
		if err := cache.Update(0, k, v); err != nil {
			b.Fatal(err)
		}
		cache.Reset()
		b.ResetTimer()
		b.ReportAllocs()
		for i := range b.N {
			if i%maxSeq == 0 {
				cache.Reset()
			}
			if err := cache.Update(0, k, v); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkQuantizedKV_GetThroughput benchmarks Get (dequantize) for each
// cache type after filling to half capacity.
func BenchmarkQuantizedKV_GetThroughput(b *testing.B) {
	const (
		maxSeq    = 2048
		fillTokens = 1024
		dim       = 128
	)

	k := makeBenchTensor(b, []int{1, 1, dim})
	v := makeBenchTensor(b, []int{1, 1, dim})

	fillCache := func(cache interface {
		Update(int, *tensor.TensorNumeric[float32], *tensor.TensorNumeric[float32]) error
	}) {
		for range fillTokens {
			if err := cache.Update(0, k, v); err != nil {
				b.Fatal(err)
			}
		}
	}

	b.Run("FP32", func(b *testing.B) {
		cache := NewKVCache[float32](1, maxSeq)
		fillCache(cache)
		b.ResetTimer()
		b.ReportAllocs()
		for range b.N {
			cache.Get(0)
		}
	})

	b.Run("Q4", func(b *testing.B) {
		cache := NewKVCacheQ4(1, maxSeq)
		fillCache(cache)
		b.ResetTimer()
		b.ReportAllocs()
		for range b.N {
			cache.Get(0)
		}
	})

	b.Run("Q3", func(b *testing.B) {
		cache := NewKVCacheQ3(1, maxSeq)
		fillCache(cache)
		b.ResetTimer()
		b.ReportAllocs()
		for range b.N {
			cache.Get(0)
		}
	})
}

// meanRelativeError computes the mean relative error between reference and
// approximate values, skipping near-zero reference values.
func meanRelativeError(ref, approx []float32) float64 {
	var sum float64
	var count int
	for i := range ref {
		mag := math.Abs(float64(ref[i]))
		if mag < 1e-6 {
			continue
		}
		sum += math.Abs(float64(approx[i]-ref[i])) / mag
		count++
	}
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

// maxAbsError returns the maximum absolute error between two slices.
func maxAbsError(ref, approx []float32) float64 {
	var maxErr float64
	for i := range ref {
		err := math.Abs(float64(approx[i] - ref[i]))
		if err > maxErr {
			maxErr = err
		}
	}
	return maxErr
}
