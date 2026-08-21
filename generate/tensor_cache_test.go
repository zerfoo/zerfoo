package generate

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func newTestTensorCache(t *testing.T) (*TensorCache[float32], compute.Engine[float32]) {
	t.Helper()
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 2, 128)
	return cache, eng
}

func TestTensorCache_ImplementsCacheProvider(t *testing.T) {
	cache, _ := newTestTensorCache(t)
	// Compile-time interface check.
	var _ CacheProvider[float32] = cache
}

func TestTensorCache_GetEmpty(t *testing.T) {
	cache, _ := newTestTensorCache(t)
	if _, ok := cache.Get(0); ok {
		t.Error("Get(0) on empty cache should return false")
	}
}

func TestTensorCache_GetOutOfRange(t *testing.T) {
	cache, _ := newTestTensorCache(t)
	if _, ok := cache.Get(5); ok {
		t.Error("Get(5) with 2 layers should return false")
	}
	if _, ok := cache.Get(-1); ok {
		t.Error("Get(-1) should return false")
	}
}

func TestTensorCache_UpdateAndGet(t *testing.T) {
	cache, _ := newTestTensorCache(t)

	k1 := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update(0) error: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true after Update")
	}

	gotK := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4}
	for i := range wantK {
		if gotK[i] != wantK[i] {
			t.Errorf("Key[%d] = %v, want %v", i, gotK[i], wantK[i])
		}
	}

	gotV := lkv.Value.Data()
	wantV := []float32{5, 6, 7, 8}
	for i := range wantV {
		if gotV[i] != wantV[i] {
			t.Errorf("Value[%d] = %v, want %v", i, gotV[i], wantV[i])
		}
	}
}

func TestTensorCache_UpdateConcat(t *testing.T) {
	cache, _ := newTestTensorCache(t)

	// First update: [1, 1, 4]
	k1 := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeTensor(t, []int{1, 1, 4}, []float32{10, 20, 30, 40})

	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update(0) first: %v", err)
	}

	// Second update: [1, 2, 4]
	k2 := makeTensor(t, []int{1, 2, 4}, []float32{5, 6, 7, 8, 9, 10, 11, 12})
	v2 := makeTensor(t, []int{1, 2, 4}, []float32{50, 60, 70, 80, 90, 100, 110, 120})

	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Update(0) second: %v", err)
	}

	if cache.SeqLen() != 3 {
		t.Errorf("SeqLen() = %d, want 3", cache.SeqLen())
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	// After concat: shape should be [1, 3, 4]
	shape := lkv.Key.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 4 {
		t.Errorf("Key shape = %v, want [1, 3, 4]", shape)
	}

	gotK := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	for i := range wantK {
		if gotK[i] != wantK[i] {
			t.Errorf("Key[%d] = %v, want %v", i, gotK[i], wantK[i])
		}
	}
}

func TestTensorCache_SeqLenEmpty(t *testing.T) {
	cache, _ := newTestTensorCache(t)
	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen() = %d, want 0", cache.SeqLen())
	}
}

func TestTensorCache_SeqLenNoLayers(t *testing.T) {
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 0, 128)
	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen() = %d, want 0 for zero-layer cache", cache.SeqLen())
	}
}

func TestTensorCache_Reset(t *testing.T) {
	cache, _ := newTestTensorCache(t)

	k := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}
	if cache.SeqLen() != 1 {
		t.Fatalf("SeqLen before reset = %d, want 1", cache.SeqLen())
	}

	cache.Reset()

	if cache.SeqLen() != 0 {
		t.Errorf("SeqLen after reset = %d, want 0", cache.SeqLen())
	}
	if _, ok := cache.Get(0); ok {
		t.Error("Get(0) after reset should return false")
	}
}

func TestTensorCache_Truncate(t *testing.T) {
	cache, _ := newTestTensorCache(t)

	k1 := makeTensor(t, []int{1, 3, 4}, []float32{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
	})
	v1 := makeTensor(t, []int{1, 3, 4}, []float32{
		10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
	})

	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update: %v", err)
	}

	cache.Truncate(1)

	if cache.layers[0].seqLen != 1 {
		t.Errorf("seqLen after Truncate = %d, want 1", cache.layers[0].seqLen)
	}

	// After truncation, Get returns a view of just the first position.
	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) after Truncate should return true")
	}
	gotK := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4}
	for i := range wantK {
		if gotK[i] != wantK[i] {
			t.Errorf("Key[%d] after Truncate = %v, want %v", i, gotK[i], wantK[i])
		}
	}
}

func TestTensorCache_TruncateNoOp(t *testing.T) {
	cache, _ := newTestTensorCache(t)

	k := makeTensor(t, []int{1, 2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	v := makeTensor(t, []int{1, 2, 4}, []float32{10, 20, 30, 40, 50, 60, 70, 80})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	// Truncate to >= current seqLen is a no-op.
	cache.Truncate(5)

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should still return true after no-op truncate")
	}
	if lkv.Key.Shape()[1] != 2 {
		t.Errorf("Key seq dim = %d, want 2", lkv.Key.Shape()[1])
	}
}

func TestTensorCache_Overflow(t *testing.T) {
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, 2)

	k := makeTensor(t, []int{1, 3, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	v := makeTensor(t, []int{1, 3, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})

	err := cache.Update(0, k, v)
	if err == nil {
		t.Error("expected overflow error, got nil")
	}
}

func TestTensorCache_LayerOutOfRange(t *testing.T) {
	cache, _ := newTestTensorCache(t)
	k := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Update(5, k, v); err == nil {
		t.Error("expected error for out-of-range layer, got nil")
	}
	if err := cache.Update(-1, k, v); err == nil {
		t.Error("expected error for negative layer, got nil")
	}
}

func TestTensorCache_WrongDimensions(t *testing.T) {
	cache, _ := newTestTensorCache(t)
	k := makeTensor(t, []int{4}, []float32{1, 2, 3, 4})
	v := makeTensor(t, []int{4}, []float32{5, 6, 7, 8})

	if err := cache.Update(0, k, v); err == nil {
		t.Error("expected error for 1D tensor, got nil")
	}
}

func TestTensorCache_MultipleLayers(t *testing.T) {
	cache, _ := newTestTensorCache(t)

	for layer := range 2 {
		k := makeTensor(t, []int{1, 1, 4}, []float32{float32(layer), 2, 3, 4})
		v := makeTensor(t, []int{1, 1, 4}, []float32{float32(layer + 10), 20, 30, 40})
		if err := cache.Update(layer, k, v); err != nil {
			t.Fatalf("Update(layer=%d) error: %v", layer, err)
		}
	}

	for layer := range 2 {
		lkv, ok := cache.Get(layer)
		if !ok {
			t.Fatalf("Get(%d) should return true", layer)
		}
		if got := lkv.Key.Data()[0]; got != float32(layer) {
			t.Errorf("layer %d Key[0] = %v, want %v", layer, got, float32(layer))
		}
	}
}

// TestTensorCache_ContextCarry verifies TensorCache works through the
// context-based cache provider mechanism used by the generator.
func TestTensorCache_ContextCarry(t *testing.T) {
	cache, _ := newTestTensorCache(t)

	ctx := WithCache[float32](context.Background(), cache)
	_ = ctx // Just verify it compiles; WithCache accepts CacheProvider[T].

	// Also verify via GetCache.
	ctx2 := WithCache[float32](context.Background(), CacheProvider[float32](cache))
	got, ok := GetCache[float32](ctx2)
	if !ok {
		t.Fatal("GetCache should return true")
	}
	if got != CacheProvider[float32](cache) {
		t.Error("GetCache should return the same cache instance")
	}
}

// makeGPUTensor creates a GPU-backed tensor by uploading CPU data. Skips the
// test if no GPU runtime is available.
func makeGPUTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	cpu := makeTensor(t, shape, data)
	gpu, err := tensor.ToGPU(cpu)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	return gpu
}

func TestTensorCache_UpdateGPU_D2D(t *testing.T) {
	// Verify that when source tensors are GPU-resident, appendGPU uses D2D
	// copy (CopyFromDevice) rather than falling back to src.Data() which
	// would trigger a D2H transfer.
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 2, 128)

	k1 := makeGPUTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v1 := makeGPUTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update(0) first: %v", err)
	}

	// Verify the cache allocated GPU buffers.
	lb := &cache.layers[0]
	if !lb.isGPU {
		t.Fatal("expected GPU-backed cache layer after GPU source update")
	}
	if lb.kStorage == nil || lb.vStorage == nil {
		t.Fatal("expected non-nil GPU storage buffers")
	}

	// Append a second GPU tensor to exercise the D2D append path.
	k2 := makeGPUTensor(t, []int{1, 2, 4}, []float32{9, 10, 11, 12, 13, 14, 15, 16})
	v2 := makeGPUTensor(t, []int{1, 2, 4}, []float32{17, 18, 19, 20, 21, 22, 23, 24})

	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Update(0) second: %v", err)
	}

	if cache.SeqLen() != 3 {
		t.Errorf("SeqLen() = %d, want 3", cache.SeqLen())
	}

	// Read back via Get and verify the concatenated data.
	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 4 {
		t.Errorf("Key shape = %v, want [1, 3, 4]", shape)
	}

	gotK := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16}
	for i := range wantK {
		if gotK[i] != wantK[i] {
			t.Errorf("Key[%d] = %v, want %v", i, gotK[i], wantK[i])
		}
	}

	gotV := lkv.Value.Data()
	wantV := []float32{5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24}
	for i := range wantV {
		if gotV[i] != wantV[i] {
			t.Errorf("Value[%d] = %v, want %v", i, gotV[i], wantV[i])
		}
	}
}

func TestTensorCache_UpdateGPU_MultipleLayers(t *testing.T) {
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 2, 128)

	for layer := range 2 {
		k := makeGPUTensor(t, []int{1, 1, 4}, []float32{float32(layer), 2, 3, 4})
		v := makeGPUTensor(t, []int{1, 1, 4}, []float32{float32(layer + 10), 20, 30, 40})
		if err := cache.Update(layer, k, v); err != nil {
			t.Fatalf("Update(layer=%d) error: %v", layer, err)
		}
	}

	for layer := range 2 {
		lkv, ok := cache.Get(layer)
		if !ok {
			t.Fatalf("Get(%d) should return true", layer)
		}
		if !cache.layers[layer].isGPU {
			t.Errorf("layer %d should be GPU-backed", layer)
		}
		if got := lkv.Key.Data()[0]; got != float32(layer) {
			t.Errorf("layer %d Key[0] = %v, want %v", layer, got, float32(layer))
		}
	}
}

func TestTensorCache_WithKVDtype_FP16_CPUFallback(t *testing.T) {
	// FP16 mode with CPU tensors should fall back to F32 storage
	// since FP16 conversion requires GPU kernels.
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 2, 128, WithKVDtype("fp16"))

	k := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	// CPU fallback: data should be stored as F32 and returned exactly.
	gotK := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4}
	for i := range wantK {
		if gotK[i] != wantK[i] {
			t.Errorf("Key[%d] = %v, want %v", i, gotK[i], wantK[i])
		}
	}
}

func TestTensorCache_WithKVDtype_FP32_Default(t *testing.T) {
	// WithKVDtype("fp32") should behave identically to no option.
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, 128, WithKVDtype("fp32"))

	k := makeTensor(t, []int{1, 2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	v := makeTensor(t, []int{1, 2, 4}, []float32{10, 20, 30, 40, 50, 60, 70, 80})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}
	if cache.SeqLen() != 2 {
		t.Errorf("SeqLen() = %d, want 2", cache.SeqLen())
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	gotK := lkv.Key.Data()
	wantK := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for i := range wantK {
		if gotK[i] != wantK[i] {
			t.Errorf("Key[%d] = %v, want %v", i, gotK[i], wantK[i])
		}
	}
}

func TestTensorCache_FP16_GPU(t *testing.T) {
	// GPU FP16 KV cache: verify F32→FP16→F32 roundtrip produces
	// reasonable output (within FP16 precision).
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, 128, WithKVDtype("fp16"))

	k := makeGPUTensor(t, []int{1, 1, 4}, []float32{1.0, 2.5, -3.0, 0.125})
	v := makeGPUTensor(t, []int{1, 1, 4}, []float32{0.5, -1.5, 4.0, 0.0})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	lb := &cache.layers[0]
	if lb.kFP16 == nil || lb.vFP16 == nil {
		t.Fatal("expected FP16 storage to be allocated for GPU tensors")
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true after FP16 Update")
	}

	// FP16 roundtrip: values representable in FP16 should survive exactly.
	wantK := []float32{1.0, 2.5, -3.0, 0.125}
	gotK := lkv.Key.Data()
	for i := range wantK {
		if diff := gotK[i] - wantK[i]; diff > 0.01 || diff < -0.01 {
			t.Errorf("Key[%d] = %v, want ~%v (diff=%v)", i, gotK[i], wantK[i], diff)
		}
	}

	wantV := []float32{0.5, -1.5, 4.0, 0.0}
	gotV := lkv.Value.Data()
	for i := range wantV {
		if diff := gotV[i] - wantV[i]; diff > 0.01 || diff < -0.01 {
			t.Errorf("Value[%d] = %v, want ~%v (diff=%v)", i, gotV[i], wantV[i], diff)
		}
	}
}

func TestTensorCache_FP16_GPU_MultiToken(t *testing.T) {
	// Test FP16 KV cache with multi-token prefill followed by single-token append.
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, 128, WithKVDtype("fp16"))

	// Prefill: 3 tokens.
	k1 := makeGPUTensor(t, []int{1, 3, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	v1 := makeGPUTensor(t, []int{1, 3, 4}, []float32{
		10, 20, 30, 40,
		50, 60, 70, 80,
		90, 100, 110, 120,
	})

	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Update prefill: %v", err)
	}
	if cache.SeqLen() != 3 {
		t.Errorf("SeqLen after prefill = %d, want 3", cache.SeqLen())
	}

	// Decode: 1 token.
	k2 := makeGPUTensor(t, []int{1, 1, 4}, []float32{13, 14, 15, 16})
	v2 := makeGPUTensor(t, []int{1, 1, 4}, []float32{130, 140, 150, 160})

	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Update decode: %v", err)
	}
	if cache.SeqLen() != 4 {
		t.Errorf("SeqLen after decode = %d, want 4", cache.SeqLen())
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 4 || shape[2] != 4 {
		t.Errorf("Key shape = %v, want [1, 4, 4]", shape)
	}

	// Verify key values survive FP16 roundtrip (all values are exactly
	// representable in FP16).
	wantK := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	gotK := lkv.Key.Data()
	for i := range wantK {
		if diff := gotK[i] - wantK[i]; diff > 0.01 || diff < -0.01 {
			t.Errorf("Key[%d] = %v, want ~%v", i, gotK[i], wantK[i])
		}
	}
}

func TestTensorCache_FP16_GPU_MultiHead(t *testing.T) {
	// Test FP16 KV cache with batch > 1 (GQA: multiple KV heads).
	// Shape: [numKVHeads, seqLen, headDim] — e.g. 4 KV heads, 2 tokens, 4-dim.
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, 128, WithKVDtype("fp16"))

	k := makeGPUTensor(t, []int{4, 2, 4}, []float32{
		// head 0
		1, 2, 3, 4,
		5, 6, 7, 8,
		// head 1
		9, 10, 11, 12,
		13, 14, 15, 16,
		// head 2
		17, 18, 19, 20,
		21, 22, 23, 24,
		// head 3
		25, 26, 27, 28,
		29, 30, 31, 32,
	})
	v := makeGPUTensor(t, []int{4, 2, 4}, []float32{
		100, 200, 300, 400,
		500, 600, 700, 800,
		110, 210, 310, 410,
		510, 610, 710, 810,
		120, 220, 320, 420,
		520, 620, 720, 820,
		130, 230, 330, 430,
		530, 630, 730, 830,
	})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}
	if cache.SeqLen() != 2 {
		t.Errorf("SeqLen = %d, want 2", cache.SeqLen())
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if len(shape) != 3 || shape[0] != 4 || shape[1] != 2 || shape[2] != 4 {
		t.Errorf("Key shape = %v, want [4, 2, 4]", shape)
	}

	wantK := []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30, 31, 32,
	}
	gotK := lkv.Key.Data()
	for i := range wantK {
		if diff := gotK[i] - wantK[i]; diff > 0.01 || diff < -0.01 {
			t.Errorf("Key[%d] = %v, want ~%v", i, gotK[i], wantK[i])
		}
	}

	wantV := []float32{
		100, 200, 300, 400, 500, 600, 700, 800,
		110, 210, 310, 410, 510, 610, 710, 810,
		120, 220, 320, 420, 520, 620, 720, 820,
		130, 230, 330, 430, 530, 630, 730, 830,
	}
	gotV := lkv.Value.Data()
	for i := range wantV {
		if diff := gotV[i] - wantV[i]; diff > 0.01 || diff < -0.01 {
			t.Errorf("Value[%d] = %v, want ~%v", i, gotV[i], wantV[i])
		}
	}
}

func TestTensorCache_FP16_GPU_MultiHead_PrefillAndDecode(t *testing.T) {
	// Test multi-head FP16 KV cache with prefill followed by single-token decode.
	// This mirrors the GQA decode path: prefill writes multiple tokens, then
	// decode appends one token at a time.
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 2, 128, WithKVDtype("fp16"))

	// Prefill layer 0: 2 KV heads, 3 tokens, headDim=4
	k1 := makeGPUTensor(t, []int{2, 3, 4}, []float32{
		// head 0: tokens 0-2
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		// head 1: tokens 0-2
		13, 14, 15, 16,
		17, 18, 19, 20,
		21, 22, 23, 24,
	})
	v1 := makeGPUTensor(t, []int{2, 3, 4}, []float32{
		100, 200, 300, 400,
		500, 600, 700, 800,
		900, 1000, 1100, 1200,
		110, 210, 310, 410,
		510, 610, 710, 810,
		910, 1010, 1110, 1210,
	})

	if err := cache.Update(0, k1, v1); err != nil {
		t.Fatalf("Prefill layer 0: %v", err)
	}

	// Prefill layer 1 with different data.
	k1L1 := makeGPUTensor(t, []int{2, 3, 4}, []float32{
		31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
		43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
	})
	v1L1 := makeGPUTensor(t, []int{2, 3, 4}, []float32{
		131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
		143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
	})

	if err := cache.Update(1, k1L1, v1L1); err != nil {
		t.Fatalf("Prefill layer 1: %v", err)
	}
	if cache.SeqLen() != 3 {
		t.Errorf("SeqLen after prefill = %d, want 3", cache.SeqLen())
	}

	// Decode: append 1 token to each layer.
	k2 := makeGPUTensor(t, []int{2, 1, 4}, []float32{
		25, 26, 27, 28, // head 0 token 3
		29, 30, 31, 32, // head 1 token 3
	})
	v2 := makeGPUTensor(t, []int{2, 1, 4}, []float32{
		1300, 1400, 1500, 1600,
		1310, 1410, 1510, 1610,
	})

	if err := cache.Update(0, k2, v2); err != nil {
		t.Fatalf("Decode layer 0: %v", err)
	}
	if cache.SeqLen() != 4 {
		t.Errorf("SeqLen after decode = %d, want 4", cache.SeqLen())
	}

	// Verify layer 0 key data: 2 heads x 4 tokens x 4 dim.
	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	shape := lkv.Key.Shape()
	if len(shape) != 3 || shape[0] != 2 || shape[1] != 4 || shape[2] != 4 {
		t.Errorf("Key shape = %v, want [2, 4, 4]", shape)
	}

	wantK := []float32{
		// head 0
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 25, 26, 27, 28,
		// head 1
		13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29, 30, 31, 32,
	}
	gotK := lkv.Key.Data()
	for i := range wantK {
		if diff := gotK[i] - wantK[i]; diff > 0.01 || diff < -0.01 {
			t.Errorf("Key[%d] = %v, want ~%v", i, gotK[i], wantK[i])
		}
	}
}

// kvFixture builds a [heads, tokens, dim] block of pairwise-distinct values
// starting at base. Distinctness is what makes a layout test able to fail: if
// two cached elements shared a value, a misplaced token could still satisfy
// the assertion (see docs/lore.md L-0009).
func kvFixture(base, heads, tokens, dim int) []float32 {
	out := make([]float32, heads*tokens*dim)
	for i := range out {
		out[i] = float32(base + i)
	}
	return out
}

// assertDistinct fails if want contains a repeated value, which would make the
// caller's layout assertion satisfiable by the wrong data.
func assertDistinct(t *testing.T, name string, want []float32) {
	t.Helper()
	seen := make(map[float32]int, len(want))
	for i, v := range want {
		if j, dup := seen[v]; dup {
			t.Fatalf("%s is not position-discriminating: want[%d] == want[%d] == %v; "+
				"a layout bug could satisfy this assertion", name, j, i, v)
		}
		seen[v] = i
	}
}

// wantHeadMajor returns the expected contiguous [heads, tokens, dim] readback
// after appending the given per-step fixtures, i.e. every head's tokens laid
// out back to back in append order. This is the layout TensorCache.Get
// documents and the layout KVCache.Get produces.
func wantHeadMajor(heads, dim int, steps [][]float32, stepTokens []int) []float32 {
	total := 0
	for _, n := range stepTokens {
		total += n
	}
	out := make([]float32, 0, heads*total*dim)
	for h := range heads {
		for s, data := range steps {
			n := stepTokens[s]
			off := h * n * dim
			out = append(out, data[off:off+n*dim]...)
		}
	}
	return out
}

// maxAbsDiff reports the largest elementwise gap between got and want, and the
// index where it occurs. Length mismatch is reported as index -1.
func maxAbsDiff(got, want []float32) (float32, int) {
	if len(got) != len(want) {
		return -1, -1
	}
	var worst float32
	idx := 0
	for i := range want {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d > worst {
			worst, idx = d, i
		}
	}
	return worst, idx
}

// TestTensorCache_MultiHeadStridedLayout is the regression test for #981.
//
// TensorCache allocates [batch, maxSeqLen, dim] but used to append at a flat
// token-major offset (seqLen*dim*batch) and read back a compacted head-major
// view. The three layouts coincide only at batch == 1, so the defect stayed
// invisible until GroupedQueryAttention folded numKVHeads > 1 into the batch
// axis, at which point a decode append landed in the wrong head's region.
//
// The test drives a multi-head prefill plus several single-token decodes and
// checks the readback against both an analytically-constructed expectation and
// KVCache, the reference implementation the layout is meant to match.
func TestTensorCache_MultiHeadStridedLayout(t *testing.T) {
	const (
		heads     = 3
		dim       = 4
		maxSeqLen = 16
	)

	stepTokens := []int{3, 1, 1, 1}
	kSteps := make([][]float32, len(stepTokens))
	vSteps := make([][]float32, len(stepTokens))
	for i, n := range stepTokens {
		kSteps[i] = kvFixture(1+i*1000, heads, n, dim)
		vSteps[i] = kvFixture(500001+i*1000, heads, n, dim)
	}

	wantK := wantHeadMajor(heads, dim, kSteps, stepTokens)
	wantV := wantHeadMajor(heads, dim, vSteps, stepTokens)
	assertDistinct(t, "wantK", wantK)
	assertDistinct(t, "wantV", wantV)

	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, maxSeqLen)
	ref := NewKVCache[float32](1, maxSeqLen)

	cursor := 0
	for i, n := range stepTokens {
		shape := []int{heads, n, dim}
		if err := cache.Update(0, makeTensor(t, shape, kSteps[i]), makeTensor(t, shape, vSteps[i])); err != nil {
			t.Fatalf("step %d: TensorCache.Update: %v", i, err)
		}
		if err := ref.Update(0, makeTensor(t, shape, kSteps[i]), makeTensor(t, shape, vSteps[i])); err != nil {
			t.Fatalf("step %d: KVCache.Update: %v", i, err)
		}
		cursor += n

		if got := cache.SeqLen(); got != cursor {
			t.Fatalf("step %d: SeqLen = %d, want %d", i, got, cursor)
		}

		lkv, ok := cache.Get(0)
		if !ok {
			t.Fatalf("step %d: Get(0) returned false", i)
		}
		if got := lkv.Key.Shape(); len(got) != 3 || got[0] != heads || got[1] != cursor || got[2] != dim {
			t.Fatalf("step %d: Key shape = %v, want [%d %d %d]", i, got, heads, cursor, dim)
		}

		// Expected readback is the head-major prefix of the full expectation:
		// head h occupies [h*cursor*dim, (h+1)*cursor*dim) once compacted.
		expK := make([]float32, 0, heads*cursor*dim)
		expV := make([]float32, 0, heads*cursor*dim)
		full := len(wantK) / heads
		for h := range heads {
			expK = append(expK, wantK[h*full:h*full+cursor*dim]...)
			expV = append(expV, wantV[h*full:h*full+cursor*dim]...)
		}

		if d, idx := maxAbsDiff(lkv.Key.Data(), expK); d != 0 {
			t.Errorf("step %d: Key layout wrong: maxAbsDiff=%v at index %d (got %v, want %v)",
				i, d, idx, lkv.Key.Data(), expK)
		}
		if d, idx := maxAbsDiff(lkv.Value.Data(), expV); d != 0 {
			t.Errorf("step %d: Value layout wrong: maxAbsDiff=%v at index %d (got %v, want %v)",
				i, d, idx, lkv.Value.Data(), expV)
		}

		// Differential check against KVCache, which implements the same
		// [batch, maxSeqLen, dim] layout and is the reference for this fix.
		refKV, ok := ref.Get(0)
		if !ok {
			t.Fatalf("step %d: KVCache.Get(0) returned false", i)
		}
		if d, idx := maxAbsDiff(lkv.Key.Data(), refKV.Key.Data()); d != 0 {
			t.Errorf("step %d: Key diverges from KVCache: maxAbsDiff=%v at index %d", i, d, idx)
		}
		if d, idx := maxAbsDiff(lkv.Value.Data(), refKV.Value.Data()); d != 0 {
			t.Errorf("step %d: Value diverges from KVCache: maxAbsDiff=%v at index %d", i, d, idx)
		}
	}
}

// TestTensorCache_MultiHeadStridedLayout_AfterTruncate checks that rolling the
// cursor back and re-appending keeps each head's tokens in that head's own
// region rather than sliding into a neighbour's.
func TestTensorCache_MultiHeadStridedLayout_AfterTruncate(t *testing.T) {
	const (
		heads     = 2
		dim       = 3
		maxSeqLen = 8
	)

	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, maxSeqLen)

	prefill := kvFixture(1, heads, 3, dim)
	if err := cache.Update(0, makeTensor(t, []int{heads, 3, dim}, prefill), makeTensor(t, []int{heads, 3, dim}, prefill)); err != nil {
		t.Fatalf("prefill: %v", err)
	}

	cache.Truncate(2)

	replaced := kvFixture(90001, heads, 1, dim)
	if err := cache.Update(0, makeTensor(t, []int{heads, 1, dim}, replaced), makeTensor(t, []int{heads, 1, dim}, replaced)); err != nil {
		t.Fatalf("re-append: %v", err)
	}

	// Head h keeps prefill tokens 0-1, then the re-appended token at slot 2.
	exp := make([]float32, 0, heads*3*dim)
	for h := range heads {
		exp = append(exp, prefill[h*3*dim:h*3*dim+2*dim]...)
		exp = append(exp, replaced[h*dim:(h+1)*dim]...)
	}
	assertDistinct(t, "exp", exp)

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) returned false")
	}
	if d, idx := maxAbsDiff(lkv.Key.Data(), exp); d != 0 {
		t.Errorf("Key layout wrong after Truncate: maxAbsDiff=%v at index %d (got %v, want %v)",
			d, idx, lkv.Key.Data(), exp)
	}
}

// TestTensorCache_FP16_MultiHeadStridedLayout is the FP16/GPU counterpart of
// TestTensorCache_MultiHeadStridedLayout. It skips when no GPU is available.
func TestTensorCache_FP16_MultiHeadStridedLayout(t *testing.T) {
	const (
		heads     = 3
		dim       = 4
		maxSeqLen = 16
	)

	stepTokens := []int{3, 1, 1}
	kSteps := make([][]float32, len(stepTokens))
	for i, n := range stepTokens {
		// Keep values small: FP16 has 11 bits of mantissa, so integers above
		// 2048 are no longer exactly representable.
		kSteps[i] = kvFixture(1+i*100, heads, n, dim)
	}

	wantK := wantHeadMajor(heads, dim, kSteps, stepTokens)
	assertDistinct(t, "wantK", wantK)
	for _, v := range wantK {
		if v > 2048 {
			t.Fatalf("fixture value %v exceeds exact FP16 integer range", v)
		}
	}

	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, maxSeqLen, WithKVDtype("fp16"))

	cursor := 0
	for i, n := range stepTokens {
		shape := []int{heads, n, dim}
		k := makeGPUTensor(t, shape, kSteps[i])
		if err := cache.Update(0, k, k); err != nil {
			t.Fatalf("step %d: Update: %v", i, err)
		}
		cursor += n
	}
	if got := cache.SeqLen(); got != cursor {
		t.Fatalf("SeqLen = %d, want %d", got, cursor)
	}

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) returned false")
	}
	if d, idx := maxAbsDiff(lkv.Key.Data(), wantK); d > 0.01 {
		t.Errorf("FP16 Key layout wrong: maxAbsDiff=%v at index %d (got %v, want %v)",
			d, idx, lkv.Key.Data(), wantK)
	}
}

func TestTensorCache_FP16_Free(t *testing.T) {
	eng := compute.NewCPUEngine(numeric.Float32Ops{})
	cache := NewTensorCache[float32](eng, 1, 128, WithKVDtype("fp16"))

	k := makeGPUTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeGPUTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})

	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	// Get triggers scratch buffer allocation.
	_, _ = cache.Get(0)

	// Free should not panic.
	cache.Free()

	// After free, Get should return false.
	if _, ok := cache.Get(0); ok {
		t.Error("Get(0) after Free should return false")
	}
}
