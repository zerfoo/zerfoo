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
