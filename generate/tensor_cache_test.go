package generate

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
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

func makeTensorF32(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	tn, err := tensor.New(shape, data)
	if err != nil {
		t.Fatalf("tensor.New(%v) error: %v", shape, err)
	}
	return tn
}
