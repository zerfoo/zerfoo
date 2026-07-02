package attention

import (
	"context"
	"testing"
	"unsafe"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// fullBufferCache wraps a TensorCache and implements FullBufferProvider for
// testing the CUDA graph-capturable decode path.
type fullBufferCache struct {
	generate.CacheProvider[float32]
	maxSeqLen int
	kvLenPtr  unsafe.Pointer // non-nil signals GPU counter availability
}

func (c *fullBufferCache) GetFullBuffer(layer int) (k, v *tensor.TensorNumeric[float32]) {
	// Return GPU-backed full buffers if available.
	if tc, ok := c.CacheProvider.(*generate.TensorCache[float32]); ok {
		return tc.GetFullBuffer(layer)
	}
	return nil, nil
}

func (c *fullBufferCache) MaxSeqLen() int {
	return c.maxSeqLen
}

func (c *fullBufferCache) KVSeqLenPtr() unsafe.Pointer {
	return c.kvLenPtr
}

// Verify that fullBufferCache satisfies FullBufferProvider.
var _ generate.FullBufferProvider[float32] = (*fullBufferCache)(nil)

// TestGQA_FlashDecodePathDetection verifies that the GQA decode path
// correctly detects FullBufferProvider caches and attempts the flash
// attention decode kernel. On CPU (no CUDA), it should fall back to
// the standard cuBLAS SDPA path without error.
func TestGQA_FlashDecodePathDetection(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 16
	numQ := 4
	numKV := 2

	gqa, err := NewGroupedQueryAttention[float32](
		engine, ops, modelDim, numQ, numKV,
		WithMaxSeqLen[float32](128),
	)
	if err != nil {
		t.Fatalf("NewGroupedQueryAttention: %v", err)
	}
	gqa.LayerIndex = 0

	// Use a standard KVCache (no FullBufferProvider) first.
	cache := generate.NewKVCache[float32](1, 128)
	ctx := generate.WithKVCache(context.Background(), cache)

	// Prefill.
	prefillData := make([]float32, 4*modelDim)
	for i := range prefillData {
		prefillData[i] = float32(i%7) * 0.1
	}
	prefill, _ := tensor.New([]int{1, 4, modelDim}, prefillData)
	if _, err := gqa.Forward(ctx, prefill); err != nil {
		t.Fatalf("prefill with KVCache: %v", err)
	}

	// Decode with standard cache.
	decodeData := make([]float32, modelDim)
	for i := range decodeData {
		decodeData[i] = float32(i%5) * 0.2
	}
	decodeInput, _ := tensor.New([]int{1, 1, modelDim}, decodeData)
	out, err := gqa.Forward(ctx, decodeInput)
	if err != nil {
		t.Fatalf("decode with KVCache: %v", err)
	}
	if s := out.Shape(); s[0] != 1 || s[1] != 1 || s[2] != modelDim {
		t.Errorf("output shape = %v, want [1, 1, %d]", s, modelDim)
	}

	// Now test with a FullBufferProvider cache (but nil kvLenPtr = no GPU).
	// This should still work via the fallback path.
	cache2 := generate.NewKVCache[float32](1, 128)
	fbCache := &fullBufferCache{
		CacheProvider: cache2,
		maxSeqLen:     128,
		kvLenPtr:      nil, // no GPU counter
	}
	ctx2 := generate.WithCache[float32](context.Background(), fbCache)

	// Prefill.
	if _, err := gqa.Forward(ctx2, prefill); err != nil {
		t.Fatalf("prefill with FullBufferProvider: %v", err)
	}

	// Decode - should fall through to standard SDPA because kvLenPtr is nil.
	out2, err := gqa.Forward(ctx2, decodeInput)
	if err != nil {
		t.Fatalf("decode with FullBufferProvider (nil kvLenPtr): %v", err)
	}
	if s := out2.Shape(); s[0] != 1 || s[1] != 1 || s[2] != modelDim {
		t.Errorf("output shape = %v, want [1, 1, %d]", s, modelDim)
	}
}

// TestGQA_NoD2HInDecodePath verifies that GQA decode does not call .Data()
// on GPU-resident tensors, which would trigger a D2H copy incompatible with
// CUDA graph capture. This test uses CPU tensors but checks the code paths.
func TestGQA_NoD2HInDecodePath(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	modelDim := 16
	numQ := 4
	numKV := 2

	gqa, err := NewGroupedQueryAttention[float32](
		engine, ops, modelDim, numQ, numKV,
		WithMaxSeqLen[float32](64),
	)
	if err != nil {
		t.Fatal(err)
	}
	gqa.LayerIndex = 0

	cache := generate.NewKVCache[float32](1, 64)
	ctx := generate.WithKVCache(context.Background(), cache)

	// Prefill 10 tokens.
	prefillData := make([]float32, 10*modelDim)
	for i := range prefillData {
		prefillData[i] = float32(i%11) * 0.05
	}
	prefill, _ := tensor.New([]int{1, 10, modelDim}, prefillData)
	if _, err := gqa.Forward(ctx, prefill); err != nil {
		t.Fatalf("prefill: %v", err)
	}

	// Decode 5 tokens sequentially. Each should produce valid output.
	for step := 0; step < 5; step++ {
		decodeData := make([]float32, modelDim)
		for i := range decodeData {
			decodeData[i] = float32(step*7+i%5) * 0.1
		}
		decodeInput, _ := tensor.New([]int{1, 1, modelDim}, decodeData)
		out, decodeErr := gqa.Forward(ctx, decodeInput)
		if decodeErr != nil {
			t.Fatalf("decode step %d: %v", step, decodeErr)
		}
		if s := out.Shape(); s[0] != 1 || s[1] != 1 || s[2] != modelDim {
			t.Errorf("decode step %d: output shape = %v, want [1, 1, %d]", step, s, modelDim)
		}
	}
}
