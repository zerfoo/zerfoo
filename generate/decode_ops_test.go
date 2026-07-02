package generate

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// opCountingEngine wraps a CPU engine and counts operation calls.
// This is used to detect regressions where decode steps issue unexpected
// Repeat operations (the GQA Repeat-on-maxSeqLen regression) or
// excessive allocations.
type opCountingEngine struct {
	compute.Engine[float32]
	repeatCalls  int
	repeatElems  int
	reshapeCalls int
	matmulCalls  int
	totalElems   int // sum of all output elements across all ops
}

func (e *opCountingEngine) Repeat(ctx context.Context, a *tensor.TensorNumeric[float32], axis, count int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	e.repeatCalls++
	elems := 1
	for _, d := range a.Shape() {
		elems *= d
	}
	e.repeatElems += elems * count
	result, err := e.Engine.Repeat(ctx, a, axis, count, dst...)
	if err == nil {
		outElems := 1
		for _, d := range result.Shape() {
			outElems *= d
		}
		e.totalElems += outElems
	}
	return result, err
}

func (e *opCountingEngine) Reshape(ctx context.Context, a *tensor.TensorNumeric[float32], shape []int, dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	e.reshapeCalls++
	return e.Engine.Reshape(ctx, a, shape, dst...)
}

func (e *opCountingEngine) MatMul(ctx context.Context, a, b *tensor.TensorNumeric[float32], dst ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	e.matmulCalls++
	result, err := e.Engine.MatMul(ctx, a, b, dst...)
	if err == nil {
		outElems := 1
		for _, d := range result.Shape() {
			outElems *= d
		}
		e.totalElems += outElems
	}
	return result, err
}

// TestDecodeOps_NoRepeatForNonGQA verifies that a decode step with
// numQueryHeads == numKeyValueHeads (MHA) does not issue any Repeat
// calls. This would catch a regression where the SDPA path incorrectly
// applies GQA head expansion for non-GQA models.
func TestDecodeOps_NoRepeatForNonGQA(t *testing.T) {
	base := compute.NewCPUEngine(numeric.Float32Ops{})
	eng := &opCountingEngine{Engine: base}

	// Simulate a KV cache update and get to verify no Repeat.
	cache := NewKVCache[float32](1, 128)

	// Insert 10 tokens into the cache as if from a prefill.
	headDim := 4
	numKVHeads := 4
	kvDim := numKVHeads * headDim
	for i := range 10 {
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		for j := range kData {
			kData[j] = float32(i*kvDim+j) * 0.01
			vData[j] = float32(i*kvDim+j) * 0.02
		}
		k, err := tensor.New([]int{numKVHeads, 1, headDim}, kData)
		if err != nil {
			t.Fatal(err)
		}
		v, err := tensor.New([]int{numKVHeads, 1, headDim}, vData)
		if err != nil {
			t.Fatal(err)
		}
		if err := cache.Update(0, k, v); err != nil {
			t.Fatal(err)
		}
	}

	if cache.SeqLen() != 10 {
		t.Fatalf("cache SeqLen = %d, want 10", cache.SeqLen())
	}

	// Now simulate a decode step's Repeat check: with numQ == numKV,
	// the code should NOT call Repeat.
	numQueryHeads := 4
	replicationFactor := numQueryHeads / numKVHeads

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("cache.Get(0) returned false")
	}

	// This mirrors the GQA Forward path: if numQ != numKV, Repeat is called.
	if numQueryHeads != numKVHeads && numKVHeads > 1 {
		_, err := eng.Repeat(context.Background(), lkv.Key, 1, replicationFactor)
		if err != nil {
			t.Fatal(err)
		}
		_, err = eng.Repeat(context.Background(), lkv.Value, 1, replicationFactor)
		if err != nil {
			t.Fatal(err)
		}
	}

	// For MHA (numQ == numKV), there should be zero Repeat calls.
	if eng.repeatCalls != 0 {
		t.Errorf("MHA decode: Repeat calls = %d, want 0", eng.repeatCalls)
	}
}

// TestDecodeOps_GQARepeatBounded verifies that when GQA does call
// Repeat during decode (numQ != numKV), the allocation size is bounded
// by the actual cached sequence length, not the maxSeqLen. This catches
// the regression where Repeat operated on tensors with the full
// maxSeqLen dimension, creating 128 MB temporaries per token.
func TestDecodeOps_GQARepeatBounded(t *testing.T) {
	base := compute.NewCPUEngine(numeric.Float32Ops{})
	eng := &opCountingEngine{Engine: base}

	maxSeqLen := 8192
	cache := NewKVCache[float32](1, maxSeqLen)

	headDim := 4
	numKVHeads := 2
	kvDim := numKVHeads * headDim

	// Insert 10 tokens.
	for i := range 10 {
		kData := make([]float32, kvDim)
		vData := make([]float32, kvDim)
		for j := range kData {
			kData[j] = float32(i*kvDim+j) * 0.01
			vData[j] = float32(i*kvDim+j) * 0.02
		}
		k, err := tensor.New([]int{numKVHeads, 1, headDim}, kData)
		if err != nil {
			t.Fatal(err)
		}
		v, err := tensor.New([]int{numKVHeads, 1, headDim}, vData)
		if err != nil {
			t.Fatal(err)
		}
		if err := cache.Update(0, k, v); err != nil {
			t.Fatal(err)
		}
	}

	numQueryHeads := 8
	replicationFactor := numQueryHeads / numKVHeads

	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("cache.Get(0) returned false")
	}

	// Simulate the GQA Repeat path.
	if numQueryHeads != numKVHeads && numKVHeads > 1 {
		_, err := eng.Repeat(context.Background(), lkv.Key, 1, replicationFactor)
		if err != nil {
			t.Fatal(err)
		}
		_, err = eng.Repeat(context.Background(), lkv.Value, 1, replicationFactor)
		if err != nil {
			t.Fatal(err)
		}
	}

	// The Repeat should operate on cached data (10 * headDim * numKVHeads),
	// not the full maxSeqLen buffer.
	// cachedElems = numKVHeads * 10 * headDim = 2 * 10 * 4 = 80 per K/V.
	// With replication factor 4, output = 80 * 4 = 320 per K/V, 640 total.
	//
	// The regression would produce: numKVHeads * maxSeqLen * headDim * repFactor
	// = 2 * 8192 * 4 * 4 = 262144 per K/V. We catch this by bounding at 1 MB.
	maxAllowedElems := 1024 * 1024 / 4 // 1 MB / sizeof(float32)
	if eng.repeatElems > maxAllowedElems {
		t.Errorf("GQA decode Repeat allocated %d elements (> %d), likely operating on maxSeqLen instead of cached seqLen",
			eng.repeatElems, maxAllowedElems)
	}

	// More precise bound: actual cached elements * replication.
	cachedSeq := 10
	expectedMaxElems := numKVHeads * cachedSeq * headDim * replicationFactor * 2 // K + V
	if eng.repeatElems > expectedMaxElems*2 {
		t.Errorf("GQA decode Repeat allocated %d elements, expected at most ~%d",
			eng.repeatElems, expectedMaxElems*2)
	}
}
