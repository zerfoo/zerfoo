package generate

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestTracingCacheProvider_InterfaceAssertion(t *testing.T) {
	var _ CacheProvider[float32] = (*TracingCacheProvider[float32])(nil)
}

func TestTracingCacheProvider_UpdateRecordsOps(t *testing.T) {
	cache := NewKVCache[float32](2, 256)
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](cache, tracer)

	k, _ := tensor.New[float32]([]int{1, 4, 64}, make([]float32, 256))
	v, _ := tensor.New[float32]([]int{1, 4, 64}, make([]float32, 256))

	err := tc.Update(0, k, v)
	if err != nil {
		t.Fatalf("Update: %v", err)
	}

	ops := tracer.TracedOps()
	if len(ops) != 2 {
		t.Fatalf("expected 2 traced ops, got %d", len(ops))
	}
	if ops[0].OpName != "KVCacheAppendK" {
		t.Errorf("ops[0].OpName = %q, want KVCacheAppendK", ops[0].OpName)
	}
	if ops[1].OpName != "KVCacheAppendV" {
		t.Errorf("ops[1].OpName = %q, want KVCacheAppendV", ops[1].OpName)
	}
	layer0, ok := ops[0].ExtraArgs["layer"]
	if !ok || layer0 != 0 {
		t.Errorf("ops[0].ExtraArgs[layer] = %v, want 0", layer0)
	}
}

func TestTracingCacheProvider_GetRecordsOps(t *testing.T) {
	cache := NewKVCache[float32](2, 256)
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](cache, tracer)

	// Must update first so Get returns data.
	k, _ := tensor.New[float32]([]int{1, 4, 64}, make([]float32, 256))
	v, _ := tensor.New[float32]([]int{1, 4, 64}, make([]float32, 256))
	_ = tc.Update(0, k, v)

	// Reset traced ops to isolate Get.
	tracer2 := compute.NewTracer[float32](nil)
	tc2 := NewTracingCacheProvider[float32](cache, tracer2)

	kv, ok := tc2.Get(0)
	if !ok {
		t.Fatal("Get returned false")
	}
	if kv == nil {
		t.Fatal("Get returned nil LayerKV")
	}

	ops := tracer2.TracedOps()
	if len(ops) != 2 {
		t.Fatalf("expected 2 traced ops, got %d", len(ops))
	}
	if ops[0].OpName != "KVCacheGetK" {
		t.Errorf("ops[0].OpName = %q, want KVCacheGetK", ops[0].OpName)
	}
	if ops[1].OpName != "KVCacheGetV" {
		t.Errorf("ops[1].OpName = %q, want KVCacheGetV", ops[1].OpName)
	}
}

func TestTracingCacheProvider_SeqLen(t *testing.T) {
	cache := NewKVCache[float32](2, 256)
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](cache, tracer)

	if tc.SeqLen() != 0 {
		t.Errorf("SeqLen = %d, want 0", tc.SeqLen())
	}
}

func TestTracingCacheProvider_ResetAndTruncate(t *testing.T) {
	cache := NewKVCache[float32](2, 256)
	tracer := compute.NewTracer[float32](nil)
	tc := NewTracingCacheProvider[float32](cache, tracer)

	k, _ := tensor.New[float32]([]int{1, 4, 64}, make([]float32, 256))
	v, _ := tensor.New[float32]([]int{1, 4, 64}, make([]float32, 256))
	_ = tc.Update(0, k, v)

	tc.Truncate(0)
	if tc.SeqLen() != 0 {
		t.Errorf("after Truncate(0): SeqLen = %d, want 0", tc.SeqLen())
	}

	_ = tc.Update(0, k, v)
	tc.Reset()
	if tc.SeqLen() != 0 {
		t.Errorf("after Reset: SeqLen = %d, want 0", tc.SeqLen())
	}
}
