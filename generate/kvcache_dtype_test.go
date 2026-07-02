package generate

import (
	"testing"
)

func TestWithGeneratorKVDtype(t *testing.T) {
	cfg := ModelConfig{
		VocabSize:  32000,
		MaxSeqLen:  128,
		EOSTokenID: 2,
		BOSTokenID: 1,
		NumLayers:  4,
	}

	t.Run("q4 creates KVCacheQ4", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithGeneratorKVDtype("q4"))
		cp, ok := gen.newQuantizedCache()
		if !ok {
			t.Fatal("expected newQuantizedCache to return true for q4")
		}
		if _, isQ4 := any(cp).(*KVCacheQ4); !isQ4 {
			t.Errorf("expected *KVCacheQ4, got %T", cp)
		}
	})

	t.Run("q3 creates KVCacheQ3", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithGeneratorKVDtype("q3"))
		cp, ok := gen.newQuantizedCache()
		if !ok {
			t.Fatal("expected newQuantizedCache to return true for q3")
		}
		if _, isQ3 := any(cp).(*KVCacheQ3); !isQ3 {
			t.Errorf("expected *KVCacheQ3, got %T", cp)
		}
	})

	t.Run("default returns false", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg)
		_, ok := gen.newQuantizedCache()
		if ok {
			t.Error("expected newQuantizedCache to return false with default kvDtype")
		}
	})

	t.Run("fp16 returns false", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithGeneratorKVDtype("fp16"))
		_, ok := gen.newQuantizedCache()
		if ok {
			t.Error("expected newQuantizedCache to return false for fp16")
		}
	})

	t.Run("fp32 returns false", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithGeneratorKVDtype("fp32"))
		_, ok := gen.newQuantizedCache()
		if ok {
			t.Error("expected newQuantizedCache to return false for fp32")
		}
	})

	t.Run("kvDtype stored on generator", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithGeneratorKVDtype("q4"))
		if gen.kvDtype != "q4" {
			t.Errorf("kvDtype = %q, want %q", gen.kvDtype, "q4")
		}
	})

	t.Run("q4 cache implements CacheProvider", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithGeneratorKVDtype("q4"))
		cp, ok := gen.newQuantizedCache()
		if !ok {
			t.Fatal("expected quantized cache")
		}
		// Verify CacheProvider interface methods work.
		if cp.SeqLen() != 0 {
			t.Errorf("SeqLen = %d, want 0", cp.SeqLen())
		}
		cp.Reset()
		cp.Truncate(0)
	})

	t.Run("q3 cache implements CacheProvider", func(t *testing.T) {
		gen := NewGenerator[float32](nil, nil, nil, cfg, WithGeneratorKVDtype("q3"))
		cp, ok := gen.newQuantizedCache()
		if !ok {
			t.Fatal("expected quantized cache")
		}
		if cp.SeqLen() != 0 {
			t.Errorf("SeqLen = %d, want 0", cp.SeqLen())
		}
		cp.Reset()
		cp.Truncate(0)
	})
}
