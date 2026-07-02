package generate

import (
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestTieredKVAdapter_CacheProviderCompliance(t *testing.T) {
	// Compile-time interface assertion.
	var _ CacheProvider[float32] = (*tieredKVAdapter[float32])(nil)

	engine := newTestEngine()
	store, err := NewTieredKVStore[float32](engine, TieredKVStoreConfig{
		NumLayers: 2,
		MaxSeqLen: 16,
		ChunkSize: 4,
	})
	if err != nil {
		t.Fatalf("NewTieredKVStore: %v", err)
	}
	defer store.Close()

	var cache CacheProvider[float32] = &tieredKVAdapter[float32]{store: store}

	// SeqLen starts at 0.
	if got := cache.SeqLen(); got != 0 {
		t.Fatalf("SeqLen() = %d, want 0", got)
	}

	// Update layer 0.
	k, err := tensor.New([]int{1, 1, 2}, []float32{1, 2})
	if err != nil {
		t.Fatalf("tensor.New key: %v", err)
	}
	v, err := tensor.New([]int{1, 1, 2}, []float32{3, 4})
	if err != nil {
		t.Fatalf("tensor.New value: %v", err)
	}
	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	// Get returns the data.
	lkv, ok := cache.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true after Update")
	}
	if s := lkv.Key.Shape(); s[0] != 1 || s[1] != 1 || s[2] != 2 {
		t.Errorf("Key shape = %v, want [1 1 2]", s)
	}

	// SeqLen is now 1.
	if got := cache.SeqLen(); got != 1 {
		t.Errorf("SeqLen() = %d, want 1", got)
	}

	// Truncate back to 0.
	cache.Truncate(0)
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() after Truncate(0) = %d, want 0", got)
	}

	// Re-populate and Reset.
	if err := cache.Update(0, k, v); err != nil {
		t.Fatalf("Update after truncate: %v", err)
	}
	cache.Reset()
	if got := cache.SeqLen(); got != 0 {
		t.Errorf("SeqLen() after Reset = %d, want 0", got)
	}
}

func TestWithTieredKV_OptionAccepted(t *testing.T) {
	cfg := TieredKVStoreConfig{
		ChunkSize:        32,
		DemoteThreshold:  3,
		PromoteThreshold: 8,
	}
	opt := WithTieredKV(cfg)

	// Apply the option and verify it sets tieredKVCfg.
	var gopts generatorOptions
	opt(&gopts)

	if gopts.tieredKVCfg == nil {
		t.Fatal("WithTieredKV did not set tieredKVCfg")
	}
	if gopts.tieredKVCfg.ChunkSize != 32 {
		t.Errorf("ChunkSize = %d, want 32", gopts.tieredKVCfg.ChunkSize)
	}
	if gopts.tieredKVCfg.DemoteThreshold != 3 {
		t.Errorf("DemoteThreshold = %d, want 3", gopts.tieredKVCfg.DemoteThreshold)
	}
	if gopts.tieredKVCfg.PromoteThreshold != 8 {
		t.Errorf("PromoteThreshold = %d, want 8", gopts.tieredKVCfg.PromoteThreshold)
	}

	// Verify NewGenerator accepts the option without panicking.
	gen := NewGenerator[float32](nil, nil, newTestEngine(), ModelConfig{
		NumLayers: 4,
		MaxSeqLen: 64,
		VocabSize: 100,
	}, opt)
	if gen.tieredKVCfg == nil {
		t.Fatal("Generator.tieredKVCfg is nil after WithTieredKV")
	}
}
