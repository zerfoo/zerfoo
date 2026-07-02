package generate

import (
	"os"
	"sync"
	"testing"
	"time"
)

func newTieredStore(t *testing.T, numLayers, maxSeqLen int) *TieredKVStore[float32] {
	t.Helper()
	engine := newTestEngine()
	store, err := NewTieredKVStore[float32](engine, TieredKVStoreConfig{
		NumLayers:        numLayers,
		MaxSeqLen:        maxSeqLen,
		ChunkSize:        2,
		DemoteThreshold:  2,
		PromoteThreshold: 5,
	})
	if err != nil {
		t.Fatalf("NewTieredKVStore: %v", err)
	}
	t.Cleanup(func() { store.Close() })
	return store
}

func TestTieredKVStore_NewAndNumLayers(t *testing.T) {
	store := newTieredStore(t, 4, 128)
	if got := store.NumLayers(); got != 4 {
		t.Errorf("NumLayers() = %d, want 4", got)
	}
}

func TestTieredKVStore_AllLayersStartHot(t *testing.T) {
	store := newTieredStore(t, 3, 128)
	for i := range 3 {
		if got := store.Tier(i); got != TierHot {
			t.Errorf("Tier(%d) = %d, want TierHot", i, got)
		}
	}
}

func TestTieredKVStore_UpdateAndGetHot(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 4}, []float32{1, 2, 3, 4})
	v := makeTensor(t, []int{1, 1, 4}, []float32{5, 6, 7, 8})
	if err := store.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}

	lkv, ok := store.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true")
	}

	kData := lkv.Key.Data()
	want := []float32{1, 2, 3, 4}
	for i, w := range want {
		if kData[i] != w {
			t.Errorf("Key[%d] = %v, want %v", i, kData[i], w)
		}
	}
}

func TestTieredKVStore_GetEmpty(t *testing.T) {
	store := newTieredStore(t, 2, 128)
	_, ok := store.Get(0)
	if ok {
		t.Error("Get on empty store should return false")
	}
}

func TestTieredKVStore_GetOutOfRange(t *testing.T) {
	store := newTieredStore(t, 2, 128)
	_, ok := store.Get(5)
	if ok {
		t.Error("Get(5) with 2 layers should return false")
	}
	_, ok = store.Get(-1)
	if ok {
		t.Error("Get(-1) should return false")
	}
}

func TestTieredKVStore_DemoteHotToWarm(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	if err := store.Demote(0); err != nil {
		t.Fatalf("Demote: %v", err)
	}

	if got := store.Tier(0); got != TierWarm {
		t.Errorf("Tier(0) after demote = %d, want TierWarm", got)
	}

	// Data should still be retrievable from warm tier.
	lkv, ok := store.Get(0)
	if !ok {
		t.Fatal("Get(0) from warm tier should return true")
	}

	kData := lkv.Key.Data()
	if kData[0] != 1 || kData[1] != 2 {
		t.Errorf("Warm tier Key = %v, want [1 2]", kData)
	}
}

func TestTieredKVStore_DemoteWarmToCold(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{10, 20})
	v := makeTensor(t, []int{1, 1, 2}, []float32{30, 40})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Hot → Warm
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	// Warm → Cold
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}

	if got := store.Tier(0); got != TierCold {
		t.Errorf("Tier(0) = %d, want TierCold", got)
	}

	// Data should be retrievable from disk.
	lkv, ok := store.Get(0)
	if !ok {
		t.Fatal("Get(0) from cold tier should return true")
	}

	kData := lkv.Key.Data()
	if kData[0] != 10 || kData[1] != 20 {
		t.Errorf("Cold tier Key = %v, want [10 20]", kData)
	}
	vData := lkv.Value.Data()
	if vData[0] != 30 || vData[1] != 40 {
		t.Errorf("Cold tier Value = %v, want [30 40]", vData)
	}
}

func TestTieredKVStore_PromoteColdToWarm(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{5, 6})
	v := makeTensor(t, []int{1, 1, 2}, []float32{7, 8})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Demote to cold.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if got := store.Tier(0); got != TierCold {
		t.Fatalf("expected TierCold, got %d", got)
	}

	// Promote one level: cold → warm.
	if err := store.Promote(0); err != nil {
		t.Fatal(err)
	}
	if got := store.Tier(0); got != TierWarm {
		t.Errorf("Tier(0) after promote = %d, want TierWarm", got)
	}
}

func TestTieredKVStore_PromoteWarmToHot(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{5, 6})
	v := makeTensor(t, []int{1, 1, 2}, []float32{7, 8})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Demote to warm.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}

	// Promote back to hot.
	if err := store.Promote(0); err != nil {
		t.Fatal(err)
	}
	if got := store.Tier(0); got != TierHot {
		t.Errorf("Tier(0) after promote = %d, want TierHot", got)
	}

	// Data should be accessible from hot tier.
	lkv, ok := store.Get(0)
	if !ok {
		t.Fatal("Get(0) should return true after promotion to hot")
	}
	kData := lkv.Key.Data()
	if kData[0] != 5 || kData[1] != 6 {
		t.Errorf("Key = %v, want [5 6]", kData)
	}
}

func TestTieredKVStore_UpdatePromotesColdToHot(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Demote to cold.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}

	// Update should auto-promote to hot.
	k2 := makeTensor(t, []int{1, 1, 2}, []float32{10, 20})
	v2 := makeTensor(t, []int{1, 1, 2}, []float32{30, 40})
	if err := store.Update(0, k2, v2); err != nil {
		t.Fatal(err)
	}

	if got := store.Tier(0); got != TierHot {
		t.Errorf("Tier(0) after Update = %d, want TierHot", got)
	}
}

func TestTieredKVStore_ManageTiers_DemotesLowAccess(t *testing.T) {
	store := newTieredStore(t, 2, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	// Update both layers once (accessCount = 1, below demoteThreshold of 2).
	for layer := range 2 {
		if err := store.Update(layer, k, v); err != nil {
			t.Fatal(err)
		}
	}

	// Access layer 1 enough times to stay above threshold.
	for range 2 {
		store.Get(1)
	}

	if err := store.ManageTiers(); err != nil {
		t.Fatal(err)
	}

	// Layer 0: accessCount was 1 (below 2) → demoted to warm.
	if got := store.Tier(0); got != TierWarm {
		t.Errorf("Tier(0) = %d, want TierWarm (demoted)", got)
	}
	// Layer 1: accessCount was 3 (1 update + 2 gets, above 2) → stays hot.
	if got := store.Tier(1); got != TierHot {
		t.Errorf("Tier(1) = %d, want TierHot (not demoted)", got)
	}
}

func TestTieredKVStore_ManageTiers_PromotesHighAccess(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Demote to warm.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}

	// Access many times (>= promoteThreshold of 5).
	for range 5 {
		store.Get(0)
	}

	if err := store.ManageTiers(); err != nil {
		t.Fatal(err)
	}

	// Should be promoted from warm to hot.
	if got := store.Tier(0); got != TierHot {
		t.Errorf("Tier(0) = %d, want TierHot (promoted)", got)
	}
}

func TestTieredKVStore_Reset(t *testing.T) {
	store := newTieredStore(t, 2, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	store.Reset()

	if got := store.SeqLen(); got != 0 {
		t.Errorf("SeqLen() after Reset = %d, want 0", got)
	}
	// Check access counts and tiers before calling Get (which increments accessCount).
	for i := range 2 {
		if got := store.Tier(i); got != TierHot {
			t.Errorf("Tier(%d) after Reset = %d, want TierHot", i, got)
		}
		if got := store.AccessCount(i); got != 0 {
			t.Errorf("AccessCount(%d) after Reset = %d, want 0", i, got)
		}
	}
	_, ok := store.Get(0)
	if ok {
		t.Error("Get(0) after Reset should return false")
	}
}

func TestTieredKVStore_SeqLen(t *testing.T) {
	store := newTieredStore(t, 1, 128)
	if got := store.SeqLen(); got != 0 {
		t.Errorf("SeqLen() on empty = %d, want 0", got)
	}

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if got := store.SeqLen(); got != 1 {
		t.Errorf("SeqLen() after 1 token = %d, want 1", got)
	}
}

func TestTieredKVStore_Truncate(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	for i := range 5 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(i), float32(i)})
		v := makeTensor(t, []int{1, 1, 2}, []float32{float32(i), float32(i)})
		if err := store.Update(0, k, v); err != nil {
			t.Fatal(err)
		}
	}

	store.Truncate(3)
	if got := store.SeqLen(); got != 3 {
		t.Errorf("SeqLen() after Truncate(3) = %d, want 3", got)
	}
}

func TestTieredKVStore_DemoteEmptyLayer(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	// Demote an empty layer should not error.
	if err := store.Demote(0); err != nil {
		t.Fatalf("Demote empty layer: %v", err)
	}
	if got := store.Tier(0); got != TierWarm {
		t.Errorf("Tier(0) = %d, want TierWarm", got)
	}
}

func TestTieredKVStore_DemoteOutOfRange(t *testing.T) {
	store := newTieredStore(t, 1, 128)
	if err := store.Demote(5); err == nil {
		t.Error("Demote(5) should return error")
	}
	if err := store.Demote(-1); err == nil {
		t.Error("Demote(-1) should return error")
	}
}

func TestTieredKVStore_PromoteOutOfRange(t *testing.T) {
	store := newTieredStore(t, 1, 128)
	if err := store.Promote(5); err == nil {
		t.Error("Promote(5) should return error")
	}
}

func TestTieredKVStore_PromoteAlreadyHot(t *testing.T) {
	store := newTieredStore(t, 1, 128)
	// Promoting a hot layer is a no-op.
	if err := store.Promote(0); err != nil {
		t.Fatalf("Promote hot layer: %v", err)
	}
	if got := store.Tier(0); got != TierHot {
		t.Errorf("Tier(0) = %d, want TierHot", got)
	}
}

func TestTieredKVStore_DemoteColdIsNoop(t *testing.T) {
	store := newTieredStore(t, 1, 128)
	// Demote twice to reach cold.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	// Demoting cold should be a no-op.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if got := store.Tier(0); got != TierCold {
		t.Errorf("Tier(0) = %d, want TierCold", got)
	}
}

func TestTieredKVStore_ColdRoundTrip(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	// Insert 2 tokens (matching chunkSize=2) so compression produces
	// a single mean-pooled vector. Use identical values so the mean
	// equals the original data for easy verification.
	k1 := makeTensor(t, []int{1, 1, 3}, []float32{1, 2, 3})
	v1 := makeTensor(t, []int{1, 1, 3}, []float32{7, 8, 9})
	k2 := makeTensor(t, []int{1, 1, 3}, []float32{1, 2, 3})
	v2 := makeTensor(t, []int{1, 1, 3}, []float32{7, 8, 9})
	if err := store.Update(0, k1, v1); err != nil {
		t.Fatal(err)
	}
	if err := store.Update(0, k2, v2); err != nil {
		t.Fatal(err)
	}

	// Demote to cold (hot → warm compresses, warm → cold writes to disk).
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}

	// Read back from cold.
	lkv, ok := store.Get(0)
	if !ok {
		t.Fatal("Get from cold should succeed")
	}

	// After compression: mean([1,2,3],[1,2,3]) = [1,2,3].
	wantK := []float32{1, 2, 3}
	kData := lkv.Key.Data()
	for i, w := range wantK {
		if kData[i] != w {
			t.Errorf("Key[%d] = %v, want %v", i, kData[i], w)
		}
	}

	wantV := []float32{7, 8, 9}
	vData := lkv.Value.Data()
	for i, w := range wantV {
		if vData[i] != w {
			t.Errorf("Value[%d] = %v, want %v", i, vData[i], w)
		}
	}
}

func TestTieredKVStore_MultiLayer(t *testing.T) {
	store := newTieredStore(t, 3, 128)

	for layer := range 3 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer * 10), float32(layer*10 + 1)})
		v := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer * 20), float32(layer*20 + 1)})
		if err := store.Update(layer, k, v); err != nil {
			t.Fatal(err)
		}
	}

	// Demote layer 1 to warm, layer 2 to cold.
	if err := store.Demote(1); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(2); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(2); err != nil {
		t.Fatal(err)
	}

	// All three layers should be readable from their respective tiers.
	for layer := range 3 {
		lkv, ok := store.Get(layer)
		if !ok {
			t.Errorf("Get(%d) should return true", layer)
			continue
		}
		kData := lkv.Key.Data()
		wantFirst := float32(layer * 10)
		if kData[0] != wantFirst {
			t.Errorf("Layer %d Key[0] = %v, want %v", layer, kData[0], wantFirst)
		}
	}
}

func TestTieredKVStore_AccessCountTracking(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	// Update increments access count.
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if got := store.AccessCount(0); got != 1 {
		t.Errorf("AccessCount after Update = %d, want 1", got)
	}

	// Get increments access count.
	store.Get(0)
	if got := store.AccessCount(0); got != 2 {
		t.Errorf("AccessCount after Get = %d, want 2", got)
	}
}

func TestTieredKVStore_DefaultConfig(t *testing.T) {
	engine := newTestEngine()
	store, err := NewTieredKVStore[float32](engine, TieredKVStoreConfig{
		NumLayers: 1,
		MaxSeqLen: 64,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	// Defaults should be applied.
	if store.chunkSize != 64 {
		t.Errorf("chunkSize = %d, want 64 (default)", store.chunkSize)
	}
	if store.demoteThreshold != 2 {
		t.Errorf("demoteThreshold = %d, want 2 (default)", store.demoteThreshold)
	}
	if store.promoteThreshold != 5 {
		t.Errorf("promoteThreshold = %d, want 5 (default)", store.promoteThreshold)
	}
}

func TestTieredKVStore_UpdateOutOfRange(t *testing.T) {
	store := newTieredStore(t, 1, 128)
	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})

	if err := store.Update(5, k, v); err == nil {
		t.Error("Update(5) should return error")
	}
	if err := store.Update(-1, k, v); err == nil {
		t.Error("Update(-1) should return error")
	}
}

func TestTieredKVStore_PrefetchAsyncFromCold(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{10, 20})
	v := makeTensor(t, []int{1, 1, 2}, []float32{30, 40})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Demote to cold.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if got := store.Tier(0); got != TierCold {
		t.Fatalf("Tier(0) = %d, want TierCold", got)
	}

	// Trigger async prefetch.
	store.PrefetchAsync([]int{0})

	// Wait for the background goroutine to process.
	deadline := time.Now().Add(2 * time.Second)
	var lkv *LayerKV[float32]
	var ok bool
	for time.Now().Before(deadline) {
		lkv, ok = store.GetPrefetched(0)
		if ok {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !ok {
		t.Fatal("PrefetchAsync did not produce data within timeout")
	}

	kData := lkv.Key.Data()
	if kData[0] != 10 || kData[1] != 20 {
		t.Errorf("Prefetched Key = %v, want [10 20]", kData)
	}
	vData := lkv.Value.Data()
	if vData[0] != 30 || vData[1] != 40 {
		t.Errorf("Prefetched Value = %v, want [30 40]", vData)
	}
}

func TestTieredKVStore_PrefetchAsyncFromWarm(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{5, 6})
	v := makeTensor(t, []int{1, 1, 2}, []float32{7, 8})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Demote to warm.
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if got := store.Tier(0); got != TierWarm {
		t.Fatalf("Tier(0) = %d, want TierWarm", got)
	}

	store.PrefetchAsync([]int{0})

	deadline := time.Now().Add(2 * time.Second)
	var lkv *LayerKV[float32]
	var ok bool
	for time.Now().Before(deadline) {
		lkv, ok = store.GetPrefetched(0)
		if ok {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !ok {
		t.Fatal("PrefetchAsync from warm did not produce data within timeout")
	}

	kData := lkv.Key.Data()
	if kData[0] != 5 || kData[1] != 6 {
		t.Errorf("Prefetched Key = %v, want [5 6]", kData)
	}
}

func TestTieredKVStore_PrefetchAsyncSkipsHot(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}

	// Layer is hot — prefetch should be a no-op.
	store.PrefetchAsync([]int{0})
	time.Sleep(50 * time.Millisecond)

	_, ok := store.GetPrefetched(0)
	if ok {
		t.Error("PrefetchAsync should skip hot layers")
	}
}

func TestTieredKVStore_PrefetchAsyncOutOfRange(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	// Should not panic for invalid positions.
	store.PrefetchAsync([]int{-1, 5, 100})
	time.Sleep(50 * time.Millisecond)
}

func TestTieredKVStore_PrefetchAsyncMultipleLayers(t *testing.T) {
	store := newTieredStore(t, 3, 128)

	for layer := range 3 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer * 10), float32(layer*10 + 1)})
		v := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer * 20), float32(layer*20 + 1)})
		if err := store.Update(layer, k, v); err != nil {
			t.Fatal(err)
		}
	}

	// Demote layers 0 and 2 to cold, keep 1 hot.
	for _, layer := range []int{0, 2} {
		if err := store.Demote(layer); err != nil {
			t.Fatal(err)
		}
		if err := store.Demote(layer); err != nil {
			t.Fatal(err)
		}
	}

	// Prefetch all three — only 0 and 2 should be fetched.
	store.PrefetchAsync([]int{0, 1, 2})

	deadline := time.Now().Add(2 * time.Second)
	got := make(map[int]bool)
	for time.Now().Before(deadline) {
		for _, layer := range []int{0, 2} {
			if !got[layer] {
				if _, ok := store.GetPrefetched(layer); ok {
					got[layer] = true
				}
			}
		}
		if len(got) == 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if len(got) != 2 {
		t.Errorf("expected 2 prefetched layers, got %d", len(got))
	}

	// Layer 1 was hot, should not be prefetched.
	_, ok := store.GetPrefetched(1)
	if ok {
		t.Error("Layer 1 (hot) should not have been prefetched")
	}
}

func TestTieredKVStore_PrefetchAsyncDedup(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}

	// First prefetch.
	store.PrefetchAsync([]int{0})
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, ok := store.GetPrefetched(0); ok {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Second prefetch of same layer should still work after consuming.
	store.PrefetchAsync([]int{0})
	for time.Now().Before(deadline) {
		if _, ok := store.GetPrefetched(0); ok {
			return // success
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("second PrefetchAsync did not produce data")
}

func TestTieredKVStore_ConcurrentAccess(t *testing.T) {
	store := newTieredStore(t, 4, 128)

	// Seed all layers with data.
	for layer := range 4 {
		k := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer), float32(layer + 1)})
		v := makeTensor(t, []int{1, 1, 2}, []float32{float32(layer + 10), float32(layer + 11)})
		if err := store.Update(layer, k, v); err != nil {
			t.Fatal(err)
		}
	}

	// Demote layers 2 and 3 to warm and cold respectively.
	if err := store.Demote(2); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(3); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(3); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	const goroutines = 8
	const iterations = 50

	// Concurrent reads from all tiers.
	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			layer := id % 4
			for range iterations {
				store.Get(layer)
			}
		}(g)
	}

	// Concurrent updates.
	for g := range goroutines / 2 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			layer := id % 2
			for range iterations {
				k := makeTensor(t, []int{1, 1, 2}, []float32{float32(id), float32(id + 1)})
				v := makeTensor(t, []int{1, 1, 2}, []float32{float32(id + 10), float32(id + 11)})
				store.Update(layer, k, v)
			}
		}(g)
	}

	// Concurrent tier queries.
	for range goroutines / 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range iterations {
				for layer := range 4 {
					store.Tier(layer)
				}
			}
		}()
	}

	wg.Wait()

	// Store should still be in a consistent state.
	for layer := range 4 {
		tier := store.Tier(layer)
		if tier < TierHot || tier > TierCold {
			t.Errorf("Tier(%d) = %d, out of valid range", layer, tier)
		}
	}
}

func TestTieredKVStore_ResetClearsPrefetched(t *testing.T) {
	store := newTieredStore(t, 1, 128)

	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}
	if err := store.Demote(0); err != nil {
		t.Fatal(err)
	}

	store.PrefetchAsync([]int{0})
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, ok := store.GetPrefetched(0); ok {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Prefetch again, then reset before consuming.
	store.PrefetchAsync([]int{0})
	time.Sleep(50 * time.Millisecond)
	store.Reset()

	_, ok := store.GetPrefetched(0)
	if ok {
		t.Error("GetPrefetched should return false after Reset")
	}
}

func TestTieredKVStore_Close_UserProvidedColdDirNotDeleted(t *testing.T) {
	// Provide an explicit ColdDir so the store does not own it.
	coldDir := t.TempDir()

	engine := newTestEngine()
	store, err := NewTieredKVStore[float32](engine, TieredKVStoreConfig{
		NumLayers: 1,
		MaxSeqLen: 16,
		ChunkSize: 2,
		ColdDir:   coldDir,
	})
	if err != nil {
		t.Fatalf("NewTieredKVStore: %v", err)
	}

	// Write data and demote it to the cold tier so a file is written.
	k := makeTensor(t, []int{1, 1, 2}, []float32{1, 2})
	v := makeTensor(t, []int{1, 1, 2}, []float32{3, 4})
	if err := store.Update(0, k, v); err != nil {
		t.Fatalf("Update: %v", err)
	}
	if err := store.Demote(0); err != nil { // hot -> warm
		t.Fatalf("Demote hot->warm: %v", err)
	}
	if err := store.Demote(0); err != nil { // warm -> cold
		t.Fatalf("Demote warm->cold: %v", err)
	}

	// Verify a cold file was written.
	entries, err := os.ReadDir(coldDir)
	if err != nil {
		t.Fatalf("ReadDir before Close: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected cold-tier file(s) in user-provided ColdDir before Close")
	}

	// Close must NOT delete the user-provided directory or its contents.
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, statErr := os.Stat(coldDir); os.IsNotExist(statErr) {
		t.Error("Close() deleted the user-provided ColdDir; it should be preserved")
	}
	remaining, err := os.ReadDir(coldDir)
	if err != nil {
		t.Fatalf("ReadDir after Close: %v", err)
	}
	if len(remaining) == 0 {
		t.Error("Close() deleted cold-tier files in user-provided ColdDir; they should be preserved")
	}
}
