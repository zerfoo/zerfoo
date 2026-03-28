package inference

import (
	"context"
	"sync"
	"testing"
)

// mockTransfer records which expert IDs were requested for prefetch.
type mockTransfer struct {
	mu          sync.Mutex
	transferred []int
}

func (m *mockTransfer) transfer(_ context.Context, expertID int) error {
	m.mu.Lock()
	m.transferred = append(m.transferred, expertID)
	m.mu.Unlock()
	return nil
}

func (m *mockTransfer) ids() []int {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]int, len(m.transferred))
	copy(out, m.transferred)
	return out
}

// newTestDeviceMap creates a MoEDeviceMap where shared experts are on GPU
// and routed experts are on CPU.
func newTestDeviceMap(numExperts, numShared int) *MoEDeviceMap {
	dm := &MoEDeviceMap{Experts: make(map[int]DeviceType, numExperts)}
	for i := 0; i < numExperts; i++ {
		if i < numShared {
			dm.Experts[i] = GPU
			dm.SharedExperts = append(dm.SharedExperts, i)
		} else {
			dm.Experts[i] = CPU
			dm.RoutedExperts = append(dm.RoutedExperts, i)
		}
	}
	return dm
}

func TestPrefetchStickyRouting(t *testing.T) {
	// 8 experts, 2 shared (GPU), 6 routed (CPU).
	dm := newTestDeviceMap(8, 2)
	mt := &mockTransfer{}
	pf := NewExpertPrefetcher(dm, mt.transfer)

	ctx := context.Background()

	// Layer 0: route to experts 3, 5 (both CPU).
	prefetched := pf.RecordAndPrefetch(ctx, 0, []int{3, 5})

	// Should have prefetched experts 3 and 5.
	if len(prefetched) != 2 {
		t.Fatalf("layer 0: expected 2 prefetched, got %d", len(prefetched))
	}
	pset := toSet(prefetched)
	if _, ok := pset[3]; !ok {
		t.Error("layer 0: expected expert 3 in prefetch set")
	}
	if _, ok := pset[5]; !ok {
		t.Error("layer 0: expected expert 5 in prefetch set")
	}

	// Layer 1: same experts (sticky) — prediction should be correct.
	pf.CheckPrediction(1, []int{3, 5})
	if rate := pf.Stats.HitRate(); rate != 1.0 {
		t.Errorf("layer 1: expected 100%% hit rate, got %.2f", rate)
	}
}

func TestPrefetchPartialHit(t *testing.T) {
	dm := newTestDeviceMap(8, 2)
	mt := &mockTransfer{}
	pf := NewExpertPrefetcher(dm, mt.transfer)

	ctx := context.Background()

	// Layer 0: route to experts 3, 4, 5.
	pf.RecordAndPrefetch(ctx, 0, []int{3, 4, 5})

	// Layer 1: actual routing is 3, 4, 6 — expert 5 was wrong, 6 was unpredicted.
	pf.CheckPrediction(1, []int{3, 4, 6})

	hits := pf.Stats.Hits.Load()
	misses := pf.Stats.Misses.Load()
	if hits != 2 {
		t.Errorf("expected 2 hits, got %d", hits)
	}
	if misses != 1 {
		t.Errorf("expected 1 miss, got %d", misses)
	}
}

func TestPrefetchSkipsGPUExperts(t *testing.T) {
	dm := newTestDeviceMap(8, 2)
	mt := &mockTransfer{}
	pf := NewExpertPrefetcher(dm, mt.transfer)

	ctx := context.Background()

	// Route to expert 0 (GPU shared) and 3 (CPU routed).
	prefetched := pf.RecordAndPrefetch(ctx, 0, []int{0, 3})

	// Only expert 3 should be prefetched.
	if len(prefetched) != 1 {
		t.Fatalf("expected 1 prefetched, got %d: %v", len(prefetched), prefetched)
	}
	if prefetched[0] != 3 {
		t.Errorf("expected expert 3, got %d", prefetched[0])
	}

	// Transfer should only have been called for expert 3.
	ids := mt.ids()
	if len(ids) != 1 || ids[0] != 3 {
		t.Errorf("expected transfer for [3], got %v", ids)
	}
}

func TestPrefetchNoDuplicateTransfers(t *testing.T) {
	dm := newTestDeviceMap(8, 2)
	mt := &mockTransfer{}
	pf := NewExpertPrefetcher(dm, mt.transfer)

	ctx := context.Background()

	// Layer 0: route to expert 3.
	pf.RecordAndPrefetch(ctx, 0, []int{3})

	// Layer 1: route to expert 3 again — should NOT trigger a second transfer
	// because expert 3 is still pending.
	prefetched := pf.RecordAndPrefetch(ctx, 1, []int{3})
	if len(prefetched) != 0 {
		t.Errorf("expected 0 prefetched (already pending), got %d", len(prefetched))
	}

	ids := mt.ids()
	if len(ids) != 1 {
		t.Errorf("expected 1 transfer total, got %d", len(ids))
	}
}

func TestPrefetchDeepSeekV3StickyPattern(t *testing.T) {
	// Simulate DeepSeek-V3 style routing: 160 routed experts, 2 shared.
	// Expert stickiness is ~70-80% across layers.
	numExperts := 64
	numShared := 2
	numRouted := numExperts - numShared
	topK := 6
	numLayers := 20
	stickyRate := 0.75 // 75% of experts stay the same across layers

	dm := newTestDeviceMap(numExperts, numShared)
	pf := NewExpertPrefetcher(dm, nil) // no actual transfer needed

	ctx := context.Background()

	// Generate synthetic routing with stickiness.
	// Start with a random set of routed experts.
	currentExperts := make([]int, topK)
	for i := range currentExperts {
		currentExperts[i] = numShared + (i * numRouted / topK) // spread across range
	}

	for layer := 0; layer < numLayers; layer++ {
		pf.RecordAndPrefetch(ctx, layer, currentExperts)

		// Generate next layer's routing with stickiness.
		nextExperts := make([]int, topK)
		stickyCount := int(float64(topK) * stickyRate)
		copy(nextExperts[:stickyCount], currentExperts[:stickyCount])

		// Fill remaining with different experts.
		used := toSet(nextExperts[:stickyCount])
		idx := stickyCount
		for e := numShared; e < numExperts && idx < topK; e++ {
			if _, ok := used[e]; !ok {
				nextExperts[idx] = e
				idx++
			}
		}

		if layer > 0 {
			pf.CheckPrediction(layer, currentExperts)
		}

		currentExperts = nextExperts
	}

	// Check last layer.
	pf.CheckPrediction(numLayers, currentExperts)

	hitRate := pf.Stats.HitRate()
	if hitRate < 0.60 {
		t.Errorf("DeepSeek-V3 sticky pattern: hit rate %.2f < 0.60 threshold", hitRate)
	}
	t.Logf("DeepSeek-V3 sticky pattern: hit rate=%.2f, hits=%d, misses=%d, total=%d",
		hitRate, pf.Stats.Hits.Load(), pf.Stats.Misses.Load(), pf.Stats.Total())
}

func TestPrefetchClearHistory(t *testing.T) {
	dm := newTestDeviceMap(8, 2)
	pf := NewExpertPrefetcher(dm, nil)

	ctx := context.Background()
	pf.RecordAndPrefetch(ctx, 0, []int{3, 5})
	pf.ClearHistory()

	// After clearing, prediction for layer 1 should have no basis.
	pf.CheckPrediction(1, []int{3, 5})
	if pf.Stats.Hits.Load() != 0 {
		t.Error("expected 0 hits after clear")
	}
}

func TestPrefetchStatsReset(t *testing.T) {
	var s PrefetchStats
	s.Hits.Add(10)
	s.Misses.Add(5)

	if s.HitRate() < 0.66 || s.HitRate() > 0.67 {
		t.Errorf("expected ~0.667, got %.3f", s.HitRate())
	}
	if s.Total() != 15 {
		t.Errorf("expected total=15, got %d", s.Total())
	}

	s.Reset()
	if s.HitRate() != 0 {
		t.Error("expected 0 after reset")
	}
	if s.Total() != 0 {
		t.Error("expected total=0 after reset")
	}
}

func TestPrefetchHitRateZeroDivision(t *testing.T) {
	var s PrefetchStats
	if s.HitRate() != 0 {
		t.Errorf("expected 0 for empty stats, got %f", s.HitRate())
	}
}

func TestPrefetchDeduplicatesExpertIDs(t *testing.T) {
	dm := newTestDeviceMap(8, 2)
	mt := &mockTransfer{}
	pf := NewExpertPrefetcher(dm, mt.transfer)

	ctx := context.Background()

	// Route with duplicate expert IDs.
	prefetched := pf.RecordAndPrefetch(ctx, 0, []int{3, 3, 5, 5, 3})

	// Should deduplicate to {3, 5}.
	if len(prefetched) != 2 {
		t.Errorf("expected 2 unique prefetched, got %d: %v", len(prefetched), prefetched)
	}

	ids := mt.ids()
	if len(ids) != 2 {
		t.Errorf("expected 2 transfers, got %d: %v", len(ids), ids)
	}
}

func toSet(ids []int) map[int]struct{} {
	s := make(map[int]struct{}, len(ids))
	for _, id := range ids {
		s[id] = struct{}{}
	}
	return s
}
