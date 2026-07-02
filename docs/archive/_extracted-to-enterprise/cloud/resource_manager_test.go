package cloud

import (
	"sync"
	"testing"
)

func TestNewResourceManagerZeroBudget(t *testing.T) {
	_, err := NewResourceManager(0)
	if err == nil {
		t.Fatal("expected error for zero budget")
	}
}

func TestResourceManagerLoadAndStats(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	if err := rm.Load("model-a", 400); err != nil {
		t.Fatalf("Load model-a: %v", err)
	}
	if err := rm.Load("model-b", 300); err != nil {
		t.Fatalf("Load model-b: %v", err)
	}

	used, budget, loaded := rm.Stats()
	if used != 700 {
		t.Errorf("used = %d, want 700", used)
	}
	if budget != 1000 {
		t.Errorf("budget = %d, want 1000", budget)
	}
	if loaded != 2 {
		t.Errorf("loaded = %d, want 2", loaded)
	}
}

func TestResourceManagerEvictsLRU(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	var evicted []string
	rm.SetEvictCallback(func(id string) {
		evicted = append(evicted, id)
	})

	// Load three models totalling 900 bytes.
	if err := rm.Load("model-a", 300); err != nil {
		t.Fatalf("Load model-a: %v", err)
	}
	if err := rm.Load("model-b", 300); err != nil {
		t.Fatalf("Load model-b: %v", err)
	}
	if err := rm.Load("model-c", 300); err != nil {
		t.Fatalf("Load model-c: %v", err)
	}

	// Touch model-a so it becomes MRU. LRU order (back to front): b, c, a.
	if err := rm.Touch("model-a"); err != nil {
		t.Fatalf("Touch model-a: %v", err)
	}

	// Load model-d (400 bytes) — must evict model-b (LRU) to fit.
	if err := rm.Load("model-d", 400); err != nil {
		t.Fatalf("Load model-d: %v", err)
	}

	if len(evicted) != 1 || evicted[0] != "model-b" {
		t.Errorf("evicted = %v, want [model-b]", evicted)
	}

	used, _, loaded := rm.Stats()
	if used != 1000 {
		t.Errorf("used = %d, want 1000", used)
	}
	if loaded != 3 {
		t.Errorf("loaded = %d, want 3", loaded)
	}
}

func TestResourceManagerEvictsMultiple(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	var evicted []string
	rm.SetEvictCallback(func(id string) {
		evicted = append(evicted, id)
	})

	// Fill with small models: 300 + 300 + 300 = 900.
	if err := rm.Load("s1", 300); err != nil {
		t.Fatalf("Load s1: %v", err)
	}
	if err := rm.Load("s2", 300); err != nil {
		t.Fatalf("Load s2: %v", err)
	}
	if err := rm.Load("s3", 300); err != nil {
		t.Fatalf("Load s3: %v", err)
	}

	// Load big (800 bytes): 900 + 800 = 1700 > 1000.
	// Need to free at least 700. s1(300) evicted -> 600 used, still 600+800=1400 > 1000.
	// s2(300) evicted -> 300 used, still 300+800=1100 > 1000.
	// s3(300) evicted -> 0 used, 0+800=800 <= 1000. OK.
	if err := rm.Load("big", 800); err != nil {
		t.Fatalf("Load big: %v", err)
	}

	if len(evicted) != 3 {
		t.Fatalf("evicted count = %d, want 3", len(evicted))
	}

	used, _, loaded := rm.Stats()
	if used != 800 {
		t.Errorf("used = %d, want 800", used)
	}
	if loaded != 1 {
		t.Errorf("loaded = %d, want 1", loaded)
	}
}

func TestResourceManagerEvictsUntilFit(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	var evicted []string
	rm.SetEvictCallback(func(id string) {
		evicted = append(evicted, id)
	})

	// Fill: 300 + 300 + 300 = 900
	if err := rm.Load("s1", 300); err != nil {
		t.Fatalf("Load s1: %v", err)
	}
	if err := rm.Load("s2", 300); err != nil {
		t.Fatalf("Load s2: %v", err)
	}
	if err := rm.Load("s3", 300); err != nil {
		t.Fatalf("Load s3: %v", err)
	}

	// Load 900-byte model: need to evict all three (900 used + 900 new > 1000).
	if err := rm.Load("huge", 900); err != nil {
		t.Fatalf("Load huge: %v", err)
	}

	if len(evicted) != 3 {
		t.Fatalf("evicted count = %d, want 3", len(evicted))
	}

	used, _, loaded := rm.Stats()
	if used != 900 {
		t.Errorf("used = %d, want 900", used)
	}
	if loaded != 1 {
		t.Errorf("loaded = %d, want 1", loaded)
	}
}

func TestResourceManagerModelTooLarge(t *testing.T) {
	rm, err := NewResourceManager(500)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	if err := rm.Load("oversized", 501); err != errModelTooLarge {
		t.Errorf("Load oversized: got %v, want errModelTooLarge", err)
	}
}

func TestResourceManagerReloadIdempotent(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	if err := rm.Load("model-a", 400); err != nil {
		t.Fatalf("Load model-a: %v", err)
	}
	// Loading the same model again should be a no-op (touch).
	if err := rm.Load("model-a", 400); err != nil {
		t.Fatalf("second Load model-a: %v", err)
	}

	used, _, loaded := rm.Stats()
	if used != 400 {
		t.Errorf("used = %d, want 400 (no double-counting)", used)
	}
	if loaded != 1 {
		t.Errorf("loaded = %d, want 1", loaded)
	}
}

func TestResourceManagerExplicitEvict(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	if err := rm.Load("model-a", 400); err != nil {
		t.Fatalf("Load: %v", err)
	}

	if err := rm.Evict("model-a"); err != nil {
		t.Fatalf("Evict: %v", err)
	}

	used, _, loaded := rm.Stats()
	if used != 0 {
		t.Errorf("used = %d, want 0 after evict", used)
	}
	if loaded != 0 {
		t.Errorf("loaded = %d, want 0 after evict", loaded)
	}

	// Evicting again should error.
	if err := rm.Evict("model-a"); err != errModelNotLoaded {
		t.Errorf("second Evict: got %v, want errModelNotLoaded", err)
	}
}

func TestResourceManagerTouchNotLoaded(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	if err := rm.Touch("nonexistent"); err != errModelNotLoaded {
		t.Errorf("Touch: got %v, want errModelNotLoaded", err)
	}
}

func TestResourceManagerLoadedModels(t *testing.T) {
	rm, err := NewResourceManager(1000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	if err := rm.Load("a", 100); err != nil {
		t.Fatalf("Load a: %v", err)
	}
	if err := rm.Load("b", 200); err != nil {
		t.Fatalf("Load b: %v", err)
	}

	models := rm.LoadedModels()
	if len(models) != 2 {
		t.Fatalf("LoadedModels len = %d, want 2", len(models))
	}
	// Front of LRU (most recent) should be first.
	if models[0].ModelID != "b" {
		t.Errorf("models[0].ModelID = %s, want b (MRU)", models[0].ModelID)
	}
	if models[1].ModelID != "a" {
		t.Errorf("models[1].ModelID = %s, want a (LRU)", models[1].ModelID)
	}
}

func TestResourceManagerConcurrentAccess(t *testing.T) {
	rm, err := NewResourceManager(10000)
	if err != nil {
		t.Fatalf("NewResourceManager: %v", err)
	}

	var wg sync.WaitGroup
	// Concurrent loads.
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := string(rune('A' + n%26))
			_ = rm.Load(id, 100)
			_ = rm.Touch(id)
		}(i)
	}
	wg.Wait()

	used, _, loaded := rm.Stats()
	if used > 10000 {
		t.Errorf("used %d exceeds budget 10000", used)
	}
	if loaded > 26 {
		t.Errorf("loaded %d exceeds 26 unique models", loaded)
	}
}
