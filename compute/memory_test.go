package compute

import (
	"errors"
	"sync"
	"testing"
)

func TestMemoryTracker_AllocWithinLimit(t *testing.T) {
	mt := NewMemoryTracker(1024)
	if err := mt.Alloc(512); err != nil {
		t.Fatalf("Alloc(512): %v", err)
	}
	if got := mt.Allocated(); got != 512 {
		t.Errorf("Allocated() = %d, want 512", got)
	}
}

func TestMemoryTracker_AllocExceedsLimit(t *testing.T) {
	mt := NewMemoryTracker(1024)
	if err := mt.Alloc(512); err != nil {
		t.Fatalf("first Alloc: %v", err)
	}
	if err := mt.Alloc(600); !errors.Is(err, ErrMemoryLimitExceeded) {
		t.Errorf("expected ErrMemoryLimitExceeded, got %v", err)
	}
	// Counter should not have changed.
	if got := mt.Allocated(); got != 512 {
		t.Errorf("Allocated() = %d, want 512 (unchanged)", got)
	}
}

func TestMemoryTracker_Free(t *testing.T) {
	mt := NewMemoryTracker(1024)
	if err := mt.Alloc(800); err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	mt.Free(300)
	if got := mt.Allocated(); got != 500 {
		t.Errorf("Allocated() = %d, want 500", got)
	}
	// Now we should be able to allocate more.
	if err := mt.Alloc(500); err != nil {
		t.Errorf("Alloc(500) after free: %v", err)
	}
}

func TestMemoryTracker_Unlimited(t *testing.T) {
	mt := NewMemoryTracker(0)
	// Should never fail.
	if err := mt.Alloc(1 << 30); err != nil {
		t.Errorf("Alloc with unlimited: %v", err)
	}
	if got := mt.Allocated(); got != 1<<30 {
		t.Errorf("Allocated() = %d, want %d", got, 1<<30)
	}
}

func TestMemoryTracker_Limit(t *testing.T) {
	mt := NewMemoryTracker(4096)
	if got := mt.Limit(); got != 4096 {
		t.Errorf("Limit() = %d, want 4096", got)
	}
}

func TestMemoryTracker_Concurrent(t *testing.T) {
	mt := NewMemoryTracker(100000)
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := mt.Alloc(100); err != nil {
				return // limit hit is fine
			}
			mt.Free(100)
		}()
	}
	wg.Wait()
}
