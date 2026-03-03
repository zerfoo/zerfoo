//go:build cuda

package cuda

import (
	"testing"
)

func TestMemPoolAllocFresh(t *testing.T) {
	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	ptr, err := pool.Alloc(1024)
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}

	if ptr == nil {
		t.Fatal("Alloc returned nil pointer")
	}

	// Return to pool
	pool.Free(ptr, 1024)
}

func TestMemPoolAllocReuse(t *testing.T) {
	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	// Allocate and free to populate cache
	ptr1, err := pool.Alloc(2048)
	if err != nil {
		t.Fatalf("first Alloc failed: %v", err)
	}

	pool.Free(ptr1, 2048)

	// Second alloc of same size should reuse
	ptr2, err := pool.Alloc(2048)
	if err != nil {
		t.Fatalf("second Alloc failed: %v", err)
	}

	if ptr1 != ptr2 {
		t.Error("expected pool to reuse cached pointer")
	}

	pool.Free(ptr2, 2048)
}

func TestMemPoolAllocDifferentSizes(t *testing.T) {
	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	ptr1, err := pool.Alloc(1024)
	if err != nil {
		t.Fatalf("Alloc(1024) failed: %v", err)
	}

	pool.Free(ptr1, 1024)

	// Different size should not reuse
	ptr2, err := pool.Alloc(2048)
	if err != nil {
		t.Fatalf("Alloc(2048) failed: %v", err)
	}

	if ptr1 == ptr2 {
		t.Error("different sizes should not reuse same pointer")
	}

	pool.Free(ptr2, 2048)
}

func TestMemPoolDrain(t *testing.T) {
	pool := NewMemPool()

	ptr, err := pool.Alloc(512)
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}

	pool.Free(ptr, 512)

	allocs, bytes := pool.Stats()
	if allocs != 1 || bytes != 512 {
		t.Errorf("before Drain: Stats() = (%d, %d), want (1, 512)", allocs, bytes)
	}

	if err := pool.Drain(); err != nil {
		t.Fatalf("Drain failed: %v", err)
	}

	allocs, bytes = pool.Stats()
	if allocs != 0 || bytes != 0 {
		t.Errorf("after Drain: Stats() = (%d, %d), want (0, 0)", allocs, bytes)
	}
}

func TestMemPoolStats(t *testing.T) {
	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	allocs, bytes := pool.Stats()
	if allocs != 0 || bytes != 0 {
		t.Errorf("empty pool Stats() = (%d, %d), want (0, 0)", allocs, bytes)
	}

	ptrs := make([]struct {
		ptr  interface{}
		size int
	}, 0)

	for _, size := range []int{1024, 1024, 2048} {
		ptr, err := pool.Alloc(size)
		if err != nil {
			t.Fatalf("Alloc(%d) failed: %v", size, err)
		}

		pool.Free(ptr, size)
		_ = ptrs // suppress unused variable
	}

	allocs, bytes = pool.Stats()
	if allocs != 3 {
		t.Errorf("Stats().allocations = %d, want 3", allocs)
	}

	if bytes != 1024+1024+2048 {
		t.Errorf("Stats().totalBytes = %d, want %d", bytes, 1024+1024+2048)
	}
}
