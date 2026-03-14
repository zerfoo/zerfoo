package cuda

import (
	"testing"
	"unsafe"
)

func TestArenaPool_BasicAllocReset(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 4096, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	ptr, err := arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}
	if ptr == nil {
		t.Fatal("Alloc returned nil")
	}

	hits, misses, _ := arena.HitMissStats()
	if hits != 1 {
		t.Errorf("hits = %d, want 1", hits)
	}
	if misses != 0 {
		t.Errorf("misses = %d, want 0", misses)
	}

	arena.Reset()
	if arena.UsedBytes() != 0 {
		t.Errorf("UsedBytes after Reset = %d, want 0", arena.UsedBytes())
	}
}

func TestArenaPool_FallbackOnExhaustion(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 512, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	// First alloc fits in arena (256 aligned).
	_, err = arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc 1: %v", err)
	}

	// Second alloc also fits (256 + 256 = 512 = capacity).
	_, err = arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc 2: %v", err)
	}

	// Third alloc should fall back.
	ptr, err := arena.Alloc(0, 256)
	if err != nil {
		t.Fatalf("Alloc 3 (fallback): %v", err)
	}
	if ptr == nil {
		t.Fatal("Alloc 3 returned nil")
	}

	_, misses, _ := arena.HitMissStats()
	if misses != 1 {
		t.Errorf("misses = %d, want 1", misses)
	}

	// Free the fallback pointer.
	arena.Free(0, ptr, 256)
}

func TestArenaPool_IsManaged(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 4096, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	managed := arena.IsManaged()
	// The arena only uses managed memory when both the device supports it AND
	// ZERFOO_ENABLE_MANAGED_MEM is set. The test verifies IsManaged() returns
	// a consistent value without asserting a specific expected result, since
	// the env var may or may not be set in CI.
	t.Logf("ArenaPool.IsManaged() = %v (device supports managed: %v)", managed, ManagedMemorySupported(0))
}

func TestArenaPool_ManagedMemoryDataIntegrity(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}
	if !ManagedMemorySupported(0) {
		t.Skip("managed memory not supported on device 0")
	}

	fallback := NewMemPool()
	arena, err := NewArenaPool(0, 4096, fallback)
	if err != nil {
		t.Fatalf("NewArenaPool: %v", err)
	}
	defer func() { _ = arena.Drain() }()

	if !arena.IsManaged() {
		t.Skip("arena not using managed memory (set ZERFOO_ENABLE_MANAGED_MEM=1)")
	}

	// Allocate from managed arena and write from CPU.
	ptr, err := arena.Alloc(0, 16)
	if err != nil {
		t.Fatalf("Alloc: %v", err)
	}

	data := unsafe.Slice((*float32)(ptr), 4)
	data[0] = 1.0
	data[1] = 2.0
	data[2] = 3.0
	data[3] = 4.0

	// Read back.
	for i, want := range []float32{1.0, 2.0, 3.0, 4.0} {
		if data[i] != want {
			t.Errorf("data[%d] = %f, want %f", i, data[i], want)
		}
	}
}
