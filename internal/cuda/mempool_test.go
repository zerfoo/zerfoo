package cuda

import (
	"testing"
)

func TestMemPoolAllocFresh(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	ptr, err := pool.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}

	if ptr == nil {
		t.Fatal("Alloc returned nil pointer")
	}

	// Return to pool
	pool.Free(0, ptr, 1024)
}

func TestMemPoolAllocReuse(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	// Allocate and free to populate cache
	ptr1, err := pool.Alloc(0, 2048)
	if err != nil {
		t.Fatalf("first Alloc failed: %v", err)
	}

	pool.Free(0, ptr1, 2048)

	// Second alloc of same size should reuse
	ptr2, err := pool.Alloc(0, 2048)
	if err != nil {
		t.Fatalf("second Alloc failed: %v", err)
	}

	if ptr1 != ptr2 {
		t.Error("expected pool to reuse cached pointer")
	}

	pool.Free(0, ptr2, 2048)
}

func TestMemPoolAllocDifferentSizes(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	ptr1, err := pool.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc(1024) failed: %v", err)
	}

	pool.Free(0, ptr1, 1024)

	// Different size should not reuse
	ptr2, err := pool.Alloc(0, 2048)
	if err != nil {
		t.Fatalf("Alloc(2048) failed: %v", err)
	}

	if ptr1 == ptr2 {
		t.Error("different sizes should not reuse same pointer")
	}

	pool.Free(0, ptr2, 2048)
}

func TestMemPoolDrain(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	pool := NewMemPool()

	ptr, err := pool.Alloc(0, 512)
	if err != nil {
		t.Fatalf("Alloc failed: %v", err)
	}

	pool.Free(0, ptr, 512)

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
	if !Available() {
		t.Skip("CUDA not available")
	}

	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	allocs, bytes := pool.Stats()
	if allocs != 0 || bytes != 0 {
		t.Errorf("empty pool Stats() = (%d, %d), want (0, 0)", allocs, bytes)
	}

	// Alloc three different sizes to avoid cache reuse between calls.
	for _, size := range []int{1024, 2048, 4096} {
		ptr, err := pool.Alloc(0, size)
		if err != nil {
			t.Fatalf("Alloc(%d) failed: %v", size, err)
		}

		pool.Free(0, ptr, size)
	}

	allocs, bytes = pool.Stats()
	if allocs != 3 {
		t.Errorf("Stats().allocations = %d, want 3", allocs)
	}

	if bytes != 1024+2048+4096 {
		t.Errorf("Stats().totalBytes = %d, want %d", bytes, 1024+2048+4096)
	}
}

func TestBucketSize(t *testing.T) {
	tests := []struct {
		input int
		want  int
	}{
		{0, 0},
		{1, 1},
		{100, 100},
		{256, 256},       // at threshold, exact
		{257, 512},       // just above threshold, rounds to next power of 2
		{1024, 1024},     // already power of 2
		{1025, 2048},     // rounds up
		{4096, 4096},     // power of 2
		{5000, 8192},     // rounds up
		{8192, 8192},     // already power of 2
		{65536, 65536},   // already power of 2
		{100000, 131072}, // rounds up to 2^17
	}
	for _, tt := range tests {
		got := bucketSize(tt.input)
		if got != tt.want {
			t.Errorf("bucketSize(%d) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestMemPoolBucketReuse(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	// Allocate 5000 bytes (buckets to 8192), free it.
	ptr1, err := pool.Alloc(0, 5000)
	if err != nil {
		t.Fatalf("Alloc(5000) failed: %v", err)
	}
	pool.Free(0, ptr1, 5000)

	// Allocate 6000 bytes (also buckets to 8192) -- should reuse.
	ptr2, err := pool.Alloc(0, 6000)
	if err != nil {
		t.Fatalf("Alloc(6000) failed: %v", err)
	}

	if ptr1 != ptr2 {
		t.Error("expected bucket reuse: 5000 and 6000 both bucket to 8192")
	}
	pool.Free(0, ptr2, 6000)
}

func TestMemPoolNoCrossDeviceReuse(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount: %v", err)
	}
	if count < 2 {
		t.Skip("need >= 2 GPUs for cross-device reuse test")
	}

	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	// Allocate on device 0 and return to pool.
	ptr0, err := pool.Alloc(0, 4096)
	if err != nil {
		t.Fatalf("Alloc on device 0: %v", err)
	}
	pool.Free(0, ptr0, 4096)

	// Allocate same size on device 1 -- must NOT reuse the device-0 pointer.
	ptr1, err := pool.Alloc(1, 4096)
	if err != nil {
		t.Fatalf("Alloc on device 1: %v", err)
	}

	if ptr0 == ptr1 {
		t.Error("pool reused device 0 pointer for device 1 allocation")
	}

	pool.Free(1, ptr1, 4096)
}

func TestMemPoolMultiDeviceStats(t *testing.T) {
	if !Available() {
		t.Skip("CUDA not available")
	}

	count, err := GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount: %v", err)
	}
	if count < 2 {
		t.Skip("need >= 2 GPUs for multi-device stats test")
	}

	pool := NewMemPool()
	defer func() { _ = pool.Drain() }()

	ptr0, err := pool.Alloc(0, 1024)
	if err != nil {
		t.Fatalf("Alloc device 0: %v", err)
	}
	pool.Free(0, ptr0, 1024)

	ptr1, err := pool.Alloc(1, 2048)
	if err != nil {
		t.Fatalf("Alloc device 1: %v", err)
	}
	pool.Free(1, ptr1, 2048)

	allocs, bytes := pool.Stats()
	if allocs != 2 {
		t.Errorf("Stats().allocations = %d, want 2", allocs)
	}
	if bytes != 1024+2048 {
		t.Errorf("Stats().totalBytes = %d, want %d", bytes, 1024+2048)
	}
}
