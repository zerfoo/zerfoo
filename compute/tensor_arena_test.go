package compute

import (
	"testing"
)

func TestTensorArena_GetReturnsCorrectSize(t *testing.T) {
	var arena TensorArena
	for _, n := range []int{1, 4, 7, 128, 2048} {
		buf := arena.Get(n)
		if len(buf) != n {
			t.Errorf("Get(%d): got len %d", n, len(buf))
		}
		// Capacity must be a power of 2 >= n.
		c := cap(buf)
		if c < n {
			t.Errorf("Get(%d): cap %d < n", n, c)
		}
		if c&(c-1) != 0 {
			t.Errorf("Get(%d): cap %d not power of 2", n, c)
		}
	}
}

func TestTensorArena_GetZero(t *testing.T) {
	var arena TensorArena
	buf := arena.Get(0)
	if buf != nil {
		t.Errorf("Get(0) should return nil, got len %d", len(buf))
	}
	buf = arena.Get(-1)
	if buf != nil {
		t.Errorf("Get(-1) should return nil, got len %d", len(buf))
	}
}

func TestTensorArena_PutAndReuse(t *testing.T) {
	var arena TensorArena
	buf1 := arena.Get(100)
	buf1[0] = 42
	arena.Put(buf1)

	buf2 := arena.Get(100)
	// Should reuse the same backing array (capacity match).
	if cap(buf2) != cap(buf1) {
		t.Errorf("expected reuse: cap %d vs %d", cap(buf2), cap(buf1))
	}
	// Data should be zeroed.
	if buf2[0] != 0 {
		t.Errorf("reused buffer not zeroed: got %v", buf2[0])
	}
}

func TestTensorArena_Reset(t *testing.T) {
	var arena TensorArena
	arena.Put(arena.Get(64))
	arena.Put(arena.Get(128))
	arena.Reset()

	// After reset, Get should allocate fresh (no reuse).
	// We can't directly test "no reuse" but we can verify it still works.
	buf := arena.Get(64)
	if len(buf) != 64 {
		t.Errorf("after Reset: Get(64) len = %d", len(buf))
	}
}

func TestBucketIndex(t *testing.T) {
	tests := []struct {
		n    int
		want int
	}{
		{0, 0}, {1, 0}, {2, 1}, {3, 2}, {4, 2},
		{5, 3}, {8, 3}, {9, 4}, {16, 4}, {17, 5},
		{128, 7}, {129, 8}, {1024, 10},
	}
	for _, tt := range tests {
		got := bucketIndex(tt.n)
		if got != tt.want {
			t.Errorf("bucketIndex(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

func BenchmarkTensorArena_GetPut(b *testing.B) {
	var arena TensorArena
	b.ResetTimer()
	for range b.N {
		buf := arena.Get(2048)
		arena.Put(buf)
	}
}
