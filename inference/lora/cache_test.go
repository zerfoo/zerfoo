package lora

import (
	"fmt"
	"sync"
	"testing"
)

func makeAdapter(rank int) *Adapter {
	return &Adapter{
		Rank:        rank,
		Alpha:       float64(rank),
		ScaleFactor: 1.0,
		Layers:      map[string]*Layer{},
	}
}

func TestAdapterCache(t *testing.T) {
	tests := []struct {
		name string
		fn   func(t *testing.T)
	}{
		{
			name: "cache hit returns same pointer",
			fn: func(t *testing.T) {
				c := NewAdapterCache(4)
				a := makeAdapter(8)
				c.Put("model-a", a)

				got := c.Get("model-a")
				if got != a {
					t.Fatal("expected same pointer on cache hit")
				}
			},
		},
		{
			name: "cache miss returns nil",
			fn: func(t *testing.T) {
				c := NewAdapterCache(4)
				if got := c.Get("nonexistent"); got != nil {
					t.Fatalf("expected nil, got %v", got)
				}
			},
		},
		{
			name: "LRU eviction removes oldest",
			fn: func(t *testing.T) {
				c := NewAdapterCache(2)
				c.Put("a", makeAdapter(1))
				c.Put("b", makeAdapter(2))
				c.Put("c", makeAdapter(3)) // evicts "a"

				if c.Get("a") != nil {
					t.Fatal("expected 'a' to be evicted")
				}
				if c.Get("b") == nil {
					t.Fatal("expected 'b' to be present")
				}
				if c.Get("c") == nil {
					t.Fatal("expected 'c' to be present")
				}
			},
		},
		{
			name: "access refreshes LRU position",
			fn: func(t *testing.T) {
				c := NewAdapterCache(2)
				c.Put("a", makeAdapter(1))
				c.Put("b", makeAdapter(2))
				c.Get("a")                 // refresh "a"
				c.Put("c", makeAdapter(3)) // evicts "b" (not "a")

				if c.Get("a") == nil {
					t.Fatal("expected 'a' to be present after refresh")
				}
				if c.Get("b") != nil {
					t.Fatal("expected 'b' to be evicted")
				}
				if c.Get("c") == nil {
					t.Fatal("expected 'c' to be present")
				}
			},
		},
		{
			name: "explicit evict",
			fn: func(t *testing.T) {
				c := NewAdapterCache(4)
				c.Put("x", makeAdapter(4))
				c.Evict("x")

				if c.Get("x") != nil {
					t.Fatal("expected nil after evict")
				}
				if c.Size() != 0 {
					t.Fatalf("expected size 0, got %d", c.Size())
				}
			},
		},
		{
			name: "size and names",
			fn: func(t *testing.T) {
				c := NewAdapterCache(4)
				c.Put("alpha", makeAdapter(1))
				c.Put("beta", makeAdapter(2))
				c.Put("gamma", makeAdapter(3))

				if c.Size() != 3 {
					t.Fatalf("expected size 3, got %d", c.Size())
				}
				names := c.Names()
				want := []string{"alpha", "beta", "gamma"}
				if len(names) != len(want) {
					t.Fatalf("expected %v, got %v", want, names)
				}
				for i := range want {
					if names[i] != want[i] {
						t.Fatalf("names[%d]: expected %q, got %q", i, want[i], names[i])
					}
				}
			},
		},
		{
			name: "concurrent access no race",
			fn: func(t *testing.T) {
				c := NewAdapterCache(10)
				var wg sync.WaitGroup
				for i := range 10 {
					wg.Add(1)
					go func(id int) {
						defer wg.Done()
						name := fmt.Sprintf("adapter-%d", id)
						c.Put(name, makeAdapter(id+1))
						c.Get(name)
						c.Size()
						c.Names()
						if id%3 == 0 {
							c.Evict(name)
						}
					}(i)
				}
				wg.Wait()
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.fn)
	}
}

func TestAdapterCache_LRU_NPlusOne(t *testing.T) {
	// Insert N+1 adapters into a cache of size N and verify the first is evicted.
	const n = 5
	c := NewAdapterCache(n)

	adapters := make([]*Adapter, n+1)
	for i := range adapters {
		adapters[i] = makeAdapter(i + 1)
		c.Put(fmt.Sprintf("adapter-%d", i), adapters[i])
	}

	if c.Size() != n {
		t.Fatalf("expected size %d, got %d", n, c.Size())
	}

	// adapter-0 should have been evicted.
	if c.Get("adapter-0") != nil {
		t.Fatal("expected adapter-0 to be evicted")
	}

	// All others should be present.
	for i := 1; i <= n; i++ {
		name := fmt.Sprintf("adapter-%d", i)
		if c.Get(name) == nil {
			t.Fatalf("expected %q to be present", name)
		}
	}
}

func TestAdapterCache_PutUpdatesExisting(t *testing.T) {
	c := NewAdapterCache(2)
	a1 := makeAdapter(4)
	a2 := makeAdapter(8)

	c.Put("x", a1)
	c.Put("x", a2) // update, not insert

	if c.Size() != 1 {
		t.Fatalf("expected size 1 after update, got %d", c.Size())
	}
	got := c.Get("x")
	if got != a2 {
		t.Fatal("expected updated adapter pointer")
	}
}

func TestAdapterCache_EvictNonexistent(t *testing.T) {
	c := NewAdapterCache(2)
	c.Put("a", makeAdapter(1))

	// Evicting a non-existent entry should be a no-op.
	c.Evict("nonexistent")
	if c.Size() != 1 {
		t.Fatalf("expected size 1, got %d", c.Size())
	}
}

func TestAdapterCache_ConcurrentPutGetEvict(t *testing.T) {
	// Stress test with many goroutines doing mixed operations.
	c := NewAdapterCache(8)
	var wg sync.WaitGroup
	const goroutines = 20

	for i := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			name := fmt.Sprintf("adapter-%d", id%10) // deliberate contention
			c.Put(name, makeAdapter(id+1))
			_ = c.Get(name)
			_ = c.Size()
			_ = c.Names()
			if id%4 == 0 {
				c.Evict(name)
			}
			// Re-insert after eviction to keep things interesting.
			c.Put(name, makeAdapter(id+1))
		}(i)
	}
	wg.Wait()

	// Cache should still be consistent.
	if s := c.Size(); s > 8 {
		t.Fatalf("cache size %d exceeds max 8", s)
	}
	names := c.Names()
	if len(names) != c.Size() {
		t.Fatalf("names length %d != size %d", len(names), c.Size())
	}
}
