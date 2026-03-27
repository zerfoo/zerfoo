package lora

import (
	"os"
	"sync"
)

// AdapterCache manages loaded LoRA adapters with LRU eviction.
// Thread-safe for concurrent access from multiple request goroutines.
type AdapterCache struct {
	mu       sync.Mutex
	adapters map[string]*cacheEntry
	order    []string // LRU order: most recent at end
	maxSize  int
}

type cacheEntry struct {
	adapter *Adapter
	path    string
}

// NewAdapterCache creates a cache that holds up to maxSize adapters.
func NewAdapterCache(maxSize int) *AdapterCache {
	return &AdapterCache{
		adapters: make(map[string]*cacheEntry),
		maxSize:  maxSize,
	}
}

// Get returns a cached adapter by name. Returns nil if not cached.
// Moves the entry to most-recently-used position.
func (c *AdapterCache) Get(name string) *Adapter {
	c.mu.Lock()
	defer c.mu.Unlock()

	e, ok := c.adapters[name]
	if !ok {
		return nil
	}
	c.touch(name)
	return e.adapter
}

// Put inserts an adapter directly into the cache. If the cache is full,
// the least-recently-used adapter is evicted.
func (c *AdapterCache) Put(name string, adapter *Adapter) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, ok := c.adapters[name]; ok {
		c.adapters[name].adapter = adapter
		c.touch(name)
		return
	}
	c.evictIfNeeded()
	c.adapters[name] = &cacheEntry{adapter: adapter}
	c.order = append(c.order, name)
}

// GetOrLoad returns a cached adapter or loads it from path.
// If loading causes the cache to exceed maxSize, the least-recently-used
// adapter is evicted.
func (c *AdapterCache) GetOrLoad(name, path string) (*Adapter, error) {
	c.mu.Lock()
	if e, ok := c.adapters[name]; ok {
		c.touch(name)
		c.mu.Unlock()
		return e.adapter, nil
	}
	c.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	adapter, err := LoadAdapter(path, f)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Check again in case another goroutine loaded it while we were loading.
	if e, ok := c.adapters[name]; ok {
		c.touch(name)
		return e.adapter, nil
	}

	c.evictIfNeeded()
	c.adapters[name] = &cacheEntry{adapter: adapter, path: path}
	c.order = append(c.order, name)
	return adapter, nil
}

// Evict removes a specific adapter from the cache.
func (c *AdapterCache) Evict(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, ok := c.adapters[name]; !ok {
		return
	}
	delete(c.adapters, name)
	c.removeFromOrder(name)
}

// Size returns the number of cached adapters.
func (c *AdapterCache) Size() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.adapters)
}

// Names returns the names of all cached adapters in LRU order (oldest first).
func (c *AdapterCache) Names() []string {
	c.mu.Lock()
	defer c.mu.Unlock()

	out := make([]string, len(c.order))
	copy(out, c.order)
	return out
}

// touch moves name to the end of the order slice (most recently used).
// Caller must hold c.mu.
func (c *AdapterCache) touch(name string) {
	c.removeFromOrder(name)
	c.order = append(c.order, name)
}

// removeFromOrder removes name from the order slice.
// Caller must hold c.mu.
func (c *AdapterCache) removeFromOrder(name string) {
	for i, n := range c.order {
		if n == name {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}

// evictIfNeeded removes the least-recently-used entry if the cache is at capacity.
// Caller must hold c.mu.
func (c *AdapterCache) evictIfNeeded() {
	if len(c.adapters) < c.maxSize {
		return
	}
	if len(c.order) == 0 {
		return
	}
	oldest := c.order[0]
	c.order = c.order[1:]
	delete(c.adapters, oldest)
}
