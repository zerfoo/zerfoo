// Package modelcache provides an LRU model file cache for pre-caching GGUF
// models on Kubernetes nodes via a DaemonSet.
package cache

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// entry tracks a cached model file and its metadata.
type entry struct {
	ref      string
	path     string
	size     int64
	lastUsed time.Time
}

// Cache manages local model files with LRU eviction.
type Cache struct {
	mu      sync.Mutex
	dir     string
	maxSize int64
	entries map[string]*entry
}

// NewCache creates a cache that stores files under dir with a maximum
// aggregate size of maxSize bytes.
func NewCache(dir string, maxSize int64) *Cache {
	return &Cache{
		dir:     dir,
		maxSize: maxSize,
		entries: make(map[string]*entry),
	}
}

// Get returns the local file path for the given model ref if it is cached.
func (c *Cache) Get(ref string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[ref]
	if !ok {
		return "", false
	}
	e.lastUsed = time.Now()
	return e.path, true
}

// Put copies the file at srcPath into the cache under the given ref.
func (c *Cache) Put(ref string, srcPath string) error {
	src, err := os.Open(srcPath)
	if err != nil {
		return fmt.Errorf("modelcache: open source: %w", err)
	}
	defer src.Close()

	info, err := src.Stat()
	if err != nil {
		return fmt.Errorf("modelcache: stat source: %w", err)
	}
	size := info.Size()

	if size > c.maxSize {
		return fmt.Errorf("modelcache: file size %d exceeds max cache size %d", size, c.maxSize)
	}

	dest := filepath.Join(c.dir, sanitize(ref))
	dst, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("modelcache: create dest: %w", err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		os.Remove(dest)
		return fmt.Errorf("modelcache: copy: %w", err)
	}

	c.mu.Lock()
	// If replacing an existing entry, remove old size accounting.
	if old, ok := c.entries[ref]; ok {
		delete(c.entries, ref)
		if old.path != dest {
			os.Remove(old.path)
		}
	}
	c.entries[ref] = &entry{
		ref:      ref,
		path:     dest,
		size:     size,
		lastUsed: time.Now(),
	}
	c.mu.Unlock()

	c.Evict()
	return nil
}

// Evict removes least-recently-used entries until the total cache size is
// at or below maxSize.
func (c *Cache) Evict() {
	c.mu.Lock()
	defer c.mu.Unlock()

	var total int64
	for _, e := range c.entries {
		total += e.size
	}
	if total <= c.maxSize {
		return
	}

	// Sort entries by lastUsed ascending (oldest first).
	sorted := make([]*entry, 0, len(c.entries))
	for _, e := range c.entries {
		sorted = append(sorted, e)
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].lastUsed.Before(sorted[j].lastUsed)
	})

	for _, e := range sorted {
		if total <= c.maxSize {
			break
		}
		os.Remove(e.path)
		total -= e.size
		delete(c.entries, e.ref)
	}
}

// Prefetch downloads models for the given refs using pullFn, which should
// fetch the model identified by ref and write it to dest. Already-cached
// refs are skipped.
func (c *Cache) Prefetch(refs []string, pullFn func(ref string, dest string) error) error {
	for _, ref := range refs {
		if _, ok := c.Get(ref); ok {
			continue
		}
		dest := filepath.Join(c.dir, sanitize(ref)+".tmp")
		if err := pullFn(ref, dest); err != nil {
			os.Remove(dest)
			return fmt.Errorf("modelcache: prefetch %q: %w", ref, err)
		}
		if err := c.Put(ref, dest); err != nil {
			os.Remove(dest)
			return err
		}
		os.Remove(dest) // Put copies the file; remove the temp
	}
	return nil
}

// sanitize converts a model ref into a safe filename.
func sanitize(ref string) string {
	out := make([]byte, len(ref))
	for i := range ref {
		switch {
		case ref[i] >= 'a' && ref[i] <= 'z',
			ref[i] >= 'A' && ref[i] <= 'Z',
			ref[i] >= '0' && ref[i] <= '9',
			ref[i] == '-', ref[i] == '_', ref[i] == '.':
			out[i] = ref[i]
		default:
			out[i] = '_'
		}
	}
	return string(out)
}
