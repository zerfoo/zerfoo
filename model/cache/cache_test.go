package cache

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// createFile writes size bytes to a file and returns its path.
func createFile(t *testing.T, dir, name string, size int) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestCache_PutGet(t *testing.T) {
	tmp := t.TempDir()
	cacheDir := filepath.Join(tmp, "cache")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		ref     string
		size    int
		wantHit bool
	}{
		{name: "put then get", ref: "model-a", size: 100, wantHit: true},
		{name: "miss before put", ref: "model-b", size: 0, wantHit: false},
	}

	c := NewCache(cacheDir, 10000)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.size > 0 {
				src := createFile(t, tmp, tt.ref+".gguf", tt.size)
				if err := c.Put(tt.ref, src); err != nil {
					t.Fatalf("Put(%q): %v", tt.ref, err)
				}
			}

			path, ok := c.Get(tt.ref)
			if ok != tt.wantHit {
				t.Fatalf("Get(%q) hit=%v, want %v", tt.ref, ok, tt.wantHit)
			}
			if ok {
				info, err := os.Stat(path)
				if err != nil {
					t.Fatalf("cached file missing: %v", err)
				}
				if int(info.Size()) != tt.size {
					t.Fatalf("cached file size=%d, want %d", info.Size(), tt.size)
				}
			}
		})
	}
}

func TestCache_Eviction(t *testing.T) {
	tmp := t.TempDir()
	cacheDir := filepath.Join(tmp, "cache")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// maxSize=150: can hold one 100-byte file or a combination up to 150
	c := NewCache(cacheDir, 150)

	tests := []struct {
		name        string
		putRef      string
		putSize     int
		touchRef    string        // ref to Get (touch) before the new Put
		sleepBefore time.Duration // ensure time ordering
		evictedRef  string        // ref expected to be evicted after Put
		survivorRef string        // ref expected to remain
	}{
		{
			name:    "add first model",
			putRef:  "model-a",
			putSize: 100,
		},
		{
			name:        "add second evicts first (LRU)",
			putRef:      "model-b",
			putSize:     100,
			sleepBefore: 10 * time.Millisecond,
			evictedRef:  "model-a",
			survivorRef: "model-b",
		},
		{
			name:        "touch model-b then add model-c evicts model-b if untouched would be older but model-b was touched",
			putRef:      "model-c",
			putSize:     100,
			touchRef:    "model-b",
			sleepBefore: 10 * time.Millisecond,
			evictedRef:  "", // model-b was touched so it survives; but total exceeds so oldest goes
			survivorRef: "model-c",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.touchRef != "" {
				c.Get(tt.touchRef)
			}
			if tt.sleepBefore > 0 {
				time.Sleep(tt.sleepBefore)
			}
			src := createFile(t, tmp, tt.putRef+".gguf", tt.putSize)
			if err := c.Put(tt.putRef, src); err != nil {
				t.Fatalf("Put(%q): %v", tt.putRef, err)
			}
			if tt.evictedRef != "" {
				if _, ok := c.Get(tt.evictedRef); ok {
					t.Fatalf("expected %q to be evicted", tt.evictedRef)
				}
			}
			if tt.survivorRef != "" {
				if _, ok := c.Get(tt.survivorRef); !ok {
					t.Fatalf("expected %q to survive eviction", tt.survivorRef)
				}
			}
		})
	}
}

func TestCache_Prefetch(t *testing.T) {
	tmp := t.TempDir()
	cacheDir := filepath.Join(tmp, "cache")
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		t.Fatal(err)
	}

	c := NewCache(cacheDir, 10000)

	tests := []struct {
		name       string
		refs       []string
		pullErr    bool
		wantErr    bool
		wantCached []string
	}{
		{
			name:       "prefetch two models",
			refs:       []string{"model-x", "model-y"},
			wantCached: []string{"model-x", "model-y"},
		},
		{
			name:       "skip already cached",
			refs:       []string{"model-x"}, // already cached from previous test
			wantCached: []string{"model-x"},
		},
		{
			name:    "pull error propagates",
			refs:    []string{"model-fail"},
			pullErr: true,
			wantErr: true,
		},
	}

	pullCalls := 0
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pullCalls = 0
			pullFn := func(ref string, dest string) error {
				pullCalls++
				if tt.pullErr {
					return os.ErrNotExist
				}
				return os.WriteFile(dest, make([]byte, 50), 0o644)
			}

			err := c.Prefetch(tt.refs, pullFn)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Prefetch() err=%v, wantErr=%v", err, tt.wantErr)
			}
			for _, ref := range tt.wantCached {
				if _, ok := c.Get(ref); !ok {
					t.Fatalf("expected %q to be cached after prefetch", ref)
				}
			}
		})
	}

	// Verify skip: prefetching an already-cached model should not call pullFn.
	pullCalls = 0
	_ = c.Prefetch([]string{"model-x"}, func(ref string, dest string) error {
		pullCalls++
		return nil
	})
	if pullCalls != 0 {
		t.Fatalf("expected 0 pull calls for cached model, got %d", pullCalls)
	}
}
