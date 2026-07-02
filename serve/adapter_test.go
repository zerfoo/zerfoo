package serve

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/inference/lora"
)

func TestAdapterCacheHandle_ResolveAdapter_MissingFile(t *testing.T) {
	h := &AdapterCacheHandle{
		cache: lora.NewAdapterCache(2),
		dir:   t.TempDir(),
	}

	_, err := h.resolveAdapter("nonexistent")
	if err == nil {
		t.Fatal("expected error for missing adapter file, got nil")
	}
}

func TestAdapterCacheHandle_ResolveAdapter_InvalidGGUF(t *testing.T) {
	dir := t.TempDir()

	// Write a file that is not valid GGUF.
	if err := os.WriteFile(filepath.Join(dir, "bad.gguf"), []byte("not a gguf file"), 0644); err != nil {
		t.Fatal(err)
	}

	h := &AdapterCacheHandle{
		cache: lora.NewAdapterCache(2),
		dir:   dir,
	}

	_, err := h.resolveAdapter("bad")
	if err == nil {
		t.Fatal("expected error for invalid GGUF, got nil")
	}
}

func TestWithAdapterCache(t *testing.T) {
	dir := t.TempDir()
	opt := WithAdapterCache(dir, 5)

	s := &Server{}
	opt(s)

	if s.adapterCache == nil {
		t.Fatal("expected adapterCache to be set")
	}
	if s.adapterCache.dir != dir {
		t.Errorf("dir = %q, want %q", s.adapterCache.dir, dir)
	}
}
