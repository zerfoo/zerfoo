package serve

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/zerfoo/zerfoo/inference/lora"
)

// AdapterCacheHandle wraps a lora.AdapterCache with the directory
// where adapter GGUF files are stored.
type AdapterCacheHandle struct {
	cache *lora.AdapterCache
	dir   string // directory containing <name>.gguf adapter files
}

// WithAdapterCache enables per-request LoRA adapter selection.
// dir is the directory containing adapter GGUF files named <adapter>.gguf.
// maxCached is the maximum number of adapters to keep in memory.
func WithAdapterCache(dir string, maxCached int) ServerOption {
	return func(s *Server) {
		s.adapterCache = &AdapterCacheHandle{
			cache: lora.NewAdapterCache(maxCached),
			dir:   dir,
		}
	}
}

// ParseModelAdapter splits a model field of the form "base_model:adapter_name"
// into the base model ID and the adapter name. If no colon is present, the
// adapter name is empty.
func ParseModelAdapter(model string) (baseModel, adapterName string) {
	idx := strings.IndexByte(model, ':')
	if idx < 0 {
		return model, ""
	}
	return model[:idx], model[idx+1:]
}

// resolveAdapter loads an adapter by name from the cache, or from disk if not cached.
// Returns the adapter or an error if the adapter cannot be found or loaded.
func (h *AdapterCacheHandle) resolveAdapter(name string) (*lora.Adapter, error) {
	path := filepath.Join(h.dir, name+".gguf")
	adapter, err := h.cache.GetOrLoad(name, path)
	if err != nil {
		return nil, fmt.Errorf("loading adapter %q: %w", name, err)
	}
	return adapter, nil
}
