package serve

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/zerfoo/zerfoo/inference/lora"
)

// adapterNameRe restricts adapter names to a safe charset so that
// filesystem-hostile input (path separators, "..", NUL bytes, etc.) is
// rejected before it ever reaches filepath.Join.
var adapterNameRe = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

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
//
// The name is attacker-controlled (parsed from the request's "model" field by
// ParseModelAdapter), so it is validated against an anchored charset and the
// resulting path is checked for directory containment before any filesystem
// access. Without this, a name like "../../../../etc/passwd" would survive
// filepath.Join (which cleans "../" segments) and let a request open any
// file on disk that the process can read.
func (h *AdapterCacheHandle) resolveAdapter(name string) (*lora.Adapter, error) {
	if !adapterNameRe.MatchString(name) {
		return nil, fmt.Errorf("invalid adapter name %q", name)
	}

	path := filepath.Join(h.dir, name+".gguf")
	clean := filepath.Clean(path)
	dirPrefix := filepath.Clean(h.dir) + string(os.PathSeparator)
	if !strings.HasPrefix(clean, dirPrefix) {
		return nil, fmt.Errorf("adapter path escapes directory")
	}

	adapter, err := h.cache.GetOrLoad(name, clean)
	if err != nil {
		return nil, fmt.Errorf("loading adapter %q: %w", name, err)
	}
	return adapter, nil
}
