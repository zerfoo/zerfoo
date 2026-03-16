package generate

import (
	"context"

	"github.com/zerfoo/ztensor/tensor"
)

// CacheProvider is the interface implemented by both KVCache (pre-allocated)
// and PagedKVCache (block-based). Attention layers use this interface to
// store and retrieve cached key-value tensors during generation.
type CacheProvider[T tensor.Numeric] interface {
	Update(layer int, newK, newV *tensor.TensorNumeric[T]) error
	Get(layer int) (*LayerKV[T], bool)
	SeqLen() int
	Reset()
	Truncate(newSeqLen int)
}

type kvCacheKey struct{}

// WithKVCache returns a new context that carries the given KVCache.
// Deprecated: Use WithCache for CacheProvider-based caching.
func WithKVCache[T tensor.Numeric](ctx context.Context, cache *KVCache[T]) context.Context {
	if cache == nil {
		return context.WithValue(ctx, kvCacheKey{}, (*KVCache[T])(nil))
	}
	return context.WithValue(ctx, kvCacheKey{}, CacheProvider[T](cache))
}

// WithCache returns a new context that carries the given CacheProvider.
func WithCache[T tensor.Numeric](ctx context.Context, cache CacheProvider[T]) context.Context {
	return context.WithValue(ctx, kvCacheKey{}, cache)
}

// GetKVCache extracts the KVCache from the context, if present.
// It handles both direct *KVCache storage and CacheProvider interface storage.
// Deprecated: Use GetCache for CacheProvider-based caching.
func GetKVCache[T tensor.Numeric](ctx context.Context) (*KVCache[T], bool) {
	val := ctx.Value(kvCacheKey{})
	if val == nil {
		return nil, false
	}
	// Try direct *KVCache.
	if cache, ok := val.(*KVCache[T]); ok {
		if cache == nil {
			return nil, false
		}
		return cache, true
	}
	// Try CacheProvider interface (WithKVCache stores as CacheProvider).
	if cp, ok := val.(CacheProvider[T]); ok {
		if cache, ok := cp.(*KVCache[T]); ok && cache != nil {
			return cache, true
		}
	}
	return nil, false
}

// GetCache extracts the CacheProvider from the context, if present.
func GetCache[T tensor.Numeric](ctx context.Context) (CacheProvider[T], bool) {
	cache, ok := ctx.Value(kvCacheKey{}).(CacheProvider[T])
	if !ok || cache == nil {
		return nil, false
	}
	return cache, true
}
