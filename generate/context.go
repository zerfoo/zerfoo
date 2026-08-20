package generate

import (
	"context"
	"unsafe"

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

// LayerSeqLenProvider is an optional CacheProvider extension that reports the
// number of positions cached for one specific layer.
//
// Attention layers MUST derive their RoPE position offset from LayerSeqLen,
// never from SeqLen. SeqLen reports layer 0's cursor, and layer 0 advances that
// cursor with its own Update partway through a forward pass, so layers
// 1..N-1 would read a value already advanced by the current chunk's length and
// rotate their Q/K to positions shifted by +chunkLen. Within one pass the shift
// is uniform and RoPE is relative, so the damage is invisible in a pure prefill
// or a pure token-by-token decode; it appears at the prefill->decode transition,
// where a decode query shifted by +1 is scored against keys cached with a
// shift of +promptLen. That was zerfoo#990: correct first token, then decode
// degrading into on-topic repetition on every architecture.
//
// Implementations whose SeqLen is already pass-stable (advanced only after the
// last layer's Update, as GPUKVCache does) satisfy this trivially.
type LayerSeqLenProvider interface {
	// LayerSeqLen returns the number of positions currently cached for the
	// given layer. Out-of-range layers return 0.
	LayerSeqLen(layer int) int
}

// FullBufferProvider is an optional interface for caches that support
// fixed-size (maxSeqLen) KV buffer access. This enables CUDA graph capture
// for the decode attention loop: the FlashAttentionDecode kernel reads the
// actual KV length from a GPU-resident counter (KVSeqLenPtr), so tensor
// shapes stay fixed across graph replays.
type FullBufferProvider[T tensor.Numeric] interface {
	// GetFullBuffer returns GPU-backed KV tensors spanning the full
	// pre-allocated buffer (maxSeqLen capacity) for the given layer.
	// Shape is [batch, maxSeqLen, dim]. Returns nil if the layer is
	// CPU-backed or not yet initialized.
	GetFullBuffer(layer int) (k, v *tensor.TensorNumeric[T])
	// MaxSeqLen returns the maximum sequence length (buffer capacity).
	MaxSeqLen() int
	// KVSeqLenPtr returns the device pointer to the GPU-resident int32
	// KV sequence length counter. Returns nil if not allocated.
	KVSeqLenPtr() unsafe.Pointer
}

type kvCacheKey struct{}

// WithKVCache returns a new context that carries the given KVCache.
//
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
//
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
