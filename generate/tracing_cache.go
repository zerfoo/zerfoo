package generate

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// TracingCacheProvider wraps a real CacheProvider and records KV cache
// operations into a Tracer during a tracing compilation pass. This allows
// the tracing compiler to capture the full attention dataflow including
// cache reads and writes.
type TracingCacheProvider[T tensor.Numeric] struct {
	real   CacheProvider[T]
	tracer *compute.Tracer[T]
}

// NewTracingCacheProvider creates a TracingCacheProvider wrapping the given
// real cache and recording ops into the tracer.
func NewTracingCacheProvider[T tensor.Numeric](real CacheProvider[T], tracer *compute.Tracer[T]) *TracingCacheProvider[T] {
	return &TracingCacheProvider[T]{real: real, tracer: tracer}
}

// Update delegates to the real cache and records KVCacheAppendK/V ops.
func (t *TracingCacheProvider[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	err := t.real.Update(layer, newK, newV)
	if err != nil {
		return err
	}
	t.tracer.Record("KVCacheAppendK", []*tensor.TensorNumeric[T]{newK}, newK, map[string]any{"layer": layer})
	t.tracer.Record("KVCacheAppendV", []*tensor.TensorNumeric[T]{newV}, newV, map[string]any{"layer": layer})
	return nil
}

// Get delegates to the real cache and records KVCacheGetK/V ops.
func (t *TracingCacheProvider[T]) Get(layer int) (*LayerKV[T], bool) {
	kv, ok := t.real.Get(layer)
	if !ok {
		return kv, ok
	}
	t.tracer.Record("KVCacheGetK", nil, kv.Key, map[string]any{"layer": layer})
	t.tracer.Record("KVCacheGetV", nil, kv.Value, map[string]any{"layer": layer})
	return kv, ok
}

// SeqLen delegates to the real cache.
func (t *TracingCacheProvider[T]) SeqLen() int {
	return t.real.SeqLen()
}

// Reset delegates to the real cache.
func (t *TracingCacheProvider[T]) Reset() {
	t.real.Reset()
}

// Truncate delegates to the real cache.
func (t *TracingCacheProvider[T]) Truncate(newSeqLen int) {
	t.real.Truncate(newSeqLen)
}
