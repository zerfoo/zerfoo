package generate

import "github.com/zerfoo/ztensor/tensor"

// tieredKVAdapter wraps a TieredKVStore to implement the CacheProvider interface.
type tieredKVAdapter[T tensor.Numeric] struct {
	store *TieredKVStore[T]
}

func (a *tieredKVAdapter[T]) Update(layer int, newK, newV *tensor.TensorNumeric[T]) error {
	return a.store.Update(layer, newK, newV)
}

func (a *tieredKVAdapter[T]) Get(layer int) (*LayerKV[T], bool) {
	return a.store.Get(layer)
}

func (a *tieredKVAdapter[T]) SeqLen() int {
	return a.store.SeqLen()
}

func (a *tieredKVAdapter[T]) Reset() {
	a.store.Reset()
}

func (a *tieredKVAdapter[T]) Truncate(newSeqLen int) {
	a.store.Truncate(newSeqLen)
}
