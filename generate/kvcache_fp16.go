package generate

import (
	"github.com/zerfoo/ztensor/tensor"
)

// fp16Storage implements QuantStorage using Float16 (IEEE 754 half-precision).
// Storing in FP16 halves the memory bandwidth vs float32 (2 bytes vs 4 bytes
// per element).
type fp16Storage struct {
	buf *tensor.Float16Storage
}

func newFP16Storage(total int) QuantStorage {
	zeros := make([]float32, total)
	return &fp16Storage{buf: tensor.NewFloat16StorageFromF32(zeros)}
}

func (s *fp16Storage) Decode() []float32 {
	return s.buf.Slice()
}

func (s *fp16Storage) EncodeRegion(batch, seqLen, dim, maxSeqLen, cursor int, src []float32) {
	all := s.buf.Slice()
	for bi := range batch {
		srcOff := bi * seqLen * dim
		dstOff := bi*maxSeqLen*dim + cursor*dim
		copy(all[dstOff:dstOff+seqLen*dim], src[srcOff:srcOff+seqLen*dim])
	}
	s.buf.Set(all)
}

// KVCacheFP16 stores key-value tensors for all attention layers using FP16 storage,
// halving the memory bandwidth compared to float32. On Update, float32 values are
// converted to FP16; on Get, FP16 values are converted back to float32.
//
// This is a drop-in replacement for KVCache[float32] with 2x bandwidth reduction
// at the cost of slight precision loss (FP16 has ~3 decimal digits of precision).
type KVCacheFP16 struct {
	*quantKVCache
}

// NewKVCacheFP16 creates a KVCacheFP16 for the specified number of layers and
// maximum sequence length. FP16 backing buffers are lazily allocated on the
// first Update call for each layer.
func NewKVCacheFP16(numLayers, maxSeqLen int) *KVCacheFP16 {
	return &KVCacheFP16{
		quantKVCache: newQuantKVCache(numLayers, maxSeqLen, newFP16Storage),
	}
}
