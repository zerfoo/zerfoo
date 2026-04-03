package generate

import (
	"github.com/zerfoo/ztensor/tensor"
)

// fp8Storage implements QuantStorage using FP8 E4M3 with per-tensor absmax scaling.
// Storing in FP8 quarters the memory bandwidth vs float32 (1 byte vs 4 bytes
// per element).
type fp8Storage struct {
	buf *tensor.FP8E4M3Storage
}

func newFP8Storage(total int) QuantStorage {
	zeros := make([]float32, total)
	return &fp8Storage{buf: tensor.NewFP8E4M3Storage(zeros)}
}

func (s *fp8Storage) Decode() []float32 {
	return s.buf.Slice()
}

func (s *fp8Storage) EncodeRegion(batch, seqLen, dim, maxSeqLen, cursor int, src []float32) {
	all := s.buf.Slice()
	for bi := range batch {
		srcOff := bi * seqLen * dim
		dstOff := bi*maxSeqLen*dim + cursor*dim
		copy(all[dstOff:dstOff+seqLen*dim], src[srcOff:srcOff+seqLen*dim])
	}
	s.buf.Set(all)
}

// KVCacheFP8 stores key-value tensors for all attention layers using FP8 E4M3
// storage, reducing memory by ~4x compared to float32 and ~2x compared to FP16.
// On Update, float32 values are quantized to FP8; on Get, FP8 values are
// dequantized back to float32.
//
// FP8 E4M3 has lower precision than FP16 (~1.5 decimal digits vs ~3) but the
// perplexity impact is typically within 0.5 for attention KV values.
type KVCacheFP8 struct {
	*quantKVCache
}

// NewKVCacheFP8 creates a KVCacheFP8 for the specified number of layers and
// maximum sequence length. FP8 backing buffers are lazily allocated on the
// first Update call for each layer.
func NewKVCacheFP8(numLayers, maxSeqLen int) *KVCacheFP8 {
	return &KVCacheFP8{
		quantKVCache: newQuantKVCache(numLayers, maxSeqLen, newFP8Storage),
	}
}
