package compute

import (
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedDecodeAttentionProvider is implemented by engines that support fused
// single-token attention in a single GPU kernel launch.
// Replaces 3 kernel launches (QK^T + ScaledSoftmax + attn*V) with 1.
type FusedDecodeAttentionProvider[T tensor.Numeric] interface {
	// GPUFusedDecodeAttention computes:
	//   scores = Q * K^T * scale
	//   weights = softmax(scores)
	//   output = weights * V
	// Q: [numQHeads, 1, headDim], K: [kvBatch, kvSeqLen, headDim],
	// V: [kvBatch, kvSeqLen, headDim].
	GPUFusedDecodeAttention(q, k, v *tensor.TensorNumeric[T], scale float32) (*tensor.TensorNumeric[T], error)
}
