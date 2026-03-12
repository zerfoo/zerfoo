//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FusedDecodeAttentionF32 runs fused single-token attention:
// scores = Q * K^T * scale, weights = softmax(scores), O = weights * V.
// Q: [numQHeads, headDim], K: [kvSeqLen, headDim], V: [kvSeqLen, headDim],
// O: [numQHeads, headDim].
func FusedDecodeAttentionF32(Q, K, V, O unsafe.Pointer, numQHeads, kvSeqLen, headDim int, scale float32, s unsafe.Pointer) error { //nolint:gocritic // match CGo API
	k := klib()
	if k == nil {
		return fmt.Errorf("fused_decode_attention_f32 kernel: kernels not available")
	}
	ret := cuda.Ccall(k.launchFusedDecodeAttentionF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(numQHeads), uintptr(kvSeqLen), uintptr(headDim),
		floatBits(scale), uintptr(s))
	return checkKernel(ret, "fused_decode_attention_f32")
}
