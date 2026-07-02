//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FlashAttentionForward computes scaled dot-product attention using a fused
// tiled kernel. All tensors are in [batch, heads, seq_len, head_dim] layout.
// When causal is true, an upper-triangular mask is applied.
func FlashAttentionForward(
	Q, K, V, O unsafe.Pointer,
	batch, heads, seqLen, headDim int,
	causal bool,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("flash_attention_forward_f32 kernel: kernels not available")
	}
	c := uintptr(0)
	if causal {
		c = 1
	}
	ret := cuda.Ccall(k.launchFlashAttentionF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(batch), uintptr(heads), uintptr(seqLen), uintptr(headDim),
		c, uintptr(stream))
	return checkKernel(ret, "flash_attention_forward_f32")
}

// FlashAttentionDecode computes single-query attention for autoregressive decode.
// Supports GQA: numQueryHeads may differ from numKVHeads (must be a multiple).
//
// Q: [batch*numQueryHeads, 1, headDim]   -- single query per head.
// K: [batch*numKVHeads, maxKVLen, headDim]  -- pre-allocated KV cache buffer.
// V: [batch*numKVHeads, maxKVLen, headDim]
// O: [batch*numQueryHeads, 1, headDim]   -- output.
//
// kvLen is the actual KV sequence length (used when kvLenPtr is nil).
// kvLenPtr is a GPU-resident int32 pointer; when non-nil the kernel reads
// the KV length from GPU memory at runtime, making it compatible with
// CUDA graph replay (the value is not frozen at capture time).
func FlashAttentionDecode(
	Q, K, V, O unsafe.Pointer,
	numBH, maxKVLen, headDim, kvLen int,
	kvLenPtr unsafe.Pointer,
	numQueryHeads, numKVHeads int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("flash_attention_decode_f32 kernel: kernels not available")
	}
	if k.launchFlashAttentionDecodeF32 == 0 {
		return fmt.Errorf("flash_attention_decode_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchFlashAttentionDecodeF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(numBH), uintptr(maxKVLen), uintptr(headDim),
		uintptr(kvLen), uintptr(kvLenPtr),
		uintptr(numQueryHeads), uintptr(numKVHeads),
		uintptr(stream))
	return checkKernel(ret, "flash_attention_decode_f32")
}
