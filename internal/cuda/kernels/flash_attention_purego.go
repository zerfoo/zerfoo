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
