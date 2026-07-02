//go:build cuda && cutlass

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "flash_attention.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
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
	c := C.int(0)
	if causal {
		c = 1
	}
	err := C.flash_attention_forward_f32(
		(*C.float)(Q), (*C.float)(K), (*C.float)(V), (*C.float)(O),
		C.int(batch), C.int(heads), C.int(seqLen), C.int(headDim),
		c, C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("flash_attention_forward_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}

// FlashAttentionDecode computes single-query attention for autoregressive decode.
// Supports GQA: numQueryHeads may differ from numKVHeads (must be a multiple).
func FlashAttentionDecode(
	Q, K, V, O unsafe.Pointer,
	numBH, maxKVLen, headDim, kvLen int,
	kvLenPtr unsafe.Pointer,
	numQueryHeads, numKVHeads int,
	stream unsafe.Pointer,
) error {
	err := C.flash_attention_decode_f32(
		(*C.float)(Q), (*C.float)(K), (*C.float)(V), (*C.float)(O),
		C.int(numBH), C.int(maxKVLen), C.int(headDim),
		C.int(kvLen), (*C.int)(kvLenPtr),
		C.int(numQueryHeads), C.int(numKVHeads),
		C.cudaStream_t(stream),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("flash_attention_decode_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
