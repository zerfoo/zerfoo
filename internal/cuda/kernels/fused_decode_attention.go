//go:build cuda

package kernels

/*
#cgo LDFLAGS: -L${SRCDIR} -lkernels -lcudart -lstdc++
#include "fused_decode_attention.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// FusedDecodeAttentionF32 runs fused single-token attention:
// scores = Q * K^T * scale, weights = softmax(scores), O = weights * V.
// Q: [numQHeads, headDim], K: [kvSeqLen, headDim], V: [kvSeqLen, headDim],
// O: [numQHeads, headDim].
func FusedDecodeAttentionF32(Q, K, V, O unsafe.Pointer, numQHeads, kvSeqLen, headDim int, scale float32, s unsafe.Pointer) error { //nolint:gocritic // match purego API
	err := C.fused_decode_attention_f32(
		(*C.float)(Q), (*C.float)(K), (*C.float)(V), (*C.float)(O),
		C.int(numQHeads), C.int(kvSeqLen), C.int(headDim),
		C.uint(math.Float32bits(scale)),
		C.cudaStream_t(s),
	)
	if err != C.cudaSuccess {
		return fmt.Errorf("fused_decode_attention_f32: %s",
			C.GoString(C.cudaGetErrorString(err)))
	}
	return nil
}
