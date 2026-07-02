//go:build !cuda

package kernels

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/zerfoo/internal/cuda"
)

// FlashDecodeSplitKV computes single-query attention for autoregressive decode
// using split-KV parallelism. The KV cache is split across thread blocks per
// head, each computing partial attention with online softmax. A reduction kernel
// merges the partial results using log-sum-exp correction.
//
// Q:           [batch*numQueryHeads, headDim]
// K:           [batch, maxKVLen, numKVHeads*headDim]
// V:           [batch, maxKVLen, numKVHeads*headDim]
// O:           [batch*numQueryHeads, headDim]
// partialO:    [numBH*numSplits, headDim] scratch buffer
// partialLSE:  [2*numBH*numSplits] scratch buffer (max + sum)
//
// kvLen is the actual KV sequence length. kvLenPtr is a GPU-resident int32;
// when non-nil the kernel reads the length at runtime for CUDA graph compat.
// chunkSize controls how many KV positions each thread block processes.
func FlashDecodeSplitKV(
	Q, K, V, O unsafe.Pointer,
	partialO, partialLSE unsafe.Pointer,
	numBH, maxKVLen, headDim, kvLen int,
	kvLenPtr unsafe.Pointer,
	numQueryHeads, numKVHeads int,
	chunkSize int,
	stream unsafe.Pointer,
) error {
	k := klib()
	if k == nil {
		return fmt.Errorf("flash_decode_splitkv_f32 kernel: kernels not available")
	}
	if k.launchFlashDecodeSplitKVF32 == 0 {
		return fmt.Errorf("flash_decode_splitkv_f32 kernel: symbol not loaded")
	}
	ret := cuda.Ccall(k.launchFlashDecodeSplitKVF32,
		uintptr(Q), uintptr(K), uintptr(V), uintptr(O),
		uintptr(partialO), uintptr(partialLSE),
		uintptr(numBH), uintptr(maxKVLen), uintptr(headDim),
		uintptr(kvLen), uintptr(kvLenPtr),
		uintptr(numQueryHeads), uintptr(numKVHeads),
		uintptr(chunkSize),
		uintptr(stream))
	return checkKernel(ret, "flash_decode_splitkv_f32")
}

// IsFlashDecodeSplitKVSupported returns true if the split-KV flash decode
// kernel symbol was loaded from libkernels.so.
func IsFlashDecodeSplitKVSupported() bool {
	k := klib()
	return k != nil && k.launchFlashDecodeSplitKVF32 != 0
}
