//go:build !arm64

package xblas

import (
	"math"
	"unsafe"
)

// VexpF32 computes out[i] = exp(x[i]) for n float32 values (scalar fallback).
func VexpF32(out, x *float32, n int) {
	outSlice := unsafe.Slice(out, n)
	xSlice := unsafe.Slice(x, n)
	for i := range n {
		outSlice[i] = float32(math.Exp(float64(xSlice[i])))
	}
}
