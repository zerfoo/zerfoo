//go:build !arm64

package xblas

import "unsafe"

func RoPEF32(out, in, cos, sin *float32, halfDim, headDim int) {
	oSlice := unsafe.Slice(out, headDim)
	iSlice := unsafe.Slice(in, headDim)
	cSlice := unsafe.Slice(cos, halfDim)
	sSlice := unsafe.Slice(sin, halfDim)
	for i := range halfDim {
		oSlice[i] = iSlice[i]*cSlice[i] - iSlice[i+halfDim]*sSlice[i]
		oSlice[i+halfDim] = iSlice[i+halfDim]*cSlice[i] + iSlice[i]*sSlice[i]
	}
	for i := halfDim * 2; i < headDim; i++ {
		oSlice[i] = iSlice[i]
	}
}
