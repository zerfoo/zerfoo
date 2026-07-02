//go:build !arm64

package xblas

import (
	"math"
	"unsafe"
)

func RMSNormF32(out, x, weight *float32, dim int, eps float32, scale *float32) {
	xSlice := unsafe.Slice(x, dim)
	wSlice := unsafe.Slice(weight, dim)
	oSlice := unsafe.Slice(out, dim)
	var sumSq float32
	for i := range dim {
		sumSq += xSlice[i] * xSlice[i]
	}
	s := float32(1.0 / math.Sqrt(float64(sumSq/float32(dim)+eps)))
	for i := range dim {
		oSlice[i] = xSlice[i] * s * wSlice[i]
	}
	*scale = s
}
