//go:build !arm64

package xblas

import (
	"math"
	"unsafe"
)

// SiLUF32 computes silu(x) = x / (1 + exp(-x)) for n float32 values.
func SiLUF32(out, x *float32, n int) {
	oSlice := unsafe.Slice(out, n)
	xSlice := unsafe.Slice(x, n)
	for i := range n {
		v := float64(xSlice[i])
		oSlice[i] = float32(v / (1 + math.Exp(-v)))
	}
}

// SiLUGateF32 computes silu(gate) * up for n float32 values (SwiGLU operation).
func SiLUGateF32(out, gate, up *float32, n int) {
	oSlice := unsafe.Slice(out, n)
	gSlice := unsafe.Slice(gate, n)
	uSlice := unsafe.Slice(up, n)
	for i := range n {
		g := float64(gSlice[i])
		sig := 1.0 / (1 + math.Exp(-g))
		oSlice[i] = float32(g*sig) * uSlice[i]
	}
}
