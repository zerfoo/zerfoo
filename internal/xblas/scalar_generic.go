//go:build !arm64

package xblas

import "unsafe"

// VmulScalarF32 computes out[i] = a[i] * scalar for n float32 values.
func VmulScalarF32(out, a *float32, scalar float32, n int) {
	o, as := unsafe.Slice(out, n), unsafe.Slice(a, n)
	for i := range n {
		o[i] = as[i] * scalar
	}
}

// VaddScalarF32 computes out[i] = a[i] + scalar for n float32 values.
func VaddScalarF32(out, a *float32, scalar float32, n int) {
	o, as := unsafe.Slice(out, n), unsafe.Slice(a, n)
	for i := range n {
		o[i] = as[i] + scalar
	}
}

// VdivScalarF32 computes out[i] = a[i] / scalar for n float32 values.
func VdivScalarF32(out, a *float32, scalar float32, n int) {
	o, as := unsafe.Slice(out, n), unsafe.Slice(a, n)
	for i := range n {
		o[i] = as[i] / scalar
	}
}
