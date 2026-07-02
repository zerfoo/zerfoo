//go:build !arm64

package xblas

import "unsafe"

// VaddF32 computes out[i] = a[i] + b[i] for n float32 values.
func VaddF32(out, a, b *float32, n int) {
	o, as, bs := unsafe.Slice(out, n), unsafe.Slice(a, n), unsafe.Slice(b, n)
	for i := range n {
		o[i] = as[i] + bs[i]
	}
}

// VmulF32 computes out[i] = a[i] * b[i] for n float32 values.
func VmulF32(out, a, b *float32, n int) {
	o, as, bs := unsafe.Slice(out, n), unsafe.Slice(a, n), unsafe.Slice(b, n)
	for i := range n {
		o[i] = as[i] * bs[i]
	}
}

// VsubF32 computes out[i] = a[i] - b[i] for n float32 values.
func VsubF32(out, a, b *float32, n int) {
	o, as, bs := unsafe.Slice(out, n), unsafe.Slice(a, n), unsafe.Slice(b, n)
	for i := range n {
		o[i] = as[i] - bs[i]
	}
}

// VdivF32 computes out[i] = a[i] / b[i] for n float32 values.
func VdivF32(out, a, b *float32, n int) {
	o, as, bs := unsafe.Slice(out, n), unsafe.Slice(a, n), unsafe.Slice(b, n)
	for i := range n {
		o[i] = as[i] / bs[i]
	}
}
