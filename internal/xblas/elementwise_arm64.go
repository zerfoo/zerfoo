//go:build arm64

package xblas

// VaddF32 computes out[i] = a[i] + b[i] for n float32 values using NEON.
//
//go:noescape
func VaddF32(out, a, b *float32, n int)

// VmulF32 computes out[i] = a[i] * b[i] for n float32 values using NEON.
//
//go:noescape
func VmulF32(out, a, b *float32, n int)

// VsubF32 computes out[i] = a[i] - b[i] for n float32 values using NEON.
//
//go:noescape
func VsubF32(out, a, b *float32, n int)

// VdivF32 computes out[i] = a[i] / b[i] for n float32 values using NEON.
//
//go:noescape
func VdivF32(out, a, b *float32, n int)
