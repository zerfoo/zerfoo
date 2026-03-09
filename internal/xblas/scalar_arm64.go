//go:build arm64

package xblas

// VmulScalarF32 computes out[i] = a[i] * scalar for n float32 values using NEON.
//
//go:noescape
func VmulScalarF32(out, a *float32, scalar float32, n int)

// VaddScalarF32 computes out[i] = a[i] + scalar for n float32 values using NEON.
//
//go:noescape
func VaddScalarF32(out, a *float32, scalar float32, n int)

// VdivScalarF32 computes out[i] = a[i] / scalar for n float32 values using NEON.
//
//go:noescape
func VdivScalarF32(out, a *float32, scalar float32, n int)
