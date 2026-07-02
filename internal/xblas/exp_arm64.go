//go:build arm64

package xblas

// VexpF32 computes out[i] = exp(x[i]) for n float32 values using NEON.
// Uses range-reduced degree-5 polynomial approximation.
// Max relative error < 2e-7 for x in [-87.3, 88.7].
//
//go:noescape
func VexpF32(out, x *float32, n int)
