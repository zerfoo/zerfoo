//go:build arm64

package xblas

// RMSNormF32 computes out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i].
// x is [D], weight is [D], out is [D]. The scale factor is written to *scale.
//
//go:noescape
func RMSNormF32(out, x, weight *float32, D int, eps float32, scale *float32)
