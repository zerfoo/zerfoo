//go:build arm64

package xblas

// SoftmaxF32 computes softmax(x) in-place for a float32 vector of length n.
// Uses 3-pass NEON: (1) find max, (2) exp(x-max), (3) normalize by 1/sum.
//
//go:noescape
func SoftmaxF32(data *float32, n int)
