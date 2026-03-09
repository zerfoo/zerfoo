//go:build arm64

package xblas

// RoPEF32 applies rotary position embeddings to one position.
// in is [headDim], cos/sin are [halfDim], out is [headDim].
// halfDim must be even and <= headDim. halfDim is the rotary dimension / 2.
//
//go:noescape
func RoPEF32(out, in, cos, sin *float32, halfDim, headDim int)
