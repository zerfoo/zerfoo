//go:build arm64

package xblas

// SiLUF32 computes silu(x) = x / (1 + exp(-x)) for n float32 values.
//
//go:noescape
func SiLUF32(out, x *float32, n int)

// SiLUGateF32 computes silu(gate) * up for n float32 values (SwiGLU operation).
// result[i] = gate[i] * sigmoid(gate[i]) * up[i]
//
//go:noescape
func SiLUGateF32(out, gate, up *float32, n int)
