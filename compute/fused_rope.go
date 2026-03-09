package compute

import (
	"fmt"

	"github.com/zerfoo/zerfoo/internal/xblas"
	"github.com/zerfoo/zerfoo/tensor"
)

// FusedRoPE applies rotary position embeddings in a single pass.
// Input shape: [batch, seq_len, head_dim] where head_dim is even.
// cos/sin shape: [seq_len, half_dim] (precomputed angles).
// rotaryDim: number of dimensions that receive rotation (<= head_dim, must be even).
// For each position (b, s):
//
//	out[..., i]            = in[..., i] * cos[s,i] - in[..., i+half] * sin[s,i]      (i < half)
//	out[..., i+half]       = in[..., i+half] * cos[s,i] + in[..., i] * sin[s,i]      (i < half)
//	out[..., rotaryDim..]  = in[..., rotaryDim..]                                      (pass-through)
func FusedRoPE(input, cosAngles, sinAngles *tensor.TensorNumeric[float32], rotaryDim int) (*tensor.TensorNumeric[float32], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("FusedRoPE: expected 3D input [batch, seq, dim], got %dD", len(shape))
	}

	batch := shape[0]
	seqLen := shape[1]
	headDim := shape[2]
	halfRotary := rotaryDim / 2

	cosShape := cosAngles.Shape()
	if len(cosShape) != 2 || cosShape[0] < seqLen || cosShape[1] < halfRotary {
		return nil, fmt.Errorf("FusedRoPE: cos shape %v incompatible with seq_len=%d half_rotary=%d", cosShape, seqLen, halfRotary)
	}

	inData := input.Data()
	cosData := cosAngles.Data()
	sinData := sinAngles.Data()
	cosStride := cosShape[1] // row stride of cos/sin tables
	outData := make([]float32, len(inData))

	for b := range batch {
		for s := range seqLen {
			inOff := (b*seqLen + s) * headDim
			csOff := s * cosStride
			xblas.RoPEF32(&outData[inOff], &inData[inOff], &cosData[csOff], &sinData[csOff], halfRotary, headDim)
		}
	}

	return tensor.New(shape, outData)
}
