package inference

import (
	"context"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// llamaAttnNode implements grouped-query attention with RoPE for vision-language
// text decoders (used by arch_qwenvl and arch_voxtral). New architectures should
// compose from layers/attention.GroupedQueryAttention instead.
type llamaAttnNode[T tensor.Numeric] struct {
	engine                      compute.Engine[T]
	ops                         numeric.Arithmetic[T]
	qW, kW, vW, oW             *tensor.TensorNumeric[T]
	numHeads, numKVHeads        int
	headDim                     int
	ropeTheta                   float64
	maxSeqLen                   int
}

func (a *llamaAttnNode[T]) OpType() string                   { return "LLaVAAttn" }
func (a *llamaAttnNode[T]) Attributes() map[string]any        { return nil }
func (a *llamaAttnNode[T]) OutputShape() []int                { return nil }
func (a *llamaAttnNode[T]) Parameters() []*graph.Parameter[T] { return nil }
func (a *llamaAttnNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{a.qW, a.kW, a.vW, a.oW}
}

func (a *llamaAttnNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	batch, seqLen, hiddenDim := shape[0], shape[1], shape[2]
	inData := input.Data()
	qWData, kWData, vWData, oWData := a.qW.Data(), a.kW.Data(), a.vW.Data(), a.oW.Data()
	kvDim := a.numKVHeads * a.headDim
	qDim := a.numHeads * a.headDim

	q := make([]T, batch*seqLen*qDim)
	k := make([]T, batch*seqLen*kvDim)
	v := make([]T, batch*seqLen*kvDim)

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < qDim; o++ {
				var sum T
				for d := 0; d < hiddenDim; d++ {
					sum = a.ops.Add(sum, a.ops.Mul(inData[b*seqLen*hiddenDim+s*hiddenDim+d], qWData[o*hiddenDim+d]))
				}
				q[b*seqLen*qDim+s*qDim+o] = sum
			}
			for o := 0; o < kvDim; o++ {
				var sumK, sumV T
				for d := 0; d < hiddenDim; d++ {
					xVal := inData[b*seqLen*hiddenDim+s*hiddenDim+d]
					sumK = a.ops.Add(sumK, a.ops.Mul(xVal, kWData[o*hiddenDim+d]))
					sumV = a.ops.Add(sumV, a.ops.Mul(xVal, vWData[o*hiddenDim+d]))
				}
				k[b*seqLen*kvDim+s*kvDim+o] = sumK
				v[b*seqLen*kvDim+s*kvDim+o] = sumV
			}
		}
	}

	applyRoPE(q, batch, seqLen, a.numHeads, a.headDim, a.ropeTheta, a.ops)
	applyRoPE(k, batch, seqLen, a.numKVHeads, a.headDim, a.ropeTheta, a.ops)

	scale := T(1.0 / math.Sqrt(float64(a.headDim)))
	kvGroupSize := a.numHeads / a.numKVHeads
	attnOut := make([]T, batch*seqLen*qDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < a.numHeads; h++ {
			kvH := h / kvGroupSize
			scores := make([]T, seqLen*seqLen)
			for qi := 0; qi < seqLen; qi++ {
				for ki := 0; ki < seqLen; ki++ {
					var dot T
					for d := 0; d < a.headDim; d++ {
						dot = a.ops.Add(dot, a.ops.Mul(q[b*seqLen*qDim+qi*qDim+h*a.headDim+d], k[b*seqLen*kvDim+ki*kvDim+kvH*a.headDim+d]))
					}
					scores[qi*seqLen+ki] = a.ops.Mul(dot, scale)
				}
			}
			for qi := 0; qi < seqLen; qi++ {
				maxVal := scores[qi*seqLen]
				for ki := 1; ki < seqLen; ki++ {
					if a.ops.GreaterThan(scores[qi*seqLen+ki], maxVal) {
						maxVal = scores[qi*seqLen+ki]
					}
				}
				var sumExp T
				for ki := 0; ki < seqLen; ki++ {
					scores[qi*seqLen+ki] = a.ops.Exp(a.ops.Sub(scores[qi*seqLen+ki], maxVal))
					sumExp = a.ops.Add(sumExp, scores[qi*seqLen+ki])
				}
				for ki := 0; ki < seqLen; ki++ {
					scores[qi*seqLen+ki] = a.ops.Div(scores[qi*seqLen+ki], sumExp)
				}
				for d := 0; d < a.headDim; d++ {
					var val T
					for ki := 0; ki < seqLen; ki++ {
						val = a.ops.Add(val, a.ops.Mul(scores[qi*seqLen+ki], v[b*seqLen*kvDim+ki*kvDim+kvH*a.headDim+d]))
					}
					attnOut[b*seqLen*qDim+qi*qDim+h*a.headDim+d] = val
				}
			}
		}
	}

	out := make([]T, batch*seqLen*hiddenDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < hiddenDim; o++ {
				var sum T
				for d := 0; d < qDim; d++ {
					sum = a.ops.Add(sum, a.ops.Mul(attnOut[b*seqLen*qDim+s*qDim+d], oWData[o*qDim+d]))
				}
				out[b*seqLen*hiddenDim+s*hiddenDim+o] = sum
			}
		}
	}
	return tensor.New[T]([]int{batch, seqLen, hiddenDim}, out)
}

func (a *llamaAttnNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// llamaFFNNode implements SwiGLU FFN for vision-language text decoders.
type llamaFFNNode[T tensor.Numeric] struct {
	engine            compute.Engine[T]
	ops               numeric.Arithmetic[T]
	gateW, upW, downW *tensor.TensorNumeric[T]
}

func (f *llamaFFNNode[T]) OpType() string                   { return "LLaVAFFN" }
func (f *llamaFFNNode[T]) Attributes() map[string]any        { return nil }
func (f *llamaFFNNode[T]) OutputShape() []int                { return nil }
func (f *llamaFFNNode[T]) Parameters() []*graph.Parameter[T] { return nil }
func (f *llamaFFNNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{f.gateW, f.upW, f.downW}
}

func (f *llamaFFNNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	shape := input.Shape()
	batch, seqLen, hiddenDim := shape[0], shape[1], shape[2]
	interDim := f.gateW.Shape()[0]
	inData, gateData, upData, downData := input.Data(), f.gateW.Data(), f.upW.Data(), f.downW.Data()
	gate := make([]T, batch*seqLen*interDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < interDim; o++ {
				var sumG, sumU T
				for d := 0; d < hiddenDim; d++ {
					xVal := inData[b*seqLen*hiddenDim+s*hiddenDim+d]
					sumG = f.ops.Add(sumG, f.ops.Mul(xVal, gateData[o*hiddenDim+d]))
					sumU = f.ops.Add(sumU, f.ops.Mul(xVal, upData[o*hiddenDim+d]))
				}
				one := f.ops.One()
				negG := f.ops.Mul(f.ops.FromFloat64(-1.0), sumG)
				sigmoid := f.ops.Div(one, f.ops.Add(one, f.ops.Exp(negG)))
				gate[b*seqLen*interDim+s*interDim+o] = f.ops.Mul(f.ops.Mul(sumG, sigmoid), sumU)
			}
		}
	}
	out := make([]T, batch*seqLen*hiddenDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < hiddenDim; o++ {
				var sum T
				for d := 0; d < interDim; d++ {
					sum = f.ops.Add(sum, f.ops.Mul(gate[b*seqLen*interDim+s*interDim+d], downData[o*interDim+d]))
				}
				out[b*seqLen*hiddenDim+s*hiddenDim+o] = sum
			}
		}
	}
	return tensor.New[T]([]int{batch, seqLen, hiddenDim}, out)
}

func (f *llamaFFNNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// applyRoPE applies rotary positional embeddings in-place.
func applyRoPE[T tensor.Numeric](data []T, batch, seqLen, numHeads, headDim int, theta float64, ops numeric.Arithmetic[T]) {
	dim := numHeads * headDim
	for b := 0; b < batch; b++ {
		for pos := 0; pos < seqLen; pos++ {
			for h := 0; h < numHeads; h++ {
				for d := 0; d < headDim/2; d++ {
					freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
					angle := float64(pos) * freq
					cosVal := ops.FromFloat64(math.Cos(angle))
					sinVal := ops.FromFloat64(math.Sin(angle))
					idx0 := b*seqLen*dim + pos*dim + h*headDim + 2*d
					idx1 := idx0 + 1
					x0, x1 := data[idx0], data[idx1]
					data[idx0] = ops.Sub(ops.Mul(x0, cosVal), ops.Mul(x1, sinVal))
					data[idx1] = ops.Add(ops.Mul(x1, cosVal), ops.Mul(x0, sinVal))
				}
			}
		}
	}
}

// newRMSNormNode creates an RMSNorm graph node.
func newRMSNormNode(
	engine compute.Engine[float32],
	ops numeric.Float32Ops,
	eps float32,
	weightParam *graph.Parameter[float32],
) (*rmsNormWrapNode, error) {
	return &rmsNormWrapNode{engine: engine, ops: ops, weight: weightParam.Value, eps: eps}, nil
}

type rmsNormWrapNode struct {
	engine compute.Engine[float32]
	ops    numeric.Float32Ops
	weight *tensor.TensorNumeric[float32]
	eps    float32
}

func (r *rmsNormWrapNode) OpType() string                         { return "RMSNorm" }
func (r *rmsNormWrapNode) Attributes() map[string]any              { return nil }
func (r *rmsNormWrapNode) OutputShape() []int                      { return nil }
func (r *rmsNormWrapNode) Parameters() []*graph.Parameter[float32] { return nil }
func (r *rmsNormWrapNode) EmbeddedFrozen() []*tensor.TensorNumeric[float32] {
	return []*tensor.TensorNumeric[float32]{r.weight}
}

func (r *rmsNormWrapNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	input := inputs[0]
	shape := input.Shape()
	data := input.Data()
	wData := r.weight.Data()
	hiddenDim := shape[len(shape)-1]
	numTokens := len(data) / hiddenDim
	out := make([]float32, len(data))
	for t := 0; t < numTokens; t++ {
		offset := t * hiddenDim
		var sumSq float32
		for d := 0; d < hiddenDim; d++ {
			v := data[offset+d]
			sumSq += v * v
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(hiddenDim)) + float64(r.eps)))
		invRMS := 1.0 / rms
		for d := 0; d < hiddenDim; d++ {
			out[offset+d] = data[offset+d] * invRMS * wData[d]
		}
	}
	return tensor.New(shape, out)
}

func (r *rmsNormWrapNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return nil, nil
}

var _ graph.EmbeddedFrozenProvider[float32] = (*llamaAttnNode[float32])(nil)
var _ graph.EmbeddedFrozenProvider[float32] = (*llamaFFNNode[float32])(nil)
var _ graph.EmbeddedFrozenProvider[float32] = (*rmsNormWrapNode)(nil)
