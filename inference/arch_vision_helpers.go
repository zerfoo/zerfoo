package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
)

// newVisionRMSNorm creates an RMSNorm graph node using the canonical
// normalization.RMSNorm implementation.
func newVisionRMSNorm(
	engine compute.Engine[float32],
	eps float32,
	weightParam *graph.Parameter[float32],
) (*normalization.RMSNorm[float32], error) {
	return normalization.NewRMSNormFromParam[float32](engine, engine.Ops(), eps, weightParam)
}

// newVisionGQA creates a grouped-query attention node using the canonical
// attention.GroupedQueryAttention implementation with RoPE. Weights are
// expected in GGUF layout [outDim, inDim] and are transposed internally
// for the canonical Linear layer which expects [inDim, outDim].
func newVisionGQA(
	engine compute.Engine[float32],
	hiddenSize, numHeads, numKVHeads, headDim, maxSeqLen int,
	ropeTheta float64,
	qW, kW, vW, oW *tensor.TensorNumeric[float32],
	prefix string,
	pw paramWrapper[float32],
) (*attention.GroupedQueryAttention[float32], error) {
	qWT, err := cpuTranspose2D(qW)
	if err != nil {
		return nil, fmt.Errorf("transpose q: %w", err)
	}
	kWT, err := cpuTranspose2D(kW)
	if err != nil {
		return nil, fmt.Errorf("transpose k: %w", err)
	}
	vWT, err := cpuTranspose2D(vW)
	if err != nil {
		return nil, fmt.Errorf("transpose v: %w", err)
	}
	oWT, err := cpuTranspose2D(oW)
	if err != nil {
		return nil, fmt.Errorf("transpose o: %w", err)
	}

	wq := core.NewDenseFromParams(
		core.NewLinearFromParam(engine, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)), nil,
	)
	wk := core.NewDenseFromParams(
		core.NewLinearFromParam(engine, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)), nil,
	)
	wv := core.NewDenseFromParams(
		core.NewLinearFromParam(engine, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)), nil,
	)
	wo := core.NewDenseFromParams(
		core.NewLinearFromParam(engine, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)), nil,
	)

	if maxSeqLen == 0 {
		maxSeqLen = 2048
	}
	// Vision-language models receive audio/image encoder output whose
	// sequence length can exceed the text decoder's configured maxSeqLen.
	// Pre-compute enough RoPE positions to cover encoder outputs.
	if maxSeqLen < 8192 {
		maxSeqLen = 8192
	}
	rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
		context.Background(), engine, headDim, maxSeqLen,
		embeddings.WithRotaryBase(ropeTheta),
	)
	if err != nil {
		return nil, fmt.Errorf("rope: %w", err)
	}

	gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
		engine, engine.Ops(), hiddenSize, numHeads, numKVHeads,
		wq, wk, wv, wo, rope, headDim,
	)
	if err != nil {
		return nil, fmt.Errorf("gqa: %w", err)
	}

	return gqa, nil
}

// newVisionSwiGLUFFN creates a SwiGLU FFN node using the canonical
// core.FFN implementation. Weights are expected in GGUF layout
// [outDim, inDim] and are transposed internally.
func newVisionSwiGLUFFN(
	engine compute.Engine[float32],
	hiddenSize int,
	gateW, upW, downW *tensor.TensorNumeric[float32],
	prefix string,
) (*core.FFN[float32], error) {
	interDim := gateW.Shape()[0]

	ffn, err := core.NewFFN[float32](
		prefix+"mlp", engine, engine.Ops(),
		hiddenSize, interDim, hiddenSize,
		core.WithSwiGLU[float32](),
		core.WithFFNNoBias[float32](),
	)
	if err != nil {
		return nil, err
	}

	gateWT, err := cpuTranspose2D(gateW)
	if err != nil {
		return nil, fmt.Errorf("transpose gate: %w", err)
	}
	upWT, err := cpuTranspose2D(upW)
	if err != nil {
		return nil, fmt.Errorf("transpose up: %w", err)
	}
	downWT, err := cpuTranspose2D(downW)
	if err != nil {
		return nil, fmt.Errorf("transpose down: %w", err)
	}

	ffnParams := ffn.Parameters()
	ffnParams[0].Value = gateWT  // w1 = gate
	ffnParams[1].Value = downWT  // w2 = down
	ffnParams[2].Value = upWT    // w3 = up

	return ffn, nil
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
