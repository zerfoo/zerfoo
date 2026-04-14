// Package inference: Gemma 4 edge PLE (Per-Layer Embedding) helper nodes.
//
// These nodes implement the PLE combiner described in ADR-086 and
// HuggingFace transformers modeling_gemma4.py Gemma4TextModel.project_per_layer_inputs
// (lines 1674-1696) and Gemma4TextDecoderLayer.forward (lines 1401-1408).
//
// The canonical Gemma 4 edge layout packs 35 per-layer 256-dim embedding
// slices into a single [vocab, 8960] table (`model.ple_embed_tokens.weight`).
// A global [hidden, 8960] projection (`model.ple_model_proj.weight`) and a
// shared [256] RMSNorm (`model.ple_proj_norm.weight`) feed the per-layer
// combiner. Per-block `input_gate`, `ple_layer_proj`, and `post_layernorm`
// weights complete the sub-block.
//
// The builder wires one `pleCombinedProducer` per graph to compute both
// full-width per-layer feature tensors once, and one `pleSliceNode` per
// transformer layer to extract that layer's [B, S, 256] slice and combine
// them into `per_layer_inputs[i]`.
package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// pleCombinedProducer computes the two full-width [B, S, numLayers*pleDim]
// per-layer feature tensors once per forward pass and caches the result for
// consumption by pleSliceNode.
//
// Inputs:
//
//	inputs[0] -- token ids [B, S]
//	inputs[1] -- hidden after main embedding lookup, scaled by sqrt(hidden).
//	             HF multiplies by hidden_size**-0.5 before the projection; we
//	             pre-apply the inverse scale inside this node so the caller
//	             passes the standard `embeds * sqrt(hidden)` (same tensor the
//	             main transformer stack consumes).
//
// Outputs an opaque placeholder tensor; consumers read results via the
// producer's cached tokenPLE / modelProj fields through pleSliceNode.
type pleCombinedProducer[T tensor.Numeric] struct {
	engine compute.Engine[T]

	pleEmbed     *tensor.TensorNumeric[T] // [vocab, numLayers*pleDim]
	pleModelProj *tensor.TensorNumeric[T] // [hidden, numLayers*pleDim] (already transposed for MatMul)

	numLayers  int
	pleDim     int
	hiddenSize int
	vocabSize  int

	tokenPLE  *tensor.TensorNumeric[T] // cached: [B, S, numLayers*pleDim]
	modelProj *tensor.TensorNumeric[T] // cached: [B, S, numLayers*pleDim]
}

func newPLECombinedProducer[T tensor.Numeric](
	engine compute.Engine[T],
	pleEmbed, pleModelProj *tensor.TensorNumeric[T],
	numLayers, pleDim, hiddenSize int,
) (*pleCombinedProducer[T], error) {
	if pleEmbed == nil || pleModelProj == nil {
		return nil, fmt.Errorf("pleCombinedProducer: pleEmbed and pleModelProj must be non-nil")
	}
	embShape := pleEmbed.Shape()
	if len(embShape) != 2 || embShape[1] != numLayers*pleDim {
		return nil, fmt.Errorf("pleCombinedProducer: pleEmbed shape %v incompatible with numLayers=%d pleDim=%d",
			embShape, numLayers, pleDim)
	}
	projShape := pleModelProj.Shape()
	if len(projShape) != 2 || projShape[0] != hiddenSize || projShape[1] != numLayers*pleDim {
		return nil, fmt.Errorf("pleCombinedProducer: pleModelProj shape %v incompatible with hidden=%d numLayers=%d pleDim=%d",
			projShape, hiddenSize, numLayers, pleDim)
	}
	return &pleCombinedProducer[T]{
		engine:       engine,
		pleEmbed:     pleEmbed,
		pleModelProj: pleModelProj,
		numLayers:    numLayers,
		pleDim:       pleDim,
		hiddenSize:   hiddenSize,
		vocabSize:    embShape[0],
	}, nil
}

func (p *pleCombinedProducer[T]) OpType() string                   { return "Gemma4PLECombinedProducer" }
func (p *pleCombinedProducer[T]) Attributes() map[string]any        { return nil }
func (p *pleCombinedProducer[T]) OutputShape() []int                { return nil }
func (p *pleCombinedProducer[T]) Parameters() []*graph.Parameter[T] { return nil }

// EmbeddedFrozen registers the PLE tables as frozen constants so the graph
// compiler treats them the same as normal embedding/weight tensors.
func (p *pleCombinedProducer[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{p.pleEmbed, p.pleModelProj}
}

// Forward computes per-layer features. Returns inputs[1] unchanged as a
// pass-through; the actual per-layer outputs are read by pleSliceNode
// consumers via p.tokenPLE and p.modelProj.
func (p *pleCombinedProducer[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: expected 2 inputs (ids, hidden), got %d", len(inputs))
	}
	ids := inputs[0]
	hidden := inputs[1]

	idsShape := ids.Shape()
	if len(idsShape) != 2 {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: ids rank %d, want 2 ([B, S])", len(idsShape))
	}
	batch, seqLen := idsShape[0], idsShape[1]
	totalPLE := p.numLayers * p.pleDim

	// --- 1. Token-identity PLE lookup.
	// Produce [B, S, totalPLE] by gathering rows of p.pleEmbed. Use the
	// same logic as embeddingLookupNode but specialized to float data here.
	idData := ids.Data()
	flatLen := batch * seqLen
	tokenFlat := make([]T, flatLen*totalPLE)
	pleData := p.pleEmbed.Data()
	if pleData == nil {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: CPU path requires dense pleEmbed data")
	}
	for i := 0; i < flatLen; i++ {
		id := int(idData[i])
		if id < 0 || id >= p.vocabSize {
			return nil, fmt.Errorf("Gemma4PLECombinedProducer: token id %d out of range [0, %d)", id, p.vocabSize)
		}
		copy(tokenFlat[i*totalPLE:(i+1)*totalPLE], pleData[id*totalPLE:(id+1)*totalPLE])
	}
	tokenPLE, err := tensor.New[T]([]int{batch, seqLen, totalPLE}, tokenFlat)
	if err != nil {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: tokenPLE alloc: %w", err)
	}

	// Scale by sqrt(pleDim). HF: per-layer embedding is multiplied by
	// sqrt(hidden_size_per_layer_input) = sqrt(256) = 16 (line 1688-1689).
	pleScale := T(math.Sqrt(float64(p.pleDim)))
	tokenPLE, err = p.engine.MulScalar(ctx, tokenPLE, pleScale)
	if err != nil {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: tokenPLE scale: %w", err)
	}
	p.tokenPLE = tokenPLE

	// --- 2. Per-layer model projection.
	// HF (line 1687): per_layer_projection = per_layer_model_proj(embeds * hidden_size**-0.5).
	// Caller passed in `embeds * sqrt(hidden)` (the standard scaled embedding that
	// feeds the transformer stack). Multiplying by hidden**-0.5 yields
	// `embeds * sqrt(hidden) * hidden**-0.5 = embeds * hidden**-0.5`, matching HF.
	invScale := T(1.0 / math.Sqrt(float64(p.hiddenSize)))
	scaled, err := p.engine.MulScalar(ctx, hidden, invScale)
	if err != nil {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: invScale hidden: %w", err)
	}
	// [B, S, hidden] @ [hidden, totalPLE] -> [B, S, totalPLE].
	proj, err := p.engine.MatMul(ctx, scaled, p.pleModelProj)
	if err != nil {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: model_proj matmul: %w", err)
	}
	p.modelProj = proj

	// Pass-through the hidden input so downstream nodes that declare a
	// dependency on the producer see a well-formed tensor.
	return hidden, nil
}

func (p *pleCombinedProducer[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// pleSliceNode extracts the per-layer [B, S, pleDim] slice from the producer's
// cached full-width tensors, applies the shared per-layer projection RMSNorm
// to the projection slice, adds the token-identity PLE slice, and scales by
// 1/sqrt(2) to produce `per_layer_inputs[layerIdx]`. See HF
// modeling_gemma4.py lines 1682-1696.
type pleSliceNode[T tensor.Numeric] struct {
	engine    compute.Engine[T]
	producer  *pleCombinedProducer[T]
	normGain  *tensor.TensorNumeric[T] // [pleDim] shared RMSNorm gain
	eps       float32
	layerIdx  int
	pleDim    int
	numLayers int
}

func newPLESliceNode[T tensor.Numeric](
	engine compute.Engine[T],
	producer *pleCombinedProducer[T],
	normGain *tensor.TensorNumeric[T],
	eps float32,
	layerIdx int,
) (*pleSliceNode[T], error) {
	if producer == nil {
		return nil, fmt.Errorf("pleSliceNode: producer must be non-nil")
	}
	if normGain == nil {
		return nil, fmt.Errorf("pleSliceNode: normGain must be non-nil")
	}
	normShape := normGain.Shape()
	if len(normShape) != 1 || normShape[0] != producer.pleDim {
		return nil, fmt.Errorf("pleSliceNode: normGain shape %v incompatible with pleDim=%d", normShape, producer.pleDim)
	}
	if layerIdx < 0 || layerIdx >= producer.numLayers {
		return nil, fmt.Errorf("pleSliceNode: layerIdx %d out of range [0, %d)", layerIdx, producer.numLayers)
	}
	return &pleSliceNode[T]{
		engine:    engine,
		producer:  producer,
		normGain:  normGain,
		eps:       eps,
		layerIdx:  layerIdx,
		pleDim:    producer.pleDim,
		numLayers: producer.numLayers,
	}, nil
}

func (n *pleSliceNode[T]) OpType() string                   { return "Gemma4PLESlice" }
func (n *pleSliceNode[T]) Attributes() map[string]any        { return map[string]any{"layer": n.layerIdx} }
func (n *pleSliceNode[T]) OutputShape() []int                { return nil }
func (n *pleSliceNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *pleSliceNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{n.normGain}
}

// Forward reads the producer's cached tensors, extracts slice [layerIdx],
// applies RMSNorm to the projection slice (dim=pleDim), adds the
// token-identity slice, scales by 1/sqrt(2), and returns [B, S, pleDim].
func (n *pleSliceNode[T]) Forward(ctx context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if n.producer.tokenPLE == nil || n.producer.modelProj == nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): producer has not run Forward yet", n.layerIdx)
	}
	tokenSlice, err := sliceLastDim[T](ctx, n.engine, n.producer.tokenPLE, n.layerIdx*n.pleDim, n.pleDim)
	if err != nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): token slice: %w", n.layerIdx, err)
	}
	projSlice, err := sliceLastDim[T](ctx, n.engine, n.producer.modelProj, n.layerIdx*n.pleDim, n.pleDim)
	if err != nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): proj slice: %w", n.layerIdx, err)
	}
	// RMSNorm over last dim using the shared normGain.
	projNormed, err := rmsNormLastDim[T](ctx, n.engine, projSlice, n.normGain, n.eps)
	if err != nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): rmsnorm: %w", n.layerIdx, err)
	}
	combined, err := n.engine.Add(ctx, projNormed, tokenSlice)
	if err != nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): add: %w", n.layerIdx, err)
	}
	// per_layer_input_scale = 1/sqrt(2) (HF line 1695).
	inputScale := T(1.0 / math.Sqrt(2.0))
	scaled, err := n.engine.MulScalar(ctx, combined, inputScale)
	if err != nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): scale: %w", n.layerIdx, err)
	}
	return scaled, nil
}

func (n *pleSliceNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// sliceLastDim extracts elements [start, start+length) along the last axis of
// a rank-3 tensor shaped [B, S, D]. Implemented via a direct data copy; the
// compute.Engine interface has no generic slice primitive today.
func sliceLastDim[T tensor.Numeric](_ context.Context, _ compute.Engine[T], src *tensor.TensorNumeric[T], start, length int) (*tensor.TensorNumeric[T], error) {
	shape := src.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("sliceLastDim: expected rank 3, got %d", len(shape))
	}
	batch, seqLen, dim := shape[0], shape[1], shape[2]
	if start < 0 || start+length > dim {
		return nil, fmt.Errorf("sliceLastDim: [%d, %d) out of range [0, %d)", start, start+length, dim)
	}
	srcData := src.Data()
	if srcData == nil {
		return nil, fmt.Errorf("sliceLastDim: CPU path requires dense src data")
	}
	out := make([]T, batch*seqLen*length)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			srcOff := (b*seqLen+s)*dim + start
			dstOff := (b*seqLen + s) * length
			copy(out[dstOff:dstOff+length], srcData[srcOff:srcOff+length])
		}
	}
	return tensor.New[T]([]int{batch, seqLen, length}, out)
}

// rmsNormLastDim computes RMSNorm over the last axis of a rank-3 tensor
// using the provided per-channel gain. Gemma-family RMSNorm uses (1 + gain)
// as the effective scale (matches normalization.RMSNorm Gemma default).
func rmsNormLastDim[T tensor.Numeric](_ context.Context, _ compute.Engine[T], src *tensor.TensorNumeric[T], gain *tensor.TensorNumeric[T], eps float32) (*tensor.TensorNumeric[T], error) {
	shape := src.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("rmsNormLastDim: expected rank 3, got %d", len(shape))
	}
	dim := shape[2]
	gainShape := gain.Shape()
	if len(gainShape) != 1 || gainShape[0] != dim {
		return nil, fmt.Errorf("rmsNormLastDim: gain shape %v incompatible with dim=%d", gainShape, dim)
	}
	gainData := gain.Data()
	srcData := src.Data()
	if srcData == nil || gainData == nil {
		return nil, fmt.Errorf("rmsNormLastDim: CPU path requires dense data")
	}
	out := make([]T, len(srcData))
	batch, seqLen := shape[0], shape[1]
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * dim
			var sumSq float64
			for d := 0; d < dim; d++ {
				v := float64(srcData[off+d])
				sumSq += v * v
			}
			invRMS := 1.0 / math.Sqrt(sumSq/float64(dim)+float64(eps))
			for d := 0; d < dim; d++ {
				g := 1.0 + float64(gainData[d])
				out[off+d] = T(float64(srcData[off+d]) * invRMS * g)
			}
		}
	}
	return tensor.New[T](shape, out)
}

// layerOutputScaleNode multiplies its input by a learned scalar stored as a
// 1-element tensor (`model.layers.N.layer_output_scale.weight`). Corresponds
// to `self.layer_scalar` in HF modeling_gemma4.py (line 1337, applied at line
// 1410: hidden_states *= self.layer_scalar).
type layerOutputScaleNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	scale  *tensor.TensorNumeric[T] // shape [1]
}

func newLayerOutputScaleNode[T tensor.Numeric](engine compute.Engine[T], scale *tensor.TensorNumeric[T]) (*layerOutputScaleNode[T], error) {
	if scale == nil {
		return nil, fmt.Errorf("layerOutputScaleNode: scale must be non-nil")
	}
	shape := scale.Shape()
	if len(shape) != 1 || shape[0] != 1 {
		return nil, fmt.Errorf("layerOutputScaleNode: scale shape %v, want [1]", shape)
	}
	return &layerOutputScaleNode[T]{engine: engine, scale: scale}, nil
}

func (n *layerOutputScaleNode[T]) OpType() string                   { return "Gemma4LayerOutputScale" }
func (n *layerOutputScaleNode[T]) Attributes() map[string]any        { return nil }
func (n *layerOutputScaleNode[T]) OutputShape() []int                { return nil }
func (n *layerOutputScaleNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *layerOutputScaleNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{n.scale}
}

func (n *layerOutputScaleNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Gemma4LayerOutputScale: expected 1 input, got %d", len(inputs))
	}
	sData := n.scale.Data()
	if len(sData) == 0 {
		return nil, fmt.Errorf("Gemma4LayerOutputScale: scale data empty")
	}
	return n.engine.MulScalar(ctx, inputs[0], sData[0])
}

func (n *layerOutputScaleNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// elementwiseMulNode computes element-wise inputs[0] * inputs[1]. Used by the
// PLE sub-block gate stage: `gelu(inp_gate(h)) * per_layer_inputs[i]`.
type elementwiseMulNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
}

func (n *elementwiseMulNode[T]) OpType() string                   { return "ElementwiseMul" }
func (n *elementwiseMulNode[T]) Attributes() map[string]any        { return nil }
func (n *elementwiseMulNode[T]) OutputShape() []int                { return nil }
func (n *elementwiseMulNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (n *elementwiseMulNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("ElementwiseMul: expected 2 inputs, got %d", len(inputs))
	}
	return n.engine.Mul(ctx, inputs[0], inputs[1])
}

func (n *elementwiseMulNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
