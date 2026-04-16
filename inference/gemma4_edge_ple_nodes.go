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

	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// pleCombinedProducer computes the two full-width [B, S, numLayers*pleDim]
// per-layer feature tensors once per forward pass and pre-slices them into
// per-layer [B, S, pleDim] GPU-resident tensors for consumption by
// pleSliceNode.
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
// Outputs the hidden input unchanged as a pass-through; consumers read
// per-layer results via the producer's cached tokenPLESlices /
// modelProjSlices fields through pleSliceNode.
//
// CUDA graph capture strategy (see ADR-088). The node is listed as
// non-capturable in ztensor's graph compiler (Gemma4PLECombinedProducer),
// so it runs in pre-capture on every forward. After computing the two
// full-width tensors, it pre-slices them into 35 per-layer slices and
// caches the slices as GPU-resident tensors with stable addresses across
// calls. On the first call, stable GPU buffers are allocated via a
// no-op MulScalar; on subsequent calls, the buffers are refreshed in
// place via GPUStorage.CopyFromHost. pleSliceNode then reads the cached
// GPU slices directly — no .Data() calls — which makes the per-layer
// PLE combiner fully capturable.
type pleCombinedProducer[T tensor.Numeric] struct {
	engine compute.Engine[T]

	pleEmbed     *tensor.TensorNumeric[T] // [vocab, numLayers*pleDim]
	pleModelProj *tensor.TensorNumeric[T] // [hidden, numLayers*pleDim] (already transposed for MatMul)

	numLayers  int
	pleDim     int
	hiddenSize int
	vocabSize  int

	// Per-layer pre-sliced tensors with stable GPU addresses (see
	// ADR-088). Allocated on the first forward pass via a no-op
	// MulScalar to obtain GPU buffers; refreshed in place on subsequent
	// passes via GPUStorage.CopyFromHost. Length == numLayers once
	// initialized.
	tokenPLESlices  []*tensor.TensorNumeric[T]
	modelProjSlices []*tensor.TensorNumeric[T]

	// cachedBatch / cachedSeqLen record the shape of the currently
	// allocated slice buffers. If the next forward pass has a different
	// shape (e.g. prefill vs decode), the buffers are reallocated.
	cachedBatch  int
	cachedSeqLen int
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
// consumers via p.tokenPLESlices and p.modelProjSlices.
//
// Side effect: allocates (first call) or refreshes (subsequent calls)
// 2*numLayers GPU-resident tensors with stable device addresses across
// calls. This stability is required for CUDA graph replay — see
// ADR-088.
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
	modelProj, err := p.engine.MatMul(ctx, scaled, p.pleModelProj)
	if err != nil {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: model_proj matmul: %w", err)
	}

	// --- 3. Pre-slice into per-layer GPU-resident tensors with stable
	// device addresses (see ADR-088). If the batch/seqLen have changed
	// since the last call, drop the cached buffers so fresh ones are
	// allocated; otherwise, refresh the existing GPU buffers in place
	// via CopyFromHost so the CUDA-graph replayed reads hit the same
	// GPU addresses on every decode step.
	if p.cachedBatch != batch || p.cachedSeqLen != seqLen {
		p.tokenPLESlices = nil
		p.modelProjSlices = nil
		p.cachedBatch = batch
		p.cachedSeqLen = seqLen
	}
	if err := p.refreshPerLayerSlices(ctx, tokenPLE, modelProj, batch, seqLen); err != nil {
		return nil, fmt.Errorf("Gemma4PLECombinedProducer: refresh slices: %w", err)
	}

	// Pass-through the hidden input so downstream nodes that declare a
	// dependency on the producer see a well-formed tensor.
	return hidden, nil
}

// refreshPerLayerSlices fills p.tokenPLESlices and p.modelProjSlices so
// each entry is a GPU-resident [B, S, pleDim] tensor. On the first
// invocation (or after a shape change) it allocates stable GPU buffers
// via a no-op MulScalar. On every subsequent invocation it reuses the
// same GPU buffers and refreshes their contents with CopyFromHost, so
// the CUDA graph replay path sees stable device addresses.
func (p *pleCombinedProducer[T]) refreshPerLayerSlices(
	ctx context.Context,
	tokenPLE, modelProj *tensor.TensorNumeric[T],
	batch, seqLen int,
) error {
	sliceElems := batch * seqLen * p.pleDim
	firstCall := p.tokenPLESlices == nil
	if firstCall {
		p.tokenPLESlices = make([]*tensor.TensorNumeric[T], p.numLayers)
		p.modelProjSlices = make([]*tensor.TensorNumeric[T], p.numLayers)
	}

	// Both tokenPLE and modelProj are rank-3 [B, S, totalPLE]. We need
	// their dense data to slice on CPU. The engine's MulScalar on a
	// GPU tensor returns a GPU tensor, but reading via .Data() forces
	// a D2H copy. Since this node runs in pre-capture, D2H is safe
	// (no capturing stream). Pull .Data() once per tensor and reuse
	// the slices.
	tokenData := tokenPLE.Data()
	if tokenData == nil {
		return fmt.Errorf("tokenPLE has no readable data")
	}
	projData := modelProj.Data()
	if projData == nil {
		return fmt.Errorf("modelProj has no readable data")
	}

	totalPLE := p.numLayers * p.pleDim

	// Per-layer buffers: stride = totalPLE, offset = layer*pleDim per row.
	flatLen := batch * seqLen
	tokenScratch := make([]T, sliceElems)
	projScratch := make([]T, sliceElems)

	for layer := 0; layer < p.numLayers; layer++ {
		offset := layer * p.pleDim
		// Gather the per-layer strided slice into a contiguous CPU buffer.
		for i := 0; i < flatLen; i++ {
			src := tokenData[i*totalPLE+offset : i*totalPLE+offset+p.pleDim]
			copy(tokenScratch[i*p.pleDim:(i+1)*p.pleDim], src)
			src = projData[i*totalPLE+offset : i*totalPLE+offset+p.pleDim]
			copy(projScratch[i*p.pleDim:(i+1)*p.pleDim], src)
		}

		if firstCall {
			// Allocate a stable GPU buffer by wrapping the CPU data as a
			// tensor and passing it through a no-op MulScalar. The GPU
			// engine's getDevicePtr uploads the contents and returns a
			// GPU-resident tensor whose device address is stable for the
			// lifetime of the tensor.
			tokenCPU, err := tensor.New[T]([]int{batch, seqLen, p.pleDim}, append([]T(nil), tokenScratch...))
			if err != nil {
				return fmt.Errorf("token slice layer=%d alloc: %w", layer, err)
			}
			tokenGPU, err := p.engine.MulScalar(ctx, tokenCPU, T(1))
			if err != nil {
				return fmt.Errorf("token slice layer=%d upload: %w", layer, err)
			}
			p.tokenPLESlices[layer] = tokenGPU

			projCPU, err := tensor.New[T]([]int{batch, seqLen, p.pleDim}, append([]T(nil), projScratch...))
			if err != nil {
				return fmt.Errorf("proj slice layer=%d alloc: %w", layer, err)
			}
			projGPU, err := p.engine.MulScalar(ctx, projCPU, T(1))
			if err != nil {
				return fmt.Errorf("proj slice layer=%d upload: %w", layer, err)
			}
			p.modelProjSlices[layer] = projGPU
			continue
		}

		// Subsequent calls: refresh the stable GPU buffers in place.
		// If the storage is CPU-backed (CPU engine), update the CPU
		// data directly; no device copy is needed.
		if err := refreshSliceStorage(p.tokenPLESlices[layer], tokenScratch); err != nil {
			return fmt.Errorf("token slice layer=%d refresh: %w", layer, err)
		}
		if err := refreshSliceStorage(p.modelProjSlices[layer], projScratch); err != nil {
			return fmt.Errorf("proj slice layer=%d refresh: %w", layer, err)
		}
	}
	return nil
}

// refreshSliceStorage updates the contents of an existing tensor in
// place. For GPU-resident tensors it issues a synchronous H2D copy via
// CopyFromHost, which is safe here because the producer runs in the
// pre-capture region (no active stream capture). For CPU-resident
// tensors it overwrites the underlying Data() slice directly.
func refreshSliceStorage[T tensor.Numeric](dst *tensor.TensorNumeric[T], src []T) error {
	if dst == nil {
		return fmt.Errorf("destination tensor is nil")
	}
	if gs, ok := dst.GetStorage().(*tensor.GPUStorage[T]); ok {
		return gs.CopyFromHost(src, 0)
	}
	dstData := dst.Data()
	if dstData == nil {
		return fmt.Errorf("destination tensor has no readable data")
	}
	if len(dstData) < len(src) {
		return fmt.Errorf("destination tensor too small: have %d, need %d", len(dstData), len(src))
	}
	copy(dstData, src)
	return nil
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
	normLayer *normalization.RMSNorm[T] // shared per-layer projection norm
	layerIdx  int
	pleDim    int
	numLayers int
}

func newPLESliceNode[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
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
	gainParam, err := graph.NewParameter[T](fmt.Sprintf("ple_proj_norm.gain.layer_%d", layerIdx), normGain, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("pleSliceNode: wrap normGain: %w", err)
	}
	normLayer, err := normalization.NewRMSNormFromParam[T](engine, ops, ops.FromFloat64(float64(eps)), gainParam)
	if err != nil {
		return nil, fmt.Errorf("pleSliceNode: build RMSNorm: %w", err)
	}
	return &pleSliceNode[T]{
		engine:    engine,
		producer:  producer,
		normLayer: normLayer,
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
	params := n.normLayer.Parameters()
	frozen := make([]*tensor.TensorNumeric[T], 0, len(params))
	for _, p := range params {
		frozen = append(frozen, p.Value)
	}
	return frozen
}

// Forward reads the producer's pre-computed per-layer GPU slices, applies
// RMSNorm to the projection slice (dim=pleDim), adds the token-identity
// slice, scales by 1/sqrt(2), and returns [B, S, pleDim].
//
// The slices are allocated and refreshed by pleCombinedProducer so that
// their GPU device addresses are stable across decode steps. Reading
// them here via Go struct fields produces only GPU-native operations
// (RMSNorm, Add, MulScalar), making this node fully CUDA-graph
// capturable. See ADR-088.
func (n *pleSliceNode[T]) Forward(ctx context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if n.producer.tokenPLESlices == nil || n.producer.modelProjSlices == nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): producer has not run Forward yet", n.layerIdx)
	}
	if n.layerIdx >= len(n.producer.tokenPLESlices) || n.layerIdx >= len(n.producer.modelProjSlices) {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): producer slice cache too short (len=%d)", n.layerIdx, len(n.producer.tokenPLESlices))
	}
	tokenSlice := n.producer.tokenPLESlices[n.layerIdx]
	projSlice := n.producer.modelProjSlices[n.layerIdx]
	if tokenSlice == nil || projSlice == nil {
		return nil, fmt.Errorf("pleSliceNode(layer=%d): producer slice cache has nil entry", n.layerIdx)
	}
	// Delegate RMSNorm to layers/normalization (architectural guard:
	// private layer reimpls are disallowed outside layers/).
	projNormed, err := n.normLayer.Forward(ctx, projSlice)
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
