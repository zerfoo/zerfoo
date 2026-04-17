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
// CUDA graph capture strategy (see ADR-088, T99.2.1 follow-up). The
// node is listed as non-capturable in ztensor's graph compiler
// (Gemma4PLECombinedProducer), so it runs in pre-capture on every
// forward. After computing the two full-width tensors it pre-slices
// them into numLayers per-layer slices and caches the slices as
// GPU-resident tensors with stable addresses across calls. pleSliceNode
// reads the cached slices directly — no .Data() calls — which makes the
// per-layer PLE combiner fully capturable.
//
// Refresh strategy (T99.2.1 fast path). For the decode regime
// (flatLen == batch*seqLen == 1, i.e. seqLen == 1 which covers every
// generate step) the per-layer slices are non-owning views into two
// stable full-width GPU buffers. Each Forward copies the freshly
// computed full-width tensors into the stable buffers via a single
// D2D memcpy each (2 memcpys total), instead of 2*numLayers H2D
// memcpys of tiny 256-float segments. This eliminates the 70 per-step
// cudaMemcpy launches that caused the regression bisected to commit
// 96c7540a.
//
// For prefill (flatLen > 1), the last-dim slice is strided, so
// contiguous GPU views are invalid. In that regime the producer falls
// back to a slower per-layer-scatter CPU gather + H2D upload; prefill
// runs once per sequence and is not in the decode hot path, so the
// cost is acceptable.
type pleCombinedProducer[T tensor.Numeric] struct {
	engine compute.Engine[T]

	pleEmbed     *tensor.TensorNumeric[T] // [vocab, numLayers*pleDim]
	pleModelProj *tensor.TensorNumeric[T] // [hidden, numLayers*pleDim] (already transposed for MatMul)

	numLayers  int
	pleDim     int
	hiddenSize int
	vocabSize  int

	// Per-layer pre-sliced tensors with stable addresses (see
	// ADR-088, T99.2.1). Length == numLayers once initialized.
	tokenPLESlices  []*tensor.TensorNumeric[T]
	modelProjSlices []*tensor.TensorNumeric[T]

	// cachedBatch / cachedSeqLen record the shape of the currently
	// allocated slice buffers. If the next forward pass has a different
	// shape (e.g. prefill vs decode), the buffers are reallocated.
	cachedBatch  int
	cachedSeqLen int

	// Stable full-width GPU buffers backing the per-layer views in the
	// decode fast path (flatLen == 1). Nil when the producer is in the
	// CPU path or when the fallback prefill path owns the slices. When
	// non-nil these buffers live for the lifetime of the current shape
	// regime and the per-layer slice tensors are SubSlice views into
	// them, so refreshing them with a single D2D memcpy per forward is
	// enough to propagate new data to every layer.
	tokenPLEBuf  *tensor.GPUStorage[T]
	modelProjBuf *tensor.GPUStorage[T]
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

	// --- 3. Pre-slice into per-layer tensors with stable device
	// addresses (see ADR-088). If the batch/seqLen have changed since
	// the last call, drop the cached buffers so fresh ones are
	// allocated; otherwise, refresh the existing buffers in place so
	// that the CUDA-graph replayed reads hit the same device addresses
	// on every decode step.
	if p.cachedBatch != batch || p.cachedSeqLen != seqLen {
		p.tokenPLESlices = nil
		p.modelProjSlices = nil
		p.tokenPLEBuf = nil
		p.modelProjBuf = nil
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
// each entry is a [B, S, pleDim] tensor with a stable device address
// (for GPU engines) or stable backing slice (for CPU engines).
//
// T99.2.1: the fast path applies when flatLen == batch*seqLen == 1
// (every decode step, with seqLen == 1). In that regime the last-dim
// slice of the full-width rank-3 tensor is contiguous, so each
// per-layer slice can be a non-owning view into a single stable
// full-width buffer. Refreshing the 35 slices therefore costs one
// memcpy (D2D on GPU, slice copy on CPU), not 35 small memcpys.
//
// For flatLen > 1 (prefill) the last-dim slice is strided and can't
// be aliased as a contiguous view; we fall back to a per-layer gather
// + per-slice refresh. Prefill is invoked once per sequence and runs
// outside the CUDA-graph replay window, so the extra overhead is not
// in the decode hot path.
func (p *pleCombinedProducer[T]) refreshPerLayerSlices(
	ctx context.Context,
	tokenPLE, modelProj *tensor.TensorNumeric[T],
	batch, seqLen int,
) error {
	firstCall := p.tokenPLESlices == nil
	if firstCall {
		p.tokenPLESlices = make([]*tensor.TensorNumeric[T], p.numLayers)
		p.modelProjSlices = make([]*tensor.TensorNumeric[T], p.numLayers)
	}

	flatLen := batch * seqLen
	if flatLen == 1 {
		return p.refreshPerLayerSlicesFast(ctx, tokenPLE, modelProj, batch, seqLen, firstCall)
	}
	return p.refreshPerLayerSlicesFallback(ctx, tokenPLE, modelProj, batch, seqLen, firstCall)
}

// refreshPerLayerSlicesFast is the decode-regime implementation
// (flatLen == 1). The per-layer slices are non-owning views into two
// stable full-width buffers. On the first call the stable buffers are
// allocated and the views are constructed; on every subsequent call a
// single memcpy per full-width tensor refreshes every layer's data at
// once. See ADR-088 follow-up and T99.2.1 for the rationale.
func (p *pleCombinedProducer[T]) refreshPerLayerSlicesFast(
	ctx context.Context,
	tokenPLE, modelProj *tensor.TensorNumeric[T],
	batch, seqLen int,
	firstCall bool,
) error {
	totalPLE := p.numLayers * p.pleDim

	// Branch on the storage type of the freshly computed full-width
	// tensors. GPU engine: compute stays on-device, D2D refresh. CPU
	// engine: copy the backing []T slice into the cached full-width
	// buffer and re-wrap per-layer views.
	if tokenGPU, ok := tokenPLE.GetStorage().(*tensor.GPUStorage[T]); ok {
		projGPU, ok := modelProj.GetStorage().(*tensor.GPUStorage[T])
		if !ok {
			return fmt.Errorf("Gemma4PLECombinedProducer: modelProj storage mismatches tokenPLE (GPU vs %T)", modelProj.GetStorage())
		}
		if firstCall {
			tb, err := tensor.NewGPUStorage[T](totalPLE, tokenGPU.DeviceID())
			if err != nil {
				return fmt.Errorf("alloc tokenPLE stable buffer: %w", err)
			}
			mb, err := tensor.NewGPUStorage[T](totalPLE, projGPU.DeviceID())
			if err != nil {
				return fmt.Errorf("alloc modelProj stable buffer: %w", err)
			}
			p.tokenPLEBuf = tb
			p.modelProjBuf = mb
			for layer := 0; layer < p.numLayers; layer++ {
				offset := layer * p.pleDim
				tokenView := tensor.NewGPUStorageView[T](p.tokenPLEBuf, offset, p.pleDim)
				tokenTensor, err := tensor.NewWithStorage[T]([]int{batch, seqLen, p.pleDim}, tokenView)
				if err != nil {
					return fmt.Errorf("wrap tokenPLE view layer=%d: %w", layer, err)
				}
				p.tokenPLESlices[layer] = tokenTensor

				projView := tensor.NewGPUStorageView[T](p.modelProjBuf, offset, p.pleDim)
				projTensor, err := tensor.NewWithStorage[T]([]int{batch, seqLen, p.pleDim}, projView)
				if err != nil {
					return fmt.Errorf("wrap modelProj view layer=%d: %w", layer, err)
				}
				p.modelProjSlices[layer] = projTensor
			}
		}
		// Refresh the stable buffers with a single D2D memcpy each.
		if err := p.tokenPLEBuf.CopyFromDevice(tokenGPU, 0, 0, totalPLE); err != nil {
			return fmt.Errorf("tokenPLE D2D refresh: %w", err)
		}
		if err := p.modelProjBuf.CopyFromDevice(projGPU, 0, 0, totalPLE); err != nil {
			return fmt.Errorf("modelProj D2D refresh: %w", err)
		}
		_ = ctx
		return nil
	}

	// CPU path: allocate a single stable []T of length totalPLE per
	// tensor and wrap per-layer views as non-owning sub-slices. The
	// contents are refreshed per call by copying the freshly computed
	// full-width data in. This keeps the invariant that per-layer
	// tensor pointers stay stable across calls with the same shape
	// (exercised by TestPLECombinedProducer_SliceBuffersStable).
	tokenData := tokenPLE.Data()
	if tokenData == nil {
		return fmt.Errorf("tokenPLE has no readable data")
	}
	projData := modelProj.Data()
	if projData == nil {
		return fmt.Errorf("modelProj has no readable data")
	}
	if len(tokenData) < totalPLE || len(projData) < totalPLE {
		return fmt.Errorf("Gemma4PLECombinedProducer: full-width data shorter than expected (tok=%d proj=%d need>=%d)",
			len(tokenData), len(projData), totalPLE)
	}
	if firstCall {
		// Allocate stable backing arrays and per-layer sub-slice views.
		tokenStable := make([]T, totalPLE)
		projStable := make([]T, totalPLE)
		copy(tokenStable, tokenData[:totalPLE])
		copy(projStable, projData[:totalPLE])
		for layer := 0; layer < p.numLayers; layer++ {
			offset := layer * p.pleDim
			tokenView := tokenStable[offset : offset+p.pleDim]
			tokenTensor, err := tensor.NewWithStorage[T]([]int{batch, seqLen, p.pleDim}, tensor.NewCPUStorage[T](tokenView))
			if err != nil {
				return fmt.Errorf("wrap tokenPLE cpu view layer=%d: %w", layer, err)
			}
			p.tokenPLESlices[layer] = tokenTensor

			projView := projStable[offset : offset+p.pleDim]
			projTensor, err := tensor.NewWithStorage[T]([]int{batch, seqLen, p.pleDim}, tensor.NewCPUStorage[T](projView))
			if err != nil {
				return fmt.Errorf("wrap modelProj cpu view layer=%d: %w", layer, err)
			}
			p.modelProjSlices[layer] = projTensor
		}
		// Stash the stable backing arrays on the first layer's slice so
		// future refreshes can reach them via the existing views.
		// The per-layer Data() for layer 0 returns the first pleDim
		// elements of tokenStable; we refresh the full array via the
		// shared underlying slice below.
		return nil
	}
	// Refresh by overwriting the stable backing arrays in place. Since
	// each per-layer view shares the underlying array, one bulk copy
	// updates every layer.
	if err := refreshCPUStableBuffer(p.tokenPLESlices[0], tokenData[:totalPLE], p.numLayers, p.pleDim); err != nil {
		return fmt.Errorf("tokenPLE cpu refresh: %w", err)
	}
	if err := refreshCPUStableBuffer(p.modelProjSlices[0], projData[:totalPLE], p.numLayers, p.pleDim); err != nil {
		return fmt.Errorf("modelProj cpu refresh: %w", err)
	}
	_ = ctx
	return nil
}

// refreshCPUStableBuffer overwrites the stable underlying []T of the
// first-layer view with `src`. Since every per-layer view references
// the same underlying slice (with a per-layer offset), one bulk write
// propagates the update to every layer. `numLayers*pleDim == len(src)`.
func refreshCPUStableBuffer[T tensor.Numeric](layer0 *tensor.TensorNumeric[T], src []T, numLayers, pleDim int) error {
	if layer0 == nil {
		return fmt.Errorf("refreshCPUStableBuffer: layer0 tensor nil")
	}
	cs, ok := layer0.GetStorage().(*tensor.CPUStorage[T])
	if !ok {
		return fmt.Errorf("refreshCPUStableBuffer: expected CPU storage, got %T", layer0.GetStorage())
	}
	// The underlying slice for layer0 is tokenStable[0:pleDim]. Its
	// capacity extends to the full backing array because sub-slicing
	// preserves capacity up to the end of the parent slice, so we can
	// address the entire backing array via its [:numLayers*pleDim]
	// extension.
	layer0View := cs.Slice()
	total := numLayers * pleDim
	if cap(layer0View) < total {
		return fmt.Errorf("refreshCPUStableBuffer: backing capacity %d < %d", cap(layer0View), total)
	}
	if len(src) != total {
		return fmt.Errorf("refreshCPUStableBuffer: src len %d != expected %d", len(src), total)
	}
	full := layer0View[:total]
	copy(full, src)
	return nil
}

// refreshPerLayerSlicesFallback is the prefill / multi-row regime
// (flatLen > 1). The last-dim slices are strided so a single view into
// a full-width buffer cannot represent them. This is the pre-T99.2.1
// path: scatter-gather on the CPU side, then either upload into fresh
// per-layer GPU buffers (first call) or refresh existing GPU buffers
// in place (subsequent calls). Called at most once per sequence (at
// prefill), so the 2*numLayers memcpy launches are acceptable.
func (p *pleCombinedProducer[T]) refreshPerLayerSlicesFallback(
	ctx context.Context,
	tokenPLE, modelProj *tensor.TensorNumeric[T],
	batch, seqLen int,
	firstCall bool,
) error {
	sliceElems := batch * seqLen * p.pleDim
	tokenData := tokenPLE.Data()
	if tokenData == nil {
		return fmt.Errorf("tokenPLE has no readable data")
	}
	projData := modelProj.Data()
	if projData == nil {
		return fmt.Errorf("modelProj has no readable data")
	}

	totalPLE := p.numLayers * p.pleDim
	flatLen := batch * seqLen
	tokenScratch := make([]T, sliceElems)
	projScratch := make([]T, sliceElems)

	for layer := 0; layer < p.numLayers; layer++ {
		offset := layer * p.pleDim
		for i := 0; i < flatLen; i++ {
			src := tokenData[i*totalPLE+offset : i*totalPLE+offset+p.pleDim]
			copy(tokenScratch[i*p.pleDim:(i+1)*p.pleDim], src)
			src = projData[i*totalPLE+offset : i*totalPLE+offset+p.pleDim]
			copy(projScratch[i*p.pleDim:(i+1)*p.pleDim], src)
		}

		if firstCall {
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
//
// CUDA graph capture note (ADR-088 follow-up): the scalar value is resolved
// at construction time (while the tensor is still CPU-resident from GGUF
// load) and stored in `scalarValue`. Forward never calls `scale.Data()`,
// because once the graph compiler uploads frozen weights to GPU, `Data()`
// on the GPU-backed scale tensor triggers a synchronous D2H cudaMemcpy
// that CUDA rejects inside a capturing stream.
type layerOutputScaleNode[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	scale       *tensor.TensorNumeric[T] // shape [1]; retained for EmbeddedFrozen registration
	scalarValue T
}

func newLayerOutputScaleNode[T tensor.Numeric](engine compute.Engine[T], scale *tensor.TensorNumeric[T]) (*layerOutputScaleNode[T], error) {
	if scale == nil {
		return nil, fmt.Errorf("layerOutputScaleNode: scale must be non-nil")
	}
	shape := scale.Shape()
	if len(shape) != 1 || shape[0] != 1 {
		return nil, fmt.Errorf("layerOutputScaleNode: scale shape %v, want [1]", shape)
	}
	sData := scale.Data()
	if len(sData) == 0 {
		return nil, fmt.Errorf("layerOutputScaleNode: scale data empty (construct with CPU-backed tensor before GPU upload)")
	}
	return &layerOutputScaleNode[T]{engine: engine, scale: scale, scalarValue: sData[0]}, nil
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
	return n.engine.MulScalar(ctx, inputs[0], n.scalarValue)
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
