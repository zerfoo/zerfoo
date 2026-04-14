// Package attention provides attention mechanisms for neural networks.
package attention

import (
	"context"
	"fmt"
	"log/slog"
	"unsafe"

	"github.com/zerfoo/zerfoo/generate"
	cudago "github.com/zerfoo/zerfoo/internal/cuda"
	"github.com/zerfoo/zerfoo/internal/cuda/kernels"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings" // For RoPE
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// GroupedQueryAttention implements grouped query attention mechanism.
type GroupedQueryAttention[T tensor.Numeric] struct {
	engine           compute.Engine[T]
	ops              numeric.Arithmetic[T]
	modelDim         int
	numQueryHeads    int
	numKeyValueHeads int
	headDim          int

	wq   *core.Dense[T]                           // Query projection
	wk   *core.Dense[T]                           // Key projection
	wv   *core.Dense[T]                           // Value projection
	wo   *core.Dense[T]                           // Output projection
	rope *embeddings.RotaryPositionalEmbedding[T] // Rotary positional embedding

	scaledDotProductAttention *ScaledDotProductAttention[T]

	// LayerIndex identifies this layer within a model for KV cache indexing.
	LayerIndex int

	// SlidingWindowSize, if > 0, restricts attention to the last N positions
	// using a causal sliding window mask during prefill (seqLen > 1).
	SlidingWindowSize int

	// blockTableReader optionally reads KV from paged block tables directly.
	blockTableReader BlockTableReader[T]

	// Optional per-head Q/K norms (Gemma 3).
	qNorm graph.Node[T]
	kNorm graph.Node[T]

	// Fused QK norm+RoPE: raw weights and epsilon for decode fast path.
	qNormWeight *tensor.TensorNumeric[T]
	kNormWeight *tensor.TensorNumeric[T]
	qkNormEps   float32

	// Merged QKV weight for single-GEMV decode optimization.
	mergedQKV *tensor.TensorNumeric[T]
	qDim      int // number of Q output elements (numQueryHeads * headDim)
	kDim      int // number of K output elements (numKVHeads * headDim)
	vDim      int // number of V output elements (numKVHeads * headDim)

	// qkNormPreReshape, when true, applies Q/K norms before the head reshape
	// (i.e., to Q of shape [batch, seq, nH*hD]) instead of after.
	// MiniMax-M2 stores attn_q_norm.weight as [nH*hD] rather than [hD].
	qkNormPreReshape bool

	// bidirectional disables causal masking so every position attends to
	// every other position (encoder-style attention).
	bidirectional bool

	// kEqV, when true, uses the K projection output as V (shared K=V).
	// Gemma 4 global attention layers use this optimization.
	kEqV bool

	// externalKV, when true, expects K and V to be supplied as forward
	// inputs (inputs[1] and inputs[2]) instead of being computed from the
	// layer's own wk/wv weights. In this mode the layer carries no wk, wv,
	// or k_norm parameters. Used by architectures with cross-layer K/V
	// sharing (e.g. Gemma 4 edge). Mutually exclusive with kEqV.
	externalKV bool

	// Cached tensors for backward pass
	qProj           *tensor.TensorNumeric[T] // Projected Q
	kProj           *tensor.TensorNumeric[T] // Projected K
	vProj           *tensor.TensorNumeric[T] // Projected V
	attnOutput      *tensor.TensorNumeric[T] // Output from scaledDotProductAttention (heads format)
	attnOutputFinal *tensor.TensorNumeric[T] // Final reshaped output passed to wo
	qHeadsRoPE      *tensor.TensorNumeric[T] // Q after RoPE
	kHeadsRoPE      *tensor.TensorNumeric[T] // K after RoPE
	outputShape     []int

	// K/V output ports (task-T95.1.2). Captured during Forward after the
	// layer-local K/V projection + optional per-head norms + post-projection
	// RoPE have been applied, and before KV-cache expansion or group
	// replication. Shapes: [batch, numKVHeads, seqLen, headDim].
	//
	// Downstream shared-KV layers consume these via KPort()/VPort() when
	// wired through the external-K/V input path (see ADR-087 and
	// task-T95.1.1 / T95.2.1). Nil until the first successful Forward.
	kOut *tensor.TensorNumeric[T]
	vOut *tensor.TensorNumeric[T]

}

// OpType returns the operation type.
func (gqa *GroupedQueryAttention[T]) OpType() string {
	return "GroupedQueryAttention"
}

// Attributes returns the attributes.
func (gqa *GroupedQueryAttention[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"model_dim":           gqa.modelDim,
		"num_query_heads":     gqa.numQueryHeads,
		"num_key_value_heads": gqa.numKeyValueHeads,
		"head_dim":            gqa.headDim,
	}
}

// GQAOptions holds configuration options for the GroupedQueryAttention layer.
type GQAOptions[T tensor.Numeric] struct {
	Base          float64
	MaxSeqLen     int
	Bidirectional bool // when true, disables causal masking for encoder-style models
	NoRoPE        bool // when true, skip RoPE creation (for models like GPT-2 that use learned position embeddings)
	ExternalKV    bool // when true, K/V are supplied as Forward inputs; wk/wv/k_norm are not instantiated
}

// GQAOption is a function that applies an option to GQAOptions.
type GQAOption[T tensor.Numeric] func(*GQAOptions[T])

// WithRopeBase sets the base for Rotary Positional Embeddings.
func WithRopeBase[T tensor.Numeric](base float64) GQAOption[T] {
	return func(o *GQAOptions[T]) {
		o.Base = base
	}
}

// WithMaxSeqLen sets the maximum sequence length for Rotary Positional Embeddings.
func WithMaxSeqLen[T tensor.Numeric](maxSeqLen int) GQAOption[T] {
	return func(o *GQAOptions[T]) {
		o.MaxSeqLen = maxSeqLen
	}
}

// WithBidirectionalGQA returns an option that disables causal masking in the
// grouped query attention layer, allowing every position to attend to every
// other position. This is required for encoder-style models such as BERT.
func WithBidirectionalGQA[T tensor.Numeric]() GQAOption[T] {
	return func(o *GQAOptions[T]) {
		o.Bidirectional = true
	}
}

// WithExternalKV returns an option that configures the GroupedQueryAttention
// layer to read K and V from additional Forward inputs instead of computing
// them from internal wk/wv projections. When set, the constructor does not
// instantiate wk, wv, or k_norm; Forward expects inputs[1] to be K and
// inputs[2] to be V, each with shape [batch, seq, numKeyValueHeads * headDim]
// (matching what the internal projection would produce). RoPE for K still
// applies (if configured). Q, q_norm, w_out, attention math, and output
// projection remain unchanged. This option is mutually exclusive with the
// kEqV mode configured via SetKEqV. See ADR-087 for rationale.
func WithExternalKV[T tensor.Numeric]() GQAOption[T] {
	return func(o *GQAOptions[T]) {
		o.ExternalKV = true
	}
}

// WithNoRoPE returns an option that disables rotary positional embeddings.
// Models like GPT-2 use learned position embeddings instead of RoPE, so the
// GQA layer should pass Q and K through without rotational encoding.
func WithNoRoPE[T tensor.Numeric]() GQAOption[T] {
	return func(o *GQAOptions[T]) {
		o.NoRoPE = true
	}
}

// NewGroupedQueryAttention creates a new GroupedQueryAttention layer.
// modelDim: The dimension of the input and output of the block (d_model).
// numQueryHeads: The number of query heads.
// numKeyValueHeads: The number of key/value heads.
func NewGroupedQueryAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads int,
	opts ...GQAOption[T],
) (*GroupedQueryAttention[T], error) {
	// Default options
	options := &GQAOptions[T]{
		Base:      10000.0,
		MaxSeqLen: 2048,
	}
	for _, opt := range opts {
		opt(options)
	}

	if numQueryHeads%numKeyValueHeads != 0 {
		return nil, fmt.Errorf("number of query heads (%d) must be divisible by number of key/value heads (%d)", numQueryHeads, numKeyValueHeads)
	}

	if modelDim%numQueryHeads != 0 {
		return nil, fmt.Errorf("model dimension (%d) must be divisible by number of query heads (%d)", modelDim, numQueryHeads)
	}

	if modelDim%numKeyValueHeads != 0 {
		return nil, fmt.Errorf("model dimension (%d) must be divisible by number of key/value heads (%d)", modelDim, numKeyValueHeads)
	}

	headDim := modelDim / numQueryHeads

	// Initialize Dense layers for Q, K, V projections
	wq, err := core.NewDense[T]("wq", engine, ops, modelDim, modelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create WQ dense layer: %w", err)
	}

	var wk, wv *core.Dense[T]
	if !options.ExternalKV {
		wk, err = core.NewDense[T]("wk", engine, ops, modelDim, headDim*numKeyValueHeads) // K projection
		if err != nil {
			return nil, fmt.Errorf("failed to create WK dense layer: %w", err)
		}

		wv, err = core.NewDense[T]("wv", engine, ops, modelDim, headDim*numKeyValueHeads) // V projection
		if err != nil {
			return nil, fmt.Errorf("failed to create WV dense layer: %w", err)
		}
	}

	// Initialize ScaledDotProductAttention. dk is headDim.
	scaledDotProductAttention := NewScaledDotProductAttention[T](engine, headDim)

	// Initialize output Dense layer.
	wo, err := core.NewDense[T]("wo", engine, ops, modelDim, modelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create WO dense layer: %w", err)
	}

	var rope *embeddings.RotaryPositionalEmbedding[T]
	if !options.NoRoPE {
		rope, err = embeddings.NewRotaryPositionalEmbedding[T](context.Background(), engine, headDim, options.MaxSeqLen, embeddings.WithRotaryBase(options.Base))
		if err != nil {
			return nil, fmt.Errorf("failed to create RotaryPositionalEmbedding: %w", err)
		}
	}

	return &GroupedQueryAttention[T]{
		engine:                    engine,
		ops:                       ops,
		numQueryHeads:             numQueryHeads,
		numKeyValueHeads:          numKeyValueHeads,
		modelDim:                  modelDim,
		headDim:                   headDim,
		wq:                        wq,
		wk:                        wk,
		wv:                        wv,
		scaledDotProductAttention: scaledDotProductAttention,
		wo:                        wo,
		rope:                      rope,
		bidirectional:             options.Bidirectional,
		externalKV:                options.ExternalKV,
	}, nil
}

// NewGroupedQueryAttentionFromParams creates a new GroupedQueryAttention layer from existing parameters.
// headDimOverride, if > 0, sets the per-head dimension explicitly instead of
// deriving it from modelDim/numQueryHeads. This is required for architectures
// like Gemma 3 where key_length differs from hidden_size/num_heads.
func NewGroupedQueryAttentionFromParams[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads int,
	wq, wk, wv, wo *core.Dense[T],
	rope *embeddings.RotaryPositionalEmbedding[T],
	headDimOverride ...int,
) (*GroupedQueryAttention[T], error) {
	headDim := modelDim / numQueryHeads
	if len(headDimOverride) > 0 && headDimOverride[0] > 0 {
		headDim = headDimOverride[0]
	}
	scaledDotProductAttention := NewScaledDotProductAttention[T](engine, headDim)

	return &GroupedQueryAttention[T]{
		engine:                    engine,
		ops:                       ops,
		numQueryHeads:             numQueryHeads,
		numKeyValueHeads:          numKeyValueHeads,
		modelDim:                  modelDim,
		headDim:                   headDim,
		wq:                        wq,
		wk:                        wk,
		wv:                        wv,
		scaledDotProductAttention: scaledDotProductAttention,
		wo:                        wo,
		rope:                      rope,
	}, nil
}

// SetDocumentBoundaries sets document boundary positions for document-wise
// RoPE. When boundaries are set, position IDs reset to 0 at each boundary
// so each document receives independent positional encoding during
// multi-document inference. Boundaries are sequence positions (0-indexed)
// where new documents begin. Pass nil to disable document-wise mode.
func (gqa *GroupedQueryAttention[T]) SetDocumentBoundaries(boundaries []int) {
	if gqa.rope != nil {
		gqa.rope.SetDocumentBoundaries(boundaries)
	}
}

// SetBidirectional enables or disables bidirectional (non-causal) attention.
func (gqa *GroupedQueryAttention[T]) SetBidirectional(bidirectional bool) {
	gqa.bidirectional = bidirectional
}

// SetKEqV configures GQA to use the K projection output for both K and V.
// When enabled, the V projection (wv) is skipped and K output is used as V.
// This implements Gemma 4's unified K=V projection for global attention layers.
// Mutually exclusive with WithExternalKV: panics if both are requested.
func (gqa *GroupedQueryAttention[T]) SetKEqV(v bool) {
	if v && gqa.externalKV {
		panic("GroupedQueryAttention: SetKEqV is mutually exclusive with WithExternalKV")
	}
	gqa.kEqV = v
}

// SetExternalKV configures the layer to read K/V from Forward inputs[1]/[2]
// instead of projecting from its own wk/wv weights. When enabled, wk, wv, and
// kNorm are cleared (the shared-KV layer does not use them per ADR-087 and
// HuggingFace transformers modeling_gemma4.py line 1167 "Layers sharing kv
// states don't need any weight matrices"). Mutually exclusive with kEqV.
//
// This setter is the FromParams-path counterpart to the WithExternalKV
// constructor option: builders that load weights from GGUF construct GQA via
// NewGroupedQueryAttentionFromParams with wk=nil, wv=nil (or disposable wk/wv),
// then invoke SetExternalKV(true) to flip the forward path. Idempotent;
// passing false re-enables internal projection iff wk/wv are still populated.
func (gqa *GroupedQueryAttention[T]) SetExternalKV(v bool) {
	if v && gqa.kEqV {
		panic("GroupedQueryAttention: SetExternalKV is mutually exclusive with SetKEqV")
	}
	gqa.externalKV = v
	if v {
		gqa.wk = nil
		gqa.wv = nil
		gqa.kNorm = nil
	}
}

// ExternalKV reports whether this layer was configured with WithExternalKV.
func (gqa *GroupedQueryAttention[T]) ExternalKV() bool {
	return gqa.externalKV
}

// OutputShape returns the output shape of the GroupedQueryAttention.
func (gqa *GroupedQueryAttention[T]) OutputShape() []int {
	return gqa.outputShape
}

// ScaleRope scales the rotary positional embeddings.
func (gqa *GroupedQueryAttention[T]) ScaleRope(ctx context.Context, factor float64) error {
	if gqa.rope == nil {
		return nil
	}
	return gqa.rope.Scale(ctx, factor)
}

// SetQKNorms sets optional per-head RMSNorm layers for Q and K projections.
// Used by architectures like Gemma 3 that normalize Q/K after projection.
// When externalKV is set, the kNorm argument is ignored (K is supplied
// pre-computed and any normalization must have been applied by the donor).
func (gqa *GroupedQueryAttention[T]) SetQKNorms(qNorm, kNorm graph.Node[T]) {
	gqa.qNorm = qNorm
	if gqa.externalKV {
		// In external-KV mode the layer carries no k_norm: the donor layer
		// is responsible for any K normalization before publishing.
		gqa.kNorm = nil
		return
	}
	gqa.kNorm = kNorm
}

// SetQKNormPreReshape configures whether QK norms are applied before the
// head reshape. Set true for architectures (e.g. MiniMax-M2) whose norm
// weights are [nH*hD] rather than [hD].
func (gqa *GroupedQueryAttention[T]) SetQKNormPreReshape(v bool) {
	gqa.qkNormPreReshape = v
}

// SetQKNormWeights stores raw RMSNorm weights for the fused QK norm+RoPE
// decode path. When set alongside SetQKNorms, the fused kernel replaces
// 4 kernel launches (Q norm, K norm, Q RoPE, K RoPE) with 1 during decode.
func (gqa *GroupedQueryAttention[T]) SetQKNormWeights(qWeight, kWeight *tensor.TensorNumeric[T], eps float32) {
	gqa.qNormWeight = qWeight
	gqa.kNormWeight = kWeight
	gqa.qkNormEps = eps
}

// SetBlockTableReader sets an optional BlockTableReader that provides KV data
// directly from paged block tables, bypassing the standard cache gather path.
func (gqa *GroupedQueryAttention[T]) SetBlockTableReader(r BlockTableReader[T]) {
	gqa.blockTableReader = r
}

// SetMergedQKV sets a merged Q/K/V weight tensor for single-GEMV decode optimization.
// During decode (seqLen=1), a single MatMul with this weight replaces three separate
// Q/K/V projections, reducing kernel launch overhead. The output is split into Q, K, V
// using zero-copy GPU storage views.
func (gqa *GroupedQueryAttention[T]) SetMergedQKV(weight *tensor.TensorNumeric[T], qDim, kDim, vDim int) {
	gqa.mergedQKV = weight
	gqa.qDim = qDim
	gqa.kDim = kDim
	gqa.vDim = vDim
}

// MergedQKVParameter returns the merged QKV parameter for GPU upload, or nil if not set.
func (gqa *GroupedQueryAttention[T]) MergedQKVParameter() *graph.Parameter[T] {
	if gqa.mergedQKV == nil {
		return nil
	}
	return &graph.Parameter[T]{Name: "merged_qkv", Value: gqa.mergedQKV}
}

// Parameters returns the parameters of the GroupedQueryAttention layer.
func (gqa *GroupedQueryAttention[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]

	params = append(params, gqa.wq.Parameters()...)
	if gqa.wk != nil {
		params = append(params, gqa.wk.Parameters()...)
	}
	if gqa.wv != nil {
		params = append(params, gqa.wv.Parameters()...)
	}
	params = append(params, gqa.wo.Parameters()...)

	if p := gqa.MergedQKVParameter(); p != nil {
		params = append(params, p)
	}

	return params
}

// Forward computes the grouped query attention.
func (gqa *GroupedQueryAttention[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("GroupedQueryAttention: expected at least 1 input tensor, got %d", len(inputs))
	}

	input := inputs[0] // (batch_size, seq_len, model_dim)
	gqa.outputShape = input.Shape()

	var (
		externalK *tensor.TensorNumeric[T]
		externalV *tensor.TensorNumeric[T]
		mask      *tensor.TensorNumeric[T]
	)
	if gqa.externalKV {
		if len(inputs) < 3 {
			return nil, fmt.Errorf("GroupedQueryAttention: externalKV mode expects at least 3 inputs (hidden, K, V), got %d", len(inputs))
		}
		externalK = inputs[1]
		externalV = inputs[2]
		if len(inputs) > 3 {
			mask = inputs[3]
		}
	} else if len(inputs) > 1 {
		mask = inputs[1]
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	// Check for KV cache in context.
	cache, hasCache := generate.GetCache[T](ctx)

	// 1. Linear projections for Q, K, V
	var err error
	var qProj, kProj, vProj *tensor.TensorNumeric[T]
	switch {
	case gqa.externalKV:
		// External K/V path: Q projected normally, K/V taken from inputs.
		qProj, err = gqa.wq.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
		expectedKVDim := gqa.numKeyValueHeads * gqa.headDim
		for name, t := range map[string]*tensor.TensorNumeric[T]{"K": externalK, "V": externalV} {
			shape := t.Shape()
			if len(shape) != 3 || shape[0] != batchSize || shape[1] != seqLen || shape[2] != expectedKVDim {
				return nil, fmt.Errorf("GroupedQueryAttention: external %s has shape %v, expected [%d, %d, %d]",
					name, shape, batchSize, seqLen, expectedKVDim)
			}
		}
		kProj = externalK
		vProj = externalV
	case gqa.mergedQKV != nil && seqLen == 1:
		// Merged QKV: single GEMV + zero-copy split for decode.
		merged, mergeErr := gqa.engine.MatMul(ctx, input, gqa.mergedQKV)
		if mergeErr != nil {
			return nil, fmt.Errorf("merged QKV MatMul: %w", mergeErr)
		}
		qProj, kProj, vProj, err = splitMergedQKV[T](merged, gqa.qDim, gqa.kDim, gqa.vDim)
		if err != nil {
			return nil, fmt.Errorf("split merged QKV: %w", err)
		}
	default:
		qProj, err = gqa.wq.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
		kProj, err = gqa.wk.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
		if gqa.kEqV {
			vProj = kProj
		} else {
			vProj, err = gqa.wv.Forward(ctx, input)
			if err != nil {
				return nil, err
			}
		}
	}

	// Cache projected Q, K, V for backward pass
	gqa.qProj = qProj
	gqa.kProj = kProj
	gqa.vProj = vProj

	// 2. Split into heads, apply optional Q/K norms, then RoPE
	var qHeadsRoPE, kHeadsRoPE *tensor.TensorNumeric[T]
	var vHeads *tensor.TensorNumeric[T]

	// V always takes the same path: reshape + transpose.
	vReshaped, err := gqa.engine.Reshape(ctx, vProj, []int{batchSize, seqLen, gqa.numKeyValueHeads, gqa.headDim})
	if err != nil {
		return nil, err
	}
	vHeads, err = gqa.engine.Transpose(ctx, vReshaped, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// T95.1.2: capture V output port (post reshape+transpose, pre cache
	// expansion / group replication). Shape: [batch, numKVHeads, seqLen, headDim].
	gqa.vOut = vHeads

	// Fused QK norm+RoPE decode path: replaces 4 kernel launches with 1.
	// Conditions: decode (seqLen=1), RoPE enabled, Q/K norm weights available, engine supports it.
	fusedQKNormRoPE := false
	if seqLen == 1 && gqa.rope != nil && gqa.qNormWeight != nil && gqa.kNormWeight != nil {
		realEngine := compute.Engine[T](gqa.engine)
		if proxy, ok := gqa.engine.(*compute.EngineProxy[T]); ok {
			realEngine = proxy.Real()
		}
		if provider, ok := realEngine.(compute.FusedQKNormRoPEProvider[T]); ok {
			totalHeads := gqa.numQueryHeads + gqa.numKeyValueHeads
			qkElems := totalHeads * gqa.headDim

			// Try zero-copy: if Q and K are adjacent GPU views from merged QKV,
			// create a single [totalHeads, headDim] view without any kernel launch.
			var qkCombined *tensor.TensorNumeric[T]
			if qGS, ok := qProj.GetStorage().(*tensor.GPUStorage[T]); ok {
				qkView := qGS.SubSlice(0, qkElems)
				qkCombined, _ = tensor.NewWithStorage[T]([]int{totalHeads, gqa.headDim}, qkView)
			}
			if qkCombined == nil {
				// Fallback: reshape and concat.
				qFlat, reshapeErr := gqa.engine.Reshape(ctx, qProj, []int{gqa.numQueryHeads, gqa.headDim})
				if reshapeErr != nil {
					return nil, reshapeErr
				}
				kFlat, reshapeErr := gqa.engine.Reshape(ctx, kProj, []int{gqa.numKeyValueHeads, gqa.headDim})
				if reshapeErr != nil {
					return nil, reshapeErr
				}
				var catErr error
				qkCombined, catErr = gqa.engine.Concat(ctx, []*tensor.TensorNumeric[T]{qFlat, kFlat}, 0)
				if catErr != nil {
					return nil, fmt.Errorf("fused QK norm+RoPE concat: %w", catErr)
				}
			}

			// Get cos/sin angles for the current position.
			// GPU path: use GPU-resident counter to avoid CPU readback.
			var cosAngles, sinAngles *tensor.TensorNumeric[T]
			var halfRotary int
			var angleErr error

			type gpuCounterProvider interface {
				GPUCounterPtr() unsafe.Pointer
			}
			if gcp, ok := cache.(gpuCounterProvider); ok && gcp.GPUCounterPtr() != nil {
				// Get stream from compute engine.
				realEng := compute.Engine[T](gqa.engine)
				if proxy, ok := gqa.engine.(*compute.EngineProxy[T]); ok {
					realEng = proxy.Real()
				}
				if sp, ok := realEng.(compute.StreamProvider); ok {
					cosAngles, sinAngles, halfRotary, angleErr = gqa.rope.GetAnglesGPU(gcp.GPUCounterPtr(), 1, sp.Stream())
				}
			}
			// CPU fallback: no GPU counter or stream unavailable.
			if cosAngles == nil && angleErr == nil {
				posOffset := 0
				if hasCache {
					posOffset = cache.SeqLen()
				}
				cosAngles, sinAngles, halfRotary, angleErr = gqa.rope.GetAngles(posOffset, 1)
			}
			if angleErr != nil {
				return nil, fmt.Errorf("fused QK norm+RoPE angles: %w", angleErr)
			}

			// For the kernel, we need 1D [halfRotary] cos/sin (single position).
			cos1D, reshapeErr := gqa.engine.Reshape(ctx, cosAngles, []int{halfRotary})
			if reshapeErr != nil {
				return nil, reshapeErr
			}
			sin1D, reshapeErr := gqa.engine.Reshape(ctx, sinAngles, []int{halfRotary})
			if reshapeErr != nil {
				return nil, reshapeErr
			}

			// Run fused kernel: RMSNorm + RoPE for all heads in one launch.
			fusedOut, fusedErr := provider.GPUFusedQKNormRoPE(
				qkCombined, gqa.qNormWeight, gqa.kNormWeight,
				cos1D, sin1D,
				gqa.qkNormEps, totalHeads, gqa.headDim, gqa.numQueryHeads, halfRotary,
			)
			if fusedErr == nil {
				fusedQKNormRoPE = true

				// Split output [totalHeads, headDim] into Q' and K' via zero-copy views.
				qElems := gqa.numQueryHeads * gqa.headDim
				kElems := gqa.numKeyValueHeads * gqa.headDim
				if gs, ok := fusedOut.GetStorage().(*tensor.GPUStorage[T]); ok {
					qView := gs.SubSlice(0, qElems)
					kView := gs.SubSlice(qElems, kElems)

					qSlice, viewErr := tensor.NewWithStorage[T]([]int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim}, qView)
					if viewErr != nil {
						return nil, fmt.Errorf("fused QK split Q view: %w", viewErr)
					}
					kSlice, viewErr := tensor.NewWithStorage[T]([]int{batchSize, gqa.numKeyValueHeads, seqLen, gqa.headDim}, kView)
					if viewErr != nil {
						return nil, fmt.Errorf("fused QK split K view: %w", viewErr)
					}
					qHeadsRoPE = qSlice
					kHeadsRoPE = kSlice
				} else if fs, ok := any(fusedOut.GetStorage()).(*tensor.Float16Storage); ok {
					qView := any(fs.SubSlice(0, qElems)).(tensor.Storage[T])
					kView := any(fs.SubSlice(qElems, kElems)).(tensor.Storage[T])

					qSlice, viewErr := tensor.NewWithStorage[T]([]int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim}, qView)
					if viewErr != nil {
						return nil, fmt.Errorf("fused QK split Q fp16 view: %w", viewErr)
					}
					kSlice, viewErr := tensor.NewWithStorage[T]([]int{batchSize, gqa.numKeyValueHeads, seqLen, gqa.headDim}, kView)
					if viewErr != nil {
						return nil, fmt.Errorf("fused QK split K fp16 view: %w", viewErr)
					}
					qHeadsRoPE = qSlice
					kHeadsRoPE = kSlice
				} else {
					return nil, fmt.Errorf("fused QK norm+RoPE: expected GPUStorage or Float16Storage but got %T; this causes a D2H copy that blocks CUDA graph capture", fusedOut.GetStorage())
				}
			}
			// Fall through to unfused path on error.
		}
	}

	if !fusedQKNormRoPE {
		// Unfused path: separate Q/K norm + transpose + RoPE.

		// Pre-reshape Q norm: architectures whose norm weight is [nH*hD]
		// (e.g. MiniMax-M2) normalize before per-head splitting.
		if gqa.qkNormPreReshape && gqa.qNorm != nil {
			qProj, err = gqa.qNorm.Forward(ctx, qProj)
			if err != nil {
				return nil, fmt.Errorf("qNorm (pre-reshape): %w", err)
			}
		}
		if gqa.qkNormPreReshape && gqa.kNorm != nil {
			kProj, err = gqa.kNorm.Forward(ctx, kProj)
			if err != nil {
				return nil, fmt.Errorf("kNorm (pre-reshape): %w", err)
			}
		}

		// Q: (batch, seq_len, num_query_heads, head_dim)
		qReshaped, reshapeErr := gqa.engine.Reshape(ctx, qProj, []int{batchSize, seqLen, gqa.numQueryHeads, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		// Apply per-head Q norm if set (Gemma 3) and not already applied above.
		if !gqa.qkNormPreReshape && gqa.qNorm != nil {
			qReshaped, err = gqa.qNorm.Forward(ctx, qReshaped)
			if err != nil {
				return nil, fmt.Errorf("qNorm: %w", err)
			}
		}

		qHeads, reshapeErr := gqa.engine.Transpose(ctx, qReshaped, []int{0, 2, 1, 3})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		// K: (batch, seq_len, num_kv_heads, head_dim)
		kReshaped, reshapeErr := gqa.engine.Reshape(ctx, kProj, []int{batchSize, seqLen, gqa.numKeyValueHeads, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		// Apply per-head K norm if set (Gemma 3) and not already applied above.
		if !gqa.qkNormPreReshape && gqa.kNorm != nil {
			kReshaped, err = gqa.kNorm.Forward(ctx, kReshaped)
			if err != nil {
				return nil, fmt.Errorf("kNorm: %w", err)
			}
		}

		kHeads, reshapeErr := gqa.engine.Transpose(ctx, kReshaped, []int{0, 2, 1, 3})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		if gqa.rope != nil {
			// Apply RoPE to Q and K
			qForRoPE, reshapeErr := gqa.engine.Reshape(ctx, qHeads, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
			if reshapeErr != nil {
				return nil, reshapeErr
			}

			kForRoPE, reshapeErr := gqa.engine.Reshape(ctx, kHeads, []int{batchSize * gqa.numKeyValueHeads, seqLen, gqa.headDim})
			if reshapeErr != nil {
				return nil, reshapeErr
			}

			// Set RoPE position offset from cached sequence length so that decode
			// tokens get the correct absolute position rotation.
			// GPU path: use GPU-resident counter to avoid CPU readback that would
			// break CUDA graph capture.
			gpuRoPEApplied := false
			if hasCache {
				type unfusedGPUCounterProvider interface {
					GPUCounterPtr() unsafe.Pointer
				}
				if gcp, ok := cache.(unfusedGPUCounterProvider); ok && gcp.GPUCounterPtr() != nil {
					realEng := compute.Engine[T](gqa.engine)
					if proxy, ok := gqa.engine.(*compute.EngineProxy[T]); ok {
						realEng = proxy.Real()
					}
					if sp, ok := realEng.(compute.StreamProvider); ok {
						cosAngles, sinAngles, _, angleErr := gqa.rope.GetAnglesGPU(gcp.GPUCounterPtr(), seqLen, sp.Stream())
						if angleErr == nil {
							if provider, ok := realEng.(compute.FusedRoPEProvider[T]); ok {
								qOut, qErr := provider.GPUFusedRoPE(qForRoPE, cosAngles, sinAngles, gqa.headDim)
								kOut, kErr := provider.GPUFusedRoPE(kForRoPE, cosAngles, sinAngles, gqa.headDim)
								if qErr == nil && kErr == nil {
									qHeadsRoPE = qOut
									kHeadsRoPE = kOut
									gpuRoPEApplied = true
								}
							}
						}
					}
				}
				if !gpuRoPEApplied {
					gqa.rope.SetPositionOffset(cache.SeqLen())
				}
			} else {
				gqa.rope.SetPositionOffset(0)
			}

			if !gpuRoPEApplied {
				qHeadsRoPE, err = gqa.rope.Forward(ctx, qForRoPE)
				if err != nil {
					return nil, err
				}

				kHeadsRoPE, err = gqa.rope.Forward(ctx, kForRoPE)
				if err != nil {
					return nil, err
				}
			}
		} else {
			// No RoPE: use Q and K heads as-is, reshaped for attention.
			qHeadsRoPE, err = gqa.engine.Reshape(ctx, qHeads, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
			if err != nil {
				return nil, err
			}
			kHeadsRoPE, err = gqa.engine.Reshape(ctx, kHeads, []int{batchSize * gqa.numKeyValueHeads, seqLen, gqa.headDim})
			if err != nil {
				return nil, err
			}
		}
	}

	gqa.qHeadsRoPE = qHeadsRoPE // Cache for backward
	gqa.kHeadsRoPE = kHeadsRoPE // Cache for backward

	// Reshape back to (batch, num_heads, seq_len, head_dim)
	qHeadsRoPE, err = gqa.engine.Reshape(ctx, qHeadsRoPE, []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	kHeadsRoPE, err = gqa.engine.Reshape(ctx, kHeadsRoPE, []int{batchSize, gqa.numKeyValueHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	// T95.1.2: capture K output port (post norm + post RoPE, pre cache
	// expansion / group replication). Shape: [batch, numKVHeads, seqLen, headDim].
	gqa.kOut = kHeadsRoPE

	// KV Cache: store K/V per KV-head in shape [batch*numKVHeads, seq_len, headDim],
	// then retrieve full cached K/V for attention computation.
	// kvSeqLen tracks the K/V sequence length (may differ from Q seqLen when cached).
	kvSeqLen := seqLen
	var attnOutputHeads *tensor.TensorNumeric[T]
	flashDecodeUsed := false

	if hasCache {
		// Flatten K/V from [batch, numKVHeads, seqLen, headDim] to [batch*numKVHeads, seqLen, headDim]
		// for storage and concat in the cache.
		kFlat, reshapeErr := gqa.engine.Reshape(ctx, kHeadsRoPE, []int{batchSize * gqa.numKeyValueHeads, seqLen, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}
		vFlat, reshapeErr := gqa.engine.Reshape(ctx, vHeads, []int{batchSize * gqa.numKeyValueHeads, seqLen, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		if err := cache.Update(gqa.LayerIndex, kFlat, vFlat); err != nil {
			return nil, fmt.Errorf("kv cache update: %w", err)
		}

		// CUDA graph-capturable path: use FlashAttentionDecode with fixed-size
		// (maxSeqLen) KV buffers and a GPU-resident KV length counter. The kernel
		// reads kvLen from GPU memory at runtime, so tensor shapes are fixed
		// across graph replays (no D2H transfer, no baked-in dimensions).
		//
		// Requirements: (1) decode (seqLen==1), (2) cache implements
		// FullBufferProvider with GPU KV counter, (3) CUDA kernels available,
		// (4) Q is GPU-resident.
		// FlashAttentionDecode is disabled: SDPA with arena-based softmax is
		// capture-safe (arena uses bump allocation, no cudaMalloc) and 15% faster
		// than the custom flash decode kernel on Gemma 3 1B (170 vs 148 tok/s).
		// The ztensor prefill-skip fix ensures capture only happens during decode
		// where GQA has the cache context and all tensors are GPU-resident.
		if false && seqLen == 1 && cudago.Available() {
			if fbp, ok := cache.(generate.FullBufferProvider[T]); ok && fbp.KVSeqLenPtr() != nil {
				fullK, fullV := fbp.GetFullBuffer(gqa.LayerIndex)
				if fullK != nil && fullV != nil {
					if qGS, ok := qHeadsRoPE.GetStorage().(*tensor.GPUStorage[T]); ok {
						kGS, kOK := fullK.GetStorage().(*tensor.GPUStorage[T])
						vGS, vOK := fullV.GetStorage().(*tensor.GPUStorage[T])
						if kOK && vOK {
							maxKVLen := fbp.MaxSeqLen()
							kvLen := cache.SeqLen() // CPU fallback value; kernel uses GPU counter
							numBH := batchSize * gqa.numQueryHeads

							// Allocate output: [batch*numQueryHeads, 1, headDim]
							oElems := numBH * gqa.headDim
							oGPU, allocErr := tensor.NewGPUStorage[T](oElems, qGS.DeviceID())
							if allocErr == nil {
								// Get stream for kernel launch.
								realEng := compute.Engine[T](gqa.engine)
								if proxy, ok := gqa.engine.(*compute.EngineProxy[T]); ok {
									realEng = proxy.Real()
								}
								var streamPtr unsafe.Pointer
								if sp, ok := realEng.(compute.StreamProvider); ok {
									streamPtr = sp.Stream()
								}

								flashErr := kernels.FlashAttentionDecode(
									qGS.Ptr(), kGS.Ptr(), vGS.Ptr(), oGPU.Ptr(),
									numBH, maxKVLen, gqa.headDim, kvLen,
									fbp.KVSeqLenPtr(),
									gqa.numQueryHeads, gqa.numKeyValueHeads,
									streamPtr,
								)
								if flashErr == nil {
									oShape := []int{batchSize * gqa.numQueryHeads, 1, gqa.headDim}
									attnOutputHeads, err = tensor.NewWithStorage[T](oShape, oGPU)
									if err == nil {
										flashDecodeUsed = true
									}
								} else {
								_ = oGPU.Free()
							}
							}
						}
					}
				}
			}
		}

		if !flashDecodeUsed {
			// Fallback: variable-size KV view + cuBLAS SDPA.
			// This path is used during prefill (seqLen > 1), CPU inference,
			// or when the cache doesn't support full-buffer access.
			var cachedK, cachedV *tensor.TensorNumeric[T]
			if gqa.blockTableReader != nil {
				cachedK, cachedV, _ = gqa.blockTableReader.ReadKV(gqa.LayerIndex)
			}

			if cachedK == nil || cachedV == nil {
				lkv, ok := cache.Get(gqa.LayerIndex)
				if !ok {
					return nil, fmt.Errorf("kv cache: layer %d missing after update", gqa.LayerIndex)
				}
				cachedK = lkv.Key
				cachedV = lkv.Value
			}

			cachedSeqLen := cachedK.Shape()[1]
			kHeadsRoPE, err = gqa.engine.Reshape(ctx, cachedK, []int{batchSize, gqa.numKeyValueHeads, cachedSeqLen, gqa.headDim})
			if err != nil {
				return nil, err
			}
			vHeads, err = gqa.engine.Reshape(ctx, cachedV, []int{batchSize, gqa.numKeyValueHeads, cachedSeqLen, gqa.headDim})
			if err != nil {
				return nil, err
			}
			kvSeqLen = cachedSeqLen
		}
	}

	// Run SDPA (prefill, decode without flash, or no-cache).
	// Skip when FlashAttentionDecode already computed the output above.
	if !flashDecodeUsed {
		// 3. Grouped Query Attention: expand K/V to match Q head count.
		// Each KV head is repeated `replicationFactor` times consecutively
		// so it pairs with its correct group of query heads:
		// [kv0, kv0, kv0, kv1, kv1, kv1, ...] (not [kv0..7, kv0..7, kv0..7]).
		//
		// We use reshape+Repeat+reshape instead of a direct Repeat on the
		// head axis. This ensures repeat-each semantics regardless of whether
		// the engine's Repeat uses tile or interleave ordering: we insert a
		// size-1 dimension per head, Repeat that (tiling 1 element = correct),
		// then merge the repeated dimension back into the head axis.
		if gqa.numQueryHeads != gqa.numKeyValueHeads && gqa.numKeyValueHeads > 1 {
			replicationFactor := gqa.numQueryHeads / gqa.numKeyValueHeads

			// Try fused RepeatInterleave: single kernel replaces
			// Reshape -> Repeat -> Reshape for each of K and V.
			type repeatInterleaver[U tensor.Numeric] interface {
				RepeatInterleave(ctx context.Context, a *tensor.TensorNumeric[U], axis int, reps int, dst ...*tensor.TensorNumeric[U]) (*tensor.TensorNumeric[U], error)
			}
			fusedOK := false
			realEng := compute.Engine[T](gqa.engine)
			if proxy, ok := gqa.engine.(*compute.EngineProxy[T]); ok {
				realEng = proxy.Real()
			}
			if ri, ok := realEng.(repeatInterleaver[T]); ok {
				kExp, kErr := ri.RepeatInterleave(ctx, kHeadsRoPE, 1, replicationFactor)
				if kErr == nil {
					vExp, vErr := ri.RepeatInterleave(ctx, vHeads, 1, replicationFactor)
					if vErr == nil {
						kHeadsRoPE = kExp
						vHeads = vExp
						fusedOK = true
					}
				}
			}

			if !fusedOK {
				// Fallback: Reshape -> Repeat -> Reshape for K and V.
				kSeqLen := kHeadsRoPE.Shape()[2]
				kr, err := gqa.engine.Reshape(ctx, kHeadsRoPE, []int{batchSize, gqa.numKeyValueHeads, 1, kSeqLen, gqa.headDim})
				if err != nil {
					return nil, err
				}
				kr, err = gqa.engine.Repeat(ctx, kr, 2, replicationFactor)
				if err != nil {
					return nil, err
				}
				kHeadsRoPE, err = gqa.engine.Reshape(ctx, kr, []int{batchSize, gqa.numQueryHeads, kSeqLen, gqa.headDim})
				if err != nil {
					return nil, err
				}

				vSeqLen := vHeads.Shape()[2]
				vr, err := gqa.engine.Reshape(ctx, vHeads, []int{batchSize, gqa.numKeyValueHeads, 1, vSeqLen, gqa.headDim})
				if err != nil {
					return nil, err
				}
				vr, err = gqa.engine.Repeat(ctx, vr, 2, replicationFactor)
				if err != nil {
					return nil, err
				}
				vHeads, err = gqa.engine.Reshape(ctx, vr, []int{batchSize, gqa.numQueryHeads, vSeqLen, gqa.headDim})
				if err != nil {
					return nil, err
				}
			}
		}

		// 4. Apply Scaled Dot-Product Attention
		qForSDPA, reshapeErr := gqa.engine.Reshape(ctx, qHeadsRoPE, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		kvBatchHeads := gqa.numQueryHeads
		if gqa.numKeyValueHeads == 1 {
			kvBatchHeads = gqa.numKeyValueHeads
		}

		kForSDPA, reshapeErr := gqa.engine.Reshape(ctx, kHeadsRoPE, []int{batchSize * kvBatchHeads, kvSeqLen, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		vForSDPA, reshapeErr := gqa.engine.Reshape(ctx, vHeads, []int{batchSize * kvBatchHeads, kvSeqLen, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		switch {
		case gqa.bidirectional:
			// Encoder-style: no causal masking, all positions attend to all others.
			gqa.scaledDotProductAttention.SetCausal(false)
		case mask == nil && seqLen > 1 && gqa.SlidingWindowSize > 0:
			// Build causal sliding window mask for prefill.
			mask = BuildCausalSlidingWindowMask[T](seqLen, gqa.SlidingWindowSize)
			gqa.scaledDotProductAttention.SetCausal(false)
		case mask == nil && seqLen > 1:
			gqa.scaledDotProductAttention.SetCausal(true)
		default:
			gqa.scaledDotProductAttention.SetCausal(false)
		}

		var sdpaErr error
		attnOutputHeads, sdpaErr = gqa.scaledDotProductAttention.Forward(ctx, qForSDPA, kForSDPA, vForSDPA, mask)
		if sdpaErr != nil {
			return nil, sdpaErr
		}
	}

	gqa.attnOutput = attnOutputHeads // Cache for backward

	// 5. Concatenate heads and reshape back
	attnOutputReshaped, err := gqa.engine.Reshape(ctx, attnOutputHeads, []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	attnOutputCombined, err := gqa.engine.Transpose(ctx, attnOutputReshaped, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	qkvDim := gqa.numQueryHeads * gqa.headDim
	attnOutputFinal, err := gqa.engine.Reshape(ctx, attnOutputCombined, []int{batchSize, seqLen, qkvDim})
	if err != nil {
		return nil, err
	}
	gqa.attnOutputFinal = attnOutputFinal // Cache for wo backward

	// 6. Final linear projection
	output, err := gqa.wo.Forward(ctx, attnOutputFinal)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// reverseHeadReplication sums gradients from replicated heads back to the
// original KV head count.  The forward uses engine.Repeat(axis=1, factor)
// which tiles the KV heads: [kv0, kv1, kv0, kv1, ...].  To reverse this,
// we reshape [batch*numQ, seq, headDim] →
// [batch, repFactor, numKV, seq, headDim] and sum along axis 1 (repFactor),
// then flatten back to [batch*numKV, seq, headDim].
func (gqa *GroupedQueryAttention[T]) reverseHeadReplication(ctx context.Context, d *tensor.TensorNumeric[T], batchSize, seqLen int) (*tensor.TensorNumeric[T], error) {
	repFactor := gqa.numQueryHeads / gqa.numKeyValueHeads
	d4, err := gqa.engine.Reshape(ctx, d, []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	d5, err := gqa.engine.Reshape(ctx, d4, []int{batchSize, repFactor, gqa.numKeyValueHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	dSum, err := gqa.engine.ReduceSum(ctx, d5, 1, false)
	if err != nil {
		return nil, err
	}
	return gqa.engine.Reshape(ctx, dSum, []int{batchSize * gqa.numKeyValueHeads, seqLen, gqa.headDim})
}

// Backward computes the gradients for GroupedQueryAttention.
//
// The backward mirrors the forward in reverse order:
//
//  1. wo backward
//  2. Reverse reshape/transpose (head concatenation)
//  3. SDPA backward
//  4. Reverse K/V head replication (sum over group copies)
//  5. RoPE backward
//  6. Reverse head split (reshape/transpose back to projection shape)
//  7. wq/wk/wv backward
func (gqa *GroupedQueryAttention[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if gqa.externalKV {
		return nil, fmt.Errorf("GroupedQueryAttention: Backward is not supported in externalKV mode")
	}
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GroupedQueryAttention: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}

	input := inputs[0] // Original input to GQA
	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	kvProjDim := gqa.numKeyValueHeads * gqa.headDim

	// 1. wo backward
	dAttnOutputFinal, err := gqa.wo.Backward(ctx, mode, dOut, gqa.attnOutputFinal)
	if err != nil {
		return nil, fmt.Errorf("wo backward: %w", err)
	}

	// 2. Reverse reshape + transpose (step 5 of Forward)
	// [batch, seq, modelDim] → [batch, seq, numQ, headDim] → [batch, numQ, seq, headDim] → [batch*numQ, seq, headDim]
	dCombined, err := gqa.engine.Reshape(ctx, dAttnOutputFinal[0], []int{batchSize, seqLen, gqa.numQueryHeads, gqa.headDim})
	if err != nil {
		return nil, fmt.Errorf("reverse concat reshape: %w", err)
	}
	dTransposed, err := gqa.engine.Transpose(ctx, dCombined, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("reverse concat transpose: %w", err)
	}
	dHeads, err := gqa.engine.Reshape(ctx, dTransposed, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, fmt.Errorf("reverse concat flatten: %w", err)
	}

	// 3. SDPA backward → dQ, dK, dV all [batch*numQ, seq, headDim]
	dQKV, err := gqa.scaledDotProductAttention.Backward(ctx, mode, dHeads,
		gqa.scaledDotProductAttention.q, gqa.scaledDotProductAttention.k, gqa.scaledDotProductAttention.v)
	if err != nil {
		return nil, fmt.Errorf("sdpa backward: %w", err)
	}
	dQ, dK, dV := dQKV[0], dQKV[1], dQKV[2]

	// 4. Reverse K/V head replication (must happen BEFORE RoPE backward for K)
	if gqa.numQueryHeads != gqa.numKeyValueHeads {
		dK, err = gqa.reverseHeadReplication(ctx, dK, batchSize, seqLen)
		if err != nil {
			return nil, fmt.Errorf("reverse K replication: %w", err)
		}
		dV, err = gqa.reverseHeadReplication(ctx, dV, batchSize, seqLen)
		if err != nil {
			return nil, fmt.Errorf("reverse V replication: %w", err)
		}
	}
	// dQ: [batch*numQ, seq, headDim], dK: [batch*numKV, seq, headDim], dV: [batch*numKV, seq, headDim]

	// 5. RoPE backward (Q and K only; V does not go through RoPE)
	var dQAfterRoPE, dKAfterRoPE []*tensor.TensorNumeric[T]
	if gqa.rope != nil {
		dQAfterRoPE, err = gqa.rope.Backward(ctx, mode, dQ)
		if err != nil {
			return nil, fmt.Errorf("rope backward Q: %w", err)
		}
		dKAfterRoPE, err = gqa.rope.Backward(ctx, mode, dK)
		if err != nil {
			return nil, fmt.Errorf("rope backward K: %w", err)
		}
	} else {
		// No RoPE: gradients pass through unchanged.
		dQAfterRoPE = []*tensor.TensorNumeric[T]{dQ}
		dKAfterRoPE = []*tensor.TensorNumeric[T]{dK}
	}

	// 6. Reverse head split: reshape/transpose back to projection shapes
	// dQ: [batch*numQ, seq, headDim] → [batch, numQ, seq, headDim] → [batch, seq, numQ, headDim] → [batch, seq, modelDim]
	dQProj, err := gqa.engine.Reshape(ctx, dQAfterRoPE[0], []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, fmt.Errorf("dQ reshape: %w", err)
	}
	dQProj, err = gqa.engine.Transpose(ctx, dQProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("dQ transpose: %w", err)
	}
	qkvDim := gqa.numQueryHeads * gqa.headDim
	dQProj, err = gqa.engine.Reshape(ctx, dQProj, []int{batchSize, seqLen, qkvDim})
	if err != nil {
		return nil, fmt.Errorf("dQ flatten: %w", err)
	}

	// dK: [batch*numKV, seq, headDim] → [batch, numKV, seq, headDim] → [batch, seq, numKV, headDim] → [batch, seq, kvProjDim]
	dKProj, err := gqa.engine.Reshape(ctx, dKAfterRoPE[0], []int{batchSize, gqa.numKeyValueHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, fmt.Errorf("dK reshape: %w", err)
	}
	dKProj, err = gqa.engine.Transpose(ctx, dKProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("dK transpose: %w", err)
	}
	dKProj, err = gqa.engine.Reshape(ctx, dKProj, []int{batchSize, seqLen, kvProjDim})
	if err != nil {
		return nil, fmt.Errorf("dK flatten: %w", err)
	}

	// dV: [batch*numKV, seq, headDim] → [batch, numKV, seq, headDim] → [batch, seq, numKV, headDim] → [batch, seq, kvProjDim]
	dVProj, err := gqa.engine.Reshape(ctx, dV, []int{batchSize, gqa.numKeyValueHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, fmt.Errorf("dV reshape: %w", err)
	}
	dVProj, err = gqa.engine.Transpose(ctx, dVProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, fmt.Errorf("dV transpose: %w", err)
	}
	dVProj, err = gqa.engine.Reshape(ctx, dVProj, []int{batchSize, seqLen, kvProjDim})
	if err != nil {
		return nil, fmt.Errorf("dV flatten: %w", err)
	}

	// 7. wq/wk/wv backward
	dInputQ, err := gqa.wq.Backward(ctx, mode, dQProj, input)
	if err != nil {
		return nil, fmt.Errorf("wq backward: %w", err)
	}
	dInputK, err := gqa.wk.Backward(ctx, mode, dKProj, input)
	if err != nil {
		return nil, fmt.Errorf("wk backward: %w", err)
	}
	dInputV, err := gqa.wv.Backward(ctx, mode, dVProj, input)
	if err != nil {
		return nil, fmt.Errorf("wv backward: %w", err)
	}

	// Sum gradients from Q, K, V paths
	dInputTotal, err := gqa.engine.Add(ctx, dInputQ[0], dInputK[0])
	if err != nil {
		return nil, fmt.Errorf("sum Q+K gradients: %w", err)
	}
	dInputTotal, err = gqa.engine.Add(ctx, dInputTotal, dInputV[0])
	if err != nil {
		return nil, fmt.Errorf("sum (Q+K)+V gradients: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dInputTotal}, nil
}

// splitMergedQKV splits a merged QKV output tensor into separate Q, K, V tensors.
// For GPU-resident tensors, this creates zero-copy views (no data movement).
// For CPU tensors, it copies the data.
func splitMergedQKV[T tensor.Numeric](merged *tensor.TensorNumeric[T], qDim, kDim, vDim int) (q, k, v *tensor.TensorNumeric[T], err error) {
	shape := merged.Shape()
	if len(shape) < 2 {
		return nil, nil, nil, fmt.Errorf("expected at least 2D tensor, got %dD", len(shape))
	}
	lastDim := shape[len(shape)-1]
	if lastDim != qDim+kDim+vDim {
		return nil, nil, nil, fmt.Errorf("last dim %d != qDim(%d)+kDim(%d)+vDim(%d)", lastDim, qDim, kDim, vDim)
	}

	// For seqLen=1 decode, the total elements are just qDim+kDim+vDim.
	// Build output shapes: same prefix dims, different last dim.
	prefix := make([]int, len(shape)-1)
	copy(prefix, shape[:len(shape)-1])
	batchElems := 1
	for _, d := range prefix {
		batchElems *= d
	}

	qShape := append(append([]int{}, prefix...), qDim)
	kShape := append(append([]int{}, prefix...), kDim)
	vShape := append(append([]int{}, prefix...), vDim)

	// GPU path: zero-copy views via GPU-side pointer arithmetic (no D2H copy).
	if gs, ok := merged.GetStorage().(*tensor.GPUStorage[T]); ok {
		qView := gs.SubSlice(0, batchElems*qDim)
		kView := gs.SubSlice(batchElems*qDim, batchElems*kDim)
		vView := gs.SubSlice(batchElems*(qDim+kDim), batchElems*vDim)

		q, err = tensor.NewWithStorage[T](qShape, qView)
		if err != nil {
			return nil, nil, nil, err
		}
		k, err = tensor.NewWithStorage[T](kShape, kView)
		if err != nil {
			return nil, nil, nil, err
		}
		v, err = tensor.NewWithStorage[T](vShape, vView)
		if err != nil {
			return nil, nil, nil, err
		}
		return q, k, v, nil
	}

	// Float16Storage GPU path: zero-copy views via FP16 SubSlice.
	if fs, ok := any(merged.GetStorage()).(*tensor.Float16Storage); ok {
		qView := any(fs.SubSlice(0, batchElems*qDim)).(tensor.Storage[T])
		kView := any(fs.SubSlice(batchElems*qDim, batchElems*kDim)).(tensor.Storage[T])
		vView := any(fs.SubSlice(batchElems*(qDim+kDim), batchElems*vDim)).(tensor.Storage[T])

		q, err = tensor.NewWithStorage[T](qShape, qView)
		if err != nil {
			return nil, nil, nil, err
		}
		k, err = tensor.NewWithStorage[T](kShape, kView)
		if err != nil {
			return nil, nil, nil, err
		}
		v, err = tensor.NewWithStorage[T](vShape, vView)
		if err != nil {
			return nil, nil, nil, err
		}
		return q, k, v, nil
	}

	// CPU path: copy data.
	slog.Warn("GQA splitMergedQKV CPU fallback triggered (D2H copy)", "storageType", fmt.Sprintf("%T", merged.GetStorage()))
	data := merged.Data()
	qData := make([]T, batchElems*qDim)
	kData := make([]T, batchElems*kDim)
	vData := make([]T, batchElems*vDim)
	for b := 0; b < batchElems; b++ {
		off := b * lastDim
		copy(qData[b*qDim:(b+1)*qDim], data[off:off+qDim])
		copy(kData[b*kDim:(b+1)*kDim], data[off+qDim:off+qDim+kDim])
		copy(vData[b*vDim:(b+1)*vDim], data[off+qDim+kDim:off+qDim+kDim+vDim])
	}
	q, err = tensor.New(qShape, qData)
	if err != nil {
		return nil, nil, nil, err
	}
	k, err = tensor.New(kShape, kData)
	if err != nil {
		return nil, nil, nil, err
	}
	v, err = tensor.New(vShape, vData)
	if err != nil {
		return nil, nil, nil, err
	}
	return q, k, v, nil
}

// BuildCausalSlidingWindowMask creates a causal attention mask that also
// restricts attention to the last windowSize positions. Positions outside
// the window or in the future are set to a large negative value.
// Shape: [1, 1, seqLen, seqLen].
func BuildCausalSlidingWindowMask[T tensor.Numeric](seqLen, windowSize int) *tensor.TensorNumeric[T] {
	// Use a runtime variable to avoid compile-time overflow check for narrow types.
	var neg = -1e9
	largeNeg := T(neg)
	data := make([]T, seqLen*seqLen)
	for i := range seqLen {
		for j := range seqLen {
			if j <= i && i-j < windowSize {
				data[i*seqLen+j] = 0
			} else {
				data[i*seqLen+j] = largeNeg
			}
		}
	}
	mask, _ := tensor.New[T]([]int{1, 1, seqLen, seqLen}, data)
	return mask
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*GroupedQueryAttention[float32])(nil)

// gqaKVPort is a lightweight graph.Node[T] adapter that exposes a cached
// tensor produced during the owning layer's Forward pass as a readable port.
// It is used by GroupedQueryAttention.KPort() / VPort() to hand the donor
// layer's K and V to downstream shared-KV attention layers (ADR-087).
//
// The adapter's Forward ignores its inputs and returns whatever tensor the
// owner most recently stored in the underlying field (kOut or vOut). Reading
// before the owner's first Forward returns nil, matching the ADR's stated
// nil-safety contract: downstream consumers must call donor.Forward(...)
// first, then pass donor.KPort()/VPort() into the shared layer.
//
// The adapter carries no parameters of its own and has no gradient path;
// backward semantics live on the owning GQA layer.
type gqaKVPort[T tensor.Numeric] struct {
	owner  *GroupedQueryAttention[T]
	selKey bool // true = K port, false = V port
}

// OpType identifies the port kind for debugging and graph dumps.
func (p *gqaKVPort[T]) OpType() string {
	if p.selKey {
		return "GroupedQueryAttention.KPort"
	}
	return "GroupedQueryAttention.VPort"
}

// Attributes returns the port's attributes.
func (p *gqaKVPort[T]) Attributes() map[string]interface{} {
	kind := "v"
	if p.selKey {
		kind = "k"
	}
	return map[string]interface{}{"port": kind}
}

// Forward returns the cached K or V tensor captured during the owner GQA's
// most recent Forward pass. Inputs are ignored -- the tensor is produced
// inside the owner and merely re-exported here. Returns an error if the
// owner has not yet run (nil cached value).
func (p *gqaKVPort[T]) Forward(_ context.Context, _ ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	t := p.owner.kOut
	name := "K"
	if !p.selKey {
		t = p.owner.vOut
		name = "V"
	}
	if t == nil {
		return nil, fmt.Errorf("GroupedQueryAttention.%sPort: owner layer has not produced %s yet; call owner.Forward first", name, name)
	}
	return t, nil
}

// Backward for a port node is a pass-through: the gradient flows back through
// the owner GQA's Backward path via the shared KV donor relationship, not
// through this adapter. The adapter therefore returns an empty slice.
func (p *gqaKVPort[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Parameters returns nil -- the port carries no trainable state of its own.
func (p *gqaKVPort[T]) Parameters() []*graph.Parameter[T] { return nil }

// OutputShape returns the shape of the cached tensor, or nil if the owner
// has not yet run.
func (p *gqaKVPort[T]) OutputShape() []int {
	t := p.owner.kOut
	if !p.selKey {
		t = p.owner.vOut
	}
	if t == nil {
		return nil
	}
	return t.Shape()
}

// Statically assert the port adapter implements graph.Node.
var _ graph.Node[float32] = (*gqaKVPort[float32])(nil)

// KPort returns a graph.Node[T] that, when executed, yields the layer's
// most-recently-computed K in shape [batch, numKVHeads, seqLen, headDim].
// In external-K/V mode, the captured K is whatever tensor was supplied
// via Forward's inputs[1]; otherwise it is the layer's own projection
// output after optional per-head K norm and post-projection RoPE.
//
// Donor -> shared wiring is performed by the caller: at graph-build time
// the builder wires donor.KPort() as an input producing K, then passes
// the resulting tensor into the shared layer's Forward via inputs[1].
// See ADR-087.
//
// The returned node's Forward must be called AFTER the owner GQA's
// Forward pass -- the port re-exports the cached tensor produced there.
// Calling Forward on the port before the owner has produced K yields a
// descriptive error (see gqaKVPort.Forward). The node carries no
// parameters and has no independent gradient.
func (gqa *GroupedQueryAttention[T]) KPort() graph.Node[T] {
	return &gqaKVPort[T]{owner: gqa, selKey: true}
}

// VPort returns a graph.Node[T] that yields the layer's most-recently-
// computed V in shape [batch, numKVHeads, seqLen, headDim]. See KPort
// for semantics, donor-wiring conventions, and nil-safety contract.
func (gqa *GroupedQueryAttention[T]) VPort() graph.Node[T] {
	return &gqaKVPort[T]{owner: gqa, selKey: false}
}
