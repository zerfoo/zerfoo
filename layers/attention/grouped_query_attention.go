// Package attention provides attention mechanisms for neural networks.
package attention

import (
	"context"
	"fmt"
	"log"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings" // For RoPE
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
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

	// Cached tensors for backward pass
	qProj           *tensor.TensorNumeric[T] // Projected Q
	kProj           *tensor.TensorNumeric[T] // Projected K
	vProj           *tensor.TensorNumeric[T] // Projected V
	attnOutput      *tensor.TensorNumeric[T] // Output from scaledDotProductAttention (heads format)
	attnOutputFinal *tensor.TensorNumeric[T] // Final reshaped output passed to wo
	qHeadsRoPE      *tensor.TensorNumeric[T] // Q after RoPE
	kHeadsRoPE      *tensor.TensorNumeric[T] // K after RoPE
	outputShape     []int
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
	Base      float64
	MaxSeqLen int
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

	wk, err := core.NewDense[T]("wk", engine, ops, modelDim, headDim*numKeyValueHeads) // K projection
	if err != nil {
		return nil, fmt.Errorf("failed to create WK dense layer: %w", err)
	}

	wv, err := core.NewDense[T]("wv", engine, ops, modelDim, headDim*numKeyValueHeads) // V projection
	if err != nil {
		return nil, fmt.Errorf("failed to create WV dense layer: %w", err)
	}

	// Initialize ScaledDotProductAttention. dk is headDim.
	scaledDotProductAttention := NewScaledDotProductAttention[T](engine, headDim)

	// Initialize output Dense layer.
	wo, err := core.NewDense[T]("wo", engine, ops, modelDim, modelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create WO dense layer: %w", err)
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[T](context.Background(), engine, headDim, options.MaxSeqLen, embeddings.WithRotaryBase(options.Base))
	if err != nil {
		return nil, fmt.Errorf("failed to create RotaryPositionalEmbedding: %w", err)
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

// OutputShape returns the output shape of the GroupedQueryAttention.
func (gqa *GroupedQueryAttention[T]) OutputShape() []int {
	return gqa.outputShape
}

// ScaleRope scales the rotary positional embeddings.
func (gqa *GroupedQueryAttention[T]) ScaleRope(ctx context.Context, factor float64) error {
	return gqa.rope.Scale(ctx, factor)
}

// SetQKNorms sets optional per-head RMSNorm layers for Q and K projections.
// Used by architectures like Gemma 3 that normalize Q/K after projection.
func (gqa *GroupedQueryAttention[T]) SetQKNorms(qNorm, kNorm graph.Node[T]) {
	gqa.qNorm = qNorm
	gqa.kNorm = kNorm
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
	params = append(params, gqa.wk.Parameters()...)
	params = append(params, gqa.wv.Parameters()...)
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

	var mask *tensor.TensorNumeric[T]
	if len(inputs) > 1 {
		mask = inputs[1]
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	// Check for KV cache in context.
	cache, hasCache := generate.GetCache[T](ctx)

	// 1. Linear projections for Q, K, V
	var err error
	var qProj, kProj, vProj *tensor.TensorNumeric[T]
	if gqa.mergedQKV != nil && seqLen == 1 {
		// Merged QKV: single GEMV + zero-copy split for decode.
		merged, mergeErr := gqa.engine.MatMul(ctx, input, gqa.mergedQKV)
		if mergeErr != nil {
			return nil, fmt.Errorf("merged QKV MatMul: %w", mergeErr)
		}
		qProj, kProj, vProj, err = splitMergedQKV[T](merged, gqa.qDim, gqa.kDim, gqa.vDim)
		if err != nil {
			return nil, fmt.Errorf("split merged QKV: %w", err)
		}
	} else {
		qProj, err = gqa.wq.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
		kProj, err = gqa.wk.Forward(ctx, input)
		if err != nil {
			return nil, err
		}
		vProj, err = gqa.wv.Forward(ctx, input)
		if err != nil {
			return nil, err
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

	// Fused QK norm+RoPE decode path: replaces 4 kernel launches with 1.
	// Conditions: decode (seqLen=1), Q/K norm weights available, engine supports it.
	fusedQKNormRoPE := false
	if seqLen == 1 && gqa.qNormWeight != nil && gqa.kNormWeight != nil {
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
			posOffset := 0
			if hasCache {
				posOffset = cache.SeqLen()
			}
			cosAngles, sinAngles, halfRotary, angleErr := gqa.rope.GetAngles(posOffset, 1)
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
				} else {
					// CPU fallback: copy data.
					log.Printf("WARNING: GQA fused QK norm+RoPE CPU fallback triggered (D2H copy); expected GPUStorage but got %T", fusedOut.GetStorage())
					data := fusedOut.Data()
					qData := make([]T, qElems)
					kData := make([]T, kElems)
					copy(qData, data[:qElems])
					copy(kData, data[qElems:qElems+kElems])
					qSlice, cpuErr := tensor.New([]int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim}, qData)
					if cpuErr != nil {
						return nil, cpuErr
					}
					kSlice, cpuErr := tensor.New([]int{batchSize, gqa.numKeyValueHeads, seqLen, gqa.headDim}, kData)
					if cpuErr != nil {
						return nil, cpuErr
					}
					qHeadsRoPE = qSlice
					kHeadsRoPE = kSlice
				}
			}
			// Fall through to unfused path on error.
		}
	}

	if !fusedQKNormRoPE {
		// Unfused path: separate Q/K norm + transpose + RoPE.
		// Q: (batch, seq_len, num_query_heads, head_dim)
		qReshaped, reshapeErr := gqa.engine.Reshape(ctx, qProj, []int{batchSize, seqLen, gqa.numQueryHeads, gqa.headDim})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

		// Apply per-head Q norm if set (Gemma 3).
		if gqa.qNorm != nil {
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

		// Apply per-head K norm if set (Gemma 3).
		if gqa.kNorm != nil {
			kReshaped, err = gqa.kNorm.Forward(ctx, kReshaped)
			if err != nil {
				return nil, fmt.Errorf("kNorm: %w", err)
			}
		}

		kHeads, reshapeErr := gqa.engine.Transpose(ctx, kReshaped, []int{0, 2, 1, 3})
		if reshapeErr != nil {
			return nil, reshapeErr
		}

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
		if hasCache {
			gqa.rope.SetPositionOffset(cache.SeqLen())
		} else {
			gqa.rope.SetPositionOffset(0)
		}

		qHeadsRoPE, err = gqa.rope.Forward(ctx, qForRoPE)
		if err != nil {
			return nil, err
		}

		kHeadsRoPE, err = gqa.rope.Forward(ctx, kForRoPE)
		if err != nil {
			return nil, err
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

	// KV Cache: store K/V per KV-head in shape [batch*numKVHeads, seq_len, headDim],
	// then retrieve full cached K/V for attention computation.
	// kvSeqLen tracks the K/V sequence length (may differ from Q seqLen when cached).
	kvSeqLen := seqLen
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

		// Try block-table reader first (avoids gather-to-contiguous copy).
		var cachedK, cachedV *tensor.TensorNumeric[T]
		if gqa.blockTableReader != nil {
			cachedK, cachedV, _ = gqa.blockTableReader.ReadKV(gqa.LayerIndex)
		}

		if cachedK == nil || cachedV == nil {
			// Fall back to standard cache gather path.
			lkv, ok := cache.Get(gqa.LayerIndex)
			if !ok {
				return nil, fmt.Errorf("kv cache: layer %d missing after update", gqa.LayerIndex)
			}
			cachedK = lkv.Key
			cachedV = lkv.Value
		}

		// Unflatten back to [batch, numKVHeads, cachedSeqLen, headDim].
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

	// 3. Grouped Query Attention: expand K/V to match Q head count.
	// When numKVHeads == 1, MatMul batch broadcasting avoids the expensive
	// Repeat that physically copies K/V data (b_batch=1 broadcast).
	// For numKVHeads > 1, use Repeat to match batch dimensions exactly.
	if gqa.numQueryHeads != gqa.numKeyValueHeads && gqa.numKeyValueHeads > 1 {
		replicationFactor := gqa.numQueryHeads / gqa.numKeyValueHeads
		kHeadsExpanded, expandErr := gqa.engine.Repeat(ctx, kHeadsRoPE, 1, replicationFactor)
		if expandErr != nil {
			return nil, expandErr
		}
		vHeadsExpanded, expandErr := gqa.engine.Repeat(ctx, vHeads, 1, replicationFactor)
		if expandErr != nil {
			return nil, expandErr
		}
		kHeadsRoPE = kHeadsExpanded
		vHeads = vHeadsExpanded
	}

	// 4. Apply Scaled Dot-Product Attention
	// Q uses numQueryHeads; K/V use numQueryHeads (after Repeat) or numKVHeads
	// (when broadcast). SDPA reshapes to 3D for batched attention.
	qForSDPA, err := gqa.engine.Reshape(ctx, qHeadsRoPE, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	kvBatchHeads := gqa.numQueryHeads
	if gqa.numKeyValueHeads == 1 {
		kvBatchHeads = gqa.numKeyValueHeads
	}

	kForSDPA, err := gqa.engine.Reshape(ctx, kHeadsRoPE, []int{batchSize * kvBatchHeads, kvSeqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	vForSDPA, err := gqa.engine.Reshape(ctx, vHeads, []int{batchSize * kvBatchHeads, kvSeqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	// Enable causal masking during prefill (seqLen > 1) when no explicit mask
	// is provided. Without this, each position attends to future tokens,
	// producing garbage output.
	if mask == nil && seqLen > 1 {
		gqa.scaledDotProductAttention.SetCausal(true)
	} else {
		gqa.scaledDotProductAttention.SetCausal(false)
	}

	attnOutputHeads, err := gqa.scaledDotProductAttention.Forward(ctx, qForSDPA, kForSDPA, vForSDPA, mask)
	if err != nil {
		return nil, err
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
// original KV head count.  It reshapes [batch*numQ, seq, headDim] →
// [batch, numKV, repFactor, seq, headDim], sums along the repFactor
// dimension (axis 2), then flattens back to [batch*numKV, seq, headDim].
func (gqa *GroupedQueryAttention[T]) reverseHeadReplication(ctx context.Context, d *tensor.TensorNumeric[T], batchSize, seqLen int) (*tensor.TensorNumeric[T], error) {
	repFactor := gqa.numQueryHeads / gqa.numKeyValueHeads
	d4, err := gqa.engine.Reshape(ctx, d, []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	d5, err := gqa.engine.Reshape(ctx, d4, []int{batchSize, gqa.numKeyValueHeads, repFactor, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	dSum, err := gqa.engine.ReduceSum(ctx, d5, 2, false)
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
	dQAfterRoPE, err := gqa.rope.Backward(ctx, mode, dQ)
	if err != nil {
		return nil, fmt.Errorf("rope backward Q: %w", err)
	}
	dKAfterRoPE, err := gqa.rope.Backward(ctx, mode, dK)
	if err != nil {
		return nil, fmt.Errorf("rope backward K: %w", err)
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

	// CPU path: copy data.
	log.Printf("WARNING: GQA splitMergedQKV CPU fallback triggered (D2H copy); expected GPUStorage but got %T", merged.GetStorage())
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

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*GroupedQueryAttention[float32])(nil)
