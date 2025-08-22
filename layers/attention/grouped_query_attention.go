// Package attention provides attention mechanisms for neural networks.
package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings" // For RoPE
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
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

	// Cached tensors for backward pass
	qProj       *tensor.TensorNumeric[T] // Projected Q
	kProj       *tensor.TensorNumeric[T] // Projected K
	vProj       *tensor.TensorNumeric[T] // Projected V
	attnOutput  *tensor.TensorNumeric[T] // Output from scaledDotProductAttention
	qHeadsRoPE  *tensor.TensorNumeric[T] // Q after RoPE
	kHeadsRoPE  *tensor.TensorNumeric[T] // K after RoPE
	outputShape []int
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
func NewGroupedQueryAttentionFromParams[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, numQueryHeads, numKeyValueHeads int,
	wq, wk, wv, wo *core.Dense[T],
	rope *embeddings.RotaryPositionalEmbedding[T],
) (*GroupedQueryAttention[T], error) {
	headDim := modelDim / numQueryHeads
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

// Parameters returns the parameters of the GroupedQueryAttention layer.
func (gqa *GroupedQueryAttention[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, gqa.wq.Parameters()...)
	params = append(params, gqa.wk.Parameters()...)
	params = append(params, gqa.wv.Parameters()...)
	params = append(params, gqa.wo.Parameters()...)

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

	// 1. Linear projections for Q, K, V
	qProj, err := gqa.wq.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	kProj, err := gqa.wk.Forward(ctx, input)
	if err != nil {
		return nil, err
	}
	vProj, err := gqa.wv.Forward(ctx, input)
	if err != nil {
		return nil, err
	}

	// Cache projected Q, K, V for backward pass
	gqa.qProj = qProj
	gqa.kProj = kProj
	gqa.vProj = vProj

	// 2. Split into heads and apply RoPE
	// Q: (batch, seq_len, num_query_heads, head_dim)
	qReshaped, err := gqa.engine.Reshape(ctx, qProj, []int{batchSize, seqLen, gqa.numQueryHeads, gqa.headDim})
	if err != nil {
		return nil, err
	}
	qHeads, err := gqa.engine.Transpose(ctx, qReshaped, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// K, V: (batch, seq_len, num_kv_heads, head_dim)
	kReshaped, err := gqa.engine.Reshape(ctx, kProj, []int{batchSize, seqLen, gqa.numKeyValueHeads, gqa.headDim})
	if err != nil {
		return nil, err
	}
	kHeads, err := gqa.engine.Transpose(ctx, kReshaped, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	vReshaped, err := gqa.engine.Reshape(ctx, vProj, []int{batchSize, seqLen, gqa.numKeyValueHeads, gqa.headDim})
	if err != nil {
		return nil, err
	}
	vHeads, err := gqa.engine.Transpose(ctx, vReshaped, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	// Apply RoPE to Q and K
	// RoPE expects (batch, seq_len, head_dim)
	// We need to apply it head by head or use a batched RoPE if engine supports.
	// Let's reshape to (batch * num_heads, seq_len, head_dim) for RoPE.
	qForRoPE, err := gqa.engine.Reshape(ctx, qHeads, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	kForRoPE, err := gqa.engine.Reshape(ctx, kHeads, []int{batchSize * gqa.numKeyValueHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	qHeadsRoPE, err := gqa.rope.Forward(ctx, qForRoPE)
	if err != nil {
		return nil, err
	}
	kHeadsRoPE, err := gqa.rope.Forward(ctx, kForRoPE)
	if err != nil {
		return nil, err
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

	// 3. Grouped Query Attention: Replicate K, V heads for each Query head group
	// (batch, num_query_heads, seq_len, head_dim)
	// (batch, num_kv_heads, seq_len, kv_head_dim)
	// Need to expand K and V to match num_query_heads
	// Example: if num_query_heads=8, num_kv_heads=2, then each KV head is replicated 4 times.
	if gqa.numQueryHeads != gqa.numKeyValueHeads {
		replicationFactor := gqa.numQueryHeads / gqa.numKeyValueHeads
		// Replicate kHeads and vHeads along the head dimension
		// (batch, num_kv_heads, seq_len, kv_head_dim) -> (batch, num_query_heads, seq_len, kv_head_dim)
		kHeadsExpanded, err := gqa.engine.Repeat(ctx, kHeadsRoPE, 1, replicationFactor)
		if err != nil {
			return nil, err
		}
		vHeadsExpanded, err := gqa.engine.Repeat(ctx, vHeads, 1, replicationFactor)
		if err != nil {
			return nil, err
		}
		kHeadsRoPE = kHeadsExpanded
		vHeads = vHeadsExpanded
	}

	// 4. Apply Scaled Dot-Product Attention
	// Reshape to (batch_size * num_heads, seq_len, head_dim) for SDPA.
	qForSDPA, err := gqa.engine.Reshape(ctx, qHeadsRoPE, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	kForSDPA, err := gqa.engine.Reshape(ctx, kHeadsRoPE, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	vForSDPA, err := gqa.engine.Reshape(ctx, vHeads, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
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
	attnOutputFinal, err := gqa.engine.Reshape(ctx, attnOutputCombined, []int{batchSize, seqLen, gqa.modelDim})
	if err != nil {
		return nil, err
	}

	// 6. Final linear projection
	output, err := gqa.wo.Forward(ctx, attnOutputFinal)
	if err != nil {
		return nil, err
	}

	return output, nil
}

func (gqa *GroupedQueryAttention[T]) backwardSplitAndReshape(ctx context.Context, dQForRoPE, dKForRoPE []*tensor.TensorNumeric[T], dVHeads *tensor.TensorNumeric[T], batchSize, seqLen, kvHeadDim int) (*tensor.TensorNumeric[T], *tensor.TensorNumeric[T], *tensor.TensorNumeric[T], error) {
	// Sum gradients from replicated K, V heads
	if gqa.numQueryHeads != gqa.numKeyValueHeads {
		// Sum dVHeads along the replicated dimension
		dVHeadsSummed, err := gqa.engine.ReduceSum(ctx, dVHeads, 1, false)
		if err != nil {
			return nil, nil, nil, err
		}
		dVHeads = dVHeadsSummed
		// Sum dKHeadsRoPE along the replicated dimension
		dKForRoPE[0], err = gqa.engine.ReduceSum(ctx, dKForRoPE[0], 1, false)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	// Reshape and transpose back to original projection shapes
	dQProj, err := gqa.engine.Reshape(ctx, dQForRoPE[0], []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, nil, nil, err
	}
	dQProj, err = gqa.engine.Transpose(ctx, dQProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, nil, nil, err
	}
	dQProj, err = gqa.engine.Reshape(ctx, dQProj, []int{batchSize, seqLen, gqa.modelDim})
	if err != nil {
		return nil, nil, nil, err
	}

	dKProj, err := gqa.engine.Reshape(ctx, dKForRoPE[0], []int{batchSize, gqa.numKeyValueHeads, seqLen, kvHeadDim})
	if err != nil {
		return nil, nil, nil, err
	}
	dKProj, err = gqa.engine.Transpose(ctx, dKProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, nil, nil, err
	}
	dKProj, err = gqa.engine.Reshape(ctx, dKProj, []int{batchSize, seqLen, gqa.modelDim / gqa.numQueryHeads * gqa.numKeyValueHeads}) // Reshape to original K proj dim
	if err != nil {
		return nil, nil, nil, err
	}

	dVProj, err := gqa.engine.Reshape(ctx, dVHeads, []int{batchSize, gqa.numKeyValueHeads, seqLen, kvHeadDim})
	if err != nil {
		return nil, nil, nil, err
	}
	dVProj, err = gqa.engine.Transpose(ctx, dVProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, nil, nil, err
	}
	dVProj, err = gqa.engine.Reshape(ctx, dVProj, []int{batchSize, seqLen, gqa.modelDim / gqa.numQueryHeads * gqa.numKeyValueHeads}) // Reshape to original V proj dim
	if err != nil {
		return nil, nil, nil, err
	}

	return dQProj, dKProj, dVProj, nil
}

// Backward computes the gradients for GroupedQueryAttention.
func (gqa *GroupedQueryAttention[T]) Backward(ctx context.Context, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GroupedQueryAttention: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := inputs[0] // Original input to GQA
	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	// 1. Backward through final linear projection (WO)
	dAttnOutputFinal, err := gqa.wo.Backward(ctx, dOut, gqa.attnOutput)
	if err != nil {
		return nil, err
	}
	dAttnOutputFinalTensor := dAttnOutputFinal[0]

	// 2. Backward through reshape and transpose (reverse of step 5 in Forward)
	dAttnOutputCombined, err := gqa.engine.Reshape(ctx, dAttnOutputFinalTensor, []int{batchSize, seqLen, gqa.numQueryHeads, gqa.headDim})
	if err != nil {
		return nil, err
	}
	dAttnOutputReshaped, err := gqa.engine.Transpose(ctx, dAttnOutputCombined, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	dAttnOutputHeads, err := gqa.engine.Reshape(ctx, dAttnOutputReshaped, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	// 3. Backward through Scaled Dot-Product Attention
	dQKVSplit, err := gqa.scaledDotProductAttention.Backward(ctx, dAttnOutputHeads, gqa.scaledDotProductAttention.q, gqa.scaledDotProductAttention.k, gqa.scaledDotProductAttention.v)
	if err != nil {
		return nil, err
	}
	dQHeadsRoPE, dKHeadsRoPE, dVHeads := dQKVSplit[0], dQKVSplit[1], dQKVSplit[2]

	// 4. Backward through RoPE
	dQForRoPE, err := gqa.rope.Backward(ctx, dQHeadsRoPE)
	if err != nil {
		return nil, err
	}
	dKForRoPE, err := gqa.rope.Backward(ctx, dKHeadsRoPE)
	if err != nil {
		return nil, err
	}

	// 5. Backward through head splitting and replication
	dQProj, dKProj, dVProj, err := gqa.backwardSplitAndReshape(ctx, dQForRoPE, dKForRoPE, dVHeads, batchSize, seqLen, gqa.headDim)
	if err != nil {
		return nil, err
	}

	// 6. Backward through linear projections (WQ, WK, WV)
	dInputQ, err := gqa.wq.Backward(ctx, dQProj, input)
	if err != nil {
		return nil, err
	}
	dInputK, err := gqa.wk.Backward(ctx, dKProj, input)
	if err != nil {
		return nil, err
	}
	dInputV, err := gqa.wv.Backward(ctx, dVProj, input)
	if err != nil {
		return nil, err
	}

	// Sum gradients from Q, K, V paths to get the total gradient for the original input
	dInputTotal, err := gqa.engine.Add(ctx, dInputQ[0], dInputK[0])
	if err != nil {
		return nil, err
	}
	dInputTotal, err = gqa.engine.Add(ctx, dInputTotal, dInputV[0])
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInputTotal}, nil
}
