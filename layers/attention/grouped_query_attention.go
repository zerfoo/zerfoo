// layers/attention/grouped_query_attention.go
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

// GroupedQueryAttention implements the Grouped Query Attention mechanism.
type GroupedQueryAttention[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops numeric.Arithmetic[T]
	numQueryHeads int // Number of query heads
	numKeyValueHeads int // Number of key/value heads (must divide numQueryHeads)
	modelDim int // d_model, the input/output dimension of the block
	headDim  int // Dimension of each head (modelDim / numQueryHeads)

	// Linear projections for Q, K, V
	wq *core.Dense[T] // Query projection
	wk *core.Dense[T] // Key projection
	wv *core.Dense[T] // Value projection

	// Scaled Dot-Product Attention
	scaledDotProductAttention *ScaledDotProductAttention[T]

	// Final linear projection
	wo *core.Dense[T] // Output projection

	// Cached tensors for backward pass
	qProj *tensor.Tensor[T] // Projected Q
	kProj *tensor.Tensor[T] // Projected K
	vProj *tensor.Tensor[T] // Projected V
	attnOutput *tensor.Tensor[T] // Output from scaledDotProductAttention
	qHeadsRoPE *tensor.Tensor[T] // Q after RoPE
	kHeadsRoPE *tensor.Tensor[T] // K after RoPE
}

// NewGroupedQueryAttention creates a new GroupedQueryAttention layer.
// modelDim: The dimension of the input and output of the block (d_model).
// numQueryHeads: The number of query heads.
// numKeyValueHeads: The number of key/value heads.
func NewGroupedQueryAttention[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim, numQueryHeads, numKeyValueHeads int) (*GroupedQueryAttention[T], error) {
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
	kvHeadDim := modelDim / numKeyValueHeads // Dimension for K and V projections

	// Initialize Dense layers for Q, K, V projections
	wq, err := core.NewDense[T]("wq", engine, ops, modelDim, modelDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create WQ dense layer: %w", err)
	}
							wk, err := core.NewDense[T]("wk", engine, ops, modelDim, kvHeadDim) // K projection to kvHeadDim
	if err != nil {
		return nil, fmt.Errorf("failed to create WK dense layer: %w", err)
	}
					wv, err := core.NewDense[T]("wv", engine, ops, modelDim, kvHeadDim) // V projection to kvHeadDim
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

	return &GroupedQueryAttention[T]{
		engine:                    engine,
		numQueryHeads:             numQueryHeads,
		numKeyValueHeads:          numKeyValueHeads,
		modelDim:                  modelDim,
		headDim:                   headDim,
		wq:                        wq,
		wk:                        wk,
		wv:                        wv,
		scaledDotProductAttention: scaledDotProductAttention,
		wo:                        wo,
	}, nil
}

// OutputShape returns the output shape of the GroupedQueryAttention.
func (gqa *GroupedQueryAttention[T]) OutputShape(inputShapes ...[]int) ([]int, error) {
	if len(inputShapes) != 1 {
		return nil, fmt.Errorf("GroupedQueryAttention: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputShapes))
	}
	inputShape := inputShapes[0]
	return inputShape, nil
}

func (gqa *GroupedQueryAttention[T]) Parameters() []graph.Parameter[T] {
	var params []graph.Parameter[T]
	for _, p := range gqa.wq.Parameters() {
		params = append(params, *p)
	}
	for _, p := range gqa.wk.Parameters() {
		params = append(params, *p)
	}
	for _, p := range gqa.wv.Parameters() {
		params = append(params, *p)
	}
	for _, p := range gqa.wo.Parameters() {
		params = append(params, *p)
	}
	return params
}

// Forward computes the grouped query attention.
func (gqa *GroupedQueryAttention[T]) Forward(ctx context.Context, inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GroupedQueryAttention: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := inputs[0] // (batch_size, seq_len, model_dim)
	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	// 1. Linear projections for Q, K, V
	qProj, err := gqa.wq.Forward(input)
	if err != nil {
		return nil, err
	}
	kProj, err := gqa.wk.Forward(input)
	if err != nil {
		return nil, err
	}
	vProj, err := gqa.wv.Forward(input)
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
	kvHeadDim := gqa.modelDim / gqa.numKeyValueHeads
	kReshaped, err := gqa.engine.Reshape(ctx, kProj, []int{batchSize, seqLen, gqa.numKeyValueHeads, kvHeadDim})
	if err != nil {
		return nil, err
	}
	kHeads, err := gqa.engine.Transpose(ctx, kReshaped, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}

	vReshaped, err := gqa.engine.Reshape(ctx, vProj, []int{batchSize, seqLen, gqa.numKeyValueHeads, kvHeadDim})
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
	kForRoPE, err := gqa.engine.Reshape(ctx, kHeads, []int{batchSize * gqa.numKeyValueHeads, seqLen, kvHeadDim})
	if err != nil {
		return nil, err
	}

	rope, err := embeddings.NewRotaryPositionalEmbedding[T](ctx, gqa.engine, gqa.headDim, seqLen)
	if err != nil {
		return nil, err
	}
	qHeadsRoPE, err := rope.Forward(ctx, qForRoPE)
	if err != nil {
		return nil, err
	}
	kHeadsRoPE, err := rope.Forward(ctx, kForRoPE)
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
	kHeadsRoPE, err = gqa.engine.Reshape(ctx, kHeadsRoPE, []int{batchSize, gqa.numKeyValueHeads, seqLen, kvHeadDim})
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

	attnOutputHeads, err := gqa.scaledDotProductAttention.Forward(ctx, qForSDPA, kForSDPA, vForSDPA)
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
	output, err := gqa.wo.Forward(attnOutputFinal)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the gradients for GroupedQueryAttention.
func (gqa *GroupedQueryAttention[T]) Backward(ctx context.Context, dOut *tensor.Tensor[T], inputs ...*tensor.Tensor[T]) ([]*tensor.Tensor[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GroupedQueryAttention: %w, expected %d, got %d", graph.ErrInvalidInputCount, 1, len(inputs))
	}
	input := inputs[0] // Original input to GQA
	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]

	// 1. Backward through final linear projection (WO)
	dAttnOutputFinal, err := gqa.wo.Backward(dOut)
	if err != nil {
		return nil, err
	}
	dAttnOutputFinalTensor := dAttnOutputFinal[0]

	// 2. Backward through reshape and transpose (reverse of step 5 in Forward)
	dAttnOutputReshaped, err := gqa.engine.Reshape(ctx, dAttnOutputFinalTensor, []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	dAttnOutputCombined, err := gqa.engine.Transpose(ctx, dAttnOutputReshaped, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	dAttnOutputHeads, err := gqa.engine.Reshape(ctx, dAttnOutputCombined, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}

	// 3. Backward through Scaled Dot-Product Attention
	// Need to pass the original Q, K, V that went into SDPA.
	// These are qHeadsRoPE, kHeadsRoPE, vHeads (after replication if any).
	qForSDPA, err := gqa.engine.Reshape(ctx, gqa.qHeadsRoPE, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	kForSDPA, err := gqa.engine.Reshape(ctx, gqa.kHeadsRoPE, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	// Re-create vForSDPA as it might have been replicated
	var kvHeadDim int = gqa.modelDim / gqa.numKeyValueHeads
	vHeadsOriginal := gqa.vProj // Use original vProj to avoid issues with replication
	var vForSDPA *tensor.Tensor[T] // Declare vForSDPA here
	if gqa.numQueryHeads != gqa.numKeyValueHeads {
		replicationFactor := gqa.numQueryHeads / gqa.numKeyValueHeads // Declare replicationFactor here
		vHeadsOriginalReshaped, err := gqa.engine.Reshape(ctx, vHeadsOriginal, []int{batchSize, seqLen, gqa.numKeyValueHeads, kvHeadDim})
		if err != nil {
			return nil, err
		}
		vHeadsOriginalTransposed, err := gqa.engine.Transpose(ctx, vHeadsOriginalReshaped, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
		vForSDPA, err = gqa.engine.Repeat(ctx, vHeadsOriginalTransposed, 1, replicationFactor)
		if err != nil {
			return nil, err
		}
		vForSDPA, err = gqa.engine.Reshape(ctx, vForSDPA, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
		if err != nil {
			return nil, err
		}
	} else {
		vForSDPA, err = gqa.engine.Reshape(ctx, vHeadsOriginal, []int{batchSize * gqa.numQueryHeads, seqLen, gqa.headDim})
		if err != nil {
			return nil, err
		}
	}

	dQKVSplit, err := gqa.scaledDotProductAttention.Backward(ctx, dAttnOutputHeads, qForSDPA, kForSDPA, vForSDPA)
	if err != nil {
		return nil, err
	}
	dQHeadsRoPE, dKHeadsRoPE, dVHeads := dQKVSplit[0], dQKVSplit[1], dQKVSplit[2]

	// 4. Backward through RoPE
	rope, err := embeddings.NewRotaryPositionalEmbedding[T](ctx, gqa.engine, gqa.headDim, seqLen)
	if err != nil {
		return nil, err
	}
	dQForRoPE, err := rope.Backward(ctx, dQHeadsRoPE, qForSDPA)
	if err != nil {
		return nil, err
	}
	dKForRoPE, err := rope.Backward(ctx, dKHeadsRoPE, kForSDPA)
	if err != nil {
		return nil, err
	}

	// 5. Backward through split and reshape (reverse of step 2 in Forward)
	// dQForRoPE: (batch * num_query_heads, seq_len, head_dim)
	// dKForRoPE: (batch * num_kv_heads, seq_len, kv_head_dim)
	// dVHeads: (batch * num_query_heads, seq_len, head_dim)

	// Sum gradients from replicated K, V heads
	if gqa.numQueryHeads != gqa.numKeyValueHeads {
		// Sum dVHeads along the replicated dimension
		dVHeadsSummed, err := gqa.engine.ReduceSum(ctx, dVHeads, 1, false)
		if err != nil {
			return nil, err
		}
		dVHeads = dVHeadsSummed
		// Sum dKHeadsRoPE along the replicated dimension
		dKHeadsRoPESummed, err := gqa.engine.ReduceSum(ctx, dKHeadsRoPE, 1, false)
		if err != nil {
			return nil, err
		}
		dKHeadsRoPE = dKHeadsRoPESummed
	}

	// Reshape and transpose back to original projection shapes
	dQProj, err := gqa.engine.Reshape(ctx, dQForRoPE[0], []int{batchSize, gqa.numQueryHeads, seqLen, gqa.headDim})
	if err != nil {
		return nil, err
	}
	dQProj, err = gqa.engine.Transpose(ctx, dQProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	dQProj, err = gqa.engine.Reshape(ctx, dQProj, []int{batchSize, seqLen, gqa.modelDim})
	if err != nil {
		return nil, err
	}

	dKProj, err := gqa.engine.Reshape(ctx, dKForRoPE[0], []int{batchSize, gqa.numKeyValueHeads, seqLen, kvHeadDim})
	if err != nil {
		return nil, err
	}
	dKProj, err = gqa.engine.Transpose(ctx, dKProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	dKProj, err = gqa.engine.Reshape(ctx, dKProj, []int{batchSize, seqLen, gqa.modelDim / gqa.numQueryHeads * gqa.numKeyValueHeads}) // Reshape to original K proj dim
	if err != nil {
		return nil, err
	}

	dVProj, err := gqa.engine.Reshape(ctx, dVHeads, []int{batchSize, gqa.numKeyValueHeads, seqLen, kvHeadDim})
	if err != nil {
		return nil, err
	}
	dVProj, err = gqa.engine.Transpose(ctx, dVProj, []int{0, 2, 1, 3})
	if err != nil {
		return nil, err
	}
	dVProj, err = gqa.engine.Reshape(ctx, dVProj, []int{batchSize, seqLen, gqa.modelDim / gqa.numQueryHeads * gqa.numKeyValueHeads}) // Reshape to original V proj dim
	if err != nil {
		return nil, err
	}

	// 6. Backward through linear projections (WQ, WK, WV)
	dInputQ, err := gqa.wq.Backward(dQProj)
	if err != nil {
		return nil, err
	}
	dInputK, err := gqa.wk.Backward(dKProj)
	if err != nil {
		return nil, err
	}
	dInputV, err := gqa.wv.Backward(dVProj)
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

	return []*tensor.Tensor[T]{dInputTotal}, nil
}
