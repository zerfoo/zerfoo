package attention

import (
	"context"
	"errors"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/tensor"
)

// AttentionHead implements a single attention head, including linear projections
// for Query, Key, and Value, followed by scaled dot-product attention.
//nolint:revive // exported type stutters as attention.AttentionHead; acceptable for clarity
type AttentionHead[T tensor.Numeric] struct {
	engine compute.Engine[T]
	qProj  *core.Dense[T]
	kProj  *core.Dense[T]
	vProj  *core.Dense[T]
	sdpa   *ScaledDotProductAttention[T]
}

// AttentionHeadOptions holds configuration options for AttentionHead.
//nolint:revive // exported type stutters as attention.AttentionHeadOptions; acceptable for clarity
type AttentionHeadOptions[T tensor.Numeric] struct {
	// No specific options for now, but kept for consistency.
}

// AttentionHeadOption applies an option to AttentionHeadOptions.
//nolint:revive // exported type stutters as attention.AttentionHeadOption; acceptable for clarity
type AttentionHeadOption[T tensor.Numeric] func(*AttentionHeadOptions[T])

// NewAttentionHead creates a new AttentionHead instance.
// inputDim is the dimension of the input features.
// headDim is the dimension of the query, key, and value vectors for this head.
func NewAttentionHead[T tensor.Numeric](engine compute.Engine[T], inputDim, headDim int, opts ...AttentionHeadOption[T]) *AttentionHead[T] {
	options := &AttentionHeadOptions[T]{}
	for _, opt := range opts {
		opt(options)
	}

	// Pass engine and its arithmetic operations to Dense layers
	qProj, err := core.NewDense[T]("q_proj", engine, engine.Ops(), inputDim, headDim)
	if err != nil {
		panic(fmt.Errorf("failed to create Q projection: %w", err))
	}
	kProj, err := core.NewDense[T]("k_proj", engine, engine.Ops(), inputDim, headDim)
	if err != nil {
		panic(fmt.Errorf("failed to create K projection: %w", err))
	}
	vProj, err := core.NewDense[T]("v_proj", engine, engine.Ops(), inputDim, headDim)
	if err != nil {
		panic(fmt.Errorf("failed to create V projection: %w", err))
	}

	return &AttentionHead[T]{
		engine: engine,
		qProj:  qProj,
		kProj:  kProj,
		vProj:  vProj,
		sdpa:   NewScaledDotProductAttention[T](engine, headDim),
	}
}

// Forward computes the output of the attention head.
// input is expected to be a 3D tensor (batch_size, seq_len, input_dim).
func (ah *AttentionHead[T]) Forward(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// Ensure input is 3D: (batch_size, seq_len, input_dim)
	if len(input.Shape()) != 3 {
		return nil, fmt.Errorf("AttentionHead: input must be a 3D tensor (batch_size, seq_len, input_dim), got shape %v", input.Shape())
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	inputDim := input.Shape()[2]

	// Reshape input for Dense layers: (batch_size * seq_len, input_dim)
	// This is because Dense layers expect 2D input (batch_size, input_dim)
	reshapedInput, err := input.Reshape([]int{batchSize * seqLen, inputDim})
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: failed to reshape input for projections: %w", err)
	}

	// Apply linear projections
	q, err := ah.qProj.Forward(ctx, reshapedInput)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: Q projection failed: %w", err)
	}
	k, err := ah.kProj.Forward(ctx, reshapedInput)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: K projection failed: %w", err)
	}
	v, err := ah.vProj.Forward(ctx, reshapedInput)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: V projection failed: %w", err)
	}

	// Reshape Q, K, V back to 3D: (batch_size, seq_len, head_dim)
	headDim := ah.qProj.OutputShape()[1] // Get headDim from the output shape of Q projection

	qReshaped, err := q.Reshape([]int{batchSize, seqLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: failed to reshape Q back to 3D: %w", err)
	}
	kReshaped, err := k.Reshape([]int{batchSize, seqLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: failed to reshape K back to 3D: %w", err)
	}
	vReshaped, err := v.Reshape([]int{batchSize, seqLen, headDim})
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: failed to reshape V back to 3D: %w", err)
	}

	// Perform scaled dot-product attention
	output, err := ah.sdpa.Forward(ctx, qReshaped, kReshaped, vReshaped, nil)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: scaled dot-product attention failed: %w", err)
	}

	return output, nil
}

// Parameters returns all trainable parameters of the AttentionHead.
func (ah *AttentionHead[T]) Parameters() []graph.Parameter[T] {
	params := []graph.Parameter[T]{}
	// Convert []*graph.Parameter[T] to []graph.Parameter[T]
	for _, p := range ah.qProj.Parameters() {
		params = append(params, *p)
	}
	for _, p := range ah.kProj.Parameters() {
		params = append(params, *p)
	}
	for _, p := range ah.vProj.Parameters() {
		params = append(params, *p)
	}

	return params
}

// OutputShape returns the output shape of the AttentionHead.
// It assumes the input shape is (batch_size, seq_len, input_dim).
// The output shape will be (batch_size, seq_len, head_dim).
func (ah *AttentionHead[T]) OutputShape() []int {
	// The output shape is determined by the output of the SDPA, which is (batch_size, seq_len, head_dim)
	// We can get head_dim from any of the projection layers' output shape.
	return []int{0, 0, ah.qProj.OutputShape()[1]} // 0 for batch_size and seq_len as they are dynamic
}

// Backward computes the gradients for the AttentionHead.
func (ah *AttentionHead[T]) Backward(_ context.Context, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// This is a placeholder. The actual backward pass for attention is complex
	// and involves backpropagating through SDPA and then through the linear projections.
	return nil, errors.New("AttentionHead backward pass not yet implemented")
}
