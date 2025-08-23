package attention

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/tensor"
)

// AttentionHead implements a single attention head, including linear projections
// for Query, Key, and Value, followed by scaled dot-product attention.
//
//nolint:revive // exported type stutters as attention.AttentionHead; acceptable for clarity
type AttentionHead[T tensor.Numeric] struct {
	engine compute.Engine[T]
	qProj  *core.Dense[T]
	kProj  *core.Dense[T]
	vProj  *core.Dense[T]
	sdpa   *ScaledDotProductAttention[T]
}

// AttentionHeadOptions holds configuration options for AttentionHead.
//
//nolint:revive // exported type stutters as attention.AttentionHeadOptions; acceptable for clarity
type AttentionHeadOptions[T tensor.Numeric] struct {
	// No specific options for now, but kept for consistency.
}

// AttentionHeadOption applies an option to AttentionHeadOptions.
//
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

	// Validate dimensions early to avoid invalid layers
	if inputDim <= 0 {
		panic(fmt.Errorf("invalid inputDim: %d; must be > 0", inputDim))
	}

	if headDim <= 0 {
		panic(fmt.Errorf("invalid headDim: %d; must be > 0", headDim))
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

	if headDim <= 0 {
		return nil, fmt.Errorf("AttentionHead: invalid headDim %d derived from qProj output shape %v", headDim, ah.qProj.OutputShape())
	}

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
// dOut has shape (batch, seq_len, head_dim). inputs[0] has shape (batch, seq_len, input_dim).
func (ah *AttentionHead[T]) Backward(ctx context.Context, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("AttentionHead: expected 1 input, got %d", len(inputs))
	}

	input := inputs[0]
	if len(input.Shape()) != 3 {
		return nil, fmt.Errorf("AttentionHead: input must be 3D, got %v", input.Shape())
	}

	batchSize := input.Shape()[0]
	seqLen := input.Shape()[1]
	inputDim := input.Shape()[2]

	// Backprop through SDPA to obtain dQ, dK, dV
	sdpaGrads, err := ah.sdpa.Backward(ctx, dOut, nil, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: SDPA backward failed: %w", err)
	}

	dQ := sdpaGrads[0] // (batch, seq, headDim)
	dK := sdpaGrads[1]
	dV := sdpaGrads[2]

	// Reshape dQ/dK/dV to 2D for Dense backward: (batch*seq, headDim)
	headDim := dQ.Shape()[2]

	dQ2D, err := dQ.Reshape([]int{batchSize * seqLen, headDim})
	if err != nil {
		return nil, err
	}

	dK2D, err := dK.Reshape([]int{batchSize * seqLen, headDim})
	if err != nil {
		return nil, err
	}

	dV2D, err := dV.Reshape([]int{batchSize * seqLen, headDim})
	if err != nil {
		return nil, err
	}

	// Prepare the 2D input used for Dense forward: (batch*seq, inputDim)
	input2D, err := input.Reshape([]int{batchSize * seqLen, inputDim})
	if err != nil {
		return nil, err
	}

	// Backprop through V, K, Q projections
	vGrads, err := ah.vProj.Backward(ctx, dV2D, input2D)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: V backward failed: %w", err)
	}

	kGrads, err := ah.kProj.Backward(ctx, dK2D, input2D)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: K backward failed: %w", err)
	}

	qGrads, err := ah.qProj.Backward(ctx, dQ2D, input2D)
	if err != nil {
		return nil, fmt.Errorf("AttentionHead: Q backward failed: %w", err)
	}

	// Sum input gradients from the three branches: shapes are (batch*seq, inputDim)
	sumVK, err := ah.engine.Add(ctx, vGrads[0], kGrads[0], nil)
	if err != nil {
		return nil, err
	}

	dInput2D, err := ah.engine.Add(ctx, sumVK, qGrads[0], nil)
	if err != nil {
		return nil, err
	}

	// Reshape back to (batch, seq, inputDim) for upstream
	dInput3D, err := dInput2D.Reshape([]int{batchSize, seqLen, inputDim})
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInput3D}, nil
}
