package core

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/tensor"
)

// RotaryEmbedding applies rotary position embedding to input tensors.
type RotaryEmbedding[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	name        string
	outputShape []int
	// configuration
	base      float64
	maxSeqLen int
	// inner RoPE implementation (lazy-initialized on first Forward)
	inner *embeddings.RotaryPositionalEmbedding[T]
}

// NewRotaryEmbedding creates a new RotaryEmbedding layer.
func NewRotaryEmbedding[T tensor.Numeric](engine compute.Engine[T]) *RotaryEmbedding[T] {
	return &RotaryEmbedding[T]{
		engine: engine,
		name:   "RotaryEmbedding",
		base:   10000.0,
	}
}

// Name returns the name of the layer.
func (r *RotaryEmbedding[T]) Name() string {
	return r.name
}

// SetName sets the name of the layer.
func (r *RotaryEmbedding[T]) SetName(name string) {
	r.name = name
}

// Parameters returns the parameters of the layer (none for RotaryEmbedding).
func (r *RotaryEmbedding[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// OutputShape returns the output shape of the layer.
func (r *RotaryEmbedding[T]) OutputShape() []int {
	return r.outputShape
}

// Forward applies rotary embedding to the input.
func (r *RotaryEmbedding[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		panic("RotaryEmbedding layer requires at least 1 input")
	}

	input := inputs[0]
	r.outputShape = input.Shape()

	// Expect input shape: (batch, seq_len, head_dim)
	if len(r.outputShape) < 3 {
		// Pass-through for non 3D tensors (defensive)
		return input, nil
	}

	seqLen := r.outputShape[1]
	headDim := r.outputShape[2]

	// Initialize inner RoPE if needed or if dimensions changed
	if r.inner == nil || r.maxSeqLen < seqLen || headDim != 0 && r.inner.OutputShape() != nil && len(r.inner.OutputShape()) > 0 && r.inner.OutputShape()[2] != headDim {
		// choose precompute seq len: maxSeqLen if provided, else current seqLen
		precomputeSeq := seqLen
		if r.maxSeqLen > 0 {
			precomputeSeq = r.maxSeqLen
		}

		rope, err := embeddings.NewRotaryPositionalEmbedding[T](context.Background(), r.engine, headDim, precomputeSeq, embeddings.WithRotaryBase(r.base))
		if err != nil {
			return nil, err
		}
		r.inner = rope
	}

	// Delegate to inner RoPE
	return r.inner.Forward(context.Background(), input)
}

// Backward computes the gradients for the RotaryEmbedding layer.
func (r *RotaryEmbedding[T]) Backward(_ context.Context, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		panic("RotaryEmbedding layer requires at least 1 input")
	}

	if r.inner == nil {
		// No-op if Forward never initialized inner; pass-through gradient
		return []*tensor.TensorNumeric[T]{outputGradient}, nil
	}

	dInputs, err := r.inner.Backward(context.Background(), outputGradient)
	if err != nil {
		return nil, err
	}
	return dInputs, nil
}

// OpType returns the operation type of the RotaryEmbedding layer.
func (r *RotaryEmbedding[T]) OpType() string {
	return r.name
}

// Attributes returns nil for the RotaryEmbedding layer.
func (r *RotaryEmbedding[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"rope_base":   r.base,
		"max_seq_len": r.maxSeqLen,
	}
}
