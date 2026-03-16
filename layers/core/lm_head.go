package core

import (
	"context"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// LMHead is a linear layer that maps hidden states to vocabulary logits.
// When tiedWeight is set, it uses MatMulTransposeB to compute the projection
// directly without an explicit transpose, avoiding a 1GB+ temporary allocation
// and a use-after-free when the graph's ref-counting releases the transpose output.
type LMHead[T tensor.Numeric] struct {
	linear     *Linear[T]
	engine     compute.Engine[T]
	tiedWeight *tensor.TensorNumeric[T] // [vocabSize, hiddenDim] from embedding
}

// NewLMHead creates a new LMHead.
func NewLMHead[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], hiddenDim, vocabSize int) (*LMHead[T], error) {
	linear, err := NewLinear[T]("lm_head", engine, ops, hiddenDim, vocabSize)
	if err != nil {
		return nil, err
	}

	return &LMHead[T]{linear: linear, engine: engine}, nil
}

// NewTiedLMHead creates an LMHead that reuses an existing embedding weight
// matrix (shape [vocabSize, hiddenDim]) instead of owning its own weight.
// Forward transposes the weight and performs the projection.
func NewTiedLMHead[T tensor.Numeric](engine compute.Engine[T], embedWeight *tensor.TensorNumeric[T]) *LMHead[T] {
	return &LMHead[T]{engine: engine, tiedWeight: embedWeight}
}

// SetWeights sets the weights of the LMHead. This is useful for sharing weights
// with a token embedding layer.
func (h *LMHead[T]) SetWeights(weights *tensor.TensorNumeric[T]) {
	h.linear.weights.Value = weights
}

// Forward computes the forward pass of the LMHead.
func (h *LMHead[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0]
	originalShape := input.Shape()
	batchSize := originalShape[0]
	seqLen := originalShape[1]
	hiddenDim := originalShape[2]

	reshapedInput, err := h.engine.Reshape(ctx, input, []int{batchSize * seqLen, hiddenDim})
	if err != nil {
		return nil, err
	}

	var output *tensor.TensorNumeric[T]
	var vocabSize int

	if h.tiedWeight != nil {
		// Tied weight is [vocabSize, hiddenDim]. Use MatMulTransposeB to compute
		// C = input * weight^T directly via cuBLAS SgemmNT, avoiding an explicit
		// Transpose allocation. The previous Transpose+MatMul approach caused a
		// use-after-free: the graph's ref-counting released the transposed tensor
		// while the LMHead cache still held a reference, resulting in a null
		// device pointer and cuBLAS status 7 (INTERNAL_ERROR).
		if tb, ok := h.engine.(compute.TransposeBMatMuler[T]); ok {
			output, err = tb.MatMulTransposeB(ctx, reshapedInput, h.tiedWeight)
		} else {
			// Fallback for engines without MatMulTransposeB: transpose each pass.
			transposed, err2 := h.engine.Transpose(ctx, h.tiedWeight, []int{1, 0})
			if err2 != nil {
				return nil, err2
			}
			output, err = h.engine.MatMul(ctx, reshapedInput, transposed)
		}
		if err != nil {
			return nil, err
		}
		vocabSize = h.tiedWeight.Shape()[0]
	} else {
		output, err = h.linear.Forward(ctx, reshapedInput)
		if err != nil {
			return nil, err
		}
		vocabSize = h.linear.OutputShape()[1]
	}

	output, err = h.engine.Reshape(ctx, output, []int{batchSize, seqLen, vocabSize})
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Backward computes the backward pass of the LMHead.
func (h *LMHead[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return h.linear.Backward(ctx, mode, dOut, inputs...)
}

// Parameters returns the parameters of the LMHead.
// Tied LMHead returns nil since the embedding layer owns the weight.
func (h *LMHead[T]) Parameters() []*graph.Parameter[T] {
	if h.tiedWeight != nil {
		return nil
	}
	return h.linear.Parameters()
}

// OutputShape returns the output shape of the LMHead.
func (h *LMHead[T]) OutputShape() []int {
	if h.tiedWeight != nil {
		return nil // shape is determined dynamically from the tied weight
	}
	return h.linear.OutputShape()
}
