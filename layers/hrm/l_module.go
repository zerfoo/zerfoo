// Package hrm implements the Hierarchical Reasoning Model.
package hrm

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/transformer"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// LModule represents the low-level recurrent module of the HRM.
// It implements graph.Node so it can be used in a computation graph.
type LModule[T tensor.Numeric] struct {
	Block       *transformer.Block[T]
	HiddenState *tensor.TensorNumeric[T]
	modelDim    int

	// Cached forward intermediates for backward pass.
	fwdCombinedInput *tensor.TensorNumeric[T]
	fwdNeedSqueeze   bool
	fwdOriginalShape []int
}

// NewLModule creates a new LModule.
func NewLModule[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, ffnDim int,
	attention graph.Node[T],
	opts ...transformer.BlockOption[T],
) (*LModule[T], error) {
	block, err := transformer.NewTransformerBlock(engine, ops, modelDim, ffnDim, attention, opts...)
	if err != nil {
		return nil, fmt.Errorf("new transformer block: %w", err)
	}

	initialState, err := tensor.New[T]([]int{1, modelDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("new initial state: %w", err)
	}

	return &LModule[T]{
		Block:       block,
		HiddenState: initialState,
		modelDim:    modelDim,
	}, nil
}

// OpType returns the operation type of the LModule.
func (m *LModule[T]) OpType() string {
	return "LModule"
}

// Attributes returns the attributes of the LModule.
func (m *LModule[T]) Attributes() map[string]any {
	return map[string]any{
		"model_dim": m.modelDim,
	}
}

// OutputShape returns the output shape of the LModule.
func (m *LModule[T]) OutputShape() []int {
	return m.Block.OutputShape()
}

// Forward performs a single step of the LModule's computation.
// inputs[0] is the H-module state (hState), inputs[1] is the projected input.
func (m *LModule[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 2 {
		return nil, fmt.Errorf("l_module: expected at least 2 inputs (hState, projectedInput), got %d", len(inputs))
	}

	hState := inputs[0]
	projectedInput := inputs[1]

	combinedInput, err := m.Block.Engine().Add(ctx, hState, projectedInput)
	if err != nil {
		return nil, fmt.Errorf("l_module add hState+projectedInput: %w", err)
	}

	// Ensure 3D shape for attention block: [batch, seq_len=1, model_dim]
	originalShape := combinedInput.Shape()
	needSqueeze := false

	if len(originalShape) == 2 {
		var expanded *tensor.TensorNumeric[T]

		expanded, err = m.Block.Engine().Reshape(ctx, combinedInput, []int{originalShape[0], 1, originalShape[1]})
		if err != nil {
			return nil, fmt.Errorf("l_module reshape to 3D: %w", err)
		}

		combinedInput = expanded
		needSqueeze = true
	}

	// Cache for backward pass.
	m.fwdCombinedInput = combinedInput
	m.fwdNeedSqueeze = needSqueeze
	m.fwdOriginalShape = originalShape

	output, err := m.Block.Forward(ctx, combinedInput)
	if err != nil {
		return nil, fmt.Errorf("l_module block forward: %w", err)
	}

	if needSqueeze {
		var squeezed *tensor.TensorNumeric[T]

		squeezed, err = m.Block.Engine().Reshape(ctx, output, []int{originalShape[0], originalShape[1]})
		if err != nil {
			return nil, fmt.Errorf("l_module reshape back to 2D: %w", err)
		}

		output = squeezed
	}

	m.HiddenState = output

	return output, nil
}

// Backward computes the gradients of the LModule.
// Returns gradients for both inputs: [dHState, dProjectedInput].
// Since forward combines inputs via Add, both gradients equal the block's input gradient.
func (m *LModule[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// If forward squeezed from 2D→3D, expand gradient back to 3D.
	if m.fwdNeedSqueeze {
		var err error

		dOut, err = m.Block.Engine().Reshape(ctx, dOut, []int{m.fwdOriginalShape[0], 1, m.fwdOriginalShape[1]})
		if err != nil {
			return nil, fmt.Errorf("l_module backward reshape to 3D: %w", err)
		}
	}

	dInput, err := m.Block.Backward(ctx, mode, dOut, m.fwdCombinedInput)
	if err != nil {
		return nil, fmt.Errorf("l_module block backward: %w", err)
	}

	// Squeeze gradient back to original shape if needed.
	if m.fwdNeedSqueeze && len(dInput) > 0 {
		squeezed, reshapeErr := m.Block.Engine().Reshape(ctx, dInput[0], m.fwdOriginalShape)
		if reshapeErr != nil {
			return nil, fmt.Errorf("l_module backward reshape to 2D: %w", reshapeErr)
		}

		dInput[0] = squeezed
	}

	// Add(hState, projectedInput) → gradient flows equally to both inputs.
	if len(dInput) > 0 {
		return []*tensor.TensorNumeric[T]{dInput[0], dInput[0]}, nil
	}

	return dInput, nil
}

// Parameters returns the parameters of the LModule.
func (m *LModule[T]) Parameters() []*graph.Parameter[T] {
	return m.Block.Parameters()
}

// Statically assert that LModule implements graph.Node.
var _ graph.Node[float32] = (*LModule[float32])(nil)
