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

// HModule represents the high-level recurrent module of the HRM.
// It implements graph.Node so it can be used in a computation graph.
type HModule[T tensor.Numeric] struct {
	Block       *transformer.Block[T]
	HiddenState *tensor.TensorNumeric[T]
	modelDim    int

	// Cached forward intermediates for backward pass.
	fwdCombinedInput *tensor.TensorNumeric[T]
	fwdNeedSqueeze   bool
	fwdOriginalShape []int
}

// NewHModule creates a new HModule.
func NewHModule[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, ffnDim int,
	attention graph.Node[T],
	opts ...transformer.BlockOption[T],
) (*HModule[T], error) {
	block, err := transformer.NewTransformerBlock(engine, ops, modelDim, ffnDim, attention, opts...)
	if err != nil {
		return nil, fmt.Errorf("new transformer block: %w", err)
	}

	initialState, err := tensor.New[T]([]int{1, modelDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("new initial state: %w", err)
	}

	return &HModule[T]{
		Block:       block,
		HiddenState: initialState,
		modelDim:    modelDim,
	}, nil
}

// OpType returns the operation type of the HModule.
func (m *HModule[T]) OpType() string {
	return "HModule"
}

// Attributes returns the attributes of the HModule.
func (m *HModule[T]) Attributes() map[string]any {
	return map[string]any{
		"model_dim": m.modelDim,
	}
}

// OutputShape returns the output shape of the HModule.
func (m *HModule[T]) OutputShape() []int {
	return m.Block.OutputShape()
}

// Forward performs a single step of the HModule's computation.
// inputs[0] is the L-module state (lState).
func (m *HModule[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 1 {
		return nil, fmt.Errorf("h_module: expected at least 1 input (lState), got %d", len(inputs))
	}

	lState := inputs[0]

	combinedInput, err := m.Block.Engine().Add(ctx, lState, m.HiddenState)
	if err != nil {
		return nil, fmt.Errorf("h_module add lState+hidden: %w", err)
	}

	// Ensure 3D shape for attention block: [batch, seq_len=1, model_dim]
	originalShape := combinedInput.Shape()
	needSqueeze := false

	if len(originalShape) == 2 {
		var expanded *tensor.TensorNumeric[T]

		expanded, err = m.Block.Engine().Reshape(ctx, combinedInput, []int{originalShape[0], 1, originalShape[1]})
		if err != nil {
			return nil, fmt.Errorf("h_module reshape to 3D: %w", err)
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
		return nil, fmt.Errorf("h_module block forward: %w", err)
	}

	if needSqueeze {
		var squeezed *tensor.TensorNumeric[T]

		squeezed, err = m.Block.Engine().Reshape(ctx, output, []int{originalShape[0], originalShape[1]})
		if err != nil {
			return nil, fmt.Errorf("h_module reshape back to 2D: %w", err)
		}

		output = squeezed
	}

	m.HiddenState = output

	return output, nil
}

// Backward computes the gradients of the HModule.
func (m *HModule[T]) Backward(ctx context.Context, mode types.BackwardMode, dOut *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	// If forward squeezed from 2D→3D, expand gradient back to 3D.
	if m.fwdNeedSqueeze {
		var err error

		dOut, err = m.Block.Engine().Reshape(ctx, dOut, []int{m.fwdOriginalShape[0], 1, m.fwdOriginalShape[1]})
		if err != nil {
			return nil, fmt.Errorf("h_module backward reshape to 3D: %w", err)
		}
	}

	dInput, err := m.Block.Backward(ctx, mode, dOut, m.fwdCombinedInput)
	if err != nil {
		return nil, fmt.Errorf("h_module block backward: %w", err)
	}

	// Squeeze gradient back to original shape if needed.
	if m.fwdNeedSqueeze && len(dInput) > 0 {
		squeezed, reshapeErr := m.Block.Engine().Reshape(ctx, dInput[0], m.fwdOriginalShape)
		if reshapeErr != nil {
			return nil, fmt.Errorf("h_module backward reshape to 2D: %w", reshapeErr)
		}

		dInput[0] = squeezed
	}

	return dInput, nil
}

// Parameters returns the parameters of the HModule.
func (m *HModule[T]) Parameters() []*graph.Parameter[T] {
	return m.Block.Parameters()
}

// Statically assert that HModule implements graph.Node.
var _ graph.Node[float32] = (*HModule[float32])(nil)
