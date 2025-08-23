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
)

// LModule represents the low-level recurrent module of the HRM.
type LModule[T tensor.Numeric] struct {
	Block       *transformer.Block[T]
	HiddenState *tensor.TensorNumeric[T]
}

// NewLModule creates a new LModule.
func NewLModule[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, ffnDim int,
	attention graph.Node[T],
	opts ...transformer.BlockOption[T],
) (*LModule[T], error) {
	// construct transformer block and initial state
	block, err := transformer.NewTransformerBlock[T](engine, ops, modelDim, ffnDim, attention, opts...)
	if err != nil {
		return nil, fmt.Errorf("new transformer block: %w", err)
	}
	// Initialize hidden state
	initialState, err := tensor.New[T]([]int{1, modelDim}, nil)
	if err != nil {
		return nil, fmt.Errorf("new initial state: %w", err)
	}
	// TODO: Initialize with truncated normal distribution as per the paper.

	return &LModule[T]{
		Block:       block,
		HiddenState: initialState,
	}, nil
}

// Forward performs a single step of the LModule's computation.
func (m *LModule[T]) Forward(
	ctx context.Context,
	hState, projectedInput *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	// The L-module update is conditioned on the H-module's state and the projected input.
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

// Parameters returns the parameters of the LModule.
func (m *LModule[T]) Parameters() []*graph.Parameter[T] {
	return m.Block.Parameters()
}
