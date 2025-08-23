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

// HModule represents the high-level recurrent module of the HRM.
type HModule[T tensor.Numeric] struct {
	Block       *transformer.Block[T]
	HiddenState *tensor.TensorNumeric[T]
}

// NewHModule creates a new HModule.
func NewHModule[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, ffnDim int,
	attention graph.Node[T],
	opts ...transformer.BlockOption[T],
) (*HModule[T], error) {
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

	return &HModule[T]{
		Block:       block,
		HiddenState: initialState,
	}, nil
}

// Forward performs a single step of the HModule's computation.
func (m *HModule[T]) Forward(ctx context.Context, lState *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// The H-module update is conditioned on its own previous state and the L-module's state.
	combinedInput, err := m.Block.Engine().Add(ctx, lState, m.HiddenState)
	if err != nil {
		return nil, fmt.Errorf("h_module add lState+hidden: %w", err)
	}

	// Ensure 3D shape for attention block: [batch, seq_len=1, model_dim]
	// If input is 2D [batch, model_dim], temporarily expand dims.
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

	output, err := m.Block.Forward(ctx, combinedInput)
	if err != nil {
		return nil, fmt.Errorf("h_module block forward: %w", err)
	}

	// Squeeze back to 2D if we expanded earlier.
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

// Parameters returns the parameters of the HModule.
func (m *HModule[T]) Parameters() []*graph.Parameter[T] {
	return m.Block.Parameters()
}
