// Package hrm implements the Hierarchical Reasoning Model.
package hrm

import (
	"context"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/hrm"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// HRM represents the Hierarchical Reasoning Model.
type HRM[T tensor.Numeric] struct {
	HModule   *hrm.HModule[T]
	LModule   *hrm.LModule[T]
	InputNet  graph.Node[T]
	OutputNet graph.Node[T]
}

// NewHRM creates a new HRM model.
func NewHRM[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	modelDim, ffnDim, inputDim, outputDim int,
	hAttention, lAttention graph.Node[T],
) (*HRM[T], error) {
	hModule, err := hrm.NewHModule[T](engine, ops, modelDim, ffnDim, hAttention)
	if err != nil {
		return nil, err
	}

	lModule, err := hrm.NewLModule[T](engine, ops, modelDim, ffnDim, lAttention)
	if err != nil {
		return nil, err
	}

	inputNet, err := core.NewDense[T]("input_net", engine, ops, inputDim, modelDim)
	if err != nil {
		return nil, err
	}

	outputNet, err := core.NewDense[T]("output_net", engine, ops, modelDim, outputDim)
	if err != nil {
		return nil, err
	}

	return &HRM[T]{
		HModule:   hModule,
		LModule:   lModule,
		InputNet:  inputNet,
		OutputNet: outputNet,
	}, nil
}

// Build wires a minimal static graph for a single-step HRM pass.
// This simplified version ignores the N and T loop structure and just
// composes: input -> InputNet -> Add(hStateIn) -> LModule.Block ->
// Add(hStateIn) -> HModule.Block -> OutputNet.
func (m *HRM[T]) Build(
	builder *graph.Builder[T],
	_N, _T int,
	input, hStateIn, lStateIn graph.Node[T],
) (graph.Node[T], error) {
	// project input to model dim
	projected := builder.AddNode(m.InputNet, input)

	// L module: combine projected input with provided H state
	addLH := builder.AddNode(core.NewAdd[T](m.HModule.Block.Engine()), projected, hStateIn)
	lOut := builder.AddNode(m.LModule.Block, addLH)

	// H module: combine L output with provided H state
	addHL := builder.AddNode(core.NewAdd[T](m.HModule.Block.Engine()), lOut, hStateIn)
	hOut := builder.AddNode(m.HModule.Block, addHL)

	// Output projection
	out := builder.AddNode(m.OutputNet, hOut)
	_ = lStateIn // currently unused in this minimal wiring

	return out, nil
}

// Forward computes the forward pass of the HRM model.
func (m *HRM[T]) Forward(ctx context.Context, nSteps, tSteps int, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	// 1. Input projection
	projectedInput, err := m.InputNet.Forward(ctx, inputs...)
	if err != nil {
		return nil, err
	}

	// Recurrent updates
	for i := 0; i < nSteps; i++ {
		for j := 0; j < tSteps; j++ {
			// L-module update
			_, err = m.LModule.Forward(ctx, m.HModule.HiddenState, projectedInput)
			if err != nil {
				return nil, err
			}
		}
		// H-module update (once per cycle)
		_, err = m.HModule.Forward(ctx, m.LModule.HiddenState)
		if err != nil {
			return nil, err
		}
	}

	// 4. Output projection
	output, err := m.OutputNet.Forward(ctx, m.HModule.HiddenState)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// Parameters returns the parameters of the HRM model.
func (m *HRM[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]

	params = append(params, m.HModule.Parameters()...)
	params = append(params, m.LModule.Parameters()...)
	params = append(params, m.InputNet.Parameters()...)
	params = append(params, m.OutputNet.Parameters()...)

	return params
}
