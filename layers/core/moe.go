package core

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// MoEGate computes sparse top-k expert routing for Mixture of Experts.
//
// Forward expects exactly two inputs:
//   - hiddenStates [seqLen, modelDim]
//   - gateWeight   [numExperts, modelDim]
//
// Steps:
//  1. logits = hiddenStates @ gateWeight.T  -> [seqLen, numExperts]
//  2. probs  = Softmax(logits, axis=1)       -> [seqLen, numExperts]
//  3. For each token row: pick topK indices by descending probability.
//  4. Normalize the topK scores so each row sums to 1.0.
//
// Returns a [seqLen, topK] tensor of normalized expert weights.
type MoEGate[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	topK        int
	outputShape []int
}

// NewMoEGate creates a MoEGate layer with the given topK value.
func NewMoEGate[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], topK int) *MoEGate[T] {
	return &MoEGate[T]{engine: engine, ops: ops, topK: topK}
}

// route returns (expertIndices [seqLen][topK], expertWeights [seqLen][topK]).
// It is called by both Forward and MixtureOfExperts.
func (g *MoEGate[T]) route(
	ctx context.Context,
	hiddenStates, gateWeight *tensor.TensorNumeric[T],
) ([][]int, [][]T, error) {
	gwT, err := g.engine.Transpose(ctx, gateWeight, []int{1, 0})
	if err != nil {
		return nil, nil, fmt.Errorf("MoEGate: transpose gateWeight: %w", err)
	}
	logits, err := g.engine.MatMul(ctx, hiddenStates, gwT)
	if err != nil {
		return nil, nil, fmt.Errorf("MoEGate: matmul: %w", err)
	}
	probs, err := g.engine.Softmax(ctx, logits, 1)
	if err != nil {
		return nil, nil, fmt.Errorf("MoEGate: softmax: %w", err)
	}

	probData := probs.Data()
	seqLen := probs.Shape()[0]
	numExperts := probs.Shape()[1]
	topK := g.topK
	if topK > numExperts {
		topK = numExperts
	}

	indices := make([][]int, seqLen)
	weights := make([][]T, seqLen)

	for t := 0; t < seqLen; t++ {
		rowData := probData[t*numExperts : (t+1)*numExperts]

		idxs := make([]int, numExperts)
		for i := range idxs {
			idxs[i] = i
		}
		sort.Slice(idxs, func(a, b int) bool {
			return g.ops.GreaterThan(rowData[idxs[a]], rowData[idxs[b]])
		})

		topIdxs := make([]int, topK)
		copy(topIdxs, idxs[:topK])

		topWeights := make([]T, topK)
		rowSum := g.ops.FromFloat64(0)
		for k, idx := range topIdxs {
			topWeights[k] = rowData[idx]
			rowSum = g.ops.Add(rowSum, topWeights[k])
		}
		for k := range topWeights {
			topWeights[k] = g.ops.Div(topWeights[k], rowSum)
		}

		indices[t] = topIdxs
		weights[t] = topWeights
	}

	return indices, weights, nil
}

// Forward returns normalized expert weights shaped [seqLen, topK].
func (g *MoEGate[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("MoEGate requires exactly 2 inputs (hiddenStates, gateWeight), got %d", len(inputs))
	}

	indices, weights, err := g.route(ctx, inputs[0], inputs[1])
	if err != nil {
		return nil, err
	}

	seqLen := len(indices)
	topK := g.topK
	outData := make([]T, seqLen*topK)
	for t := 0; t < seqLen; t++ {
		for k := 0; k < topK; k++ {
			outData[t*topK+k] = weights[t][k]
		}
	}

	out, err := tensor.New[T]([]int{seqLen, topK}, outData)
	if err != nil {
		return nil, fmt.Errorf("MoEGate: create output tensor: %w", err)
	}
	g.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only).
func (g *MoEGate[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "MoEGate".
func (g *MoEGate[T]) OpType() string { return "MoEGate" }

// Attributes returns the gate configuration.
func (g *MoEGate[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{"top_k": g.topK}
}

// OutputShape returns the output shape from the last forward call.
func (g *MoEGate[T]) OutputShape() []int { return g.outputShape }

// Parameters returns nil (no trainable parameters).
func (g *MoEGate[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildMoEGate constructs a MoEGate layer from ZMF attributes.
// Required attribute: "top_k" (int or int64).
func BuildMoEGate[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	topKAttr, ok := attributes["top_k"]
	if !ok {
		return nil, errors.New("MoEGate: missing required attribute 'top_k'")
	}
	var topK int
	switch v := topKAttr.(type) {
	case int:
		topK = v
	case int64:
		topK = int(v)
	default:
		return nil, fmt.Errorf("MoEGate: attribute 'top_k' has unsupported type %T", topKAttr)
	}
	return NewMoEGate(engine, ops, topK), nil
}

// Statically assert that MoEGate implements graph.Node.
var _ graph.Node[float32] = (*MoEGate[float32])(nil)

// MixtureOfExperts routes each token to topK experts selected by MoEGate and
// returns the weighted sum of expert outputs.
//
// Forward expects exactly two inputs:
//   - hiddenStates [seqLen, modelDim]
//   - gateWeight   [numExperts, modelDim]
//
// Experts must be set at construction time as graph.Node[T] instances.
// Tech debt: ZMF sub-graph loading is not yet supported; experts are not
// populated by BuildMixtureOfExperts.
type MixtureOfExperts[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	gate        *MoEGate[T]
	experts     []graph.Node[T]
	numExperts  int
	topK        int
	outputShape []int
}

// NewMixtureOfExperts creates a MixtureOfExperts layer.
func NewMixtureOfExperts[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	gate *MoEGate[T],
	experts []graph.Node[T],
	numExperts, topK int,
) *MixtureOfExperts[T] {
	return &MixtureOfExperts[T]{
		engine:     engine,
		ops:        ops,
		gate:       gate,
		experts:    experts,
		numExperts: numExperts,
		topK:       topK,
	}
}

// Forward routes tokens to topK experts and returns the weighted sum [seqLen, modelDim].
func (m *MixtureOfExperts[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) < 2 {
		return nil, fmt.Errorf("MixtureOfExperts requires 2 inputs (hiddenStates, gateWeight), got %d", len(inputs))
	}
	if len(m.experts) == 0 {
		return nil, errors.New("MixtureOfExperts: no experts configured (tech debt: ZMF sub-graph loading not yet supported)")
	}

	hiddenStates, gateWeight := inputs[0], inputs[1]
	hsShape := hiddenStates.Shape()
	if len(hsShape) < 2 {
		return nil, fmt.Errorf("MixtureOfExperts: hiddenStates must be 2D [seqLen, modelDim], got shape %v", hsShape)
	}
	seqLen, modelDim := hsShape[0], hsShape[1]
	hsData := hiddenStates.Data()

	indices, weights, err := m.gate.route(ctx, hiddenStates, gateWeight)
	if err != nil {
		return nil, fmt.Errorf("MixtureOfExperts: gate routing: %w", err)
	}

	outData := make([]T, seqLen*modelDim)
	for t := 0; t < seqLen; t++ {
		tokenSlice := make([]T, modelDim)
		copy(tokenSlice, hsData[t*modelDim:(t+1)*modelDim])
		token, terr := tensor.New[T]([]int{1, modelDim}, tokenSlice)
		if terr != nil {
			return nil, fmt.Errorf("MixtureOfExperts: create token tensor: %w", terr)
		}

		for k := 0; k < m.topK; k++ {
			expertIdx := indices[t][k]
			if expertIdx >= len(m.experts) {
				return nil, fmt.Errorf("MixtureOfExperts: expert index %d out of range (have %d experts)", expertIdx, len(m.experts))
			}
			expertOut, eerr := m.experts[expertIdx].Forward(ctx, token)
			if eerr != nil {
				return nil, fmt.Errorf("MixtureOfExperts: expert %d forward: %w", expertIdx, eerr)
			}
			expertData := expertOut.Data()
			w := weights[t][k]
			for d := 0; d < modelDim && d < len(expertData); d++ {
				outData[t*modelDim+d] = m.ops.Add(outData[t*modelDim+d], m.ops.Mul(w, expertData[d]))
			}
		}
	}

	out, err := tensor.New[T]([]int{seqLen, modelDim}, outData)
	if err != nil {
		return nil, fmt.Errorf("MixtureOfExperts: create output tensor: %w", err)
	}
	m.outputShape = out.Shape()
	return out, nil
}

// Backward returns nil (inference-only).
func (m *MixtureOfExperts[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// OpType returns "MixtureOfExperts".
func (m *MixtureOfExperts[T]) OpType() string { return "MixtureOfExperts" }

// Attributes returns the layer configuration.
func (m *MixtureOfExperts[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_experts": m.numExperts,
		"top_k":       m.topK,
	}
}

// OutputShape returns the output shape from the last forward call.
func (m *MixtureOfExperts[T]) OutputShape() []int { return m.outputShape }

// Parameters returns nil (no trainable parameters).
func (m *MixtureOfExperts[T]) Parameters() []*graph.Parameter[T] { return nil }

// BuildMixtureOfExperts constructs a MixtureOfExperts layer from ZMF attributes.
// Required attributes: "num_experts" (int or int64), "top_k" (int or int64).
// Note: Expert sub-graphs are not populated (tech debt: ZMF sub-graph support
// not yet implemented). Experts must be injected manually for real use.
func BuildMixtureOfExperts[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	numExpertsAttr, ok := attributes["num_experts"]
	if !ok {
		return nil, errors.New("MixtureOfExperts: missing required attribute 'num_experts'")
	}
	var numExperts int
	switch v := numExpertsAttr.(type) {
	case int:
		numExperts = v
	case int64:
		numExperts = int(v)
	default:
		return nil, fmt.Errorf("MixtureOfExperts: attribute 'num_experts' has unsupported type %T", numExpertsAttr)
	}

	topKAttr, ok := attributes["top_k"]
	if !ok {
		return nil, errors.New("MixtureOfExperts: missing required attribute 'top_k'")
	}
	var topK int
	switch v := topKAttr.(type) {
	case int:
		topK = v
	case int64:
		topK = int(v)
	default:
		return nil, fmt.Errorf("MixtureOfExperts: attribute 'top_k' has unsupported type %T", topKAttr)
	}

	gate := NewMoEGate[T](engine, ops, topK)
	return NewMixtureOfExperts(engine, ops, gate, nil, numExperts, topK), nil
}

// Statically assert that MixtureOfExperts implements graph.Node.
var _ graph.Node[float32] = (*MixtureOfExperts[float32])(nil)
