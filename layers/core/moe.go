package core

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
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
	engine       compute.Engine[T]
	ops          numeric.Arithmetic[T]
	gate         *MoEGate[T]
	experts      []graph.Node[T]
	SharedExpert graph.Node[T] // optional: runs on every token, output added to routed sum
	numExperts   int
	topK         int
	outputShape  []int
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

// expertAssignment records a token's position and routing weight for one expert slot.
type expertAssignment[T tensor.Numeric] struct {
	tokenIdx int
	weight   T
}

// Forward routes tokens to topK experts and returns the weighted sum [seqLen, modelDim].
// For batch sizes > 1, tokens are grouped by expert and processed in batched forward calls
// instead of sequential per-token dispatch.
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

	// Gate routing: topK selection requires data access (no engine.TopK primitive).
	indices, weights, err := m.gate.route(ctx, hiddenStates, gateWeight)
	if err != nil {
		return nil, fmt.Errorf("MixtureOfExperts: gate routing: %w", err)
	}

	// Validate expert indices early.
	for t := 0; t < seqLen; t++ {
		for k := 0; k < m.topK; k++ {
			if indices[t][k] >= len(m.experts) {
				return nil, fmt.Errorf("MixtureOfExperts: expert index %d out of range (have %d experts)", indices[t][k], len(m.experts))
			}
		}
	}

	// Initialize output accumulator [seqLen, modelDim] with zeros.
	outData := make([]T, seqLen*modelDim)
	out, err := tensor.New[T]([]int{seqLen, modelDim}, outData)
	if err != nil {
		return nil, fmt.Errorf("MixtureOfExperts: create output tensor: %w", err)
	}

	// Shared expert: runs on all tokens at once.
	if m.SharedExpert != nil {
		sharedOut, serr := m.SharedExpert.Forward(ctx, hiddenStates)
		if serr != nil {
			return nil, fmt.Errorf("MixtureOfExperts: shared expert forward: %w", serr)
		}
		out, err = m.engine.Add(ctx, out, sharedOut)
		if err != nil {
			return nil, fmt.Errorf("MixtureOfExperts: add shared expert: %w", err)
		}
	}

	// Sequential path for batch_size=1: no grouping overhead needed.
	if seqLen == 1 {
		for k := 0; k < m.topK; k++ {
			expertIdx := indices[0][k]
			expertOut, eerr := m.experts[expertIdx].Forward(ctx, hiddenStates)
			if eerr != nil {
				return nil, fmt.Errorf("MixtureOfExperts: expert %d forward: %w", expertIdx, eerr)
			}
			scaled, serr := m.engine.MulScalar(ctx, expertOut, weights[0][k])
			if serr != nil {
				return nil, fmt.Errorf("MixtureOfExperts: scale expert %d: %w", expertIdx, serr)
			}
			out, err = m.engine.Add(ctx, out, scaled)
			if err != nil {
				return nil, fmt.Errorf("MixtureOfExperts: add expert %d: %w", expertIdx, err)
			}
		}
		m.outputShape = out.Shape()
		return out, nil
	}

	// Batched path: group tokens by expert, process each expert once with a batch.
	expertTokens := make(map[int][]expertAssignment[T])
	for t := 0; t < seqLen; t++ {
		for k := 0; k < m.topK; k++ {
			eid := indices[t][k]
			expertTokens[eid] = append(expertTokens[eid], expertAssignment[T]{
				tokenIdx: t,
				weight:   weights[t][k],
			})
		}
	}

	hsData := hiddenStates.Data()

	for eid, assignments := range expertTokens {
		batchSize := len(assignments)

		// Gather assigned token rows into a batch tensor [batchSize, modelDim].
		batchData := make([]T, batchSize*modelDim)
		for i, a := range assignments {
			copy(batchData[i*modelDim:(i+1)*modelDim], hsData[a.tokenIdx*modelDim:(a.tokenIdx+1)*modelDim])
		}
		batchIn, terr := tensor.New[T]([]int{batchSize, modelDim}, batchData)
		if terr != nil {
			return nil, fmt.Errorf("MixtureOfExperts: create batch tensor for expert %d: %w", eid, terr)
		}

		// Single batched forward through the expert.
		batchOut, eerr := m.experts[eid].Forward(ctx, batchIn)
		if eerr != nil {
			return nil, fmt.Errorf("MixtureOfExperts: expert %d batched forward: %w", eid, eerr)
		}

		// Scatter weighted results back to output.
		eData := batchOut.Data()
		outData = out.Data()
		for i, a := range assignments {
			w := a.weight
			outOff := a.tokenIdx * modelDim
			inOff := i * modelDim
			for d := 0; d < modelDim; d++ {
				outData[outOff+d] = m.ops.Add(outData[outOff+d], m.ops.Mul(eData[inOff+d], w))
			}
		}
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
func (m *MixtureOfExperts[T]) Attributes() map[string]any {
	attrs := map[string]any{
		"num_experts": m.numExperts,
		"top_k":       m.topK,
	}
	if m.SharedExpert != nil {
		attrs["has_shared_expert"] = true
	}
	return attrs
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
