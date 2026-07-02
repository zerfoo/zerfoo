package core

import (
	"context"
	"errors"
	"fmt"
	"sort"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// MoEGateOption configures optional MoEGate behavior.
type MoEGateOption[T tensor.Numeric] func(*MoEGate[T])

// WithSigmoidGating enables sigmoid gating instead of softmax for expert routing.
// When enabled, element-wise sigmoid is applied to logits instead of softmax.
func WithSigmoidGating[T tensor.Numeric]() MoEGateOption[T] {
	return func(g *MoEGate[T]) { g.sigmoidGating = true }
}

// WithRoutingBias sets an optional bias tensor added to logits before gating.
// The bias shape must be [numExperts] and is broadcast across the sequence dimension.
func WithRoutingBias[T tensor.Numeric](bias *tensor.TensorNumeric[T]) MoEGateOption[T] {
	return func(g *MoEGate[T]) { g.routingBias = bias }
}

// MoEGate computes sparse top-k expert routing for Mixture of Experts.
//
// Forward expects exactly two inputs:
//   - hiddenStates [seqLen, modelDim]
//   - gateWeight   [numExperts, modelDim]
//
// Steps:
//  1. logits = hiddenStates @ gateWeight.T  -> [seqLen, numExperts]
//  2. If routingBias is set, add bias to logits.
//  3. probs  = Softmax(logits, axis=1) or Sigmoid(logits) if sigmoidGating is enabled.
//  4. For each token row: pick topK indices by descending probability.
//  5. Normalize the topK scores so each row sums to 1.0.
//
// Returns a [seqLen, topK] tensor of normalized expert weights.
type MoEGate[T tensor.Numeric] struct {
	engine        compute.Engine[T]
	ops           numeric.Arithmetic[T]
	topK          int
	sigmoidGating bool
	routingBias   *tensor.TensorNumeric[T]
	outputShape   []int

	// Cached forward state for backward pass.
	cachedHiddenStates *tensor.TensorNumeric[T]
	cachedGateWeight   *tensor.TensorNumeric[T]
	cachedProbs        []T // softmax/sigmoid output flattened [seqLen * numExperts]
	cachedIndices      [][]int
	cachedWeights      [][]T
	cachedNumExperts   int
}

// NewMoEGate creates a MoEGate layer with the given topK value and optional configuration.
func NewMoEGate[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], topK int, opts ...MoEGateOption[T]) *MoEGate[T] {
	g := &MoEGate[T]{engine: engine, ops: ops, topK: topK}
	for _, opt := range opts {
		opt(g)
	}
	return g
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

	// Add routing bias if present: logits += bias (broadcast across seqLen).
	if g.routingBias != nil {
		logits, err = g.engine.Add(ctx, logits, g.routingBias)
		if err != nil {
			return nil, nil, fmt.Errorf("MoEGate: add routing bias: %w", err)
		}
	}

	var probs *tensor.TensorNumeric[T]
	if g.sigmoidGating {
		// Delegate to the canonical Sigmoid Node (T124.2.3) so the math
		// has a single source of truth shared with layers/activations.
		probs, err = activations.NewSigmoid(g.engine, g.ops).Forward(ctx, logits)
		if err != nil {
			return nil, nil, fmt.Errorf("MoEGate: sigmoid: %w", err)
		}
	} else {
		probs, err = g.engine.Softmax(ctx, logits, 1)
		if err != nil {
			return nil, nil, fmt.Errorf("MoEGate: softmax: %w", err)
		}
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

	// Cache state for backward pass.
	g.cachedHiddenStates = hiddenStates
	g.cachedGateWeight = gateWeight
	g.cachedProbs = make([]T, len(probData))
	copy(g.cachedProbs, probData)
	g.cachedIndices = indices
	g.cachedWeights = weights
	g.cachedNumExperts = numExperts

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

// Backward computes gradients for the MoE gate using the straight-through estimator.
//
// The top-K selection is a discrete operation. STE treats it as identity in the
// backward pass: gradients flow through the softmax as if all experts were selected,
// but only the top-K experts receive non-zero upstream gradient from the MoE output.
//
// Inputs must match the forward call: [hiddenStates, gateWeight].
// outputGradient has shape [seqLen, topK] — gradient w.r.t. the normalized gate weights.
//
// Returns gradients [dHiddenStates, dGateWeight].
func (g *MoEGate[T]) Backward(ctx context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if outputGradient == nil || g.cachedIndices == nil {
		return nil, nil
	}

	hiddenStates := g.cachedHiddenStates
	gateWeight := g.cachedGateWeight
	if hiddenStates == nil || gateWeight == nil {
		return nil, nil
	}

	seqLen := len(g.cachedIndices)
	numExperts := g.cachedNumExperts
	topK := g.topK
	if topK > numExperts {
		topK = numExperts
	}

	zero := g.ops.FromFloat64(0)
	ogData := outputGradient.Data()

	// Step 1: Scatter outputGradient [seqLen, topK] into dProbs [seqLen, numExperts].
	// STE: treat top-K selection as identity — place gradients at selected expert positions.
	// Account for normalization: weights[k] = probs[k] / S where S = sum(top-K probs).
	// Jacobian: d(p_i/S)/d(p_i) = (S - p_i) / S^2
	//           d(p_i/S)/d(p_j) = -p_i / S^2  (for other selected j)
	dProbs := make([]T, seqLen*numExperts)
	for i := range dProbs {
		dProbs[i] = zero
	}

	for t := 0; t < seqLen; t++ {
		tIndices := g.cachedIndices[t]
		probs := g.cachedProbs[t*numExperts : (t+1)*numExperts]

		// Sum of selected probs.
		sumProbs := zero
		for _, idx := range tIndices {
			sumProbs = g.ops.Add(sumProbs, probs[idx])
		}
		sumProbs2 := g.ops.Mul(sumProbs, sumProbs)

		for ki, idxI := range tIndices {
			dOut := ogData[t*topK+ki]
			pI := probs[idxI]
			for _, idxJ := range tIndices {
				pJ := probs[idxJ]
				if idxI == idxJ {
					// d(p_i/S)/d(p_i) = (S - p_i) / S^2
					num := g.ops.Sub(sumProbs, pI)
					dProbs[t*numExperts+idxJ] = g.ops.Add(dProbs[t*numExperts+idxJ],
						g.ops.Mul(dOut, g.ops.Div(num, sumProbs2)))
				} else {
					// d(p_i/S)/d(p_j) = -p_i / S^2
					_ = pJ
					neg := g.ops.Sub(zero, pI)
					dProbs[t*numExperts+idxJ] = g.ops.Add(dProbs[t*numExperts+idxJ],
						g.ops.Mul(dOut, g.ops.Div(neg, sumProbs2)))
				}
			}
		}
	}

	// Step 2: Backprop through softmax.
	// dLogits[t,i] = probs[t,i] * (dProbs[t,i] - dot(probs[t,:], dProbs[t,:]))
	dLogits := make([]T, seqLen*numExperts)
	for t := 0; t < seqLen; t++ {
		probs := g.cachedProbs[t*numExperts : (t+1)*numExperts]
		dot := zero
		for j := 0; j < numExperts; j++ {
			dot = g.ops.Add(dot, g.ops.Mul(probs[j], dProbs[t*numExperts+j]))
		}
		for j := 0; j < numExperts; j++ {
			dLogits[t*numExperts+j] = g.ops.Mul(probs[j], g.ops.Sub(dProbs[t*numExperts+j], dot))
		}
	}

	// Step 3: Backprop through logits = hiddenStates @ gateWeight.T.
	// dHiddenStates = dLogits @ gateWeight  [seqLen, modelDim]
	// dGateWeight = dLogits.T @ hiddenStates [numExperts, modelDim]
	dLogitsTensor, err := tensor.New[T]([]int{seqLen, numExperts}, dLogits)
	if err != nil {
		return nil, fmt.Errorf("MoEGate.Backward: create dLogits tensor: %w", err)
	}

	// dHiddenStates = dLogits @ gateWeight
	dHS, err := g.engine.MatMul(ctx, dLogitsTensor, gateWeight)
	if err != nil {
		return nil, fmt.Errorf("MoEGate.Backward: matmul dHiddenStates: %w", err)
	}

	// dGateWeight = dLogits.T @ hiddenStates
	dLogitsT, err := g.engine.Transpose(ctx, dLogitsTensor, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("MoEGate.Backward: transpose dLogits: %w", err)
	}
	dGW, err := g.engine.MatMul(ctx, dLogitsT, hiddenStates)
	if err != nil {
		return nil, fmt.Errorf("MoEGate.Backward: matmul dGateWeight: %w", err)
	}

	return []*tensor.TensorNumeric[T]{dHS, dGW}, nil
}

// OpType returns "MoEGate".
func (g *MoEGate[T]) OpType() string { return "MoEGate" }

// Attributes returns the gate configuration.
func (g *MoEGate[T]) Attributes() map[string]interface{} {
	attrs := map[string]interface{}{"top_k": g.topK}
	if g.sigmoidGating {
		attrs["sigmoid_gating"] = true
	}
	return attrs
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

	// Cached forward state for backward pass.
	cachedHiddenStates *tensor.TensorNumeric[T]
	cachedGateWeight   *tensor.TensorNumeric[T]
	cachedIndices      [][]int
	cachedWeights      [][]T
	cachedExpertOuts   map[int]*tensor.TensorNumeric[T] // expert index -> batched output
	cachedAssignments  map[int][]expertAssignment[T]    // expert index -> token assignments
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

	// Cache forward state for backward.
	m.cachedHiddenStates = hiddenStates
	m.cachedGateWeight = gateWeight
	m.cachedIndices = indices
	m.cachedWeights = weights
	m.cachedExpertOuts = make(map[int]*tensor.TensorNumeric[T])
	m.cachedAssignments = make(map[int][]expertAssignment[T])

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
			// Cache expert output for backward.
			m.cachedExpertOuts[expertIdx] = expertOut
			m.cachedAssignments[expertIdx] = []expertAssignment[T]{{tokenIdx: 0, weight: weights[0][k]}}

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

		// Cache expert output and assignments for backward.
		m.cachedExpertOuts[eid] = batchOut
		m.cachedAssignments[eid] = assignments

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

// Backward computes gradients for the MixtureOfExperts layer.
//
// The forward computation is:
//
//	output[t] = sum_k(weights[t][k] * expert_k(hiddenStates[t]))
//
// where weights come from softmax + top-K + normalize on router logits.
//
// Backward pass:
//  1. dX from experts: sum_k(weights[k] * dOut) — assuming identity-like expert gradients
//  2. dWeights: for each selected expert k, dot(dOut[t], expert_k(x[t]))
//  3. Route dWeights through the gate backward (softmax + STE for top-K)
//
// Inputs must match forward: [hiddenStates, gateWeight].
// Returns gradients [dHiddenStates, dGateWeight].
func (m *MixtureOfExperts[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if outputGradient == nil || m.cachedIndices == nil {
		return nil, nil
	}

	hiddenStates := m.cachedHiddenStates
	gateWeight := m.cachedGateWeight
	if hiddenStates == nil || gateWeight == nil {
		return nil, nil
	}

	hsShape := hiddenStates.Shape()
	seqLen, modelDim := hsShape[0], hsShape[1]
	topK := m.topK
	zero := m.ops.FromFloat64(0)

	ogData := outputGradient.Data()

	// dX accumulates gradient w.r.t. hiddenStates from the expert outputs.
	dXData := make([]T, seqLen*modelDim)
	for i := range dXData {
		dXData[i] = zero
	}

	// dWeights[t][k] = dot(dOut[t], expertOut[t][k]) for the gate backward.
	dWeightsData := make([]T, seqLen*topK)

	// Pre-compute expert backward passes for each expert that was used.
	// Each expert receives dOut weighted by the routing weight, but for the
	// expert backward we need the unweighted dOut per token.
	// We compute per-expert batched backward once, then scatter results.
	expertDInputs := make(map[int]*tensor.TensorNumeric[T])
	for eid, assignments := range m.cachedAssignments {
		batchSize := len(assignments)
		// Build batched dOut for this expert: for each assigned token,
		// the gradient flowing into the expert is weight * dOut.
		batchDOutData := make([]T, batchSize*modelDim)
		for i, a := range assignments {
			for d := 0; d < modelDim; d++ {
				batchDOutData[i*modelDim+d] = ogData[a.tokenIdx*modelDim+d]
			}
		}
		batchDOut, err := tensor.New[T]([]int{batchSize, modelDim}, batchDOutData)
		if err != nil {
			return nil, fmt.Errorf("MixtureOfExperts.Backward: create batch dOut for expert %d: %w", eid, err)
		}

		// Try expert backward.
		expertGrads, err := m.experts[eid].Backward(ctx, mode, batchDOut)
		if err != nil {
			return nil, fmt.Errorf("MixtureOfExperts.Backward: expert %d backward: %w", eid, err)
		}
		if expertGrads != nil && len(expertGrads) > 0 && expertGrads[0] != nil {
			expertDInputs[eid] = expertGrads[0]
		}
	}

	for t := 0; t < seqLen; t++ {
		for k := 0; k < topK; k++ {
			eid := m.cachedIndices[t][k]
			w := m.cachedWeights[t][k]

			// Find this token's position in the expert's batch output.
			expertOut := m.cachedExpertOuts[eid]
			assignments := m.cachedAssignments[eid]
			var batchIdx int
			for bi, a := range assignments {
				if a.tokenIdx == t {
					batchIdx = bi
					break
				}
			}

			eData := expertOut.Data()

			// dL/dWeight[t][k] = sum_d(dOut[t][d] * expertOut[batchIdx][d])
			dw := zero
			for d := 0; d < modelDim; d++ {
				dOut := ogData[t*modelDim+d]
				eVal := eData[batchIdx*modelDim+d]
				dw = m.ops.Add(dw, m.ops.Mul(dOut, eVal))
			}
			dWeightsData[t*topK+k] = dw

			// dX[t] += weight[t][k] * expert_grad[t]
			// If expert provides backward, use its gradient; otherwise use dOut directly.
			if eDIn, ok := expertDInputs[eid]; ok {
				eDInData := eDIn.Data()
				for d := 0; d < modelDim; d++ {
					dXData[t*modelDim+d] = m.ops.Add(dXData[t*modelDim+d],
						m.ops.Mul(w, eDInData[batchIdx*modelDim+d]))
				}
			} else {
				for d := 0; d < modelDim; d++ {
					dXData[t*modelDim+d] = m.ops.Add(dXData[t*modelDim+d],
						m.ops.Mul(w, ogData[t*modelDim+d]))
				}
			}
		}
	}

	// Build dWeights tensor and pass through gate backward.
	dWeightsTensor, err := tensor.New[T]([]int{seqLen, topK}, dWeightsData)
	if err != nil {
		return nil, fmt.Errorf("MixtureOfExperts.Backward: create dWeights: %w", err)
	}

	gateGrads, err := m.gate.Backward(ctx, mode, dWeightsTensor, hiddenStates, gateWeight)
	if err != nil {
		return nil, fmt.Errorf("MixtureOfExperts.Backward: gate backward: %w", err)
	}

	// Combine dX from experts with dX from gate routing.
	dXFromExperts, err := tensor.New[T](hsShape, dXData)
	if err != nil {
		return nil, fmt.Errorf("MixtureOfExperts.Backward: create dX: %w", err)
	}

	if gateGrads != nil && len(gateGrads) >= 1 && gateGrads[0] != nil {
		dXFromExperts, err = m.engine.Add(ctx, dXFromExperts, gateGrads[0])
		if err != nil {
			return nil, fmt.Errorf("MixtureOfExperts.Backward: add gate dX: %w", err)
		}
	}

	var dGW *tensor.TensorNumeric[T]
	if gateGrads != nil && len(gateGrads) >= 2 {
		dGW = gateGrads[1]
	}

	return []*tensor.TensorNumeric[T]{dXFromExperts, dGW}, nil
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
