package residual

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/layers/normalization"
)

// AttnRes implements full Attention Residuals (arXiv:2603.15031).
// Each layer has a learned pseudo-query w_l that attends over all previous
// layer outputs via softmax attention, replacing fixed additive residuals.
//
// Forward computes:
//
//	keys_i = RMSNorm(layerOutput_i)  for each previous layer output
//	logit_i = dot(w_l, keys_i)       per-layer scalar logit
//	alpha = softmax(logits)           attention weights over depth
//	h_l = sum(alpha_i * layerOutput_i)
type AttnRes[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	norm   *normalization.RMSNorm[T]
	query  *graph.Parameter[T] // Learned pseudo-query w_l [modelDim]
}

// NewAttnRes creates a new AttnRes layer.
// modelDim is the hidden dimension of the model.
func NewAttnRes[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim int) (*AttnRes[T], error) {
	if modelDim <= 0 {
		return nil, fmt.Errorf("AttnRes: modelDim must be > 0, got %d", modelDim)
	}

	norm, err := normalization.NewRMSNorm[T](name+"_norm", engine, ops, modelDim)
	if err != nil {
		return nil, fmt.Errorf("AttnRes: failed to create RMSNorm: %w", err)
	}

	// Initialize query parameter with ones (will be trained).
	queryData := make([]T, modelDim)
	for i := range queryData {
		queryData[i] = ops.One()
	}
	queryTensor, err := tensor.New[T]([]int{1, modelDim}, queryData)
	if err != nil {
		return nil, fmt.Errorf("AttnRes: failed to create query tensor: %w", err)
	}

	queryParam, err := graph.NewParameter[T](name+"_query", queryTensor, tensor.New[T])
	if err != nil {
		return nil, fmt.Errorf("AttnRes: failed to create query parameter: %w", err)
	}

	return &AttnRes[T]{
		engine: engine,
		ops:    ops,
		norm:   norm,
		query:  queryParam,
	}, nil
}

// Forward computes the attention-weighted residual combination.
// layerOutputs contains the outputs of all previous layers, each with shape
// compatible for dot product with the query (typically [1, modelDim] or [modelDim]).
func (a *AttnRes[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("AttnRes: requires at least 1 layer output, got 0")
	}

	numLayers := len(inputs)

	// Compute per-layer logits: logit_i = dot(w_l, RMSNorm(v_i))
	// Collect logits into a 1D tensor of shape [numLayers].
	logitData := make([]T, numLayers)

	for i, layerOut := range inputs {
		// Normalize the layer output.
		normed, err := a.norm.Forward(ctx, layerOut)
		if err != nil {
			return nil, fmt.Errorf("AttnRes: RMSNorm failed for layer %d: %w", i, err)
		}

		// Ensure normed is 2D [1, modelDim] for MatMul.
		normed2D := normed
		if len(normed.Shape()) == 1 {
			normed2D, err = normed.Reshape([]int{1, normed.Shape()[0]})
			if err != nil {
				return nil, fmt.Errorf("AttnRes: reshape normed failed for layer %d: %w", i, err)
			}
		}

		// Transpose query from [1, modelDim] to [modelDim, 1] for dot product.
		queryT, err := a.engine.Transpose(ctx, a.query.Value, []int{1, 0})
		if err != nil {
			return nil, fmt.Errorf("AttnRes: transpose query failed: %w", err)
		}

		// MatMul: [1, modelDim] x [modelDim, 1] = [1, 1]
		dot, err := a.engine.MatMul(ctx, normed2D, queryT)
		if err != nil {
			return nil, fmt.Errorf("AttnRes: dot product failed for layer %d: %w", i, err)
		}

		logitData[i] = dot.Data()[0]
	}

	// Build logits tensor [1, numLayers] and apply softmax over the depth dimension.
	logitsTensor, err := tensor.New[T]([]int{1, numLayers}, logitData)
	if err != nil {
		return nil, fmt.Errorf("AttnRes: failed to create logits tensor: %w", err)
	}

	// Softmax over axis 1 (the depth dimension).
	alpha, err := a.engine.Softmax(ctx, logitsTensor, 1)
	if err != nil {
		return nil, fmt.Errorf("AttnRes: softmax failed: %w", err)
	}

	alphaData := alpha.Data()

	// Compute weighted sum: h_l = sum(alpha_i * v_i).
	// Start with alpha_0 * v_0.
	scalar := alphaData[0]
	result, err := a.engine.MulScalar(ctx, inputs[0], scalar)
	if err != nil {
		return nil, fmt.Errorf("AttnRes: MulScalar failed for layer 0: %w", err)
	}

	for i := 1; i < numLayers; i++ {
		scaled, err := a.engine.MulScalar(ctx, inputs[i], alphaData[i])
		if err != nil {
			return nil, fmt.Errorf("AttnRes: MulScalar failed for layer %d: %w", i, err)
		}
		result, err = a.engine.Add(ctx, result, scaled)
		if err != nil {
			return nil, fmt.Errorf("AttnRes: Add failed at layer %d: %w", i, err)
		}
	}

	return result, nil
}

// Backward computes the backward pass of the AttnRes layer.
func (a *AttnRes[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, fmt.Errorf("AttnRes: backward pass not yet implemented")
}

// Parameters returns the trainable parameters of the AttnRes layer.
func (a *AttnRes[T]) Parameters() []*graph.Parameter[T] {
	params := []*graph.Parameter[T]{a.query}
	params = append(params, a.norm.Parameters()...)
	return params
}

// OutputShape returns the output shape of the AttnRes layer.
func (a *AttnRes[T]) OutputShape() []int {
	return nil
}

// Attributes returns the attributes of the AttnRes layer.
func (a *AttnRes[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"model_dim": a.query.Value.Shape()[1],
	}
}

// OpType returns the operation type of the AttnRes layer.
func (a *AttnRes[T]) OpType() string {
	return "AttnRes"
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*AttnRes[float32])(nil)
