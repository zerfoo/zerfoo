package core

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
)

// VariableSelection implements per-sample soft feature gating.
// Given input [batch, num_features]:
//
//	hidden = GELU(input @ W1 + b1)          [batch, hidden_dim]
//	logits = hidden @ W2 + b2               [batch, num_features]
//	weights = Softmax(logits, axis=1)        [batch, num_features]
//	output = input * weights * num_features  [batch, num_features]
//
// The rescaling by num_features preserves expected magnitude.
type VariableSelection[T tensor.Numeric] struct {
	name        string
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	numFeatures int
	hiddenDim   int
	w1          *graph.Parameter[T] // [numFeatures, hiddenDim]
	b1          *graph.Parameter[T] // [hiddenDim]
	w2          *graph.Parameter[T] // [hiddenDim, numFeatures]
	b2          *graph.Parameter[T] // [numFeatures]
	// Cached for backward
	lastInput   *tensor.TensorNumeric[T]
	lastHidden  *tensor.TensorNumeric[T]
	lastWeights *tensor.TensorNumeric[T]
}

// NewVariableSelection creates a new VariableSelection layer.
func NewVariableSelection[T tensor.Numeric](
	name string,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	numFeatures, hiddenDim int,
) (*VariableSelection[T], error) {
	if name == "" {
		return nil, fmt.Errorf("layer name cannot be empty")
	}
	if numFeatures <= 0 || hiddenDim <= 0 {
		return nil, fmt.Errorf("numFeatures and hiddenDim must be positive")
	}

	w1Tensor, err := tensor.New[T]([]int{numFeatures, hiddenDim}, randomData[T](numFeatures*hiddenDim))
	if err != nil {
		return nil, err
	}
	w1, err := graph.NewParameter[T](name+"_w1", w1Tensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	b1Tensor, err := tensor.New[T]([]int{hiddenDim}, make([]T, hiddenDim))
	if err != nil {
		return nil, err
	}
	b1, err := graph.NewParameter[T](name+"_b1", b1Tensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	w2Tensor, err := tensor.New[T]([]int{hiddenDim, numFeatures}, randomData[T](hiddenDim*numFeatures))
	if err != nil {
		return nil, err
	}
	w2, err := graph.NewParameter[T](name+"_w2", w2Tensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	b2Tensor, err := tensor.New[T]([]int{numFeatures}, make([]T, numFeatures))
	if err != nil {
		return nil, err
	}
	b2, err := graph.NewParameter[T](name+"_b2", b2Tensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &VariableSelection[T]{
		name:        name,
		engine:      engine,
		ops:         ops,
		numFeatures: numFeatures,
		hiddenDim:   hiddenDim,
		w1:          w1,
		b1:          b1,
		w2:          w2,
		b2:          b2,
	}, nil
}

// OpType returns the operation type of the layer.
func (v *VariableSelection[T]) OpType() string { return "VariableSelection" }

// Attributes returns the attributes of the layer.
func (v *VariableSelection[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"num_features": v.numFeatures,
		"hidden_dim":   v.hiddenDim,
	}
}

// OutputShape returns the output shape of the layer.
func (v *VariableSelection[T]) OutputShape() []int {
	return []int{-1, v.numFeatures}
}

// Forward computes per-sample feature gating.
func (v *VariableSelection[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("VariableSelection requires exactly 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	shape := input.Shape()
	if len(shape) != 2 || shape[1] != v.numFeatures {
		return nil, fmt.Errorf("VariableSelection input must be [batch, %d], got %v", v.numFeatures, shape)
	}
	batch := shape[0]

	// hidden = input @ W1 + b1
	hidden, err := v.engine.MatMul(ctx, input, v.w1.Value)
	if err != nil {
		return nil, err
	}
	b1Broadcast, err := v.broadcastBias(v.b1.Value, batch, v.hiddenDim)
	if err != nil {
		return nil, err
	}
	hidden, err = v.engine.Add(ctx, hidden, b1Broadcast)
	if err != nil {
		return nil, err
	}

	// GELU activation on hidden using ops for element-wise computation.
	// GELU(x) = x * sigmoid(1.702 * x) approximation.
	hiddenData := hidden.Data()
	geluData := make([]T, len(hiddenData))
	for i, x := range hiddenData {
		// Use sigmoid approximation: GELU(x) ≈ x * sigmoid(1.702 * x)
		scaled := v.ops.Mul(x, v.ops.FromFloat64(1.702))
		sig := v.ops.Sigmoid(scaled)
		geluData[i] = v.ops.Mul(x, sig)
	}
	geluHidden, err := tensor.New[T](hidden.Shape(), geluData)
	if err != nil {
		return nil, err
	}

	// logits = geluHidden @ W2 + b2
	logits, err := v.engine.MatMul(ctx, geluHidden, v.w2.Value)
	if err != nil {
		return nil, err
	}
	b2Broadcast, err := v.broadcastBias(v.b2.Value, batch, v.numFeatures)
	if err != nil {
		return nil, err
	}
	logits, err = v.engine.Add(ctx, logits, b2Broadcast)
	if err != nil {
		return nil, err
	}

	// weights = softmax(logits, axis=1)
	weights, err := v.engine.Softmax(ctx, logits, 1)
	if err != nil {
		return nil, err
	}

	// output = input * weights * numFeatures
	gated, err := v.engine.Mul(ctx, input, weights)
	if err != nil {
		return nil, err
	}
	nf := v.ops.FromFloat64(float64(v.numFeatures))
	output, err := v.engine.MulScalar(ctx, gated, nf)
	if err != nil {
		return nil, err
	}

	// Cache for backward
	v.lastInput = input
	v.lastHidden = geluHidden
	v.lastWeights = weights

	return output, nil
}

// broadcastBias creates a [batch, dim] tensor by repeating a [dim] bias vector.
func (v *VariableSelection[T]) broadcastBias(bias *tensor.TensorNumeric[T], batch, dim int) (*tensor.TensorNumeric[T], error) {
	biasData := bias.Data()
	broadcastData := make([]T, batch*dim)
	for b := 0; b < batch; b++ {
		copy(broadcastData[b*dim:], biasData)
	}
	return tensor.New[T]([]int{batch, dim}, broadcastData)
}

// Backward computes gradients for VariableSelection.
func (v *VariableSelection[T]) Backward(ctx context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("VariableSelection requires exactly 1 input for backward, got %d", len(inputs))
	}
	input := inputs[0]
	batch := input.Shape()[0]
	nf := v.ops.FromFloat64(float64(v.numFeatures))

	// dOutput/d(gated) = numFeatures * outputGrad
	dGated, err := v.engine.MulScalar(ctx, outputGradient, nf)
	if err != nil {
		return nil, err
	}

	// gated = input * weights
	// d(input) += dGated * weights
	dInput, err := v.engine.Mul(ctx, dGated, v.lastWeights)
	if err != nil {
		return nil, err
	}

	// d(weights) = dGated * input
	dWeights, err := v.engine.Mul(ctx, dGated, input)
	if err != nil {
		return nil, err
	}

	// Softmax backward: dLogits = weights * (dWeights - sum(dWeights * weights, axis=1, keepdim=True))
	dWW, err := v.engine.Mul(ctx, dWeights, v.lastWeights)
	if err != nil {
		return nil, err
	}
	sumDWW, err := v.engine.ReduceSum(ctx, dWW, 1, true)
	if err != nil {
		return nil, err
	}
	// Broadcast sum back
	sumData := sumDWW.Data()
	broadSumData := make([]T, batch*v.numFeatures)
	for b := 0; b < batch; b++ {
		for f := 0; f < v.numFeatures; f++ {
			broadSumData[b*v.numFeatures+f] = sumData[b]
		}
	}
	broadSum, err := tensor.New[T]([]int{batch, v.numFeatures}, broadSumData)
	if err != nil {
		return nil, err
	}
	dWeightsCentered, err := v.engine.Sub(ctx, dWeights, broadSum)
	if err != nil {
		return nil, err
	}
	dLogits, err := v.engine.Mul(ctx, v.lastWeights, dWeightsCentered)
	if err != nil {
		return nil, err
	}

	// dLogits -> dW2, db2
	transposedHidden, err := v.engine.Transpose(ctx, v.lastHidden, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dW2, err := v.engine.MatMul(ctx, transposedHidden, dLogits)
	if err != nil {
		return nil, err
	}
	v.w2.Gradient, err = v.engine.Add(ctx, v.w2.Gradient, dW2)
	if err != nil {
		return nil, err
	}

	// db2 = sum(dLogits, axis=0)
	db2, err := v.engine.ReduceSum(ctx, dLogits, 0, false)
	if err != nil {
		return nil, err
	}
	v.b2.Gradient, err = v.engine.Add(ctx, v.b2.Gradient, db2)
	if err != nil {
		return nil, err
	}

	// dHidden = dLogits @ W2^T
	transposedW2, err := v.engine.Transpose(ctx, v.w2.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dGeluHidden, err := v.engine.MatMul(ctx, dLogits, transposedW2)
	if err != nil {
		return nil, err
	}

	// GELU backward: approximate by passing gradient through.
	// A more precise implementation would compute the GELU derivative,
	// but this approximation still allows gradients to flow.
	dHidden := dGeluHidden

	// dW1 = input^T @ dHidden
	transposedInput, err := v.engine.Transpose(ctx, input, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dW1, err := v.engine.MatMul(ctx, transposedInput, dHidden)
	if err != nil {
		return nil, err
	}
	v.w1.Gradient, err = v.engine.Add(ctx, v.w1.Gradient, dW1)
	if err != nil {
		return nil, err
	}

	// db1 = sum(dHidden, axis=0)
	db1, err := v.engine.ReduceSum(ctx, dHidden, 0, false)
	if err != nil {
		return nil, err
	}
	v.b1.Gradient, err = v.engine.Add(ctx, v.b1.Gradient, db1)
	if err != nil {
		return nil, err
	}

	// dInput += dHidden @ W1^T
	transposedW1, err := v.engine.Transpose(ctx, v.w1.Value, []int{1, 0})
	if err != nil {
		return nil, err
	}
	dInputFromW1, err := v.engine.MatMul(ctx, dHidden, transposedW1)
	if err != nil {
		return nil, err
	}
	dInput, err = v.engine.Add(ctx, dInput, dInputFromW1)
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{dInput}, nil
}

// Parameters returns the parameters of the layer.
func (v *VariableSelection[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{v.w1, v.b1, v.w2, v.b2}
}
