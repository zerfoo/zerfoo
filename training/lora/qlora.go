package lora

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// QLoRATrainer wraps a model whose base Linear weights are NF4-quantized.
// Only LoRA A and B matrices are trained; base weights stay frozen.
// During forward, NF4 weights are dequantized to the working precision on the fly.
type QLoRATrainer[T tensor.Numeric] struct {
	model  Model[T]
	engine compute.Engine[T]
	opt    optimizer.Optimizer[T]
	rank   int
	alpha  float32
}

// NewQLoRATrainer creates a QLoRA trainer. It injects LoRA adapters into the
// model's target modules. Base weights are expected to already be NF4-quantized
// (via NF4Storage). Only the LoRA A and B matrices are trainable.
func NewQLoRATrainer[T tensor.Numeric](
	model Model[T],
	rank int,
	alpha float32,
	targetModules []string,
	engine compute.Engine[T],
	opt optimizer.Optimizer[T],
) (*QLoRATrainer[T], error) {
	if rank <= 0 {
		return nil, fmt.Errorf("qlora: rank must be positive, got %d", rank)
	}
	if len(targetModules) == 0 {
		return nil, fmt.Errorf("qlora: targetModules must not be empty")
	}

	// Inject LoRA adapters into the target modules.
	if err := InjectLoRA[T](model, rank, alpha, targetModules, engine); err != nil {
		return nil, fmt.Errorf("qlora: failed to inject LoRA: %w", err)
	}

	return &QLoRATrainer[T]{
		model:  model,
		engine: engine,
		opt:    opt,
		rank:   rank,
		alpha:  alpha,
	}, nil
}

// TrainableParams returns the LoRA A and B parameters from all injected layers.
func (q *QLoRATrainer[T]) TrainableParams() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	for _, layer := range q.model.Layers() {
		if ll, ok := layer.(*LoraLinear[T]); ok {
			params = append(params, ll.Parameters()...)
		}
	}
	return params
}

// Step performs a single training step: forward pass through the model,
// compute loss (MSE against target), backward pass through LoRA layers,
// and optimizer step on LoRA parameters only.
//
// Returns the scalar loss value.
func (q *QLoRATrainer[T]) Step(
	ctx context.Context,
	input *tensor.TensorNumeric[T],
	target *tensor.TensorNumeric[T],
) (float32, error) {
	// Clear gradients on all LoRA parameters.
	params := q.TrainableParams()
	for _, p := range params {
		p.ClearGradient()
	}

	// Forward pass through all layers sequentially.
	x := input
	var layers []Layer[T]
	for _, layer := range q.model.Layers() {
		layers = append(layers, layer)
	}

	// Store intermediate activations for backward pass.
	activations := make([]*tensor.TensorNumeric[T], len(layers)+1)
	activations[0] = input

	for i, layer := range layers {
		out, err := layer.Forward(ctx, x)
		if err != nil {
			return 0, fmt.Errorf("qlora: forward failed at layer %q: %w", layer.Name(), err)
		}
		activations[i+1] = out
		x = out
	}

	// Compute MSE loss: mean((output - target)^2)
	diff, err := q.engine.Sub(ctx, x, target)
	if err != nil {
		return 0, fmt.Errorf("qlora: loss sub failed: %w", err)
	}
	diffSq, err := q.engine.Mul(ctx, diff, diff)
	if err != nil {
		return 0, fmt.Errorf("qlora: loss mul failed: %w", err)
	}

	// Reduce sum over all axes using explicit positive indices.
	reduced := diffSq
	for dim := len(diffSq.Shape()) - 1; dim >= 0; dim-- {
		reduced, err = q.engine.ReduceSum(ctx, reduced, dim, false)
		if err != nil {
			return 0, fmt.Errorf("qlora: reduce sum failed: %w", err)
		}
	}

	// Get scalar loss.
	lossData := reduced.Data()
	var lossVal float32
	switch v := any(lossData[0]).(type) {
	case float32:
		lossVal = v
	case float64:
		lossVal = float32(v)
	}

	// Compute the number of elements for mean.
	nElem := 1
	for _, d := range target.Shape() {
		nElem *= d
	}
	lossVal /= float32(nElem)

	// dLoss/dOutput = 2 * (output - target) / nElem
	scaleFactor := q.engine.Ops().FromFloat64(2.0 / float64(nElem))
	dLoss, err := q.engine.MulScalar(ctx, diff, scaleFactor)
	if err != nil {
		return 0, fmt.Errorf("qlora: loss grad failed: %w", err)
	}

	// Backward pass through layers in reverse order.
	grad := dLoss
	for i := len(layers) - 1; i >= 0; i-- {
		grads, err := layers[i].Backward(ctx, types.FullBackprop, grad, activations[i])
		if err != nil {
			return 0, fmt.Errorf("qlora: backward failed at layer %q: %w", layers[i].Name(), err)
		}
		if len(grads) > 0 && grads[0] != nil {
			grad = grads[0]
		}
	}

	// Optimizer step on LoRA parameters only.
	if err := q.opt.Step(ctx, params); err != nil {
		return 0, fmt.Errorf("qlora: optimizer step failed: %w", err)
	}

	return lossVal, nil
}

// Rank returns the LoRA rank.
func (q *QLoRATrainer[T]) Rank() int { return q.rank }

// Alpha returns the LoRA alpha scaling factor.
func (q *QLoRATrainer[T]) Alpha() float32 { return q.alpha }
