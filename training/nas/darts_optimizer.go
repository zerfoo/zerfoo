package nas

import (
	"context"
	"errors"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// DARTSOptimizerConfig holds configuration for the DARTS bilevel optimizer.
type DARTSOptimizerConfig[T tensor.Numeric] struct {
	// WeightLR is the learning rate for network weight updates (inner loop).
	WeightLR T
	// AlphaLR is the learning rate for architecture parameter updates (outer loop).
	AlphaLR T
}

// DARTSOptimizer implements bilevel optimization for DARTS (Liu et al. 2019).
// Each step alternates between:
//   - Inner loop: update network weights w using training loss gradient.
//   - Outer loop: update architecture parameters alpha using validation loss gradient.
type DARTSOptimizer[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	layer  *DARTSLayer[T]
	cfg    DARTSOptimizerConfig[T]
}

// NewDARTSOptimizer creates a new DARTS bilevel optimizer.
func NewDARTSOptimizer[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T], layer *DARTSLayer[T], cfg DARTSOptimizerConfig[T]) (*DARTSOptimizer[T], error) {
	if layer == nil {
		return nil, errors.New("nas: DARTSOptimizer requires a non-nil DARTSLayer")
	}

	zero := ops.FromFloat64(0)
	if !ops.GreaterThan(cfg.WeightLR, zero) {
		return nil, errors.New("nas: WeightLR must be positive")
	}
	if !ops.GreaterThan(cfg.AlphaLR, zero) {
		return nil, errors.New("nas: AlphaLR must be positive")
	}

	return &DARTSOptimizer[T]{
		engine: engine,
		ops:    ops,
		layer:  layer,
		cfg:    cfg,
	}, nil
}

// Step performs one bilevel optimization step:
//  1. Inner update: forward on trainInput, compute training loss, backprop, update network weights w.
//  2. Outer update: forward on valInput, compute validation loss, backprop, update architecture alpha.
func (d *DARTSOptimizer[T]) Step(
	ctx context.Context,
	trainInput, trainTarget *tensor.TensorNumeric[T],
	valInput, valTarget *tensor.TensorNumeric[T],
) error {
	// --- Inner loop: update network weights using training data ---
	if err := d.innerStep(ctx, trainInput, trainTarget); err != nil {
		return err
	}

	// --- Outer loop: update alpha using validation data ---
	if err := d.outerStep(ctx, valInput, valTarget); err != nil {
		return err
	}

	return nil
}

// innerStep updates network weights w by gradient descent on training loss.
func (d *DARTSOptimizer[T]) innerStep(ctx context.Context, input, target *tensor.TensorNumeric[T]) error {
	// Collect all network weight parameters from candidates.
	var weightParams []*graph.Parameter[T]
	for _, candidate := range d.layer.candidates {
		weightParams = append(weightParams, candidate.Parameters()...)
	}

	// Clear gradients.
	for _, p := range weightParams {
		p.ClearGradient()
	}

	// Forward pass.
	pred, err := d.layer.Forward(ctx, input)
	if err != nil {
		return err
	}

	// Compute MSE loss and gradient.
	_, dLoss, err := d.mseLoss(pred, target)
	if err != nil {
		return err
	}

	// Backward pass — computes gradients for weight parameters.
	_, err = d.layer.Backward(ctx, types.FullBackprop, dLoss, input)
	if err != nil {
		return err
	}

	// SGD update: w = w - lr * grad.
	for _, p := range weightParams {
		d.sgdUpdate(p, d.cfg.WeightLR)
	}

	return nil
}

// outerStep updates architecture parameters alpha by gradient descent on validation loss.
func (d *DARTSOptimizer[T]) outerStep(ctx context.Context, input, target *tensor.TensorNumeric[T]) error {
	// Clear alpha gradients.
	alphaParam := d.layer.alpha
	alphaParam.ClearGradient()

	// Forward pass on validation data.
	pred, err := d.layer.Forward(ctx, input)
	if err != nil {
		return err
	}

	// Compute MSE loss and gradient.
	_, dLoss, err := d.mseLoss(pred, target)
	if err != nil {
		return err
	}

	// Backward pass — computes gradient for alpha.
	_, err = d.layer.Backward(ctx, types.FullBackprop, dLoss, input)
	if err != nil {
		return err
	}

	// SGD update: alpha = alpha - lr * grad.
	d.sgdUpdate(alphaParam, d.cfg.AlphaLR)

	return nil
}

// sgdUpdate performs w = w - lr * grad on a single parameter.
func (d *DARTSOptimizer[T]) sgdUpdate(p *graph.Parameter[T], lr T) {
	wData := p.Value.Data()
	gData := p.Gradient.Data()
	for i := range wData {
		wData[i] = d.ops.Sub(wData[i], d.ops.Mul(lr, gData[i]))
	}
}

// mseLoss computes mean squared error and its gradient.
// loss = (1/n) * sum((pred - target)^2)
// dLoss/dPred = (2/n) * (pred - target)
func (d *DARTSOptimizer[T]) mseLoss(pred, target *tensor.TensorNumeric[T]) (T, *tensor.TensorNumeric[T], error) {
	pData := pred.Data()
	tData := target.Data()
	n := len(pData)
	nT := d.ops.FromFloat64(float64(n))
	two := d.ops.FromFloat64(2.0)

	var loss T
	gradData := make([]T, n)

	for i := range pData {
		diff := d.ops.Sub(pData[i], tData[i])
		loss = d.ops.Add(loss, d.ops.Mul(diff, diff))
		gradData[i] = d.ops.Div(d.ops.Mul(two, diff), nT)
	}
	loss = d.ops.Div(loss, nT)

	grad, err := tensor.New[T](pred.Shape(), gradData)
	if err != nil {
		var zero T
		return zero, nil, err
	}
	return loss, grad, nil
}
