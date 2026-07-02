package loss

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// BCELoss calculates binary cross-entropy loss between predictions and targets.
// BCE(y, p) = -[y*log(p) + (1-y)*log(1-p)]
// Predictions are clamped to [eps, 1-eps] for numerical stability.
type BCELoss[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Cached tensors for backward pass
	predictions *tensor.TensorNumeric[T]
	targets     *tensor.TensorNumeric[T]
}

// NewBCELoss creates a new BCELoss loss function.
func NewBCELoss[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *BCELoss[T] {
	return &BCELoss[T]{engine: engine, ops: ops}
}

// clampOp returns a unary function that restricts values to [eps, 1-eps].
func (b *BCELoss[T]) clampOp() func(T) T {
	eps := b.ops.FromFloat64(1e-7)
	one := b.ops.One()
	oneMinusEps := b.ops.Sub(one, eps)
	return func(p T) T {
		if b.ops.GreaterThan(eps, p) {
			return eps
		}
		if b.ops.GreaterThan(p, oneMinusEps) {
			return oneMinusEps
		}
		return p
	}
}

// Forward computes the mean binary cross-entropy loss.
func (b *BCELoss[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("BCELoss expects 2 inputs, got %d", len(inputs))
	}
	predictions := inputs[0]
	targets := inputs[1]

	b.predictions = predictions
	b.targets = targets

	// Clamp predictions to [eps, 1-eps] for numerical stability.
	pClamped, err := b.engine.UnaryOp(ctx, predictions, b.clampOp())
	if err != nil {
		return nil, err
	}

	// ones tensor for (1-y) and (1-p) computations
	ones, err := tensor.New[T](targets.Shape(), nil)
	if err != nil {
		return nil, err
	}
	if err := b.engine.Fill(ctx, ones, b.ops.One()); err != nil {
		return nil, err
	}

	// log(p)
	logP, err := b.engine.Log(ctx, pClamped)
	if err != nil {
		return nil, err
	}

	// 1 - p
	oneMinusP, err := b.engine.Sub(ctx, ones, pClamped, nil)
	if err != nil {
		return nil, err
	}

	// log(1 - p)
	logOneMinusP, err := b.engine.Log(ctx, oneMinusP)
	if err != nil {
		return nil, err
	}

	// y * log(p)
	yLogP, err := b.engine.Mul(ctx, targets, logP, nil)
	if err != nil {
		return nil, err
	}

	// 1 - y
	oneMinusY, err := b.engine.Sub(ctx, ones, targets, nil)
	if err != nil {
		return nil, err
	}

	// (1-y) * log(1-p)
	oneMinusYLogOneMinusP, err := b.engine.Mul(ctx, oneMinusY, logOneMinusP, nil)
	if err != nil {
		return nil, err
	}

	// y*log(p) + (1-y)*log(1-p)
	sum, err := b.engine.Add(ctx, yLogP, oneMinusYLogOneMinusP, nil)
	if err != nil {
		return nil, err
	}

	// mean of the sum
	mean, err := b.engine.ReduceMean(ctx, sum, -1, false)
	if err != nil {
		return nil, err
	}

	// negate: BCE = -mean(y*log(p) + (1-y)*log(1-p))
	negOne := b.ops.FromFloat64(-1)
	loss, err := b.engine.MulScalar(ctx, mean, negOne)
	if err != nil {
		return nil, err
	}

	return loss, nil
}

// Backward computes the gradients for BCELoss with respect to predictions.
// Gradient: -(y/p - (1-y)/(1-p)) / N, chained with upstream dOut.
func (b *BCELoss[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	preds := b.predictions
	targs := b.targets
	if len(inputs) > 0 {
		preds = inputs[0]
		if len(inputs) > 1 {
			targs = inputs[1]
		}
	}
	if preds == nil || targs == nil {
		return nil, graph.ErrInvalidInputCount
	}

	// Clamp predictions for numerical stability.
	pClamped, err := b.engine.UnaryOp(ctx, preds, b.clampOp())
	if err != nil {
		return nil, err
	}

	// ones tensor
	ones, err := tensor.New[T](targs.Shape(), nil)
	if err != nil {
		return nil, err
	}
	if err := b.engine.Fill(ctx, ones, b.ops.One()); err != nil {
		return nil, err
	}

	// y / p
	yOverP, err := b.engine.Div(ctx, targs, pClamped, nil)
	if err != nil {
		return nil, err
	}

	// 1 - y
	oneMinusY, err := b.engine.Sub(ctx, ones, targs, nil)
	if err != nil {
		return nil, err
	}

	// 1 - p
	oneMinusP, err := b.engine.Sub(ctx, ones, pClamped, nil)
	if err != nil {
		return nil, err
	}

	// (1-y) / (1-p)
	oneMinusYOverOneMinusP, err := b.engine.Div(ctx, oneMinusY, oneMinusP, nil)
	if err != nil {
		return nil, err
	}

	// ((1-y)/(1-p) - y/p)
	rawGrad, err := b.engine.Sub(ctx, oneMinusYOverOneMinusP, yOverP, nil)
	if err != nil {
		return nil, err
	}

	// Divide by N
	n := 1
	for _, d := range preds.Shape() {
		n *= d
	}
	nT := b.ops.FromFloat64(float64(n))
	invN := b.ops.Div(b.ops.One(), nT)
	grad, err := b.engine.MulScalar(ctx, rawGrad, invN)
	if err != nil {
		return nil, err
	}

	// Chain with upstream gradient (broadcasts [1] against [N])
	gradPred, err := b.engine.Mul(ctx, grad, dOut, nil)
	if err != nil {
		return nil, err
	}

	// Zero gradient for targets
	zeroGrad, err := tensor.New[T](targs.Shape(), nil)
	if err != nil {
		return nil, err
	}
	if err := b.engine.Zeros(ctx, zeroGrad, targs.Shape()); err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{gradPred, zeroGrad}, nil
}

// OutputShape returns the output shape of the BCELoss function.
func (b *BCELoss[T]) OutputShape() []int {
	return []int{1}
}

// OpType returns the operation type of the BCELoss function.
func (b *BCELoss[T]) OpType() string {
	return "BCELoss"
}

// Attributes returns the attributes of the BCELoss function.
func (b *BCELoss[T]) Attributes() map[string]interface{} {
	return nil
}

// Parameters returns the parameters of the BCELoss function.
func (b *BCELoss[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Statically assert that the type implements the graph.Node interface.
var _ graph.Node[float32] = (*BCELoss[float32])(nil)
