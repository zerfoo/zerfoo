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

// clamp restricts p to [eps, 1-eps] for numerical stability.
func (b *BCELoss[T]) clamp(p T) T {
	eps := b.ops.FromFloat64(1e-7)
	one := b.ops.One()
	oneMinusEps := b.ops.Sub(one, eps)

	// If p < eps, return eps
	if b.ops.GreaterThan(eps, p) {
		return eps
	}
	// If p > 1-eps, return 1-eps
	if b.ops.GreaterThan(p, oneMinusEps) {
		return oneMinusEps
	}
	return p
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

	one := b.ops.One()
	pData := predictions.Data()
	yData := targets.Data()
	n := len(pData)

	var sum T
	for i := 0; i < n; i++ {
		p := b.clamp(pData[i])
		y := yData[i]

		// -[y*log(p) + (1-y)*log(1-p)]
		logP := b.ops.Log(p)
		logOneMinusP := b.ops.Log(b.ops.Sub(one, p))
		term := b.ops.Add(b.ops.Mul(y, logP), b.ops.Mul(b.ops.Sub(one, y), logOneMinusP))
		sum = b.ops.Sub(sum, term)
	}

	mean := b.ops.Div(sum, b.ops.FromFloat64(float64(n)))
	loss, err := tensor.New[T]([]int{1}, []T{mean})
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

	one := b.ops.One()
	pData := preds.Data()
	yData := targs.Data()
	n := len(pData)
	nT := b.ops.FromFloat64(float64(n))
	dOutVal := dOut.Data()[0]

	gradData := make([]T, n)
	for i := 0; i < n; i++ {
		p := b.clamp(pData[i])
		y := yData[i]

		// grad = -(y/p - (1-y)/(1-p)) / N = ((1-y)/(1-p) - y/p) / N
		yOverP := b.ops.Div(y, p)
		oneMinusYOverOneMinusP := b.ops.Div(b.ops.Sub(one, y), b.ops.Sub(one, p))
		grad := b.ops.Div(b.ops.Sub(oneMinusYOverOneMinusP, yOverP), nT)

		// Chain with upstream gradient
		gradData[i] = b.ops.Mul(grad, dOutVal)
	}

	gradTensor, err := tensor.New[T](preds.Shape(), gradData)
	if err != nil {
		return nil, err
	}

	zeroGrad, err := tensor.New[T](targs.Shape(), make([]T, len(yData)))
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{gradTensor, zeroGrad}, nil
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
