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

// CorrLoss computes -PearsonCorrelation(predictions, targets) as a differentiable
// scalar loss. Minimizing this loss maximizes the Pearson correlation between
// predictions and targets. Since Numerai targets are rank-normalized, Pearson
// closely approximates Spearman rank correlation.
//
// Forward: loss = -sum(p_c * t_c) / (sqrt(sum(p_c^2) * sum(t_c^2)) + eps)
//
//	where p_c = p - mean(p), t_c = t - mean(t)
//
// Backward: grad_i = -(t_c_i / denom - corr * p_c_i / sum_pp) * dOut
//
// All tensor operations use the engine, keeping data on GPU when available.
// Only scalar intermediate values (means, sums) are read back to CPU.
type CorrLoss[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	predictions *tensor.TensorNumeric[T]
	targets     *tensor.TensorNumeric[T]
}

// NewCorrLoss creates a new correlation loss function.
func NewCorrLoss[T tensor.Numeric](engine compute.Engine[T], ops numeric.Arithmetic[T]) *CorrLoss[T] {
	return &CorrLoss[T]{engine: engine, ops: ops}
}

// Forward computes -PearsonCorrelation(predictions, targets).
func (c *CorrLoss[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("CorrLoss expects 2 inputs, got %d", len(inputs))
	}

	predictions, targets := inputs[0], inputs[1]
	c.predictions = predictions
	c.targets = targets

	eng := c.engine
	ops := c.ops
	zero := ops.FromFloat64(0)
	eps := ops.FromFloat64(1e-8)

	// Compute means via engine (GPU-accelerated reduction).
	meanP, err := eng.ReduceMean(ctx, predictions, 0, false)
	if err != nil {
		return nil, err
	}

	meanT, err := eng.ReduceMean(ctx, targets, 0, false)
	if err != nil {
		return nil, err
	}

	// Center predictions and targets via engine.
	// Only scalar mean values are read back (1 float D2H each).
	pc, err := eng.AddScalar(ctx, predictions, ops.Sub(zero, meanP.Data()[0]), nil)
	if err != nil {
		return nil, err
	}

	tc, err := eng.AddScalar(ctx, targets, ops.Sub(zero, meanT.Data()[0]), nil)
	if err != nil {
		return nil, err
	}

	// Element-wise products via engine.
	pcTc, err := eng.Mul(ctx, pc, tc, nil)
	if err != nil {
		return nil, err
	}

	pcPc, err := eng.Mul(ctx, pc, pc, nil)
	if err != nil {
		return nil, err
	}

	tcTc, err := eng.Mul(ctx, tc, tc, nil)
	if err != nil {
		return nil, err
	}

	// Sum reductions via engine.
	sumPT, err := eng.Sum(ctx, pcTc, 0, false)
	if err != nil {
		return nil, err
	}

	sumPP, err := eng.Sum(ctx, pcPc, 0, false)
	if err != nil {
		return nil, err
	}

	sumTT, err := eng.Sum(ctx, tcTc, 0, false)
	if err != nil {
		return nil, err
	}

	// Read scalar sums and compute correlation.
	sumPTVal := sumPT.Data()[0]
	sumPPVal := sumPP.Data()[0]
	sumTTVal := sumTT.Data()[0]

	denom := ops.Add(ops.Sqrt(ops.Mul(sumPPVal, sumTTVal)), eps)
	corr := ops.Div(sumPTVal, denom)
	loss := ops.Sub(zero, corr)

	return tensor.New[T]([]int{1}, []T{loss})
}

// Backward computes the gradient of -PearsonCorrelation with respect to predictions.
// Returns [dPredictions, dTargets(zeros)].
func (c *CorrLoss[T]) Backward(ctx context.Context, _ types.BackwardMode, dOut *tensor.TensorNumeric[T], inputs ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	preds := c.predictions
	targs := c.targets
	if len(inputs) > 0 {
		preds = inputs[0]
		if len(inputs) > 1 {
			targs = inputs[1]
		}
	}

	if preds == nil || targs == nil {
		return nil, graph.ErrInvalidInputCount
	}

	eng := c.engine
	ops := c.ops
	zero := ops.FromFloat64(0)
	eps := ops.FromFloat64(1e-8)
	scale := dOut.Data()[0]

	// Compute centered values via engine.
	meanP, err := eng.ReduceMean(ctx, preds, 0, false)
	if err != nil {
		return nil, err
	}

	meanT, err := eng.ReduceMean(ctx, targs, 0, false)
	if err != nil {
		return nil, err
	}

	pc, err := eng.AddScalar(ctx, preds, ops.Sub(zero, meanP.Data()[0]), nil)
	if err != nil {
		return nil, err
	}

	tc, err := eng.AddScalar(ctx, targs, ops.Sub(zero, meanT.Data()[0]), nil)
	if err != nil {
		return nil, err
	}

	// Element-wise products.
	pcTc, err := eng.Mul(ctx, pc, tc, nil)
	if err != nil {
		return nil, err
	}

	pcPc, err := eng.Mul(ctx, pc, pc, nil)
	if err != nil {
		return nil, err
	}

	tcTc, err := eng.Mul(ctx, tc, tc, nil)
	if err != nil {
		return nil, err
	}

	// Sum reductions.
	sumPT, err := eng.Sum(ctx, pcTc, 0, false)
	if err != nil {
		return nil, err
	}

	sumPP, err := eng.Sum(ctx, pcPc, 0, false)
	if err != nil {
		return nil, err
	}

	sumTT, err := eng.Sum(ctx, tcTc, 0, false)
	if err != nil {
		return nil, err
	}

	// Scalar correlation components.
	sumPTVal := sumPT.Data()[0]
	sumPPVal := sumPP.Data()[0]
	sumTTVal := sumTT.Data()[0]

	denom := ops.Add(ops.Sqrt(ops.Mul(sumPPVal, sumTTVal)), eps)
	corr := ops.Div(sumPTVal, denom)
	sumPPEps := ops.Add(sumPPVal, eps)

	// grad_i = -(tc_i/denom - corr * pc_i / sumPPEps) * scale
	//        = (-scale/denom) * tc_i + (scale * corr / sumPPEps) * pc_i
	negScale := ops.Sub(zero, scale)
	coeff1 := ops.Div(negScale, denom)
	coeff2 := ops.Div(ops.Mul(scale, corr), sumPPEps)

	term1, err := eng.MulScalar(ctx, tc, coeff1, nil)
	if err != nil {
		return nil, err
	}

	term2, err := eng.MulScalar(ctx, pc, coeff2, nil)
	if err != nil {
		return nil, err
	}

	gradPred, err := eng.Add(ctx, term1, term2, nil)
	if err != nil {
		return nil, err
	}

	// Zero gradient for targets.
	numElems := 1
	for _, d := range targs.Shape() {
		numElems *= d
	}

	zeroGrad, err := tensor.New[T](targs.Shape(), make([]T, numElems))
	if err != nil {
		return nil, err
	}

	return []*tensor.TensorNumeric[T]{gradPred, zeroGrad}, nil
}

// OutputShape returns [1] (scalar loss).
func (c *CorrLoss[T]) OutputShape() []int {
	return []int{1}
}

// OpType returns "CorrLoss".
func (c *CorrLoss[T]) OpType() string {
	return "CorrLoss"
}

// Attributes returns nil (no configurable attributes).
func (c *CorrLoss[T]) Attributes() map[string]any {
	return nil
}

// Parameters returns nil (no trainable parameters).
func (c *CorrLoss[T]) Parameters() []*graph.Parameter[T] {
	return nil
}

// Statically assert that CorrLoss implements graph.Node.
var _ graph.Node[float32] = (*CorrLoss[float32])(nil)
