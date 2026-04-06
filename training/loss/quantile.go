package loss

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/float16"
	"github.com/zerfoo/float8"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// QuantileLoss computes the pinball (quantile regression) loss using engine ops.
// preds has shape [batch, num_quantiles], targets has shape [batch],
// and quantiles is a slice of quantile levels (e.g., 0.1, 0.5, 0.9).
//
// For each quantile q and sample i:
//
//	error = target_i - pred_i_q
//	loss_q = q * error   if error >= 0
//	loss_q = (q-1) * error   if error < 0
//
// Returns the mean loss over all samples and quantiles.
func QuantileLoss[T tensor.Numeric](engine compute.Engine[T], preds, targets *tensor.TensorNumeric[T], quantiles []float32) (float32, error) {
	predShape := preds.Shape()
	targetShape := targets.Shape()

	if len(predShape) != 2 {
		return 0, fmt.Errorf("QuantileLoss: preds must be 2D [batch, num_quantiles], got shape %v", predShape)
	}
	batch := predShape[0]
	numQ := predShape[1]

	if len(quantiles) != numQ {
		return 0, fmt.Errorf("QuantileLoss: len(quantiles)=%d must match preds dim 1=%d", len(quantiles), numQ)
	}

	targetBatch := targetShape[0]
	if targetBatch != batch {
		return 0, fmt.Errorf("QuantileLoss: batch mismatch preds=%d targets=%d", batch, targetBatch)
	}

	ctx := context.Background()
	ops := engine.Ops()

	// Broadcast targets [batch] -> [batch, 1] -> [batch, numQ] via Repeat.
	tgt1D := targets
	if len(targetShape) == 1 {
		var err error
		tgt1D, err = engine.Reshape(ctx, targets, []int{batch, 1})
		if err != nil {
			return 0, fmt.Errorf("QuantileLoss: reshape targets: %w", err)
		}
	}
	tgtBroad, err := engine.Repeat(ctx, tgt1D, 1, numQ)
	if err != nil {
		return 0, fmt.Errorf("QuantileLoss: repeat targets: %w", err)
	}

	// error = targets - preds, shape [batch, numQ]
	errTensor, err := engine.Sub(ctx, tgtBroad, preds)
	if err != nil {
		return 0, fmt.Errorf("QuantileLoss: sub: %w", err)
	}

	// pos = max(error, 0), neg = max(-error, 0)
	pos, err := engine.UnaryOp(ctx, errTensor, ops.ReLU)
	if err != nil {
		return 0, fmt.Errorf("QuantileLoss: relu pos: %w", err)
	}
	negErr, err := engine.MulScalar(ctx, errTensor, ops.FromFloat64(-1))
	if err != nil {
		return 0, fmt.Errorf("QuantileLoss: negate: %w", err)
	}
	neg, err := engine.UnaryOp(ctx, negErr, ops.ReLU)
	if err != nil {
		return 0, fmt.Errorf("QuantileLoss: relu neg: %w", err)
	}

	// Build quantile weight tensors [1, numQ].
	qData := make([]T, numQ)
	qm1Data := make([]T, numQ)
	for j := 0; j < numQ; j++ {
		qData[j] = ops.FromFloat64(float64(quantiles[j]))
		qm1Data[j] = ops.FromFloat64(float64(1 - quantiles[j]))
	}
	qTensor, err := tensor.New[T]([]int{1, numQ}, qData)
	if err != nil {
		return 0, err
	}
	qm1Tensor, err := tensor.New[T]([]int{1, numQ}, qm1Data)
	if err != nil {
		return 0, err
	}

	// loss = q * pos + (1-q) * neg  (both non-negative)
	qPos, err := engine.Mul(ctx, pos, qTensor)
	if err != nil {
		return 0, err
	}
	qm1Neg, err := engine.Mul(ctx, neg, qm1Tensor)
	if err != nil {
		return 0, err
	}
	total, err := engine.Add(ctx, qPos, qm1Neg)
	if err != nil {
		return 0, err
	}

	// Mean over all elements: sum across numQ, then sum across batch, divide by count.
	sumQ, err := engine.ReduceSum(ctx, total, 1, false)
	if err != nil {
		return 0, err
	}
	sumAll, err := engine.ReduceSum(ctx, sumQ, 0, false)
	if err != nil {
		return 0, err
	}

	count := batch * numQ
	result := numericToFloat64(sumAll.Data()[0]) / float64(count)
	return float32(result), nil
}

// SharpeLoss computes the negative Sharpe ratio as a differentiable loss for
// portfolio optimization.
//
// weights has shape [batch, num_assets] — interpreted as portfolio weights
// (softmax-normalized internally to ensure long-only, sum-to-one constraint).
// returns_ has shape [batch, num_assets] — per-asset log returns for each
// time step in the batch.
//
// Portfolio return for time step i = sum_j(w_j * r_ij)
// Sharpe = mean(portfolio_returns) / std(portfolio_returns)
// SharpeLoss = -Sharpe (to minimize)
func SharpeLoss[T tensor.Numeric](weights, returns_ *tensor.TensorNumeric[T]) (float32, error) {
	wShape := weights.Shape()
	rShape := returns_.Shape()

	if len(wShape) != 2 || len(rShape) != 2 {
		return 0, fmt.Errorf("SharpeLoss: weights and returns must be 2D, got %v and %v", wShape, rShape)
	}
	batch := wShape[0]
	numAssets := wShape[1]
	if rShape[0] != batch || rShape[1] != numAssets {
		return 0, fmt.Errorf("SharpeLoss: shape mismatch weights=%v returns=%v", wShape, rShape)
	}

	wData := weights.Data()
	rData := returns_.Data()

	// Softmax normalize weights per time step to get portfolio allocations.
	softW := make([]float64, batch*numAssets)
	for i := 0; i < batch; i++ {
		maxW := math.Inf(-1)
		for j := 0; j < numAssets; j++ {
			v := numericToFloat64(wData[i*numAssets+j])
			if v > maxW {
				maxW = v
			}
		}
		var sumExp float64
		for j := 0; j < numAssets; j++ {
			v := numericToFloat64(wData[i*numAssets+j])
			softW[i*numAssets+j] = math.Exp(v - maxW)
			sumExp += softW[i*numAssets+j]
		}
		for j := 0; j < numAssets; j++ {
			softW[i*numAssets+j] /= sumExp
		}
	}

	// Compute portfolio returns for each time step.
	portReturns := make([]float64, batch)
	for i := 0; i < batch; i++ {
		var pr float64
		for j := 0; j < numAssets; j++ {
			r := numericToFloat64(rData[i*numAssets+j])
			pr += softW[i*numAssets+j] * r
		}
		portReturns[i] = pr
	}

	// Compute mean and std of portfolio returns.
	var meanRet float64
	for _, r := range portReturns {
		meanRet += r
	}
	meanRet /= float64(batch)

	var variance float64
	for _, r := range portReturns {
		d := r - meanRet
		variance += d * d
	}
	variance /= float64(batch)
	stdRet := math.Sqrt(variance)

	if stdRet < 1e-8 {
		return 0, nil
	}

	sharpe := meanRet / stdRet
	return float32(-sharpe), nil
}

// numericToFloat64 converts a tensor.Numeric value to float64.
func numericToFloat64[T tensor.Numeric](v T) float64 {
	switch val := any(v).(type) {
	case float32:
		return float64(val)
	case float64:
		return val
	case int:
		return float64(val)
	case int8:
		return float64(val)
	case int16:
		return float64(val)
	case int32:
		return float64(val)
	case int64:
		return float64(val)
	case uint:
		return float64(val)
	case uint8:
		return float64(val)
	case uint32:
		return float64(val)
	case uint64:
		return float64(val)
	case float16.Float16:
		return float64(val.ToFloat32())
	case float16.BFloat16:
		return float64(val.ToFloat32())
	case float8.Float8:
		return val.ToFloat64()
	default:
		return 0
	}
}
