package loss

import (
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/tensor"
)

// QuantileLoss computes the pinball (quantile regression) loss.
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
func QuantileLoss[T tensor.Numeric](preds, targets *tensor.TensorNumeric[T], quantiles []float32) (float32, error) {
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

	// targets can be [batch] or [batch, 1]
	targetBatch := targetShape[0]
	if targetBatch != batch {
		return 0, fmt.Errorf("QuantileLoss: batch mismatch preds=%d targets=%d", batch, targetBatch)
	}

	predData := preds.Data()
	targetData := targets.Data()

	var totalLoss float64
	count := 0

	for i := 0; i < batch; i++ {
		tgt := float64(any(targetData[i]).(float32))
		for j := 0; j < numQ; j++ {
			pred := float64(any(predData[i*numQ+j]).(float32))
			q := float64(quantiles[j])
			err := tgt - pred
			var l float64
			if err >= 0 {
				l = q * err
			} else {
				l = (q - 1) * err
			}
			totalLoss += l
			count++
		}
	}

	if count == 0 {
		return 0, nil
	}
	return float32(totalLoss / float64(count)), nil
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
			v := float64(any(wData[i*numAssets+j]).(float32))
			if v > maxW {
				maxW = v
			}
		}
		var sumExp float64
		for j := 0; j < numAssets; j++ {
			v := float64(any(wData[i*numAssets+j]).(float32))
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
			r := float64(any(rData[i*numAssets+j]).(float32))
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
