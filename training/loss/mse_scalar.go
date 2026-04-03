package loss

// MSEFlat computes MSE loss and gradient for flat prediction/target vectors.
// Returns (loss, dPred) where dPred[i] = 2*(pred[i]-target[i])/n.
func MSEFlat(pred, target []float64) (float64, []float64) {
	n := len(pred)
	loss := 0.0
	dPred := make([]float64, n)
	for i := range pred {
		diff := pred[i] - target[i]
		loss += diff * diff
		dPred[i] = 2 * diff / float64(n)
	}
	loss /= float64(n)
	return loss, dPred
}
