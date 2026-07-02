package generate

import (
	"math"
	"math/rand/v2"
	"sort"
)

// applyTemperature divides all logits by the given temperature.
// Higher temperature increases randomness; lower sharpens the distribution.
func applyTemperature(logits []float64, temp float64) {
	if temp <= 0 {
		return
	}
	for i := range logits {
		logits[i] /= temp
	}
}

// applyTopK sets all logits outside the top-K values to negative infinity.
func applyTopK(logits []float64, k int) {
	if k <= 0 || k >= len(logits) {
		return
	}

	// Find the k-th largest value using a partial sort via a heap-like approach.
	// For simplicity, sort indices by value descending, keep top-k.
	type iv struct {
		idx int
		val float64
	}
	items := make([]iv, len(logits))
	for i, v := range logits {
		items[i] = iv{i, v}
	}
	sort.Slice(items, func(a, b int) bool {
		return items[a].val > items[b].val
	})

	// Threshold is the value at position k-1.
	threshold := items[k-1].val

	for i, v := range logits {
		if v < threshold {
			logits[i] = math.Inf(-1)
		}
	}
}

// applyTopP (nucleus sampling) keeps the smallest set of tokens whose
// cumulative probability is at least p, setting the rest to negative infinity.
func applyTopP(logits []float64, p float64) {
	if p <= 0 || p >= 1.0 {
		return
	}

	// Convert logits to probabilities via softmax.
	probs := softmax(logits)

	// Sort by probability descending.
	type iv struct {
		idx  int
		prob float64
	}
	items := make([]iv, len(probs))
	for i, prob := range probs {
		items[i] = iv{i, prob}
	}
	sort.Slice(items, func(a, b int) bool {
		return items[a].prob > items[b].prob
	})

	// Find cutoff: cumulative probability exceeds p.
	keep := make(map[int]bool)
	cumulative := 0.0
	for _, item := range items {
		keep[item.idx] = true
		cumulative += item.prob
		if cumulative >= p {
			break
		}
	}

	for i := range logits {
		if !keep[i] {
			logits[i] = math.Inf(-1)
		}
	}
}

// applyRepetitionPenalty penalizes previously generated tokens.
// For positive logits, divides by penalty; for negative, multiplies by penalty.
// penalty > 1.0 discourages repetition; 1.0 is a no-op.
func applyRepetitionPenalty(logits []float64, generatedTokens []int, penalty float64) {
	if penalty == 1.0 {
		return
	}
	for _, tokenID := range generatedTokens {
		if tokenID < 0 || tokenID >= len(logits) {
			continue
		}
		if logits[tokenID] > 0 {
			logits[tokenID] /= penalty
		} else {
			logits[tokenID] *= penalty
		}
	}
}

// sampleFromDistribution applies softmax to logits and samples a token index
// using weighted random selection.
func sampleFromDistribution(logits []float64) int {
	probs := softmax(logits)

	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	// Fallback to last token (rounding errors).
	return len(probs) - 1
}

// softmax converts logits to a probability distribution.
func softmax(logits []float64) []float64 {
	n := len(logits)
	probs := make([]float64, n)
	if n == 0 {
		return probs
	}

	// Find max for numerical stability.
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	// If all logits are -Inf, return uniform distribution.
	if math.IsInf(maxVal, -1) {
		uniform := 1.0 / float64(n)
		for i := range probs {
			probs[i] = uniform
		}
		return probs
	}

	sum := 0.0
	for i, v := range logits {
		exp := math.Exp(v - maxVal)
		probs[i] = exp
		sum += exp
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}
	return probs
}

// argmax returns the index of the maximum value in logits.
func argmax(logits []float64) int {
	best := 0
	bestVal := math.Inf(-1)
	for i, v := range logits {
		if v > bestVal {
			bestVal = v
			best = i
		}
	}
	return best
}
