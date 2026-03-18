package speculative

import (
	"math"
	"math/rand/v2"
)

// AcceptTokens implements the rejection sampling algorithm from Leviathan et al.
// 2023 ("Fast Inference from Transformers via Speculative Decoding").
//
// For each draft token x_i with draft probability q(x_i) and target probability
// p(x_i):
//   - Accept with probability min(1, p(x_i) / q(x_i))
//   - If rejected at position i, sample a correction token from the
//     renormalized max(0, p - q) distribution
//   - If all K tokens accepted, sample a bonus token from the target
//     distribution at position K
//
// Parameters:
//   - draftTokens: K token IDs proposed by the draft model
//   - draftProbs: full probability distributions from the draft model, one
//     []float32 per draft position (each slice has length vocabSize)
//   - targetProbs: full probability distributions from the target model, one
//     []float32 per draft position (each slice has length vocabSize)
//   - rng: random number generator (if nil, uses deterministic acceptance
//     where r=0 — equivalent to temperature 0 / greedy behavior)
//
// Returns the accepted token sequence (up to K+1 including a possible bonus
// token) and the acceptance rate (accepted / proposed).
func AcceptTokens(draftTokens []int32, draftProbs [][]float32, targetProbs [][]float32, rng *rand.Rand) ([]int32, float32) {
	k := len(draftTokens)
	if k == 0 {
		return nil, 0
	}

	accepted := make([]int32, 0, k+1)

	for i := 0; i < k; i++ {
		tokenID := draftTokens[i]
		q := float64(draftProbs[i][tokenID])
		p := float64(targetProbs[i][tokenID])

		if q <= 0 {
			// Draft assigned zero probability — always reject.
			correction := sampleResidual(targetProbs[i], draftProbs[i], rng)
			accepted = append(accepted, correction)
			return accepted, float32(len(accepted)-1) / float32(k)
		}

		acceptProb := math.Min(1.0, p/q)

		r := randFloat64(rng)
		if r < acceptProb {
			accepted = append(accepted, tokenID)
			continue
		}

		// Rejected: sample correction from max(0, p - q) renormalized.
		correction := sampleResidual(targetProbs[i], draftProbs[i], rng)
		accepted = append(accepted, correction)
		return accepted, float32(len(accepted)-1) / float32(k)
	}

	// All K tokens accepted — sample a bonus token from the target distribution
	// at the last position.
	bonus := sampleCategorical(targetProbs[k-1], rng)
	accepted = append(accepted, bonus)
	return accepted, 1.0
}

// sampleResidual samples from the renormalized max(0, p(x) - q(x)) distribution.
func sampleResidual(targetP []float32, draftP []float32, rng *rand.Rand) int32 {
	n := len(targetP)
	residual := make([]float64, n)
	sum := 0.0

	for i := 0; i < n; i++ {
		r := math.Max(0, float64(targetP[i])-float64(draftP[i]))
		residual[i] = r
		sum += r
	}

	if sum <= 0 {
		// Degenerate case: fall back to target distribution.
		return sampleCategorical(targetP, rng)
	}

	r := randFloat64(rng) * sum
	cumulative := 0.0
	for i := 0; i < n; i++ {
		cumulative += residual[i]
		if r < cumulative {
			return int32(i)
		}
	}
	return int32(n - 1)
}

// sampleCategorical samples a token from the probability distribution.
func sampleCategorical(probs []float32, rng *rand.Rand) int32 {
	r := randFloat64(rng)
	cumulative := 0.0
	for i, p := range probs {
		cumulative += float64(p)
		if r < cumulative {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}

// randFloat64 returns a uniform random value in [0, 1). If rng is nil,
// returns 0.0, which makes acceptance deterministic (temperature 0 behavior).
func randFloat64(rng *rand.Rand) float64 {
	if rng == nil {
		return 0.0
	}
	return rng.Float64()
}
