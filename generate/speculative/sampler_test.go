package speculative

import (
	"math/rand/v2"
	"testing"
)

// TestAcceptTokens_Deterministic_AllAccepted verifies that when p >= q for all
// draft tokens, all are accepted deterministically (nil rng = temperature 0).
func TestAcceptTokens_Deterministic_AllAccepted(t *testing.T) {
	draftTokens := []int32{1, 2, 3}
	// Draft and target assign equal probability to chosen tokens.
	draftProbs := [][]float32{
		{0.1, 0.7, 0.1, 0.1},
		{0.1, 0.1, 0.7, 0.1},
		{0.1, 0.1, 0.1, 0.7},
	}
	targetProbs := [][]float32{
		{0.1, 0.7, 0.1, 0.1},
		{0.1, 0.1, 0.7, 0.1},
		{0.1, 0.1, 0.1, 0.7},
	}

	accepted, rate := AcceptTokens(draftTokens, draftProbs, targetProbs, nil)

	if rate != 1.0 {
		t.Errorf("acceptance rate = %v, want 1.0", rate)
	}
	if len(accepted) != 4 {
		t.Fatalf("accepted length = %d, want 4 (3 draft + 1 bonus)", len(accepted))
	}
	for i, tok := range draftTokens {
		if accepted[i] != tok {
			t.Errorf("accepted[%d] = %d, want %d", i, accepted[i], tok)
		}
	}
}

// TestAcceptTokens_Deterministic_TargetDominates verifies that when p >> q,
// all tokens are accepted.
func TestAcceptTokens_Deterministic_TargetDominates(t *testing.T) {
	draftTokens := []int32{0, 1}
	draftProbs := [][]float32{
		{0.1, 0.3, 0.3, 0.3},
		{0.3, 0.1, 0.3, 0.3},
	}
	targetProbs := [][]float32{
		{0.9, 0.033, 0.033, 0.034},
		{0.033, 0.9, 0.033, 0.034},
	}

	accepted, rate := AcceptTokens(draftTokens, draftProbs, targetProbs, nil)
	if rate != 1.0 {
		t.Errorf("acceptance rate = %v, want 1.0 (p >> q)", rate)
	}
	if len(accepted) != 3 {
		t.Errorf("accepted length = %d, want 3 (2 draft + 1 bonus)", len(accepted))
	}
}

// TestAcceptTokens_DraftDominates verifies near-zero acceptance when q >> p.
func TestAcceptTokens_DraftDominates(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))

	draftTokens := []int32{0}
	draftProbs := [][]float32{
		{0.95, 0.017, 0.017, 0.016},
	}
	targetProbs := [][]float32{
		{0.01, 0.33, 0.33, 0.33},
	}

	const trials = 1000
	totalAccepted := 0
	for range trials {
		accepted, _ := AcceptTokens(draftTokens, draftProbs, targetProbs, rng)
		if len(accepted) > 0 && accepted[0] == draftTokens[0] {
			totalAccepted++
		}
	}

	observedRate := float64(totalAccepted) / float64(trials)
	if observedRate > 0.05 {
		t.Errorf("acceptance rate = %.3f, want < 0.05 (p << q)", observedRate)
	}
}

// TestAcceptTokens_Empty verifies behavior with no draft tokens.
func TestAcceptTokens_Empty(t *testing.T) {
	accepted, rate := AcceptTokens(nil, nil, nil, nil)
	if len(accepted) != 0 {
		t.Errorf("accepted = %v, want empty", accepted)
	}
	if rate != 0 {
		t.Errorf("rate = %v, want 0", rate)
	}
}

// TestAcceptTokens_DistributionChiSquare verifies that the output distribution
// matches the target distribution using a chi-square goodness-of-fit test with
// a uniform draft distribution.
func TestAcceptTokens_DistributionChiSquare(t *testing.T) {
	targetDist := []float32{0.4, 0.3, 0.2, 0.1}
	draftDist := []float32{0.25, 0.25, 0.25, 0.25}

	const trials = 10000
	observed := make([]float64, 4)
	rng := rand.New(rand.NewPCG(123, 456))

	for range trials {
		draftToken := sampleCategorical(draftDist, rng)

		accepted, _ := AcceptTokens(
			[]int32{draftToken},
			[][]float32{draftDist},
			[][]float32{targetDist},
			rng,
		)

		if len(accepted) > 0 {
			observed[accepted[0]]++
		}
	}

	chiSq := chiSquare(observed, targetDist, trials)
	// df=3, p=0.05 critical value = 7.815
	if chiSq > 7.815 {
		t.Errorf("chi-square = %.3f (> 7.815, p < 0.05), output does not match target. observed=%v",
			chiSq, observed)
	}
}

// TestAcceptTokens_DistributionChiSquare_NonUniformDraft tests with a skewed
// draft distribution to verify correctness of the residual sampling.
func TestAcceptTokens_DistributionChiSquare_NonUniformDraft(t *testing.T) {
	targetDist := []float32{0.1, 0.2, 0.3, 0.4}
	draftDist := []float32{0.5, 0.3, 0.15, 0.05}

	const trials = 10000
	observed := make([]float64, 4)
	rng := rand.New(rand.NewPCG(789, 101))

	for range trials {
		draftToken := sampleCategorical(draftDist, rng)

		accepted, _ := AcceptTokens(
			[]int32{draftToken},
			[][]float32{draftDist},
			[][]float32{targetDist},
			rng,
		)

		if len(accepted) > 0 {
			observed[accepted[0]]++
		}
	}

	chiSq := chiSquare(observed, targetDist, trials)
	if chiSq > 7.815 {
		t.Errorf("chi-square = %.3f (> 7.815, p < 0.05), output does not match target. observed=%v",
			chiSq, observed)
	}
}

// TestAcceptTokens_NoDeterminismViolation verifies that with nil rng
// (temperature 0), the same inputs always produce the same outputs.
func TestAcceptTokens_NoDeterminismViolation(t *testing.T) {
	draftTokens := []int32{2, 1, 3}
	draftProbs := [][]float32{
		{0.1, 0.1, 0.6, 0.2},
		{0.1, 0.6, 0.1, 0.2},
		{0.1, 0.1, 0.2, 0.6},
	}
	targetProbs := [][]float32{
		{0.1, 0.1, 0.6, 0.2},
		{0.1, 0.6, 0.1, 0.2},
		{0.1, 0.1, 0.2, 0.6},
	}

	first, firstRate := AcceptTokens(draftTokens, draftProbs, targetProbs, nil)
	for i := 1; i < 100; i++ {
		got, gotRate := AcceptTokens(draftTokens, draftProbs, targetProbs, nil)
		if gotRate != firstRate {
			t.Fatalf("determinism violation at iteration %d: rate %v != %v", i, gotRate, firstRate)
		}
		if len(got) != len(first) {
			t.Fatalf("determinism violation at iteration %d: len %d != %d", i, len(got), len(first))
		}
		for j := range first {
			if got[j] != first[j] {
				t.Fatalf("determinism violation at iteration %d position %d: %d != %d",
					i, j, got[j], first[j])
			}
		}
	}
}

// TestAcceptTokens_ZeroDraftProb verifies handling of zero draft probability.
func TestAcceptTokens_ZeroDraftProb(t *testing.T) {
	draftTokens := []int32{0}
	draftProbs := [][]float32{
		{0.0, 0.33, 0.33, 0.34},
	}
	targetProbs := [][]float32{
		{0.25, 0.25, 0.25, 0.25},
	}

	accepted, rate := AcceptTokens(draftTokens, draftProbs, targetProbs, nil)
	if rate != 0.0 {
		t.Errorf("rate = %v, want 0.0 (draft prob is zero)", rate)
	}
	if len(accepted) != 1 {
		t.Fatalf("accepted length = %d, want 1 (correction token)", len(accepted))
	}
}

// TestAcceptTokens_MultiToken_PartialAcceptance verifies partial acceptance
// with multiple draft tokens.
func TestAcceptTokens_MultiToken_PartialAcceptance(t *testing.T) {
	rng := rand.New(rand.NewPCG(55, 77))

	// Token 0 has p/q = 0.8/0.8 = 1 → always accept.
	// Token 1 has p/q = 0.01/0.9 ≈ 0.011 → almost always reject.
	draftTokens := []int32{0, 1}
	draftProbs := [][]float32{
		{0.8, 0.1, 0.1, 0.0},
		{0.05, 0.9, 0.025, 0.025},
	}
	targetProbs := [][]float32{
		{0.8, 0.1, 0.1, 0.0},
		{0.05, 0.01, 0.47, 0.47},
	}

	const trials = 500
	fullAccept := 0
	for range trials {
		accepted, _ := AcceptTokens(draftTokens, draftProbs, targetProbs, rng)
		if len(accepted) == 3 && accepted[0] == 0 && accepted[1] == 1 {
			fullAccept++
		}
	}

	// Second token should be rejected most of the time.
	if float64(fullAccept)/float64(trials) > 0.05 {
		t.Errorf("full acceptance rate = %.3f, want < 0.05", float64(fullAccept)/float64(trials))
	}
}

// chiSquare computes the chi-square statistic for observed counts against
// expected proportions.
func chiSquare(observed []float64, expected []float32, total int) float64 {
	chiSq := 0.0
	for i := range observed {
		exp := float64(total) * float64(expected[i])
		if exp > 0 {
			diff := observed[i] - exp
			chiSq += (diff * diff) / exp
		}
	}
	return chiSq
}

// TestAcceptTokens_DistributionChiSquare_HighEntropy tests with a nearly
// uniform target to verify edge case behavior.
func TestAcceptTokens_DistributionChiSquare_HighEntropy(t *testing.T) {
	targetDist := []float32{0.24, 0.26, 0.25, 0.25}
	draftDist := []float32{0.7, 0.1, 0.1, 0.1}

	const trials = 10000
	observed := make([]float64, 4)
	rng := rand.New(rand.NewPCG(999, 888))

	for range trials {
		draftToken := sampleCategorical(draftDist, rng)

		accepted, _ := AcceptTokens(
			[]int32{draftToken},
			[][]float32{draftDist},
			[][]float32{targetDist},
			rng,
		)

		if len(accepted) > 0 {
			observed[accepted[0]]++
		}
	}

	chiSq := chiSquare(observed, targetDist, trials)
	if chiSq > 7.815 {
		t.Errorf("chi-square = %.3f (> 7.815, p < 0.05), observed=%v", chiSq, observed)
	}
}

