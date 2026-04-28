package regime

import (
	"math"
	"math/rand"
	"testing"
)

// generateRegimeData generates a synthetic time series with regime switches.
// The data alternates between regimes with distinct statistical properties:
//   - State 0: mean=-2, low variance (mean-reverting)
//   - State 1: mean=+2, low variance (trending)
//   - State 2: mean=0, high variance (volatile)
//
// Returns the observations and the true state labels.
func generateRegimeData(nPerRegime int, seed int64) (obs []float64, states []int) {
	rng := rand.New(rand.NewSource(seed))
	means := []float64{-2.0, 2.0, 0.0}
	stddevs := []float64{0.3, 0.3, 2.0}
	pattern := []int{0, 1, 2, 0, 1}

	for _, state := range pattern {
		for i := 0; i < nPerRegime; i++ {
			obs = append(obs, means[state]+stddevs[state]*rng.NormFloat64())
			states = append(states, state)
		}
	}
	return
}

// generate2StateData generates data switching between two well-separated states.
func generate2StateData(n int, seed int64) (obs []float64, trueStates []int) {
	rng := rand.New(rand.NewSource(seed))
	state := 0
	for i := 0; i < n; i++ {
		if state == 0 {
			obs = append(obs, -3.0+0.3*rng.NormFloat64())
		} else {
			obs = append(obs, 3.0+0.3*rng.NormFloat64())
		}
		trueStates = append(trueStates, state)
		// Switch state with low probability to create blocks.
		if rng.Float64() < 0.02 {
			state = 1 - state
		}
	}
	return
}

func TestNewHMM(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr string
	}{
		{
			name:    "valid 2 states",
			cfg:     Config{NStates: 2},
			wantErr: "",
		},
		{
			name:    "valid 4 states",
			cfg:     Config{NStates: 4, Seed: 99},
			wantErr: "",
		},
		{
			name:    "too few states",
			cfg:     Config{NStates: 1},
			wantErr: "NStates must be >= 2",
		},
		{
			name:    "zero states",
			cfg:     Config{NStates: 0},
			wantErr: "NStates must be >= 2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, err := NewHMM(tt.cfg)
			if tt.wantErr != "" {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if !containsSubstr(err.Error(), tt.wantErr) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Verify initial probabilities sum to 1.
			s := 0.0
			for _, p := range h.Initial {
				s += p
			}
			if math.Abs(s-1.0) > 1e-10 {
				t.Errorf("initial probabilities sum to %f, want 1.0", s)
			}

			// Verify each transition row sums to 1.
			for i, row := range h.Transition {
				rs := 0.0
				for _, p := range row {
					rs += p
				}
				if math.Abs(rs-1.0) > 1e-10 {
					t.Errorf("transition row %d sums to %f, want 1.0", i, rs)
				}
			}

			// Verify positive variances.
			for i, v := range h.Variances {
				if v <= 0 {
					t.Errorf("variance[%d] = %f, want > 0", i, v)
				}
			}
		})
	}
}

func TestHMM_BaumWelch(t *testing.T) {
	obs, _ := generate2StateData(500, 42)
	h, err := NewHMM(Config{NStates: 2, Seed: 7})
	if err != nil {
		t.Fatalf("NewHMM: %v", err)
	}

	ll, iters, err := h.Fit(obs, Config{
		NStates:       2,
		MaxIterations: 200,
		Tolerance:     1e-6,
	})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	t.Logf("converged in %d iterations, log-likelihood: %.2f", iters, ll)

	// After training, the two means should be well-separated.
	// One should be near -3 and the other near +3.
	m0, m1 := h.Means[0], h.Means[1]
	if math.Abs(m0-m1) < 3.0 {
		t.Errorf("means not well separated: %.2f, %.2f", m0, m1)
	}

	// Both variances should be small (data has stddev 0.3 => var ~ 0.09).
	for i, v := range h.Variances {
		if v > 1.0 {
			t.Errorf("variance[%d] = %.4f, expected < 1.0", i, v)
		}
	}

	// Transition matrix should be sticky (diagonal dominant).
	for i := 0; i < 2; i++ {
		if h.Transition[i][i] < 0.8 {
			t.Errorf("transition[%d][%d] = %.4f, expected > 0.8 (sticky)", i, i, h.Transition[i][i])
		}
	}

	// Log-likelihood should be finite and negative.
	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("log-likelihood is not finite: %f", ll)
	}
}

func TestHMM_BaumWelch_Errors(t *testing.T) {
	h, _ := NewHMM(Config{NStates: 2})
	_, _, err := h.Fit([]float64{1.0}, Config{NStates: 2})
	if err == nil {
		t.Fatal("expected error for short sequence")
	}
	if !containsSubstr(err.Error(), "at least 2") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestHMM_Viterbi(t *testing.T) {
	obs, trueStates := generate2StateData(300, 42)

	h, err := NewHMM(Config{NStates: 2, Seed: 7})
	if err != nil {
		t.Fatalf("NewHMM: %v", err)
	}
	_, _, err = h.Fit(obs, Config{
		NStates:       2,
		MaxIterations: 200,
		Tolerance:     1e-6,
	})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	decoded, err := h.Viterbi(obs)
	if err != nil {
		t.Fatalf("Viterbi: %v", err)
	}

	if len(decoded) != len(obs) {
		t.Fatalf("decoded length %d != obs length %d", len(decoded), len(obs))
	}

	// The decoded states may be permuted relative to the true labels
	// (HMM states are unordered). Check accuracy up to a label permutation.
	acc := maxPermutationAccuracy(decoded, trueStates, 2)
	t.Logf("Viterbi accuracy (best permutation): %.2f%%", acc*100)

	if acc < 0.85 {
		t.Errorf("Viterbi accuracy %.2f%% is below 85%% threshold", acc*100)
	}
}

func TestHMM_Viterbi_Errors(t *testing.T) {
	h, _ := NewHMM(Config{NStates: 2})
	_, err := h.Viterbi(nil)
	if err == nil {
		t.Fatal("expected error for empty sequence")
	}
}

func TestHMM_RegimeDetection(t *testing.T) {
	// Generate data with 3 regimes and verify the HMM can detect regime
	// transitions.
	obs, trueStates := generateRegimeData(100, 42)

	h, err := NewHMM(Config{NStates: 3, Seed: 123})
	if err != nil {
		t.Fatalf("NewHMM: %v", err)
	}
	ll, iters, err := h.Fit(obs, Config{
		NStates:       3,
		MaxIterations: 300,
		Tolerance:     1e-8,
	})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}
	t.Logf("converged in %d iterations, log-likelihood: %.2f", iters, ll)

	decoded, err := h.Viterbi(obs)
	if err != nil {
		t.Fatalf("Viterbi: %v", err)
	}

	// Verify regime transitions are detected: the decoded sequence should
	// not be constant (all one state).
	stateSet := make(map[int]bool)
	for _, s := range decoded {
		stateSet[s] = true
	}
	if len(stateSet) < 2 {
		t.Errorf("expected multiple regimes detected, got %d distinct states", len(stateSet))
	}

	// Check accuracy up to label permutation.
	acc := maxPermutationAccuracy(decoded, trueStates, 3)
	t.Logf("Regime detection accuracy (best permutation): %.2f%%", acc*100)
	if acc < 0.70 {
		t.Errorf("regime detection accuracy %.2f%% is below 70%% threshold", acc*100)
	}

	// Verify transition points: there should be transitions near the true
	// regime boundaries (at multiples of 100).
	transitionPoints := []int{}
	for i := 1; i < len(decoded); i++ {
		if decoded[i] != decoded[i-1] {
			transitionPoints = append(transitionPoints, i)
		}
	}
	t.Logf("detected %d transitions at positions: %v", len(transitionPoints), transitionPoints)
	if len(transitionPoints) < 2 {
		t.Error("expected at least 2 regime transitions")
	}
}

func TestHMM_Predict(t *testing.T) {
	obs, _ := generate2StateData(200, 42)

	h, err := NewHMM(Config{NStates: 2, Seed: 7})
	if err != nil {
		t.Fatalf("NewHMM: %v", err)
	}
	_, _, err = h.Fit(obs, Config{
		NStates:       2,
		MaxIterations: 200,
		Tolerance:     1e-6,
	})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	state, probs, err := h.Predict(obs)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if state < 0 || state >= 2 {
		t.Errorf("predicted state %d out of range [0, 2)", state)
	}

	// Probabilities should sum to 1.
	s := 0.0
	for _, p := range probs {
		s += p
	}
	if math.Abs(s-1.0) > 1e-6 {
		t.Errorf("posterior probabilities sum to %f, want 1.0", s)
	}

	// The winning state should have the highest probability.
	for i, p := range probs {
		if i != state && p > probs[state] {
			t.Errorf("state %d has prob %.4f > predicted state %d prob %.4f", i, p, state, probs[state])
		}
	}
}

func TestHMM_LogLikelihood(t *testing.T) {
	obs, _ := generate2StateData(100, 42)

	h, err := NewHMM(Config{NStates: 2, Seed: 7})
	if err != nil {
		t.Fatalf("NewHMM: %v", err)
	}
	_, _, err = h.Fit(obs, Config{NStates: 2, MaxIterations: 100})
	if err != nil {
		t.Fatalf("Fit: %v", err)
	}

	ll, err := h.LogLikelihood(obs)
	if err != nil {
		t.Fatalf("LogLikelihood: %v", err)
	}
	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("log-likelihood is not finite: %f", ll)
	}

	// Log-likelihood on training data should be reasonable (not extremely
	// negative for well-fit data).
	perObs := ll / float64(len(obs))
	t.Logf("log-likelihood per observation: %.4f", perObs)
	if perObs < -10 {
		t.Errorf("log-likelihood per observation %.4f seems too low", perObs)
	}
}

func TestHMM_LogLikelihood_Errors(t *testing.T) {
	h, _ := NewHMM(Config{NStates: 2})
	_, err := h.LogLikelihood(nil)
	if err == nil {
		t.Fatal("expected error for empty sequence")
	}
}

func TestGaussianPDF(t *testing.T) {
	tests := []struct {
		name     string
		x, mu, v float64
		want     float64
	}{
		{
			name: "standard normal at zero",
			x:    0, mu: 0, v: 1,
			want: 1.0 / math.Sqrt(2*math.Pi),
		},
		{
			name: "standard normal at one sigma",
			x:    1, mu: 0, v: 1,
			want: math.Exp(-0.5) / math.Sqrt(2*math.Pi),
		},
		{
			name: "shifted mean",
			x:    5, mu: 5, v: 2,
			want: 1.0 / math.Sqrt(2*math.Pi*2),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gaussianPDF(tt.x, tt.mu, tt.v)
			if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("gaussianPDF(%f, %f, %f) = %f, want %f", tt.x, tt.mu, tt.v, got, tt.want)
			}
		})
	}
}

// maxPermutationAccuracy computes classification accuracy under the best
// label permutation (since HMM state indices are arbitrary).
func maxPermutationAccuracy(decoded, truth []int, nStates int) float64 {
	perms := permutations(nStates)
	bestAcc := 0.0
	for _, perm := range perms {
		correct := 0
		for i := range decoded {
			if perm[decoded[i]] == truth[i] {
				correct++
			}
		}
		acc := float64(correct) / float64(len(decoded))
		if acc > bestAcc {
			bestAcc = acc
		}
	}
	return bestAcc
}

// permutations returns all permutations of [0, 1, ..., n-1].
func permutations(n int) [][]int {
	if n == 0 {
		return [][]int{{}}
	}
	var result [][]int
	generatePerms(make([]int, 0, n), make([]bool, n), n, &result)
	return result
}

func generatePerms(current []int, used []bool, n int, result *[][]int) {
	if len(current) == n {
		perm := make([]int, n)
		copy(perm, current)
		*result = append(*result, perm)
		return
	}
	for i := 0; i < n; i++ {
		if !used[i] {
			used[i] = true
			generatePerms(append(current, i), used, n, result)
			used[i] = false
		}
	}
}

func containsSubstr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
