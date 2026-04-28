package federated

import (
	"math"
	"testing"
)

func mustNewDPStrategy(t *testing.T, inner Strategy, config DPConfig) *DPStrategy {
	t.Helper()
	dp, err := NewDPStrategy(inner, config)
	if err != nil {
		t.Fatalf("NewDPStrategy: %v", err)
	}
	return dp
}

func TestDP_NoiseInjection(t *testing.T) {
	tests := []struct {
		name      string
		mechanism string
		epsilon   float64
		delta     float64
		clipNorm  float64
	}{
		{
			name:      "gaussian mechanism adds noise",
			mechanism: "gaussian",
			epsilon:   1.0,
			delta:     1e-5,
			clipNorm:  1.0,
		},
		{
			name:      "laplacian mechanism adds noise",
			mechanism: "laplacian",
			epsilon:   1.0,
			delta:     1e-5,
			clipNorm:  1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp := mustNewDPStrategy(t, NewFedAvg(), DPConfig{
				Epsilon:   tt.epsilon,
				Delta:     tt.delta,
				ClipNorm:  tt.clipNorm,
				Mechanism: tt.mechanism,
			})

			updates := []ModelUpdate{
				{ClientID: "a", Weights: []float64{0.5, 0.5, 0.5}, NSamples: 100},
				{ClientID: "b", Weights: []float64{0.5, 0.5, 0.5}, NSamples: 100},
			}

			agg, err := dp.Aggregate(updates)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// With identical inputs and equal samples, FedAvg would produce
			// exactly [0.5, 0.5, 0.5]. DP noise should perturb at least one weight.
			allExact := true
			for _, w := range agg.Weights {
				if math.Abs(w-0.5) > 1e-15 {
					allExact = false
					break
				}
			}
			if allExact {
				t.Error("expected noise to perturb weights, but all weights are exact")
			}
		})
	}
}

func TestDP_NonDeterministicNoise(t *testing.T) {
	config := DPConfig{
		Epsilon:   1.0,
		Delta:     1e-5,
		ClipNorm:  1.0,
		Mechanism: "gaussian",
	}
	dp1 := mustNewDPStrategy(t, NewFedAvg(), config)
	dp2 := mustNewDPStrategy(t, NewFedAvg(), config)

	updates := []ModelUpdate{
		{ClientID: "a", Weights: []float64{1.0, 2.0, 3.0}, NSamples: 100},
	}

	agg1, err := dp1.Aggregate(updates)
	if err != nil {
		t.Fatalf("dp1.Aggregate: %v", err)
	}
	agg2, err := dp2.Aggregate(updates)
	if err != nil {
		t.Fatalf("dp2.Aggregate: %v", err)
	}

	// Two independently constructed DPStrategy instances should produce
	// different noise (crypto/rand seeding). This could theoretically fail
	// with probability ~0, but 3 float64 values matching is vanishingly unlikely.
	allSame := true
	for i := range agg1.Weights {
		if agg1.Weights[i] != agg2.Weights[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Error("two NewDPStrategy instances produced identical noise; expected non-deterministic seeding")
	}
}

func TestDP_InvalidConfig(t *testing.T) {
	tests := []struct {
		name   string
		config DPConfig
	}{
		{"zero epsilon", DPConfig{Epsilon: 0, Delta: 1e-5, ClipNorm: 1.0, Mechanism: "gaussian"}},
		{"negative epsilon", DPConfig{Epsilon: -1, Delta: 1e-5, ClipNorm: 1.0, Mechanism: "gaussian"}},
		{"zero delta", DPConfig{Epsilon: 1.0, Delta: 0, ClipNorm: 1.0, Mechanism: "gaussian"}},
		{"delta equals 1", DPConfig{Epsilon: 1.0, Delta: 1.0, ClipNorm: 1.0, Mechanism: "gaussian"}},
		{"negative delta", DPConfig{Epsilon: 1.0, Delta: -0.1, ClipNorm: 1.0, Mechanism: "gaussian"}},
		{"zero clip norm", DPConfig{Epsilon: 1.0, Delta: 1e-5, ClipNorm: 0, Mechanism: "gaussian"}},
		{"negative clip norm", DPConfig{Epsilon: 1.0, Delta: 1e-5, ClipNorm: -1.0, Mechanism: "gaussian"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewDPStrategy(NewFedAvg(), tt.config)
			if err == nil {
				t.Error("expected error for invalid config, got nil")
			}
		})
	}
}

func TestDP_GradientClipping(t *testing.T) {
	tests := []struct {
		name     string
		weights  []float64
		clipNorm float64
		wantNorm float64
	}{
		{
			name:     "within norm is unchanged",
			weights:  []float64{0.3, 0.4},
			clipNorm: 1.0,
			wantNorm: 0.5,
		},
		{
			name:     "exceeds norm is clipped",
			weights:  []float64{3.0, 4.0},
			clipNorm: 1.0,
			wantNorm: 1.0,
		},
		{
			name:     "exactly at norm is unchanged",
			weights:  []float64{0.6, 0.8},
			clipNorm: 1.0,
			wantNorm: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clipped := clipL2(tt.weights, tt.clipNorm)
			norm := 0.0
			for _, w := range clipped {
				norm += w * w
			}
			norm = math.Sqrt(norm)
			if math.Abs(norm-tt.wantNorm) > 1e-9 {
				t.Errorf("clipped norm = %f, want %f", norm, tt.wantNorm)
			}
		})
	}
}

func TestDP_GaussianSigma(t *testing.T) {
	dp := mustNewDPStrategy(t, NewFedAvg(), DPConfig{
		Epsilon:   1.0,
		Delta:     1e-5,
		ClipNorm:  1.0,
		Mechanism: "gaussian",
	})
	// σ = ClipNorm * sqrt(2 * ln(1.25/δ)) / ε
	// σ = 1.0 * sqrt(2 * ln(125000)) / 1.0
	want := math.Sqrt(2 * math.Log(1.25/1e-5))
	got := dp.gaussianSigma()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("gaussianSigma() = %f, want %f", got, want)
	}
}

func TestDP_PrivacyBudget(t *testing.T) {
	tests := []struct {
		name         string
		rounds       int
		epsilon      float64
		delta        float64
		maxEpsilon   float64
		wantEpsilon  float64
		wantDelta    float64
		wantContinue bool
	}{
		{
			name:         "single round within budget",
			rounds:       1,
			epsilon:      1.0,
			delta:        1e-5,
			maxEpsilon:   10.0,
			wantEpsilon:  1.0,
			wantDelta:    1e-5,
			wantContinue: true,
		},
		{
			name:         "multiple rounds accumulate",
			rounds:       5,
			epsilon:      1.0,
			delta:        1e-5,
			maxEpsilon:   10.0,
			wantEpsilon:  5.0,
			wantDelta:    5e-5,
			wantContinue: true,
		},
		{
			name:         "budget exceeded",
			rounds:       10,
			epsilon:      1.0,
			delta:        1e-5,
			maxEpsilon:   5.0,
			wantEpsilon:  10.0,
			wantDelta:    10e-5,
			wantContinue: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp := mustNewDPStrategy(t, NewFedAvg(), DPConfig{
				Epsilon:   tt.epsilon,
				Delta:     tt.delta,
				ClipNorm:  1.0,
				Mechanism: "gaussian",
			})

			updates := []ModelUpdate{
				{ClientID: "a", Weights: []float64{0.5, 0.5}, NSamples: 100},
			}

			for i := 0; i < tt.rounds; i++ {
				_, err := dp.Aggregate(updates)
				if err != nil {
					t.Fatalf("round %d: unexpected error: %v", i, err)
				}
			}

			acc := dp.Accountant()
			gotEps, gotDelta := acc.Spent()
			if math.Abs(gotEps-tt.wantEpsilon) > 1e-9 {
				t.Errorf("epsilon spent = %f, want %f", gotEps, tt.wantEpsilon)
			}
			if math.Abs(gotDelta-tt.wantDelta) > 1e-9 {
				t.Errorf("delta spent = %f, want %f", gotDelta, tt.wantDelta)
			}
			if acc.CanContinue(tt.maxEpsilon) != tt.wantContinue {
				t.Errorf("CanContinue(%f) = %v, want %v", tt.maxEpsilon, acc.CanContinue(tt.maxEpsilon), tt.wantContinue)
			}
		})
	}
}

func TestDP_UnsupportedMechanism(t *testing.T) {
	dp := mustNewDPStrategy(t, NewFedAvg(), DPConfig{
		Epsilon:   1.0,
		Delta:     1e-5,
		ClipNorm:  1.0,
		Mechanism: "unknown",
	})
	updates := []ModelUpdate{
		{ClientID: "a", Weights: []float64{1.0}, NSamples: 100},
	}
	_, err := dp.Aggregate(updates)
	if err == nil {
		t.Fatal("expected error for unsupported mechanism")
	}
}

func TestDP_SelectClientsDelegates(t *testing.T) {
	dp := mustNewDPStrategy(t, NewFedAvg(), DPConfig{
		Epsilon:   1.0,
		Delta:     1e-5,
		ClipNorm:  1.0,
		Mechanism: "gaussian",
	})
	ids := []ClientID{"a", "b", "c"}
	selected := dp.SelectClients(0, ids)
	if len(selected) != len(ids) {
		t.Fatalf("expected %d selected, got %d", len(ids), len(selected))
	}
}
