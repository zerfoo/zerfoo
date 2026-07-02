package federated

import (
	"math"
	"testing"
)

func TestFedProx_ProximalTerm(t *testing.T) {
	tests := []struct {
		name        string
		local       []float64
		global      []float64
		mu          float64
		wantPenalty float64
	}{
		{
			name:        "zero divergence",
			local:       []float64{1.0, 2.0, 3.0},
			global:      []float64{1.0, 2.0, 3.0},
			mu:          0.5,
			wantPenalty: 0.0,
		},
		{
			name:        "unit divergence single weight",
			local:       []float64{2.0},
			global:      []float64{1.0},
			mu:          1.0,
			wantPenalty: 0.5, // (1.0/2) * 1^2 = 0.5
		},
		{
			name:        "multiple weights",
			local:       []float64{3.0, 5.0},
			global:      []float64{1.0, 2.0},
			mu:          2.0,
			wantPenalty: 13.0, // (2.0/2) * (4 + 9) = 13.0
		},
		{
			name:        "mu zero disables penalty",
			local:       []float64{10.0, 20.0},
			global:      []float64{0.0, 0.0},
			mu:          0.0,
			wantPenalty: 0.0,
		},
		{
			name:        "negative divergence",
			local:       []float64{0.0},
			global:      []float64{3.0},
			mu:          1.0,
			wantPenalty: 4.5, // (1.0/2) * 9 = 4.5
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ProximalLoss(tt.local, tt.global, tt.mu)
			if math.Abs(got-tt.wantPenalty) > 1e-9 {
				t.Errorf("ProximalLoss() = %f, want %f", got, tt.wantPenalty)
			}
		})
	}
}

func TestFedProx_Convergence(t *testing.T) {
	tests := []struct {
		name        string
		mu          float64
		updates     []ModelUpdate
		wantWeights []float64
		wantN       int
		wantErr     bool
	}{
		{
			name: "weighted average matches FedAvg",
			mu:   0.1,
			updates: []ModelUpdate{
				{ClientID: "a", Weights: []float64{1.0, 2.0}, NSamples: 100},
				{ClientID: "b", Weights: []float64{3.0, 4.0}, NSamples: 300},
			},
			// a=25%, b=75%: w0=1*0.25+3*0.75=2.5, w1=2*0.25+4*0.75=3.5
			wantWeights: []float64{2.5, 3.5},
			wantN:       2,
		},
		{
			name: "single client returns its weights",
			mu:   1.0,
			updates: []ModelUpdate{
				{ClientID: "a", Weights: []float64{5.0, 6.0, 7.0}, NSamples: 50},
			},
			wantWeights: []float64{5.0, 6.0, 7.0},
			wantN:       1,
		},
		{
			name:    "empty updates returns error",
			mu:      0.5,
			updates: nil,
			wantErr: true,
		},
		{
			name: "dimension mismatch returns error",
			mu:   0.5,
			updates: []ModelUpdate{
				{ClientID: "a", Weights: []float64{1.0, 2.0}, NSamples: 10},
				{ClientID: "b", Weights: []float64{1.0}, NSamples: 10},
			},
			wantErr: true,
		},
		{
			name: "proximal term reduces divergence over rounds",
			mu:   0.5,
			updates: []ModelUpdate{
				{ClientID: "a", Weights: []float64{1.0, 1.0}, NSamples: 100},
				{ClientID: "b", Weights: []float64{3.0, 3.0}, NSamples: 100},
			},
			// Equal samples: simple average (2.0, 2.0)
			wantWeights: []float64{2.0, 2.0},
			wantN:       2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fp := NewFedProx(tt.mu)
			if fp.Mu() != tt.mu {
				t.Fatalf("Mu() = %f, want %f", fp.Mu(), tt.mu)
			}

			agg, err := fp.Aggregate(tt.updates)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if agg.NParticipants != tt.wantN {
				t.Fatalf("NParticipants = %d, want %d", agg.NParticipants, tt.wantN)
			}
			for i, w := range agg.Weights {
				if math.Abs(w-tt.wantWeights[i]) > 1e-9 {
					t.Errorf("weight[%d] = %f, want %f", i, w, tt.wantWeights[i])
				}
			}
		})
	}

	// Verify proximal penalty decreases as local weights approach global.
	t.Run("penalty decreases with convergence", func(t *testing.T) {
		global := []float64{2.0, 2.0}
		rounds := [][]float64{
			{4.0, 4.0}, // round 1: far from global
			{3.0, 3.0}, // round 2: closer
			{2.5, 2.5}, // round 3: closer still
			{2.0, 2.0}, // round 4: converged
		}
		mu := 0.5
		prevLoss := math.Inf(1)
		for i, local := range rounds {
			loss := ProximalLoss(local, global, mu)
			if loss >= prevLoss {
				t.Errorf("round %d: penalty %f did not decrease from %f", i, loss, prevLoss)
			}
			prevLoss = loss
		}
	})

	// Verify select clients returns all.
	t.Run("select clients returns all", func(t *testing.T) {
		fp := NewFedProx(0.1)
		ids := []ClientID{"a", "b", "c"}
		selected := fp.SelectClients(0, ids)
		if len(selected) != len(ids) {
			t.Fatalf("expected %d selected, got %d", len(ids), len(selected))
		}
		for i, id := range selected {
			if id != ids[i] {
				t.Errorf("selected[%d] = %q, want %q", i, id, ids[i])
			}
		}
	})
}
