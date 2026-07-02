package federated

import (
	"math"
	"testing"
)

// mockClient implements Client for testing.
type mockClient struct {
	id       ClientID
	nSamples int
	weights  []float64
	err      error
}

func (m *mockClient) ID() ClientID { return m.id }

func (m *mockClient) Train(_ []float64) (*ModelUpdate, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &ModelUpdate{
		ClientID: m.id,
		Weights:  m.weights,
		NSamples: m.nSamples,
		Metrics:  map[string]float64{"loss": 0.1},
	}, nil
}

func TestFederatedStrategy_FedAvg(t *testing.T) {
	avg := NewFedAvg()

	t.Run("weighted average is correct", func(t *testing.T) {
		updates := []ModelUpdate{
			{ClientID: "a", Weights: []float64{1.0, 2.0, 3.0}, NSamples: 100},
			{ClientID: "b", Weights: []float64{3.0, 4.0, 5.0}, NSamples: 300},
		}
		// Expected: a contributes 25%, b contributes 75%.
		// w0 = 1.0*0.25 + 3.0*0.75 = 2.5
		// w1 = 2.0*0.25 + 4.0*0.75 = 3.5
		// w2 = 3.0*0.25 + 5.0*0.75 = 4.5
		agg, err := avg.Aggregate(updates)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if agg.NParticipants != 2 {
			t.Fatalf("expected 2 participants, got %d", agg.NParticipants)
		}
		want := []float64{2.5, 3.5, 4.5}
		for i, w := range agg.Weights {
			if math.Abs(w-want[i]) > 1e-9 {
				t.Errorf("weight[%d] = %f, want %f", i, w, want[i])
			}
		}
	})

	t.Run("equal samples gives simple average", func(t *testing.T) {
		updates := []ModelUpdate{
			{ClientID: "a", Weights: []float64{2.0, 4.0}, NSamples: 50},
			{ClientID: "b", Weights: []float64{6.0, 8.0}, NSamples: 50},
		}
		agg, err := avg.Aggregate(updates)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []float64{4.0, 6.0}
		for i, w := range agg.Weights {
			if math.Abs(w-want[i]) > 1e-9 {
				t.Errorf("weight[%d] = %f, want %f", i, w, want[i])
			}
		}
	})

	t.Run("single client", func(t *testing.T) {
		updates := []ModelUpdate{
			{ClientID: "a", Weights: []float64{1.0, 2.0, 3.0}, NSamples: 100},
		}
		agg, err := avg.Aggregate(updates)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		for i, w := range agg.Weights {
			if w != updates[0].Weights[i] {
				t.Errorf("weight[%d] = %f, want %f", i, w, updates[0].Weights[i])
			}
		}
	})

	t.Run("empty updates returns error", func(t *testing.T) {
		_, err := avg.Aggregate(nil)
		if err == nil {
			t.Fatal("expected error for empty updates")
		}
	})

	t.Run("dimension mismatch returns error", func(t *testing.T) {
		updates := []ModelUpdate{
			{ClientID: "a", Weights: []float64{1.0, 2.0}, NSamples: 10},
			{ClientID: "b", Weights: []float64{1.0}, NSamples: 10},
		}
		_, err := avg.Aggregate(updates)
		if err == nil {
			t.Fatal("expected error for dimension mismatch")
		}
	})

	t.Run("select clients returns all", func(t *testing.T) {
		ids := []ClientID{"a", "b", "c"}
		selected := avg.SelectClients(0, ids)
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

func TestFederatedStrategy_RoundCoordination(t *testing.T) {
	clients := []Client{
		&mockClient{id: "c1", nSamples: 100, weights: []float64{1.0, 2.0}},
		&mockClient{id: "c2", nSamples: 200, weights: []float64{4.0, 5.0}},
		&mockClient{id: "c3", nSamples: 300, weights: []float64{7.0, 8.0}},
	}

	coord := NewCoordinator(NewFedAvg(), CoordinatorConfig{
		MinClients: 2,
		MaxRounds:  10,
	})

	t.Run("round runs end-to-end", func(t *testing.T) {
		result, err := coord.RunRound(clients)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result.Model == nil {
			t.Fatal("expected non-nil model")
		}
		if result.Model.NParticipants != 3 {
			t.Fatalf("expected 3 participants, got %d", result.Model.NParticipants)
		}
		if result.Model.Round != 1 {
			t.Fatalf("expected round 1, got %d", result.Model.Round)
		}
		if len(result.Updates) != 3 {
			t.Fatalf("expected 3 updates, got %d", len(result.Updates))
		}
		// Verify weighted average: total=600
		// w0 = 1*100/600 + 4*200/600 + 7*300/600 = 100/600 + 800/600 + 2100/600 = 3000/600 = 5.0
		// w1 = 2*100/600 + 5*200/600 + 8*300/600 = 200/600 + 1000/600 + 2400/600 = 3600/600 = 6.0
		want := []float64{5.0, 6.0}
		for i, w := range result.Model.Weights {
			if math.Abs(w-want[i]) > 1e-9 {
				t.Errorf("weight[%d] = %f, want %f", i, w, want[i])
			}
		}
	})

	t.Run("round counter increments", func(t *testing.T) {
		if coord.Round() != 1 {
			t.Fatalf("expected round 1 after first round, got %d", coord.Round())
		}
		result, err := coord.RunRound(clients)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if result.Model.Round != 2 {
			t.Fatalf("expected round 2, got %d", result.Model.Round)
		}
		if coord.Round() != 2 {
			t.Fatalf("expected round 2, got %d", coord.Round())
		}
	})

	t.Run("no clients returns error", func(t *testing.T) {
		c := NewCoordinator(NewFedAvg(), CoordinatorConfig{MinClients: 1})
		_, err := c.RunRound(nil)
		if err == nil {
			t.Fatal("expected error for no clients")
		}
	})

	t.Run("not enough clients returns error", func(t *testing.T) {
		c := NewCoordinator(NewFedAvg(), CoordinatorConfig{MinClients: 5})
		_, err := c.RunRound(clients)
		if err == nil {
			t.Fatal("expected error when fewer clients than MinClients")
		}
	})
}
