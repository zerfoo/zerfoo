package registry

import (
	"fmt"
	"math"
	"testing"
)

func TestABRouterDeterminism(t *testing.T) {
	r := NewABRouter(ABConfig{
		ChampionID:       "champion-v1",
		ChallengerID:     "challenger-v2",
		ChallengerWeight: 0.5,
	})
	for i := 0; i < 100; i++ {
		reqID := fmt.Sprintf("req-%d", i)
		first := r.Route(reqID)
		for j := 0; j < 100; j++ {
			got := r.Route(reqID)
			if got != first {
				t.Fatalf("Route(%q) returned %q on call %d, expected %q", reqID, got, j, first)
			}
		}
	}
}

func TestABRouterDistribution(t *testing.T) {
	const (
		n              = 10000
		targetWeight   = 0.1
		tolerancePct   = 0.02
	)
	r := NewABRouter(ABConfig{
		ChampionID:       "champion",
		ChallengerID:     "challenger",
		ChallengerWeight: targetWeight,
	})

	challengerCount := 0
	for i := 0; i < n; i++ {
		if r.Route(fmt.Sprintf("unique-request-%d", i)) == "challenger" {
			challengerCount++
		}
	}

	ratio := float64(challengerCount) / float64(n)
	if math.Abs(ratio-targetWeight) > tolerancePct {
		t.Fatalf("challenger ratio = %.4f, want %.2f ± %.2f", ratio, targetWeight, tolerancePct)
	}
}

func TestABRouterStats(t *testing.T) {
	r := NewABRouter(ABConfig{
		ChampionID:       "champion",
		ChallengerID:     "challenger",
		ChallengerWeight: 0.5,
	})

	const n = 1000
	for i := 0; i < n; i++ {
		r.Route(fmt.Sprintf("stats-req-%d", i))
	}

	s := r.Stats()
	total := s.ChampionRequests + s.ChallengerRequests
	if total != n {
		t.Fatalf("total requests = %d, want %d", total, n)
	}
	if s.ChampionRequests == 0 || s.ChallengerRequests == 0 {
		t.Fatalf("expected both counters > 0, got champion=%d challenger=%d",
			s.ChampionRequests, s.ChallengerRequests)
	}
}

func TestABRouterUpdateWeights(t *testing.T) {
	r := NewABRouter(ABConfig{
		ChampionID:       "champion",
		ChallengerID:     "challenger",
		ChallengerWeight: 0.0,
	})

	// With weight 0, everything goes to champion.
	for i := 0; i < 100; i++ {
		if got := r.Route(fmt.Sprintf("w0-%d", i)); got != "champion" {
			t.Fatalf("with weight 0, Route returned %q, want champion", got)
		}
	}

	// Update to weight 1.0 — everything goes to challenger.
	if err := r.UpdateWeights(1.0); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		if got := r.Route(fmt.Sprintf("w1-%d", i)); got != "challenger" {
			t.Fatalf("with weight 1, Route returned %q, want challenger", got)
		}
	}

	// Invalid weights return errors.
	if err := r.UpdateWeights(-0.1); err == nil {
		t.Fatal("expected error for negative weight")
	}
	if err := r.UpdateWeights(1.1); err == nil {
		t.Fatal("expected error for weight > 1")
	}

	// Update to 0.5 and verify distribution shifts.
	if err := r.UpdateWeights(0.5); err != nil {
		t.Fatal(err)
	}
	challengerCount := 0
	const n = 10000
	for i := 0; i < n; i++ {
		if r.Route(fmt.Sprintf("w05-%d", i)) == "challenger" {
			challengerCount++
		}
	}
	ratio := float64(challengerCount) / float64(n)
	if math.Abs(ratio-0.5) > 0.05 {
		t.Fatalf("after update to 0.5, challenger ratio = %.4f, want ~0.5", ratio)
	}
}
