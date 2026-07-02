package main

import "testing"

func TestSimulation_SharedPromptHitsAboveZero(t *testing.T) {
	stats := RunSimulation(Config{
		Users:             10,
		Turns:             5,
		SysPromptTokens:   256,
		HistTokensPerTurn: 32,
		BlockSize:         16,
	})

	if stats.TotalRequests != 50 {
		t.Errorf("TotalRequests = %d, want 50", stats.TotalRequests)
	}

	if stats.Hits == 0 {
		t.Error("expected at least one cache hit with shared system prompt")
	}

	if stats.HitRate <= 0 {
		t.Errorf("HitRate = %f, want > 0", stats.HitRate)
	}

	t.Logf("hit rate: %.1f%%, TTFT reduction: %.1f%%", stats.HitRate*100, stats.TTFTReduction*100)
}

func TestSimulation_HitRateAbove60Percent(t *testing.T) {
	stats := RunSimulation(Config{
		Users:             10,
		Turns:             5,
		SysPromptTokens:   256,
		HistTokensPerTurn: 32,
		BlockSize:         16,
	})

	if stats.HitRate < 0.60 {
		t.Errorf("HitRate = %.1f%%, want >= 60%%", stats.HitRate*100)
	}

	t.Logf("hit rate: %.1f%% (target >= 60%%)", stats.HitRate*100)
}

func TestSimulation_TTFTReductionAbove40Percent(t *testing.T) {
	stats := RunSimulation(Config{
		Users:             10,
		Turns:             5,
		SysPromptTokens:   256,
		HistTokensPerTurn: 32,
		BlockSize:         16,
	})

	if stats.TTFTReduction < 0.40 {
		t.Errorf("TTFTReduction = %.1f%%, want >= 40%%", stats.TTFTReduction*100)
	}

	t.Logf("TTFT reduction: %.1f%% (target >= 40%%)", stats.TTFTReduction*100)
}

func TestSimulation_SingleUser(t *testing.T) {
	stats := RunSimulation(Config{
		Users:             1,
		Turns:             3,
		SysPromptTokens:   64,
		HistTokensPerTurn: 16,
		BlockSize:         16,
	})

	if stats.TotalRequests != 3 {
		t.Errorf("TotalRequests = %d, want 3", stats.TotalRequests)
	}

	// First turn is always a miss, subsequent turns should hit on system prompt.
	if stats.Misses < 1 {
		t.Error("expected at least one miss (first turn)")
	}

	t.Logf("hits=%d misses=%d rate=%.1f%%", stats.Hits, stats.Misses, stats.HitRate*100)
}

func TestSimulation_NoUsers(t *testing.T) {
	stats := RunSimulation(Config{
		Users:             0,
		Turns:             5,
		SysPromptTokens:   256,
		HistTokensPerTurn: 32,
		BlockSize:         16,
	})

	if stats.TotalRequests != 0 {
		t.Errorf("TotalRequests = %d, want 0", stats.TotalRequests)
	}
}
