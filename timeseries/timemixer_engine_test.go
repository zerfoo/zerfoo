package timeseries

import (
	"context"
	"math"
	"testing"
)

// TestTimeMixer_ForwardEngine_Parity verifies that the engine-accelerated
// forward pass produces output matching the pure-Go forward within 1e-4.
func TestTimeMixer_ForwardEngine_Parity(t *testing.T) {
	engine, ops := newTestEngine()

	cfg := TimeMixerConfig{
		InputLen:    32,
		OutputLen:   8,
		NumFeatures: 3,
		NumScales:   4,
	}

	// Create model with engine.
	m := NewTimeMixer(cfg, WithTimeMixerEngine(engine, ops))

	// Deterministic input.
	input := make([][]float64, cfg.NumFeatures)
	for f := 0; f < cfg.NumFeatures; f++ {
		input[f] = make([]float64, cfg.InputLen)
		for i := 0; i < cfg.InputLen; i++ {
			input[f][i] = math.Sin(float64(i)*0.3+float64(f)) + float64(f)*0.5
		}
	}

	// Pure-Go forward.
	cpuOut, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Engine forward.
	engineOut, err := m.ForwardEngine(context.Background(), input)
	if err != nil {
		t.Fatalf("ForwardEngine failed: %v", err)
	}

	if len(engineOut.Scales) != len(cpuOut.Scales) {
		t.Fatalf("scale count mismatch: engine=%d cpu=%d", len(engineOut.Scales), len(cpuOut.Scales))
	}

	const tol = 1e-4
	for s := 0; s < cfg.NumScales; s++ {
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.InputLen; i++ {
				trendDiff := math.Abs(engineOut.Scales[s].trend[f][i] - cpuOut.Scales[s].trend[f][i])
				if trendDiff > tol {
					t.Errorf("scale %d feature %d index %d: trend diff %.6e (engine=%.8f cpu=%.8f)",
						s, f, i, trendDiff, engineOut.Scales[s].trend[f][i], cpuOut.Scales[s].trend[f][i])
				}
				seasonDiff := math.Abs(engineOut.Scales[s].seasonal[f][i] - cpuOut.Scales[s].seasonal[f][i])
				if seasonDiff > tol {
					t.Errorf("scale %d feature %d index %d: seasonal diff %.6e (engine=%.8f cpu=%.8f)",
						s, f, i, seasonDiff, engineOut.Scales[s].seasonal[f][i], cpuOut.Scales[s].seasonal[f][i])
				}
			}
		}
	}
}

// TestTimeMixer_ForwardEngine_DecomposeRoundTrip verifies trend+seasonal=input
// at the decomposition stage (before mixing transforms the components).
func TestTimeMixer_ForwardEngine_DecomposeRoundTrip(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    24,
		OutputLen:   6,
		NumFeatures: 2,
		NumScales:   3,
	}
	m := NewTimeMixer(cfg)

	input := make([][]float64, cfg.NumFeatures)
	for f := range input {
		input[f] = make([]float64, cfg.InputLen)
		for i := range input[f] {
			input[f][i] = math.Cos(float64(i)*0.2) * float64(f+1)
		}
	}

	// Test decomposition directly (before mixing).
	scales := m.decompose(input)

	const tol = 1e-10
	for s, sc := range scales {
		for f := 0; f < cfg.NumFeatures; f++ {
			for i := 0; i < cfg.InputLen; i++ {
				reconstructed := sc.trend[f][i] + sc.seasonal[f][i]
				diff := math.Abs(reconstructed - input[f][i])
				if diff > tol {
					t.Errorf("scale %d feature %d index %d: roundtrip diff %.6e (got=%.8f want=%.8f)",
						s, f, i, diff, reconstructed, input[f][i])
				}
			}
		}
	}
}

// TestTimeMixer_ForwardEngine_NilEngine falls back to CPU path.
func TestTimeMixer_ForwardEngine_NilEngine(t *testing.T) {
	cfg := TimeMixerConfig{
		InputLen:    16,
		OutputLen:   4,
		NumFeatures: 1,
		NumScales:   2,
	}
	m := NewTimeMixer(cfg) // no engine

	input := make([][]float64, 1)
	input[0] = make([]float64, cfg.InputLen)
	for i := range input[0] {
		input[0][i] = float64(i) * 0.1
	}

	out, err := m.ForwardEngine(context.Background(), input)
	if err != nil {
		t.Fatalf("ForwardEngine (nil engine) failed: %v", err)
	}
	if len(out.Scales) != cfg.NumScales {
		t.Errorf("expected %d scales, got %d", cfg.NumScales, len(out.Scales))
	}
}

// TestTimeMixer_ForwardEngine_Validation checks input validation in engine path.
func TestTimeMixer_ForwardEngine_Validation(t *testing.T) {
	engine, ops := newTestEngine()

	cfg := TimeMixerConfig{
		InputLen:    16,
		OutputLen:   4,
		NumFeatures: 2,
		NumScales:   2,
	}
	m := NewTimeMixer(cfg, WithTimeMixerEngine(engine, ops))

	ctx := context.Background()

	if _, err := m.ForwardEngine(ctx, nil); err == nil {
		t.Error("expected error for nil input")
	}
	if _, err := m.ForwardEngine(ctx, make([][]float64, 3)); err == nil {
		t.Error("expected error for wrong feature count")
	}
	bad := make([][]float64, 2)
	bad[0] = make([]float64, 16)
	bad[1] = make([]float64, 10)
	if _, err := m.ForwardEngine(ctx, bad); err == nil {
		t.Error("expected error for wrong input length")
	}
}
