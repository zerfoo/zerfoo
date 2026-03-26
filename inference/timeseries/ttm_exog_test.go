package timeseries

import (
	"math"
	"testing"
)

func testTTMExogConfig() *TTMConfig {
	return &TTMConfig{
		ContextLen:     64,
		ForecastLen:    16,
		NumChannels:    2,
		PatchLen:       8,
		DModel:         16,
		NumMixerLayers: 1,
		ChannelMixing:  false,
		Expansion:      2,
		NumExogenous:   3,
	}
}

func makeExogenousInput(forecastLen, numExog int) [][]float64 {
	exog := make([][]float64, forecastLen)
	for t := range forecastLen {
		exog[t] = make([]float64, numExog)
		for e := range numExog {
			exog[t][e] = float64(t+1) * float64(e+1) * 0.5
		}
	}
	return exog
}

func TestTTMExogForecastShape(t *testing.T) {
	cfg := testTTMExogConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	exog := makeExogenousInput(cfg.ForecastLen, cfg.NumExogenous)

	output, err := inf.ForecastWithExogenous(input, exog)
	if err != nil {
		t.Fatalf("ForecastWithExogenous: %v", err)
	}

	if len(output) != cfg.ForecastLen {
		t.Errorf("output forecast_len: got %d, want %d", len(output), cfg.ForecastLen)
	}
	for i, row := range output {
		if len(row) != cfg.NumChannels {
			t.Errorf("output[%d] channels: got %d, want %d", i, len(row), cfg.NumChannels)
		}
	}
}

func TestTTMExogDifferentFromBaseline(t *testing.T) {
	cfg := testTTMExogConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	exog := makeExogenousInput(cfg.ForecastLen, cfg.NumExogenous)

	// Forecast with exogenous.
	withExog, err := inf.ForecastWithExogenous(input, exog)
	if err != nil {
		t.Fatalf("ForecastWithExogenous: %v", err)
	}

	// Forecast without exogenous (baseline). Because ForecastWithExogenous
	// sets exogenous on the node, a plain Forecast should produce different
	// results since there are no exogenous variables influencing it.
	withoutExog, err := inf.Forecast(input)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	// The forecasts should differ because exogenous variables modify the
	// forecast head input.
	allSame := true
	for ti := range cfg.ForecastLen {
		for c := range cfg.NumChannels {
			if math.Abs(withExog[ti][c]-withoutExog[ti][c]) > 1e-10 {
				allSame = false
				break
			}
		}
		if !allSame {
			break
		}
	}
	if allSame {
		t.Error("forecast with exogenous should differ from forecast without, but they are identical")
	}
}

func TestTTMExogWrongForecastLen(t *testing.T) {
	cfg := testTTMExogConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	// Wrong forecast length for exogenous.
	exog := makeExogenousInput(cfg.ForecastLen+5, cfg.NumExogenous)

	_, err = inf.ForecastWithExogenous(input, exog)
	if err == nil {
		t.Error("expected error for wrong exogenous forecast_len, got nil")
	}
}

func TestTTMExogWrongNumExog(t *testing.T) {
	cfg := testTTMExogConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	// Wrong number of exogenous channels.
	exog := makeExogenousInput(cfg.ForecastLen, cfg.NumExogenous+2)

	_, err = inf.ForecastWithExogenous(input, exog)
	if err == nil {
		t.Error("expected error for wrong num_exog, got nil")
	}
}

func TestTTMExogZeroBackwardCompat(t *testing.T) {
	// With NumExogenous=0, the model should work exactly like before.
	cfg := testTTMConfig() // uses NumExogenous=0 (default)
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	output, err := inf.Forecast(input)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	if len(output) != cfg.ForecastLen {
		t.Errorf("output forecast_len: got %d, want %d", len(output), cfg.ForecastLen)
	}
	for i, row := range output {
		if len(row) != cfg.NumChannels {
			t.Errorf("output[%d] channels: got %d, want %d", i, len(row), cfg.NumChannels)
		}
	}
}

func TestTTMExogNotConfiguredError(t *testing.T) {
	// A model with NumExogenous=0 should reject ForecastWithExogenous.
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	exog := makeExogenousInput(cfg.ForecastLen, 2)

	_, err = inf.ForecastWithExogenous(input, exog)
	if err == nil {
		t.Error("expected error when calling ForecastWithExogenous on model with NumExogenous=0")
	}
}

func TestTTMExogNonZeroOutput(t *testing.T) {
	cfg := testTTMExogConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	exog := makeExogenousInput(cfg.ForecastLen, cfg.NumExogenous)

	output, err := inf.ForecastWithExogenous(input, exog)
	if err != nil {
		t.Fatalf("ForecastWithExogenous: %v", err)
	}

	allZero := true
	for _, row := range output {
		for _, v := range row {
			if v != 0 {
				allZero = false
				break
			}
		}
	}
	if allZero {
		t.Error("output is all zeros, expected non-zero predictions")
	}
}

func TestTTMExogFiniteOutput(t *testing.T) {
	cfg := testTTMExogConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	exog := makeExogenousInput(cfg.ForecastLen, cfg.NumExogenous)

	output, err := inf.ForecastWithExogenous(input, exog)
	if err != nil {
		t.Fatalf("ForecastWithExogenous: %v", err)
	}

	for ti, row := range output {
		for c, v := range row {
			if math.IsNaN(v) {
				t.Errorf("output[%d][%d] is NaN", ti, c)
			}
			if math.IsInf(v, 0) {
				t.Errorf("output[%d][%d] is Inf", ti, c)
			}
		}
	}
}
