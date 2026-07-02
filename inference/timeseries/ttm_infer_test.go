package timeseries

import (
	"math"
	"testing"
)

func testTTMConfig() *TTMConfig {
	return &TTMConfig{
		ContextLen:     64,
		ForecastLen:    16,
		NumChannels:    2,
		PatchLen:       8,
		DModel:         16,
		NumMixerLayers: 1,
		ChannelMixing:  false,
		Expansion:      2,
	}
}

func makeSinusoidalInput(contextLen, numChannels int) [][]float64 {
	input := make([][]float64, contextLen)
	for t := range contextLen {
		input[t] = make([]float64, numChannels)
		for c := range numChannels {
			// Sinusoidal with channel-dependent frequency and offset.
			freq := 2.0 * math.Pi * float64(c+1) / float64(contextLen)
			input[t][c] = 100.0*math.Sin(freq*float64(t)) + 500.0*float64(c+1)
		}
	}
	return input
}

func TestTTMInferForecastSinusoidalShape(t *testing.T) {
	cfg := testTTMConfig()
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

func TestTTMInferForecastSinusoidalNonZero(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	output, err := inf.Forecast(input)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
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

func TestTTMInferNormalizationOutputScale(t *testing.T) {
	// Verify that the output is in the original input scale, not the
	// normalized scale. We use a large-valued input (mean ~500) and check
	// that the output is not all near zero.
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	output, err := inf.Forecast(input)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	// Compute the mean of the input for comparison.
	var inputMean float64
	for _, row := range input {
		for _, v := range row {
			inputMean += v
		}
	}
	inputMean /= float64(cfg.ContextLen * cfg.NumChannels)

	// The output should have values in a comparable range to the input
	// (not near zero, which would indicate it's still in normalized scale).
	// We check that the mean of absolute output values is > 1.0 (input
	// mean is ~750, so denormalized output should be in that range).
	var outputAbsMean float64
	for _, row := range output {
		for _, v := range row {
			outputAbsMean += math.Abs(v)
		}
	}
	outputAbsMean /= float64(cfg.ForecastLen * cfg.NumChannels)

	// With random weights, exact values are unpredictable, but with
	// denormalization the output should not be near zero when input mean
	// is ~750.
	if outputAbsMean < 1.0 && inputMean > 100.0 {
		t.Errorf("output appears to be in normalized scale: outputAbsMean=%.4f, inputMean=%.4f",
			outputAbsMean, inputMean)
	}
}

func TestTTMInferForecastBatchShape(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	batchSize := 3
	inputs := make([][][]float64, batchSize)
	for b := range batchSize {
		inputs[b] = makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
		// Add batch-dependent offset so each series differs.
		for ti := range cfg.ContextLen {
			for c := range cfg.NumChannels {
				inputs[b][ti][c] += float64(b) * 100.0
			}
		}
	}

	outputs, err := inf.ForecastBatch(inputs)
	if err != nil {
		t.Fatalf("ForecastBatch: %v", err)
	}

	if len(outputs) != batchSize {
		t.Fatalf("output batch size: got %d, want %d", len(outputs), batchSize)
	}
	for b, batchOut := range outputs {
		if len(batchOut) != cfg.ForecastLen {
			t.Errorf("batch[%d] forecast_len: got %d, want %d", b, len(batchOut), cfg.ForecastLen)
		}
		for ti, row := range batchOut {
			if len(row) != cfg.NumChannels {
				t.Errorf("batch[%d][%d] channels: got %d, want %d", b, ti, len(row), cfg.NumChannels)
			}
		}
	}
}

func TestTTMInferForecastBatchNonZero(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	batchSize := 2
	inputs := make([][][]float64, batchSize)
	for b := range batchSize {
		inputs[b] = makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels)
	}

	outputs, err := inf.ForecastBatch(inputs)
	if err != nil {
		t.Fatalf("ForecastBatch: %v", err)
	}

	for b, batchOut := range outputs {
		allZero := true
		for _, row := range batchOut {
			for _, v := range row {
				if v != 0 {
					allZero = false
					break
				}
			}
		}
		if allZero {
			t.Errorf("batch[%d] output is all zeros", b)
		}
	}
}

func TestTTMInferInputValidationWrongContextLen(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	// Wrong context length.
	input := makeSinusoidalInput(cfg.ContextLen+10, cfg.NumChannels)
	_, err = inf.Forecast(input)
	if err == nil {
		t.Error("expected error for wrong context_len, got nil")
	}
}

func TestTTMInferInputValidationWrongChannels(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	// Wrong number of channels.
	input := makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels+1)
	_, err = inf.Forecast(input)
	if err == nil {
		t.Error("expected error for wrong channels, got nil")
	}
}

func TestTTMInferBatchValidationWrongContextLen(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	inputs := [][][]float64{
		makeSinusoidalInput(cfg.ContextLen, cfg.NumChannels),
		makeSinusoidalInput(cfg.ContextLen+5, cfg.NumChannels), // bad
	}
	_, err = inf.ForecastBatch(inputs)
	if err == nil {
		t.Error("expected error for batch with wrong context_len, got nil")
	}
}

func TestTTMInferBatchEmpty(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	_, err = inf.ForecastBatch(nil)
	if err == nil {
		t.Error("expected error for empty batch, got nil")
	}
}

func TestTTMInferConfig(t *testing.T) {
	cfg := testTTMConfig()
	inf, err := newTTMInferenceFromConfig(cfg)
	if err != nil {
		t.Fatalf("newTTMInferenceFromConfig: %v", err)
	}

	got := inf.Config()
	if got.ContextLen != cfg.ContextLen {
		t.Errorf("Config().ContextLen: got %d, want %d", got.ContextLen, cfg.ContextLen)
	}
	if got.ForecastLen != cfg.ForecastLen {
		t.Errorf("Config().ForecastLen: got %d, want %d", got.ForecastLen, cfg.ForecastLen)
	}
	if got.NumChannels != cfg.NumChannels {
		t.Errorf("Config().NumChannels: got %d, want %d", got.NumChannels, cfg.NumChannels)
	}
}

func TestSelectTTMVariantExactMatch(t *testing.T) {
	tests := []struct {
		contextLen  int
		forecastLen int
		want        string
	}{
		{512, 96, "ttm-512-96"},
		{1024, 96, "ttm-1024-96"},
		{1536, 96, "ttm-1536-96"},
		{512, 192, "ttm-512-192"},
		{512, 336, "ttm-512-336"},
		{512, 720, "ttm-512-720"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := SelectTTMVariant(tt.contextLen, tt.forecastLen)
			if got != tt.want {
				t.Errorf("SelectTTMVariant(%d, %d) = %q, want %q",
					tt.contextLen, tt.forecastLen, got, tt.want)
			}
		})
	}
}

func TestSelectTTMVariantSmallestCovering(t *testing.T) {
	// Request 256 context, 48 forecast -> smallest covering is 512-96.
	got := SelectTTMVariant(256, 48)
	if got != "ttm-512-96" {
		t.Errorf("SelectTTMVariant(256, 48) = %q, want %q", got, "ttm-512-96")
	}

	// Request 600 context, 96 forecast -> smallest covering is 1024-96.
	got = SelectTTMVariant(600, 96)
	if got != "ttm-1024-96" {
		t.Errorf("SelectTTMVariant(600, 96) = %q, want %q", got, "ttm-1024-96")
	}

	// Request 512 context, 200 forecast -> smallest covering is 512-336.
	got = SelectTTMVariant(512, 200)
	if got != "ttm-512-336" {
		t.Errorf("SelectTTMVariant(512, 200) = %q, want %q", got, "ttm-512-336")
	}
}

func TestSelectTTMVariantFallbackToLargest(t *testing.T) {
	// Request exceeding all variants -> falls back to largest.
	got := SelectTTMVariant(2000, 1000)
	if got != "ttm-1536-96" {
		t.Errorf("SelectTTMVariant(2000, 1000) = %q, want %q", got, "ttm-1536-96")
	}
}

func TestChannelMeanStd(t *testing.T) {
	contextLen := 100
	numChannels := 2
	input := make([][]float64, contextLen)
	for i := range contextLen {
		input[i] = make([]float64, numChannels)
		input[i][0] = 10.0 // constant channel 0
		input[i][1] = float64(i)
	}

	mean, std := channelMeanStd(input, contextLen, numChannels)

	// Channel 0: mean=10.0, std=0.0
	if math.Abs(mean[0]-10.0) > 1e-10 {
		t.Errorf("mean[0]: got %f, want 10.0", mean[0])
	}
	if std[0] != 0.0 {
		t.Errorf("std[0]: got %f, want 0.0", std[0])
	}

	// Channel 1: mean=49.5, std~28.87
	if math.Abs(mean[1]-49.5) > 1e-10 {
		t.Errorf("mean[1]: got %f, want 49.5", mean[1])
	}
	expectedStd := math.Sqrt(float64(99*100*199) / float64(6*100*100))
	// Population std of 0..99 = sqrt(sum((i-49.5)^2)/100)
	expectedStd = 0
	var sumSq float64
	for i := range contextLen {
		diff := float64(i) - 49.5
		sumSq += diff * diff
	}
	expectedStd = math.Sqrt(sumSq / float64(contextLen))
	if math.Abs(std[1]-expectedStd) > 0.01 {
		t.Errorf("std[1]: got %f, want ~%f", std[1], expectedStd)
	}
}

func TestNormalizeDenormalizeRoundTrip(t *testing.T) {
	contextLen := 50
	numChannels := 3
	input := makeSinusoidalInput(contextLen, numChannels)

	mean, std := channelMeanStd(input, contextLen, numChannels)
	normalized := normalizeChannels(input, mean, std, contextLen, numChannels)
	recovered := denormalizeChannels(normalized, mean, std, contextLen, numChannels)

	for ti := range contextLen {
		for c := range numChannels {
			if math.Abs(recovered[ti][c]-input[ti][c]) > 1e-6 {
				t.Errorf("round-trip mismatch at [%d][%d]: got %f, want %f",
					ti, c, recovered[ti][c], input[ti][c])
			}
		}
	}
}
