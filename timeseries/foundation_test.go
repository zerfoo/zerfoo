package timeseries

import (
	"context"
	"math"
	"testing"

	its "github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func newTestForecaster(t *testing.T, numLayers, inputDim, hiddenDim, horizon, numVars int) *FoundationForecaster {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	cfg := &its.TiRexConfig{
		NumLayers:  numLayers,
		InputDim:   inputDim,
		HiddenDim:  hiddenDim,
		Horizon:    horizon,
		NumVars:    numVars,
		BlockTypes: []string{"slstm", "mlstm"},
	}
	f, err := newFoundationForecasterFromConfig(cfg, engine)
	if err != nil {
		t.Fatalf("newFoundationForecasterFromConfig: %v", err)
	}
	return f
}

func TestForecastOutputShape(t *testing.T) {
	fc := newTestForecaster(t, 2, 3, 8, 6, 3)
	seqLen := 10

	input := make([][]float64, seqLen)
	for i := range input {
		input[i] = []float64{float64(i) * 0.1, float64(i) * 0.2, float64(i) * 0.3}
	}

	ctx := context.Background()
	result, err := fc.Forecast(ctx, input, 6)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	if len(result) != 6 {
		t.Errorf("result horizon: got %d, want 6", len(result))
	}
	for i, row := range result {
		if len(row) != 3 {
			t.Errorf("result[%d] num_vars: got %d, want 3", i, len(row))
		}
	}
}

func TestForecastNonDegenerate(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	seqLen := 8

	// Create input with a clear trend.
	input := make([][]float64, seqLen)
	for i := range input {
		input[i] = []float64{float64(i)*10.0 + 100.0, float64(i)*5.0 + 50.0}
	}

	ctx := context.Background()
	result, err := fc.Forecast(ctx, input, 4)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	// Check that predictions are finite and not all identical.
	allSame := true
	first := result[0][0]
	for _, row := range result {
		for _, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("non-finite prediction: %v", v)
			}
			if v != first {
				allSame = false
			}
		}
	}
	if allSame && len(result) > 1 {
		t.Error("all predictions are identical — possible degenerate output")
	}
}

func TestForecastHorizonClamping(t *testing.T) {
	fc := newTestForecaster(t, 2, 1, 8, 4, 1)

	input := make([][]float64, 10)
	for i := range input {
		input[i] = []float64{float64(i)}
	}

	ctx := context.Background()
	// Request more than model horizon — should clamp.
	result, err := fc.Forecast(ctx, input, 100)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}
	if len(result) != 4 {
		t.Errorf("clamped horizon: got %d, want 4", len(result))
	}
}

func TestForecastValidation(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	ctx := context.Background()

	// Empty input.
	_, err := fc.Forecast(ctx, nil, 4)
	if err == nil {
		t.Error("expected error for nil input")
	}

	// Wrong number of variables.
	_, err = fc.Forecast(ctx, [][]float64{{1.0}}, 4)
	if err == nil {
		t.Error("expected error for wrong num_vars")
	}

	// Zero horizon.
	_, err = fc.Forecast(ctx, [][]float64{{1.0, 2.0}}, 0)
	if err == nil {
		t.Error("expected error for zero horizon")
	}
}

func TestBatchForecastOutputShape(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	batchSize := 3
	seqLen := 6

	inputs := make([][][]float64, batchSize)
	for b := range inputs {
		inputs[b] = make([][]float64, seqLen)
		for i := range inputs[b] {
			inputs[b][i] = []float64{float64(b*10+i) * 0.1, float64(b*10+i) * 0.2}
		}
	}

	ctx := context.Background()
	results, err := fc.BatchForecast(ctx, inputs, 4)
	if err != nil {
		t.Fatalf("BatchForecast: %v", err)
	}

	if len(results) != batchSize {
		t.Fatalf("batch size: got %d, want %d", len(results), batchSize)
	}
	for b, batch := range results {
		if len(batch) != 4 {
			t.Errorf("batch[%d] horizon: got %d, want 4", b, len(batch))
		}
		for ti, row := range batch {
			if len(row) != 2 {
				t.Errorf("batch[%d][%d] num_vars: got %d, want 2", b, ti, len(row))
			}
		}
	}
}

func TestBatchForecastNonDegenerate(t *testing.T) {
	fc := newTestForecaster(t, 2, 1, 8, 4, 1)
	batchSize := 2
	seqLen := 8

	inputs := make([][][]float64, batchSize)
	for b := range inputs {
		inputs[b] = make([][]float64, seqLen)
		for i := range inputs[b] {
			inputs[b][i] = []float64{float64(b*100+i*10) + 50.0}
		}
	}

	ctx := context.Background()
	results, err := fc.BatchForecast(ctx, inputs, 4)
	if err != nil {
		t.Fatalf("BatchForecast: %v", err)
	}

	for b, batch := range results {
		for ti, row := range batch {
			for c, v := range row {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("batch[%d][%d][%d] non-finite: %v", b, ti, c, v)
				}
			}
		}
	}
}

func TestBatchForecastValidation(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	ctx := context.Background()

	// Empty batch.
	_, err := fc.BatchForecast(ctx, nil, 4)
	if err == nil {
		t.Error("expected error for empty batch")
	}

	// Mismatched sequence lengths.
	_, err = fc.BatchForecast(ctx, [][][]float64{
		{{1.0, 2.0}, {3.0, 4.0}},
		{{1.0, 2.0}},
	}, 4)
	if err == nil {
		t.Error("expected error for mismatched seq lengths")
	}

	// Wrong num vars.
	_, err = fc.BatchForecast(ctx, [][][]float64{
		{{1.0}},
	}, 4)
	if err == nil {
		t.Error("expected error for wrong num_vars")
	}
}

func TestInstanceNorm(t *testing.T) {
	input := [][]float64{
		{10.0, 20.0},
		{20.0, 40.0},
		{30.0, 60.0},
	}

	mean, std := instanceNorm(input, 3, 2)

	if math.Abs(mean[0]-20.0) > 1e-10 {
		t.Errorf("mean[0]: got %f, want 20.0", mean[0])
	}
	if math.Abs(mean[1]-40.0) > 1e-10 {
		t.Errorf("mean[1]: got %f, want 40.0", mean[1])
	}
	// std of [10, 20, 30] = sqrt(200/3) ≈ 8.165
	expectedStd := math.Sqrt(200.0 / 3.0)
	if math.Abs(std[0]-expectedStd) > 1e-10 {
		t.Errorf("std[0]: got %f, want %f", std[0], expectedStd)
	}
}

func TestLoadTiRexConfigFromMeta(t *testing.T) {
	meta := map[string]interface{}{
		"tirex.block_count": uint32(4),
		"tirex.hidden_dim":  uint32(64),
		"tirex.input_dim":   uint32(3),
		"tirex.horizon":     uint32(12),
		"tirex.num_vars":    uint32(3),
		"tirex.block_types": []any{"slstm", "mlstm", "slstm", "mlstm"},
	}

	cfg, err := loadTiRexConfigFromMeta(meta)
	if err != nil {
		t.Fatalf("loadTiRexConfigFromMeta: %v", err)
	}

	if cfg.NumLayers != 4 {
		t.Errorf("NumLayers: got %d, want 4", cfg.NumLayers)
	}
	if cfg.HiddenDim != 64 {
		t.Errorf("HiddenDim: got %d, want 64", cfg.HiddenDim)
	}
	if cfg.InputDim != 3 {
		t.Errorf("InputDim: got %d, want 3", cfg.InputDim)
	}
	if cfg.Horizon != 12 {
		t.Errorf("Horizon: got %d, want 12", cfg.Horizon)
	}
	if cfg.NumVars != 3 {
		t.Errorf("NumVars: got %d, want 3", cfg.NumVars)
	}
	if len(cfg.BlockTypes) != 4 {
		t.Errorf("BlockTypes length: got %d, want 4", len(cfg.BlockTypes))
	}
}

func TestFineTuneDecreasingLoss(t *testing.T) {
	// The TiRex sLSTM/mLSTM backbone is numerically delicate for some random
	// weight initializations — about 15-20% of random inits produce NaN on
	// the first forward pass (exploding recurrent activations). That is a
	// separate model-stability concern tracked in #350. For this test we
	// just want to verify the fine-tune loop reduces the loss when given a
	// usable starting point, so we retry construction until the first epoch
	// loss is finite. If no seed produces a usable init in 20 tries,
	// something is genuinely broken.
	ctx := context.Background()

	// Generate synthetic training data: simple linear trend.
	nSamples := 20
	seqLen := 8
	numVars := 2
	horizon := 4

	data := make([][]float64, nSamples)
	labels := make([][]float64, nSamples)
	for s := range nSamples {
		data[s] = make([]float64, seqLen*numVars)
		labels[s] = make([]float64, horizon*numVars)
		base := float64(s) * 0.5
		for t := range seqLen {
			for c := range numVars {
				data[s][t*numVars+c] = base + float64(t)*0.1*float64(c+1)
			}
		}
		for t := range horizon {
			for c := range numVars {
				labels[s][t*numVars+c] = base + float64(seqLen+t)*0.1*float64(c+1)
			}
		}
	}

	cfg := FineTuneConfig{
		Epochs:         30,
		LearningRate:   1e-2,
		BatchSize:      0,
		FreezeBackbone: true,
	}

	var result *TrainResult
	const maxAttempts = 20
	for attempt := 0; attempt < maxAttempts; attempt++ {
		fc := newTestForecaster(t, 2, 2, 8, 4, 2)
		r, err := fc.FineTune(ctx, data, labels, cfg)
		if err != nil {
			t.Fatalf("FineTune: %v", err)
		}
		if len(r.LossHistory) == 0 || math.IsNaN(r.LossHistory[0]) || math.IsInf(r.LossHistory[0], 0) {
			continue
		}
		result = r
		break
	}
	if result == nil {
		t.Fatalf("no usable init after %d attempts (all produced NaN/Inf on first epoch)", maxAttempts)
	}

	if len(result.LossHistory) != 30 {
		t.Fatalf("LossHistory length: got %d, want 30", len(result.LossHistory))
	}

	// Verify loss decreased: compare average of first 5 epochs vs last 5 epochs.
	// Using averages instead of single values avoids flakiness from random init.
	earlyAvg := float64(0)
	lateAvg := float64(0)
	n := 5
	for i := 0; i < n; i++ {
		earlyAvg += result.LossHistory[i]
		lateAvg += result.LossHistory[len(result.LossHistory)-n+i]
	}
	earlyAvg /= float64(n)
	lateAvg /= float64(n)
	if lateAvg >= earlyAvg {
		t.Errorf("loss did not decrease: early_avg=%f, late_avg=%f", earlyAvg, lateAvg)
	}

	// Verify all losses are finite.
	for i, l := range result.LossHistory {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Fatalf("LossHistory[%d] is non-finite: %v", i, l)
		}
	}
}

func TestFineTuneValidation(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	ctx := context.Background()
	cfg := FineTuneConfig{Epochs: 5, LearningRate: 1e-3, FreezeBackbone: true}

	// Empty data.
	_, err := fc.FineTune(ctx, nil, nil, cfg)
	if err == nil {
		t.Error("expected error for empty data")
	}

	// Mismatched lengths.
	_, err = fc.FineTune(ctx, [][]float64{{1, 2, 3, 4}}, nil, cfg)
	if err == nil {
		t.Error("expected error for mismatched data/labels")
	}

	// Wrong label length.
	_, err = fc.FineTune(ctx, [][]float64{{1, 2, 3, 4, 5, 6, 7, 8}}, [][]float64{{1}}, cfg)
	if err == nil {
		t.Error("expected error for wrong label length")
	}
}

func TestLoadTiRexConfigDefaults(t *testing.T) {
	meta := map[string]interface{}{
		"tirex.block_count": uint32(2),
		"tirex.hidden_dim":  uint32(32),
	}

	cfg, err := loadTiRexConfigFromMeta(meta)
	if err != nil {
		t.Fatalf("loadTiRexConfigFromMeta: %v", err)
	}

	if cfg.InputDim != 1 {
		t.Errorf("default InputDim: got %d, want 1", cfg.InputDim)
	}
	if cfg.Horizon != 1 {
		t.Errorf("default Horizon: got %d, want 1", cfg.Horizon)
	}
	if cfg.NumVars != 1 {
		t.Errorf("default NumVars: got %d, want 1 (should equal InputDim)", cfg.NumVars)
	}
}
