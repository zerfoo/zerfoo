package timeseries

import (
	"context"
	"math"
	"testing"
)

// TestTiRexDeterminism verifies that running the same input through the model
// multiple times produces identical outputs.
func TestTiRexDeterminism(t *testing.T) {
	fc := newTestForecaster(t, 2, 3, 8, 6, 3)
	ctx := context.Background()

	inputs := make([][]float64, 10)
	for i := range inputs {
		inputs[i] = []float64{
			float64(i)*1.5 + 10.0,
			float64(i)*0.7 + 5.0,
			float64(i)*2.3 + 20.0,
		}
	}

	// Run forecast 3 times and verify all outputs are identical.
	var runs [3][][]float64
	for r := range 3 {
		result, err := fc.Forecast(ctx, inputs, 6)
		if err != nil {
			t.Fatalf("run %d: Forecast: %v", r, err)
		}
		runs[r] = result
	}

	for r := 1; r < 3; r++ {
		if len(runs[r]) != len(runs[0]) {
			t.Fatalf("run %d: horizon %d != run 0 horizon %d", r, len(runs[r]), len(runs[0]))
		}
		for ti := range runs[0] {
			for c := range runs[0][ti] {
				if runs[r][ti][c] != runs[0][ti][c] {
					t.Errorf("run %d: result[%d][%d] = %v, want %v", r, ti, c, runs[r][ti][c], runs[0][ti][c])
				}
			}
		}
	}
}

// TestTiRexMultipleInputsDeterminism runs 10 different synthetic inputs and
// verifies each is deterministic and produces non-degenerate output.
func TestTiRexMultipleInputsDeterminism(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	ctx := context.Background()

	for sample := range 10 {
		seqLen := 8 + sample%5 // vary sequence length between 8 and 12
		input := make([][]float64, seqLen)
		for i := range input {
			// Each sample has a different offset/scale pattern.
			input[i] = []float64{
				float64(sample*100+i*10) + 42.0,
				float64(sample*50+i*7) + 17.0,
			}
		}

		r1, err := fc.Forecast(ctx, input, 4)
		if err != nil {
			t.Fatalf("sample %d, run 1: %v", sample, err)
		}
		r2, err := fc.Forecast(ctx, input, 4)
		if err != nil {
			t.Fatalf("sample %d, run 2: %v", sample, err)
		}

		// Determinism check.
		for ti := range r1 {
			for c := range r1[ti] {
				if r1[ti][c] != r2[ti][c] {
					t.Errorf("sample %d: r1[%d][%d]=%v != r2[%d][%d]=%v",
						sample, ti, c, r1[ti][c], ti, c, r2[ti][c])
				}
			}
		}

		// Non-degenerate check.
		allZero := true
		allSame := true
		first := r1[0][0]
		for _, row := range r1 {
			for _, v := range row {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("sample %d: non-finite value %v", sample, v)
				}
				if v != 0 {
					allZero = false
				}
				if v != first {
					allSame = false
				}
			}
		}
		if allZero {
			t.Errorf("sample %d: all outputs are zero", sample)
		}
		if allSame && len(r1)*len(r1[0]) > 1 {
			t.Errorf("sample %d: all outputs are identical (%v)", sample, first)
		}
	}
}

// TestTiRexBatchVsSingleConsistency verifies that BatchForecast produces the
// same results as calling Forecast individually for each sample.
func TestTiRexBatchVsSingleConsistency(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	ctx := context.Background()

	batchSize := 4
	seqLen := 8
	horizon := 4

	inputs := make([][][]float64, batchSize)
	for b := range inputs {
		inputs[b] = make([][]float64, seqLen)
		for i := range inputs[b] {
			inputs[b][i] = []float64{
				float64(b*100+i*10) + 30.0,
				float64(b*50+i*5) + 15.0,
			}
		}
	}

	// Get batch result.
	batchResults, err := fc.BatchForecast(ctx, inputs, horizon)
	if err != nil {
		t.Fatalf("BatchForecast: %v", err)
	}

	if len(batchResults) != batchSize {
		t.Fatalf("batch size: got %d, want %d", len(batchResults), batchSize)
	}

	// Get individual results and compare.
	for b := range batchSize {
		single, err := fc.Forecast(ctx, inputs[b], horizon)
		if err != nil {
			t.Fatalf("Forecast[%d]: %v", b, err)
		}

		if len(batchResults[b]) != len(single) {
			t.Fatalf("batch[%d] horizon: got %d, want %d", b, len(batchResults[b]), len(single))
		}

		for ti := range single {
			for c := range single[ti] {
				got := batchResults[b][ti][c]
				want := single[ti][c]
				if math.Abs(got-want) > 1e-4 {
					t.Errorf("batch[%d][%d][%d]: got %v, want %v (diff=%v)",
						b, ti, c, got, want, math.Abs(got-want))
				}
			}
		}
	}
}

// TestTiRexSingleVariate tests the model with a single variable (univariate).
func TestTiRexSingleVariate(t *testing.T) {
	fc := newTestForecaster(t, 2, 1, 8, 4, 1)
	ctx := context.Background()

	input := make([][]float64, 12)
	for i := range input {
		input[i] = []float64{float64(i)*3.0 + 100.0}
	}

	result, err := fc.Forecast(ctx, input, 4)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	if len(result) != 4 {
		t.Errorf("horizon: got %d, want 4", len(result))
	}
	for ti, row := range result {
		if len(row) != 1 {
			t.Errorf("result[%d]: got %d vars, want 1", ti, len(row))
		}
		if math.IsNaN(row[0]) || math.IsInf(row[0], 0) {
			t.Errorf("result[%d][0]: non-finite %v", ti, row[0])
		}
	}
}

// TestTiRexManyVariates tests the model with a higher number of variates.
func TestTiRexManyVariates(t *testing.T) {
	numVars := 8
	fc := newTestForecaster(t, 2, numVars, 16, 6, numVars)
	ctx := context.Background()

	seqLen := 10
	input := make([][]float64, seqLen)
	for i := range input {
		row := make([]float64, numVars)
		for c := range numVars {
			row[c] = float64(i*numVars+c)*0.5 + float64(c)*10.0
		}
		input[i] = row
	}

	result, err := fc.Forecast(ctx, input, 6)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	if len(result) != 6 {
		t.Errorf("horizon: got %d, want 6", len(result))
	}
	for ti, row := range result {
		if len(row) != numVars {
			t.Errorf("result[%d]: got %d vars, want %d", ti, len(row), numVars)
		}
		for c, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("result[%d][%d]: non-finite %v", ti, c, v)
			}
		}
	}

	// Verify determinism for many-variate case.
	result2, err := fc.Forecast(ctx, input, 6)
	if err != nil {
		t.Fatalf("Forecast (repeat): %v", err)
	}
	for ti := range result {
		for c := range result[ti] {
			if result[ti][c] != result2[ti][c] {
				t.Errorf("non-deterministic: result[%d][%d] = %v vs %v",
					ti, c, result[ti][c], result2[ti][c])
			}
		}
	}
}

// TestTiRexShortHorizon tests with the minimum horizon of 1.
func TestTiRexShortHorizon(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 1, 2)
	ctx := context.Background()

	input := make([][]float64, 6)
	for i := range input {
		input[i] = []float64{float64(i) + 10.0, float64(i)*2.0 + 5.0}
	}

	result, err := fc.Forecast(ctx, input, 1)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	if len(result) != 1 {
		t.Fatalf("horizon: got %d, want 1", len(result))
	}
	if len(result[0]) != 2 {
		t.Errorf("num_vars: got %d, want 2", len(result[0]))
	}
	for c, v := range result[0] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("result[0][%d]: non-finite %v", c, v)
		}
	}
}

// TestTiRexLongHorizon tests with a longer prediction horizon.
func TestTiRexLongHorizon(t *testing.T) {
	horizon := 24
	fc := newTestForecaster(t, 2, 2, 8, horizon, 2)
	ctx := context.Background()

	seqLen := 20
	input := make([][]float64, seqLen)
	for i := range input {
		// Sine wave + linear trend for realistic input.
		input[i] = []float64{
			math.Sin(float64(i)*0.5)*10.0 + float64(i)*2.0,
			math.Cos(float64(i)*0.3)*5.0 + float64(i)*1.5,
		}
	}

	result, err := fc.Forecast(ctx, input, horizon)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	if len(result) != horizon {
		t.Errorf("horizon: got %d, want %d", len(result), horizon)
	}

	// All values must be finite.
	for ti, row := range result {
		if len(row) != 2 {
			t.Errorf("result[%d]: got %d vars, want 2", ti, len(row))
		}
		for c, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("result[%d][%d]: non-finite %v", ti, c, v)
			}
		}
	}

	// Non-degenerate: not all outputs identical.
	allSame := true
	first := result[0][0]
	for _, row := range result {
		for _, v := range row {
			if v != first {
				allSame = false
			}
		}
	}
	if allSame {
		t.Error("all long-horizon outputs are identical")
	}
}

// TestTiRexBatchSingleSample verifies BatchForecast with a single sample
// matches Forecast.
func TestTiRexBatchSingleSample(t *testing.T) {
	fc := newTestForecaster(t, 2, 2, 8, 4, 2)
	ctx := context.Background()

	input := make([][]float64, 8)
	for i := range input {
		input[i] = []float64{float64(i)*5.0 + 50.0, float64(i)*3.0 + 25.0}
	}

	single, err := fc.Forecast(ctx, input, 4)
	if err != nil {
		t.Fatalf("Forecast: %v", err)
	}

	batch, err := fc.BatchForecast(ctx, [][][]float64{input}, 4)
	if err != nil {
		t.Fatalf("BatchForecast: %v", err)
	}

	if len(batch) != 1 {
		t.Fatalf("batch size: got %d, want 1", len(batch))
	}
	for ti := range single {
		for c := range single[ti] {
			if math.Abs(batch[0][ti][c]-single[ti][c]) > 1e-4 {
				t.Errorf("[%d][%d]: batch=%v, single=%v",
					ti, c, batch[0][ti][c], single[ti][c])
			}
		}
	}
}
