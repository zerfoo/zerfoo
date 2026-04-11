package parity_test

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/tabular"
	tsmodels "github.com/zerfoo/zerfoo/timeseries"
)

// getFloat64s extracts a float64 slice from a JSON array.
func getFloat64s(m map[string]interface{}, key string) []float64 {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([]float64, len(arr))
	for i, v := range arr {
		out[i] = v.(float64)
	}
	return out
}

// reshapeFloat64 reshapes a flat float64 slice into a 2D slice [rows][cols].
func reshapeFloat64(flat []float64, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		result[r] = flat[r*cols : (r+1)*cols]
	}
	return result
}

// ---------------------------------------------------------------------------
// T86.4.3: DLinear golden-file forward parity (PyTorch reference)
// ---------------------------------------------------------------------------

func TestParity_DLinear(t *testing.T) {
	g := loadGolden(t, "timeseries_dlinear")
	tol := getFloat(g, "tolerance")

	inputLen := int(getFloat(g, "input_len"))
	outputLen := int(getFloat(g, "output_len"))
	channels := int(getFloat(g, "channels"))
	kernelSize := int(getFloat(g, "kernel_size"))

	m, err := tsmodels.NewDLinear(inputLen, outputLen, channels, kernelSize)
	if err != nil {
		t.Fatalf("NewDLinear: %v", err)
	}

	// Load golden weights into the model via SaveWeights/loadWeights round-trip.
	trendW := getFloat64s(g, "trend_w")
	trendB := getFloat64s(g, "trend_b")
	seasonalW := getFloat64s(g, "seasonal_w")
	seasonalB := getFloat64s(g, "seasonal_b")

	// Write golden weights to a temp file in DLinear's JSON format.
	dir := t.TempDir()
	weightsPath := filepath.Join(dir, "dlinear_golden.json")
	weightsJSON := map[string]interface{}{
		"config": map[string]interface{}{
			"InputLen":   inputLen,
			"OutputLen":  outputLen,
			"Channels":   channels,
			"KernelSize": kernelSize,
		},
		"trend_w":    reshapeFloat64(trendW, channels, outputLen*inputLen),
		"trend_b":    reshapeFloat64(trendB, channels, outputLen),
		"seasonal_w": reshapeFloat64(seasonalW, channels, outputLen*inputLen),
		"seasonal_b": reshapeFloat64(seasonalB, channels, outputLen),
	}
	wData, err := json.Marshal(weightsJSON)
	if err != nil {
		t.Fatalf("marshal weights: %v", err)
	}
	if err := os.WriteFile(weightsPath, wData, 0o644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	// Build input from golden data.
	inputFlat := getFloat64s(g, "input")
	input := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		input[c] = inputFlat[c*inputLen : (c+1)*inputLen]
	}

	// Run prediction, loading weights from the golden file.
	preds, err := m.PredictWindowed(weightsPath, [][][]float64{input})
	if err != nil {
		t.Fatalf("PredictWindowed: %v", err)
	}

	// Compare against golden expected output.
	expectedOutput := getFloat64s(g, "expected_output")
	if len(preds) != len(expectedOutput) {
		t.Fatalf("output length: got %d, want %d", len(preds), len(expectedOutput))
	}
	for i := range preds {
		diff := math.Abs(preds[i] - expectedOutput[i])
		if diff > tol {
			t.Errorf("output[%d]: got %g, want %g (diff=%g, tol=%g)", i, preds[i], expectedOutput[i], diff, tol)
		}
	}
}

// ---------------------------------------------------------------------------
// T86.4.1: PatchTST structural forward parity
// Structural test - not golden-file parity
// ---------------------------------------------------------------------------

func TestParity_PatchTST_Structural(t *testing.T) {
	engine, ops := setup()

	config := tsmodels.PatchTSTConfig{
		InputLength:        16,
		PatchLength:        4,
		Stride:             4,
		DModel:             8,
		NHeads:             2,
		NLayers:            2,
		OutputDim:          4,
		ChannelIndependent: false,
	}

	m, err := tsmodels.NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	batch := 2
	inputData := make([]float32, batch*config.InputLength)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.01
	}
	input := makeTensor(t, inputData, []int{batch, config.InputLength})

	output, err := m.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Verify shape: [batch, output_dim].
	wantShape := []int{batch, config.OutputDim}
	gotShape := output.Shape()
	if len(gotShape) != 2 || gotShape[0] != wantShape[0] || gotShape[1] != wantShape[1] {
		t.Errorf("shape: got %v, want %v", gotShape, wantShape)
	}

	// Verify no NaN/Inf.
	for i, v := range output.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("output[%d] = %v (NaN/Inf)", i, v)
		}
	}

	// Verify output changes when input changes.
	inputData2 := make([]float32, batch*config.InputLength)
	for i := range inputData2 {
		inputData2[i] = float32(i+1) * 0.5
	}
	input2 := makeTensor(t, inputData2, []int{batch, config.InputLength})
	output2, err := m.Forward(context.Background(), input2)
	if err != nil {
		t.Fatalf("Forward (input2): %v", err)
	}
	allSame := true
	for i := range output.Data() {
		if output.Data()[i] != output2.Data()[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("output is constant: same result for different inputs")
	}
}

// ---------------------------------------------------------------------------
// T86.4.2: N-BEATS structural forward parity
// Structural test - not golden-file parity
// ---------------------------------------------------------------------------

func TestParity_NBEATS_Structural(t *testing.T) {
	engine, ops := setup()

	config := tsmodels.NBEATSConfig{
		InputLength:     12,
		OutputLength:    6,
		StackTypes:      []tsmodels.StackType{tsmodels.StackTrend, tsmodels.StackSeasonality},
		NBlocksPerStack: 2,
		HiddenDim:       16,
		NHarmonics:      4,
	}

	m, err := tsmodels.NewNBEATS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	batch := 3
	inputData := make([]float32, batch*config.InputLength)
	for i := range inputData {
		inputData[i] = float32(i+1) * 0.1
	}
	input := makeTensor(t, inputData, []int{batch, config.InputLength})

	result, err := m.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	forecast := result.Forecast
	wantShape := []int{batch, config.OutputLength}
	gotShape := forecast.Shape()
	if len(gotShape) != 2 || gotShape[0] != wantShape[0] || gotShape[1] != wantShape[1] {
		t.Errorf("forecast shape: got %v, want %v", gotShape, wantShape)
	}

	// Verify no NaN/Inf.
	for i, v := range forecast.Data() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("forecast[%d] = %v (NaN/Inf)", i, v)
		}
	}

	// Verify output changes when input changes.
	inputData2 := make([]float32, batch*config.InputLength)
	for i := range inputData2 {
		inputData2[i] = float32(i+1) * 0.5
	}
	input2 := makeTensor(t, inputData2, []int{batch, config.InputLength})
	result2, err := m.Forward(context.Background(), input2)
	if err != nil {
		t.Fatalf("Forward (input2): %v", err)
	}
	allSame := true
	for i := range forecast.Data() {
		if forecast.Data()[i] != result2.Forecast.Data()[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("forecast is constant: same result for different inputs")
	}
}

// ---------------------------------------------------------------------------
// T86.4.4: ITransformer structural forward parity
// Structural test - not golden-file parity
// ---------------------------------------------------------------------------

func TestParity_ITransformer_Structural(t *testing.T) {
	config := tsmodels.ITransformerConfig{
		Channels:  3,
		InputLen:  8,
		OutputLen: 4,
		DModel:    8,
		DFF:       16,
		NHeads:    2,
		NLayers:   1,
	}

	m, err := tsmodels.NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	// Build input: [channels][inputLen].
	input := make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		input[c] = make([]float64, config.InputLen)
		for i := 0; i < config.InputLen; i++ {
			input[c][i] = float64(c*config.InputLen+i+1) * 0.01
		}
	}

	output, cache, err := m.ForwardSample(input)
	if err != nil {
		t.Fatalf("ForwardSample: %v", err)
	}
	_ = cache

	// Verify output length: channels * outputLen.
	wantLen := config.Channels * config.OutputLen
	if len(output) != wantLen {
		t.Fatalf("output length: got %d, want %d", len(output), wantLen)
	}

	// Verify no NaN/Inf.
	for i, v := range output {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("output[%d] = %v (NaN/Inf)", i, v)
		}
	}

	// Verify output changes when input changes.
	input2 := make([][]float64, config.Channels)
	for c := 0; c < config.Channels; c++ {
		input2[c] = make([]float64, config.InputLen)
		for i := 0; i < config.InputLen; i++ {
			input2[c][i] = float64(c*config.InputLen+i+1) * 0.5
		}
	}
	output2, _, err := m.ForwardSample(input2)
	if err != nil {
		t.Fatalf("ForwardSample (input2): %v", err)
	}
	allSame := true
	for i := range output {
		if output[i] != output2[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("output is constant: same result for different inputs")
	}
}

// ---------------------------------------------------------------------------
// T86.4.5: TFT structural forward parity
// Structural test - not golden-file parity
// ---------------------------------------------------------------------------

func TestParity_TFT_Structural(t *testing.T) {
	engine, ops := setup()

	config := tsmodels.TFTConfig{
		NumStaticFeatures: 3,
		NumTimeFeatures:   4,
		DModel:            8,
		NHeads:            2,
		NHorizons:         3,
		Quantiles:         []float64{0.1, 0.5, 0.9},
	}

	m, err := tsmodels.NewTFT(config, engine, ops)
	if err != nil {
		t.Fatalf("NewTFT: %v", err)
	}

	staticFeatures := []float64{1.0, 2.0, 3.0}
	timeFeatures := [][]float64{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
		{0.9, 1.0, 1.1, 1.2},
	}

	result, err := m.Predict(staticFeatures, timeFeatures)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	// Verify shape: [nHorizons][nQuantiles].
	if len(result) != config.NHorizons {
		t.Fatalf("horizons: got %d, want %d", len(result), config.NHorizons)
	}
	for h, row := range result {
		if len(row) != len(config.Quantiles) {
			t.Errorf("horizon %d: got %d quantiles, want %d", h, len(row), len(config.Quantiles))
		}
		for q, val := range row {
			if math.IsNaN(val) || math.IsInf(val, 0) {
				t.Fatalf("result[%d][%d] = %v (NaN/Inf)", h, q, val)
			}
		}
	}

	// Verify output changes when input changes.
	staticFeatures2 := []float64{10.0, 20.0, 30.0}
	timeFeatures2 := [][]float64{
		{1.1, 1.2, 1.3, 1.4},
		{1.5, 1.6, 1.7, 1.8},
		{1.9, 2.0, 2.1, 2.2},
	}
	result2, err := m.Predict(staticFeatures2, timeFeatures2)
	if err != nil {
		t.Fatalf("Predict (input2): %v", err)
	}
	allSame := true
	for h := range result {
		for q := range result[h] {
			if result[h][q] != result2[h][q] {
				allSame = false
				break
			}
		}
		if !allSame {
			break
		}
	}
	if allSame {
		t.Errorf("output is constant: same result for different inputs")
	}
}

// ---------------------------------------------------------------------------
// T86.4.6: CfC structural forward parity
// Structural test - not golden-file parity
// ---------------------------------------------------------------------------

func TestParity_CfC_Structural(t *testing.T) {
	config := tsmodels.CfCConfig{
		InputSize:  3,
		HiddenSize: 8,
		OutputSize: 2,
		NumLayers:  2,
		OutputLen:  4,
	}

	m, err := tsmodels.NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Build input: [channels][seqLen] for ForwardSample (uses transposeWindow internally).
	channels := config.InputSize
	seqLen := 5
	input := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		input[c] = make([]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			input[c][i] = float64(c*seqLen+i+1) * 0.1
		}
	}

	output, cache, err := m.ForwardSample(input)
	if err != nil {
		t.Fatalf("ForwardSample: %v", err)
	}
	_ = cache

	// Verify output length: outputSize * outputLen.
	wantLen := config.OutputSize * config.OutputLen
	if len(output) != wantLen {
		t.Fatalf("output length: got %d, want %d", len(output), wantLen)
	}

	// Verify no NaN/Inf.
	for i, v := range output {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("output[%d] = %v (NaN/Inf)", i, v)
		}
	}

	// Verify output changes when input changes.
	input2 := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		input2[c] = make([]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			input2[c][i] = float64(c*seqLen+i+1) * 0.9
		}
	}
	output2, _, err := m.ForwardSample(input2)
	if err != nil {
		t.Fatalf("ForwardSample (input2): %v", err)
	}
	allSame := true
	for i := range output {
		if output[i] != output2[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Errorf("output is constant: same result for different inputs")
	}
}

// ---------------------------------------------------------------------------
// T86.4.7: FTTransformer structural forward parity
// Structural test - not golden-file parity
// ---------------------------------------------------------------------------

func TestParity_FTTransformer_Structural(t *testing.T) {
	engine, ops := setup()

	config := tabular.FTTransformerConfig{
		NumFeatures: 4,
		DToken:      8,
		NHeads:      2,
		NLayers:     2,
		DFFN:        16,
		DropoutRate: 0.0,
	}

	ft, err := tabular.NewFTTransformer(config, engine, ops)
	if err != nil {
		t.Fatalf("NewFTTransformer: %v", err)
	}

	features := []float64{1.0, -0.5, 2.0, 0.3}
	dir, conf, err := ft.Predict(features)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	// Verify direction is valid.
	if dir < tabular.Long || dir > tabular.Flat {
		t.Errorf("direction %d is not in valid range", dir)
	}

	// Verify confidence is in (0, 1].
	if conf <= 0 || conf > 1 {
		t.Errorf("confidence %f is not in (0, 1]", conf)
	}

	// Verify no NaN/Inf in confidence.
	if math.IsNaN(conf) || math.IsInf(conf, 0) {
		t.Fatalf("confidence = %v (NaN/Inf)", conf)
	}

	// Verify output changes when input changes.
	features2 := []float64{10.0, -5.0, 20.0, 3.0}
	dir2, conf2, err := ft.Predict(features2)
	if err != nil {
		t.Fatalf("Predict (input2): %v", err)
	}
	if dir == dir2 && conf == conf2 {
		t.Errorf("output is constant: same (direction, confidence) for different inputs")
	}
}
