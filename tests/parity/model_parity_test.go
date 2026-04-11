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

// getFloat64s2D extracts a 2D float64 slice from a JSON nested array.
func getFloat64s2D(m map[string]interface{}, key string) [][]float64 {
	arr, ok := m[key].([]interface{})
	if !ok {
		return nil
	}
	out := make([][]float64, len(arr))
	for i, row := range arr {
		rowArr := row.([]interface{})
		out[i] = make([]float64, len(rowArr))
		for j, v := range rowArr {
			out[i][j] = v.(float64)
		}
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
// T86.4.1b: PatchTST golden-file forward parity (PyTorch reference)
// ---------------------------------------------------------------------------

func TestParity_PatchTST(t *testing.T) {
	g := loadGolden(t, "model_patchtst")
	tol := getFloat(g, "tolerance")

	inputLen := int(getFloat(g, "input_len"))
	patchLen := int(getFloat(g, "patch_len"))
	stride := int(getFloat(g, "stride"))
	dModel := int(getFloat(g, "d_model"))
	nHeads := int(getFloat(g, "n_heads"))
	nLayers := int(getFloat(g, "n_layers"))
	outputDim := int(getFloat(g, "output_dim"))
	batch := int(getFloat(g, "batch"))
	goldenParamCount := int(getFloat(g, "param_count"))

	config := tsmodels.PatchTSTConfig{
		InputLength: inputLen,
		PatchLength: patchLen,
		Stride:      stride,
		DModel:      dModel,
		NHeads:      nHeads,
		NLayers:     nLayers,
		OutputDim:   outputDim,
	}

	engine, ops := setup()
	m, err := tsmodels.NewPatchTST(config, engine, ops)
	if err != nil {
		t.Fatalf("NewPatchTST: %v", err)
	}

	// Verify param count matches golden data.
	if pc := m.ParamCount(); pc != goldenParamCount {
		t.Fatalf("param count: got %d, want %d", pc, goldenParamCount)
	}

	// Extract golden flat params and weights from golden data.
	flatParams := getFloat64s(g, "flat_params")

	// Build the patchTSTWeights JSON using golden flat_params.
	// Flat param order: patchEmbW, patchEmbB, posEmb,
	//   per-layer: qW, qB, kW, kB, vW, vB, oW, oB,
	//              ffn1W, ffn1B, ffn2W, ffn2B, norm1, bias1, norm2, bias2,
	//   headW, headB.
	numPatches := (inputLen - patchLen) / stride + 1
	ffnDim := dModel * 4
	off := 0

	take := func(n int) []float64 {
		s := flatParams[off : off+n]
		off += n
		return s
	}

	patchEmbW := take(patchLen * dModel)
	patchEmbB := take(dModel)
	posEmb := take(numPatches * dModel)

	type layerJSON struct {
		QW    []float64 `json:"q_w"`
		QB    []float64 `json:"q_b"`
		KW    []float64 `json:"k_w"`
		KB    []float64 `json:"k_b"`
		VW    []float64 `json:"v_w"`
		VB    []float64 `json:"v_b"`
		OW    []float64 `json:"o_w"`
		OB    []float64 `json:"o_b"`
		FFN1W []float64 `json:"ffn1_w"`
		FFN1B []float64 `json:"ffn1_b"`
		FFN2W []float64 `json:"ffn2_w"`
		FFN2B []float64 `json:"ffn2_b"`
		Norm1 []float64 `json:"norm1"`
		Bias1 []float64 `json:"bias1"`
		Norm2 []float64 `json:"norm2"`
		Bias2 []float64 `json:"bias2"`
	}

	layers := make([]layerJSON, nLayers)
	for i := 0; i < nLayers; i++ {
		layers[i] = layerJSON{
			QW:    take(dModel * dModel),
			QB:    take(dModel),
			KW:    take(dModel * dModel),
			KB:    take(dModel),
			VW:    take(dModel * dModel),
			VB:    take(dModel),
			OW:    take(dModel * dModel),
			OB:    take(dModel),
			FFN1W: take(dModel * ffnDim),
			FFN1B: take(ffnDim),
			FFN2W: take(ffnDim * dModel),
			FFN2B: take(dModel),
			Norm1: take(dModel),
			Bias1: take(dModel),
			Norm2: take(dModel),
			Bias2: take(dModel),
		}
	}

	headW := take(numPatches * dModel * outputDim)
	headB := take(outputDim)

	if off != len(flatParams) {
		t.Fatalf("flat_params offset mismatch: consumed %d, total %d", off, len(flatParams))
	}

	// Write golden weights to a temp file in PatchTST's JSON format.
	weightsJSON := map[string]interface{}{
		"config": map[string]interface{}{
			"InputLength":        inputLen,
			"PatchLength":        patchLen,
			"Stride":             stride,
			"DModel":             dModel,
			"NHeads":             nHeads,
			"NLayers":            nLayers,
			"OutputDim":          outputDim,
			"ChannelIndependent": false,
		},
		"patch_emb_w": patchEmbW,
		"patch_emb_b": patchEmbB,
		"pos_emb":     posEmb,
		"layers":      layers,
		"head_w":      headW,
		"head_b":      headB,
	}
	dir := t.TempDir()
	weightsPath := filepath.Join(dir, "patchtst_golden.json")
	wData, err := json.Marshal(weightsJSON)
	if err != nil {
		t.Fatalf("marshal weights: %v", err)
	}
	if err := os.WriteFile(weightsPath, wData, 0o644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	// Build input windows: [batch][1 channel][inputLen].
	inputFlat := getFloat64s(g, "input")
	windows := make([][][]float64, batch)
	for b := 0; b < batch; b++ {
		windows[b] = [][]float64{inputFlat[b*inputLen : (b+1)*inputLen]}
	}

	// Run prediction through PredictWindowed (loads weights, runs f64 forward).
	preds, err := m.PredictWindowed(weightsPath, windows)
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
// N-BEATS golden-file forward parity (Python reference)
// ---------------------------------------------------------------------------

func TestParity_NBEATS(t *testing.T) {
	engine, ops := setup()

	g := loadGolden(t, "model_nbeats")
	tol := getFloat(g, "tolerance")

	inputLen := int(getFloat(g, "input_length"))
	outputLen := int(getFloat(g, "output_length"))
	hiddenDim := int(getFloat(g, "hidden_dim"))
	nHarmonics := int(getFloat(g, "n_harmonics"))
	batch := int(getFloat(g, "batch"))

	config := tsmodels.NBEATSConfig{
		InputLength:     inputLen,
		OutputLength:    outputLen,
		StackTypes:      []tsmodels.StackType{tsmodels.StackTrend, tsmodels.StackSeasonality},
		NBlocksPerStack: 1,
		HiddenDim:       hiddenDim,
		NHarmonics:      nHarmonics,
	}

	m, err := tsmodels.NewNBEATS(config, engine, ops)
	if err != nil {
		t.Fatalf("NewNBEATS: %v", err)
	}

	// Load golden params into model.
	goldenParams := getFloat64s(g, "params")
	flatPtrs := m.FlatParams()
	if len(goldenParams) != len(flatPtrs) {
		t.Fatalf("param count mismatch: golden=%d, model=%d", len(goldenParams), len(flatPtrs))
	}
	for i, v := range goldenParams {
		*flatPtrs[i] = float32(v)
	}

	// Build input tensor.
	inputFlat := getFloat32s(g, "input")
	input := makeTensor(t, inputFlat, []int{batch, inputLen})

	result, err := m.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	expectedOutput := getFloat32s(g, "expected_output")
	got := result.Forecast.Data()

	if len(got) != len(expectedOutput) {
		t.Fatalf("output length: got %d, want %d", len(got), len(expectedOutput))
	}
	assertClose(t, "nbeats_forecast", got, expectedOutput, tol)
}

// ---------------------------------------------------------------------------
// ITransformer golden-file forward parity (Python reference)
// ---------------------------------------------------------------------------

func TestParity_ITransformer(t *testing.T) {
	g := loadGolden(t, "model_itransformer")
	tol := getFloat(g, "tolerance")

	channels := int(getFloat(g, "channels"))
	inputLen := int(getFloat(g, "input_len"))
	outputLen := int(getFloat(g, "output_len"))
	dModel := int(getFloat(g, "d_model"))
	dFF := int(getFloat(g, "d_ff"))
	nHeads := int(getFloat(g, "n_heads"))
	nLayers := int(getFloat(g, "n_layers"))

	config := tsmodels.ITransformerConfig{
		Channels:  channels,
		InputLen:  inputLen,
		OutputLen: outputLen,
		DModel:    dModel,
		DFF:       dFF,
		NHeads:    nHeads,
		NLayers:   nLayers,
	}

	m, err := tsmodels.NewITransformer(config, nil, nil)
	if err != nil {
		t.Fatalf("NewITransformer: %v", err)
	}

	// Load golden params via flatParams pointers.
	goldenParams := getFloat64s(g, "params")
	flatPtrs := m.FlatParams()
	if len(goldenParams) != len(flatPtrs) {
		t.Fatalf("param count mismatch: golden=%d, model=%d", len(goldenParams), len(flatPtrs))
	}
	for i, v := range goldenParams {
		*flatPtrs[i] = v
	}

	// Build input: [channels][inputLen] from golden JSON.
	inputRaw := g["input"].([]interface{})
	input := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		row := inputRaw[c].([]interface{})
		input[c] = make([]float64, inputLen)
		for j := 0; j < inputLen; j++ {
			input[c][j] = row[j].(float64)
		}
	}

	// Run forward via ForwardSample (returns flat [channels*outputLen]).
	output, _, err := m.ForwardSample(input)
	if err != nil {
		t.Fatalf("ForwardSample: %v", err)
	}

	// Compare against expected output (JSON is [channels][outputLen]).
	expectedRaw := g["expected_output"].([]interface{})
	idx := 0
	for c := 0; c < channels; c++ {
		row := expectedRaw[c].([]interface{})
		for j := 0; j < outputLen; j++ {
			want := row[j].(float64)
			got := output[idx]
			diff := math.Abs(got - want)
			if diff > tol {
				t.Errorf("output[%d][%d]: got %g, want %g (diff=%g, tol=%g)", c, j, got, want, diff, tol)
			}
			idx++
		}
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
// E88: CfC golden-file forward parity (NumPy reference)
// ---------------------------------------------------------------------------

func TestParity_CfC(t *testing.T) {
	g := loadGolden(t, "model_cfc")
	tol := getFloat(g, "tolerance")

	inputSize := int(getFloat(g, "input_size"))
	hiddenSize := int(getFloat(g, "hidden_size"))
	outputSize := int(getFloat(g, "output_size"))
	numLayers := int(getFloat(g, "num_layers"))
	outputLen := int(getFloat(g, "output_len"))
	seqLen := int(getFloat(g, "seq_len"))

	config := tsmodels.CfCConfig{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
		NumLayers:  numLayers,
		OutputLen:  outputLen,
	}

	m, err := tsmodels.NewCfC(config)
	if err != nil {
		t.Fatalf("NewCfC: %v", err)
	}

	// Extract golden weights from JSON.
	whFlat := getFloat64s2D(g, "wh")
	wxFlat := getFloat64s2D(g, "wx")
	bhFlat := getFloat64s(g, "bh")
	wtauFlat := getFloat64s2D(g, "wtau")
	btauFlat := getFloat64s(g, "btau")
	outWFlat := getFloat64s2D(g, "out_w")
	outBFlat := getFloat64s(g, "out_b")

	// Write golden weights to a temp file in CfC's JSON format.
	dir := t.TempDir()
	weightsPath := filepath.Join(dir, "cfc_golden.json")
	weightsJSON := map[string]interface{}{
		"config": map[string]interface{}{
			"InputSize":  inputSize,
			"HiddenSize": hiddenSize,
			"OutputSize": outputSize,
			"NumLayers":  numLayers,
			"OutputLen":  outputLen,
		},
		"layers": []map[string]interface{}{
			{
				"wh":   whFlat,
				"wx":   wxFlat,
				"bh":   bhFlat,
				"wtau": wtauFlat,
				"btau": btauFlat,
			},
		},
		"out_w": outWFlat,
		"out_b": outBFlat,
	}
	wData, err := json.Marshal(weightsJSON)
	if err != nil {
		t.Fatalf("marshal weights: %v", err)
	}
	if err := os.WriteFile(weightsPath, wData, 0o644); err != nil {
		t.Fatalf("write weights: %v", err)
	}

	// Build input: [channels][seqLen] from golden flat data.
	inputFlat := getFloat64s(g, "input")
	channels := inputSize
	input := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		input[c] = inputFlat[c*seqLen : (c+1)*seqLen]
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

// ---------------------------------------------------------------------------
// FreTS golden-file forward parity (NumPy reference)
// ---------------------------------------------------------------------------

func TestParity_FreTS(t *testing.T) {
	g := loadGolden(t, "model_frets")
	tol := getFloat(g, "tolerance")

	channels := int(getFloat(g, "channels"))
	inputLen := int(getFloat(g, "input_len"))
	outputLen := int(getFloat(g, "output_len"))
	topK := int(getFloat(g, "top_k"))
	hiddenSize := int(getFloat(g, "hidden_size"))

	config := tsmodels.FreTSConfig{
		Channels:   channels,
		InputLen:   inputLen,
		OutputLen:  outputLen,
		TopK:       topK,
		HiddenSize: hiddenSize,
	}

	m, err := tsmodels.NewFreTS(config)
	if err != nil {
		t.Fatalf("NewFreTS: %v", err)
	}

	// Load golden weights via FlatParams pointers.
	// FlatParams order: chanW1, chanB1, chanW2, chanB2, tempW1, tempB1, tempW2, tempB2, outW, outB
	chanW1 := getFloat64s(g, "chan_w1")
	chanB1 := getFloat64s(g, "chan_b1")
	chanW2 := getFloat64s(g, "chan_w2")
	chanB2 := getFloat64s(g, "chan_b2")
	tempW1 := getFloat64s(g, "temp_w1")
	tempB1 := getFloat64s(g, "temp_b1")
	tempW2 := getFloat64s(g, "temp_w2")
	tempB2 := getFloat64s(g, "temp_b2")
	outW := getFloat64s(g, "out_w")
	outB := getFloat64s(g, "out_b")

	allWeights := make([]float64, 0)
	allWeights = append(allWeights, chanW1...)
	allWeights = append(allWeights, chanB1...)
	allWeights = append(allWeights, chanW2...)
	allWeights = append(allWeights, chanB2...)
	allWeights = append(allWeights, tempW1...)
	allWeights = append(allWeights, tempB1...)
	allWeights = append(allWeights, tempW2...)
	allWeights = append(allWeights, tempB2...)
	allWeights = append(allWeights, outW...)
	allWeights = append(allWeights, outB...)

	flatPtrs := m.FlatParams()
	if len(allWeights) != len(flatPtrs) {
		t.Fatalf("param count mismatch: golden=%d, model=%d", len(allWeights), len(flatPtrs))
	}
	for i, v := range allWeights {
		*flatPtrs[i] = v
	}

	// Build input: [channels][inputLen] from golden flat data.
	inputFlat := getFloat64s(g, "input")
	input := make([][]float64, channels)
	for c := 0; c < channels; c++ {
		input[c] = inputFlat[c*inputLen : (c+1)*inputLen]
	}

	// Run forward via ForwardSample (returns flat [channels*outputLen]).
	output, _, err := m.ForwardSample(input)
	if err != nil {
		t.Fatalf("ForwardSample: %v", err)
	}

	// Compare against expected output.
	expectedOutput := getFloat64s(g, "expected_output")
	if len(output) != len(expectedOutput) {
		t.Fatalf("output length: got %d, want %d", len(output), len(expectedOutput))
	}
	for i := range output {
		diff := math.Abs(output[i] - expectedOutput[i])
		if diff > tol {
			t.Errorf("output[%d]: got %g, want %g (diff=%g, tol=%g)", i, output[i], expectedOutput[i], diff, tol)
		}
	}
}

// ---------------------------------------------------------------------------
// E88: TimeMixer golden-file forward parity (NumPy reference)
// ---------------------------------------------------------------------------

func TestParity_TimeMixer(t *testing.T) {
	g := loadGolden(t, "model_timemixer")
	tol := getFloat(g, "tolerance")

	inputLen := int(getFloat(g, "input_len"))
	outputLen := int(getFloat(g, "output_len"))
	numFeatures := int(getFloat(g, "num_features"))
	numScales := int(getFloat(g, "num_scales"))
	hiddenSize := int(getFloat(g, "hidden_size"))
	numLayers := int(getFloat(g, "num_layers"))

	cfg := tsmodels.TimeMixerConfig{
		InputLen:    inputLen,
		OutputLen:   outputLen,
		NumFeatures: numFeatures,
		NumScales:   numScales,
		HiddenSize:  hiddenSize,
		NumLayers:   numLayers,
	}

	m := tsmodels.NewTimeMixer(cfg)

	// Inject golden flat params (MA weights + MLP weights) via FlatParams pointers.
	flatParams := getFloat64s(g, "flat_params")
	ptrs := m.FlatParams()
	if len(ptrs) != len(flatParams) {
		t.Fatalf("flat_params length mismatch: got %d pointers, want %d values", len(ptrs), len(flatParams))
	}
	for i, v := range flatParams {
		*ptrs[i] = v
	}

	// Inject trend/seasonal heads and mix weights.
	trendHeadsRaw, ok := g["trend_heads"].([]interface{})
	if !ok {
		t.Fatalf("trend_heads not found or wrong type")
	}
	trendHeads := make([][][]float64, numScales)
	for s := 0; s < numScales; s++ {
		flat := toFloat64Slice(trendHeadsRaw[s])
		trendHeads[s] = reshapeFloat64(flat, inputLen, outputLen)
	}
	m.SetTrendHeads(trendHeads)

	seasonalHeadsRaw, ok := g["seasonal_heads"].([]interface{})
	if !ok {
		t.Fatalf("seasonal_heads not found or wrong type")
	}
	seasonalHeads := make([][][]float64, numScales)
	for s := 0; s < numScales; s++ {
		flat := toFloat64Slice(seasonalHeadsRaw[s])
		seasonalHeads[s] = reshapeFloat64(flat, inputLen, outputLen)
	}
	m.SetSeasonalHeads(seasonalHeads)

	mixWeights := getFloat64s(g, "mix_weights")
	m.SetMixWeights(mixWeights)

	// Build input [numFeatures][inputLen].
	inputFlat := getFloat64s(g, "input")
	input := make([][]float64, numFeatures)
	for f := 0; f < numFeatures; f++ {
		input[f] = inputFlat[f*inputLen : (f+1)*inputLen]
	}

	// Run forward pass.
	out, err := m.Forward(input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// Compare against golden expected output.
	expectedOutput := getFloat64s(g, "expected_output")
	got := make([]float64, 0, numFeatures*outputLen)
	for f := 0; f < numFeatures; f++ {
		got = append(got, out.Forecast[f]...)
	}
	if len(got) != len(expectedOutput) {
		t.Fatalf("output length: got %d, want %d", len(got), len(expectedOutput))
	}
	for i := range got {
		diff := math.Abs(got[i] - expectedOutput[i])
		if diff > tol {
			t.Errorf("output[%d]: got %g, want %g (diff=%g, tol=%g)", i, got[i], expectedOutput[i], diff, tol)
		}
	}
}

// toFloat64Slice converts a JSON array ([]interface{}) to []float64.
func toFloat64Slice(v interface{}) []float64 {
	arr := v.([]interface{})
	out := make([]float64, len(arr))
	for i, x := range arr {
		out[i] = x.(float64)
	}
	return out
}
