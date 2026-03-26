package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// newTestFlowStateModel creates a FlowStateModel directly from a config and
// graph, bypassing GGUF loading. This enables tests without model files.
func newTestFlowStateModel(cfg *FlowStateConfig) (*FlowStateModel, error) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		return nil, err
	}

	return &FlowStateModel{
		graph:  g,
		config: cfg,
		engine: engine,
	}, nil
}

func testFlowStateConfig() *FlowStateConfig {
	return &FlowStateConfig{
		ContextLen:   64,
		ForecastLen:  16,
		NumChannels:  2,
		PatchLen:     8,
		DModel:       16,
		NumSSMLayers: 1,
		DState:       8,
		NumBasis:     4,
		ScaleFactor:  1.0,
	}
}

func makeInput(contextLen, numChannels int) [][]float64 {
	input := make([][]float64, contextLen)
	for t := range contextLen {
		input[t] = make([]float64, numChannels)
		for c := range numChannels {
			input[t][c] = float64(t*numChannels+c) * 0.01
		}
	}
	return input
}

func TestFlowStateInferForecastOutputShape(t *testing.T) {
	cfg := testFlowStateConfig()
	m, err := newTestFlowStateModel(cfg)
	if err != nil {
		t.Fatalf("newTestFlowStateModel: %v", err)
	}

	input := makeInput(cfg.ContextLen, cfg.NumChannels)
	output, err := m.Forecast(input)
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

func TestFlowStateInferForecastNonZero(t *testing.T) {
	cfg := testFlowStateConfig()
	m, err := newTestFlowStateModel(cfg)
	if err != nil {
		t.Fatalf("newTestFlowStateModel: %v", err)
	}

	input := makeInput(cfg.ContextLen, cfg.NumChannels)
	output, err := m.Forecast(input)
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

func TestFlowStateInferForecastWithFreqDiffers(t *testing.T) {
	cfg := testFlowStateConfig()
	cfg.ScaleFactor = 1.0

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Build two models with the same weights to compare frequency adaptation.
	g1, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}
	m1 := &FlowStateModel{graph: g1, config: &(*cfg), engine: engine}

	g2, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}
	// Copy weights from g1 to g2 so they match.
	copyGraphWeights(g1, g2)
	cfg2 := *cfg
	m2 := &FlowStateModel{graph: g2, config: &cfg2, engine: engine}

	input := makeInput(cfg.ContextLen, cfg.NumChannels)

	out1, err := m1.Forecast(input)
	if err != nil {
		t.Fatalf("Forecast (hourly): %v", err)
	}

	out2, err := m2.ForecastWithFreq(input, "daily")
	if err != nil {
		t.Fatalf("ForecastWithFreq (daily): %v", err)
	}

	// Outputs should differ because scale factors are different.
	var maxDiff float64
	for i := range out1 {
		for j := range out1[i] {
			diff := math.Abs(out1[i][j] - out2[i][j])
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}
	if maxDiff < 1e-6 {
		t.Errorf("forecasts with different frequencies should differ, maxDiff=%e", maxDiff)
	}
}

func TestFlowStateInferForecastWithFreqAllSupported(t *testing.T) {
	cfg := testFlowStateConfig()

	for freq, expectedSF := range FrequencyScaleFactors {
		t.Run(freq, func(t *testing.T) {
			m, err := newTestFlowStateModel(cfg)
			if err != nil {
				t.Fatalf("newTestFlowStateModel: %v", err)
			}

			input := makeInput(cfg.ContextLen, cfg.NumChannels)
			_, err = m.ForecastWithFreq(input, freq)
			if err != nil {
				t.Fatalf("ForecastWithFreq(%q): %v", freq, err)
			}

			// Verify scale factor was applied.
			if m.config.ScaleFactor != expectedSF {
				t.Errorf("scale_factor after %q: got %f, want %f",
					freq, m.config.ScaleFactor, expectedSF)
			}
		})
	}
}

func TestFlowStateInferForecastWithFreqInvalid(t *testing.T) {
	cfg := testFlowStateConfig()
	m, err := newTestFlowStateModel(cfg)
	if err != nil {
		t.Fatalf("newTestFlowStateModel: %v", err)
	}

	input := makeInput(cfg.ContextLen, cfg.NumChannels)
	_, err = m.ForecastWithFreq(input, "every_3_seconds")
	if err == nil {
		t.Error("expected error for unsupported frequency, got nil")
	}
}

func TestFlowStateInferInputValidationWrongContextLen(t *testing.T) {
	cfg := testFlowStateConfig()
	m, err := newTestFlowStateModel(cfg)
	if err != nil {
		t.Fatalf("newTestFlowStateModel: %v", err)
	}

	// Wrong context length.
	input := makeInput(cfg.ContextLen+10, cfg.NumChannels)
	_, err = m.Forecast(input)
	if err == nil {
		t.Error("expected error for wrong context_len, got nil")
	}
}

func TestFlowStateInferInputValidationWrongChannels(t *testing.T) {
	cfg := testFlowStateConfig()
	m, err := newTestFlowStateModel(cfg)
	if err != nil {
		t.Fatalf("newTestFlowStateModel: %v", err)
	}

	// Wrong number of channels.
	input := makeInput(cfg.ContextLen, cfg.NumChannels+1)
	_, err = m.Forecast(input)
	if err == nil {
		t.Error("expected error for wrong channels, got nil")
	}
}

func TestFlowStateInferConfig(t *testing.T) {
	cfg := testFlowStateConfig()
	m, err := newTestFlowStateModel(cfg)
	if err != nil {
		t.Fatalf("newTestFlowStateModel: %v", err)
	}

	got := m.Config()
	if got.ContextLen != cfg.ContextLen {
		t.Errorf("Config().ContextLen: got %d, want %d", got.ContextLen, cfg.ContextLen)
	}
	if got.ForecastLen != cfg.ForecastLen {
		t.Errorf("Config().ForecastLen: got %d, want %d", got.ForecastLen, cfg.ForecastLen)
	}
}

func TestFlowStateInferCopyGraphWeights(t *testing.T) {
	cfg := testFlowStateConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g1, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}

	g2, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}

	copyGraphWeights(g1, g2)

	// Verify all parameters were copied by running the same input through both
	// graphs and checking outputs match.
	batch := 1
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i%50) * 0.02
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	ctx := context.Background()
	out1, err := g1.Forward(ctx, input)
	if err != nil {
		t.Fatalf("g1.Forward: %v", err)
	}
	out2, err := g2.Forward(ctx, input)
	if err != nil {
		t.Fatalf("g2.Forward: %v", err)
	}

	d1, d2 := out1.Data(), out2.Data()
	for i := range d1 {
		if d1[i] != d2[i] {
			t.Errorf("output[%d]: got %f vs %f after weight copy", i, d1[i], d2[i])
			break
		}
	}
}

func TestFlowStateInferFlowStateConfigFromGranite(t *testing.T) {
	granite := &GraniteTimeSeriesConfig{
		ModelType:    "flowstate",
		ContextLen:   512,
		ForecastLen:  96,
		NumSSMLayers: 4,
		ScaleFactor:  3.43,
	}
	granite.InputFeatures = 7
	granite.PatchLen = 16
	granite.HiddenDim = 64

	cfg, err := flowStateConfigFromGranite(granite)
	if err != nil {
		t.Fatalf("flowStateConfigFromGranite: %v", err)
	}

	if cfg.ContextLen != 512 {
		t.Errorf("ContextLen: got %d, want 512", cfg.ContextLen)
	}
	if cfg.ForecastLen != 96 {
		t.Errorf("ForecastLen: got %d, want 96", cfg.ForecastLen)
	}
	if cfg.NumChannels != 7 {
		t.Errorf("NumChannels: got %d, want 7", cfg.NumChannels)
	}
	if cfg.PatchLen != 16 {
		t.Errorf("PatchLen: got %d, want 16", cfg.PatchLen)
	}
	if cfg.DModel != 64 {
		t.Errorf("DModel: got %d, want 64", cfg.DModel)
	}
	if cfg.NumSSMLayers != 4 {
		t.Errorf("NumSSMLayers: got %d, want 4", cfg.NumSSMLayers)
	}
	if cfg.ScaleFactor != 3.43 {
		t.Errorf("ScaleFactor: got %f, want 3.43", cfg.ScaleFactor)
	}
}

func TestFlowStateInferFlowStateConfigFromGraniteDefaults(t *testing.T) {
	granite := &GraniteTimeSeriesConfig{
		ModelType:   "flowstate",
		ContextLen:  512,
		ForecastLen: 96,
	}
	granite.InputFeatures = 3

	cfg, err := flowStateConfigFromGranite(granite)
	if err != nil {
		t.Fatalf("flowStateConfigFromGranite: %v", err)
	}

	// Verify defaults are applied.
	if cfg.PatchLen != 8 {
		t.Errorf("PatchLen default: got %d, want 8", cfg.PatchLen)
	}
	if cfg.DModel != 128 {
		t.Errorf("DModel default: got %d, want 128", cfg.DModel)
	}
	if cfg.NumSSMLayers != 4 {
		t.Errorf("NumSSMLayers default: got %d, want 4", cfg.NumSSMLayers)
	}
	if cfg.ScaleFactor != 1.0 {
		t.Errorf("ScaleFactor default: got %f, want 1.0", cfg.ScaleFactor)
	}
}

func TestFlowStateInferFrequencyScaleFactors(t *testing.T) {
	expected := map[string]float32{
		"15min":          0.25,
		"quarter_hourly": 0.25,
		"hourly":         1.0,
		"daily":          3.43,
		"weekly":         24.0,
		"monthly":        104.0,
	}

	for freq, want := range expected {
		got, ok := FrequencyScaleFactors[freq]
		if !ok {
			t.Errorf("FrequencyScaleFactors missing %q", freq)
			continue
		}
		if got != want {
			t.Errorf("FrequencyScaleFactors[%q]: got %f, want %f", freq, got, want)
		}
	}
}

// TestFlowStateInferApplyFlowStateWeights verifies that weight loading by
// parameter name works correctly.
func TestFlowStateInferApplyFlowStateWeights(t *testing.T) {
	cfg := testFlowStateConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}

	// Create synthetic weight tensors matching parameter names.
	params := g.Parameters()
	tensors := make(map[string]*tensor.TensorNumeric[float32])
	for _, p := range params {
		shape := p.Value.Shape()
		size := 1
		for _, s := range shape {
			size *= s
		}
		data := make([]float32, size)
		for i := range data {
			data[i] = 42.0
		}
		w, wErr := tensor.New[float32](shape, data)
		if wErr != nil {
			t.Fatalf("create weight tensor %s: %v", p.Name, wErr)
		}
		tensors[p.Name] = w
	}

	applyFlowStateWeights(g, tensors)

	// Verify weights were applied.
	for _, p := range g.Parameters() {
		d := p.Value.Data()
		if len(d) == 0 {
			continue
		}
		if d[0] != 42.0 {
			t.Errorf("parameter %s: expected 42.0, got %f", p.Name, d[0])
		}
	}
}
