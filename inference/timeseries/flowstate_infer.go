package timeseries

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// FrequencyScaleFactors maps sampling frequency strings to their temporal
// scale factors used by the FlowState Functional Basis Decoder.
var FrequencyScaleFactors = map[string]float32{
	"15min":          0.25,
	"quarter_hourly": 0.25,
	"hourly":         1.0,
	"daily":          3.43, // 24/7
	"weekly":         24.0,
	"monthly":        104.0,
}

// FlowStateModel wraps a compiled FlowState computation graph for inference.
type FlowStateModel struct {
	graph   *graph.Graph[float32]
	config  *FlowStateConfig
	engine  compute.Engine[float32]
	granite *GraniteTimeSeriesConfig
}

// LoadFlowState loads a FlowState model from a GGUF file and returns an
// inference-ready model. The GGUF metadata must contain ts.signal.model_type
// set to "flowstate" along with the required FlowState configuration fields.
func LoadFlowState(path string, opts ...Option) (*FlowStateModel, error) {
	o := defaultOptions()
	for _, opt := range opts {
		opt(o)
	}

	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	// Load Granite time-series config from GGUF metadata.
	graniteCfg, err := LoadGraniteTimeSeriesConfig(gf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("load granite config: %w", err)
	}

	cfg, err := flowStateConfigFromGranite(graniteCfg)
	if err != nil {
		return nil, fmt.Errorf("build FlowState config: %w", err)
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	g, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build FlowState graph: %w", err)
	}

	// Load tensor weights from GGUF and apply to graph parameters.
	tensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return nil, fmt.Errorf("load GGUF tensors: %w", err)
	}
	applyFlowStateWeights(g, tensors)

	return &FlowStateModel{
		graph:   g,
		config:  cfg,
		engine:  engine,
		granite: graniteCfg,
	}, nil
}

// flowStateConfigFromGranite builds a FlowStateConfig from the Granite config.
func flowStateConfigFromGranite(g *GraniteTimeSeriesConfig) (*FlowStateConfig, error) {
	cfg := &FlowStateConfig{
		ContextLen:   g.ContextLen,
		ForecastLen:  g.ForecastLen,
		NumChannels:  g.InputFeatures,
		PatchLen:     g.PatchLen,
		DModel:       g.HiddenDim,
		NumSSMLayers: g.NumSSMLayers,
		DState:       g.HiddenDim, // default: same as DModel
		NumBasis:     16,          // default Fourier basis count
		ScaleFactor:  g.ScaleFactor,
	}

	// Apply overrides from base config where applicable.
	if cfg.PatchLen == 0 {
		cfg.PatchLen = 8
	}
	if cfg.DModel == 0 {
		cfg.DModel = 128
	}
	if cfg.NumSSMLayers == 0 {
		cfg.NumSSMLayers = 4
	}
	if cfg.DState == 0 {
		cfg.DState = cfg.DModel
	}
	if cfg.ScaleFactor == 0 {
		cfg.ScaleFactor = 1.0
	}

	return cfg, nil
}

// applyFlowStateWeights loads GGUF tensor weights into the graph parameters.
func applyFlowStateWeights(g *graph.Graph[float32], tensors map[string]*tensor.TensorNumeric[float32]) {
	for _, p := range g.Parameters() {
		if w, ok := tensors[p.Name]; ok {
			p.Value = w
		}
	}
}

// Forecast produces time series forecasts using the loaded FlowState model.
// Input shape: [context_len][channels] as [][]float64.
// Output shape: [forecast_len][channels] as [][]float64.
func (m *FlowStateModel) Forecast(input [][]float64) ([][]float64, error) {
	return m.forecastWithScaleFactor(input, m.config.ScaleFactor)
}

// ForecastWithFreq produces forecasts adapted to a specific sampling frequency.
// The frequency string adjusts the internal scale_factor used by the Fourier
// basis decoder. Supported values: "15min", "quarter_hourly", "hourly",
// "daily", "weekly", "monthly".
func (m *FlowStateModel) ForecastWithFreq(input [][]float64, freq string) ([][]float64, error) {
	sf, ok := FrequencyScaleFactors[freq]
	if !ok {
		return nil, fmt.Errorf("unsupported frequency %q; supported: 15min, quarter_hourly, hourly, daily, weekly, monthly", freq)
	}
	return m.forecastWithScaleFactor(input, sf)
}

// forecastWithScaleFactor runs the FlowState inference pipeline with the
// given scale factor. It rebuilds the graph if the scale factor differs from
// the current config to ensure the Fourier basis decoder uses the correct
// temporal scaling.
func (m *FlowStateModel) forecastWithScaleFactor(input [][]float64, scaleFactor float32) ([][]float64, error) {
	if err := m.validateInput(input); err != nil {
		return nil, err
	}

	// If scale factor changed, rebuild the graph with the updated config.
	if scaleFactor != m.config.ScaleFactor {
		adapted := *m.config
		adapted.ScaleFactor = scaleFactor
		g, err := BuildFlowState[float32](&adapted, m.engine)
		if err != nil {
			return nil, fmt.Errorf("rebuild graph for scale_factor %f: %w", scaleFactor, err)
		}
		// Copy weights from current graph to the new one.
		copyGraphWeights(m.graph, g)
		m.graph = g
		m.config.ScaleFactor = scaleFactor
	}

	// Convert [][]float64 to flat float32 tensor [1, context_len, channels].
	batch := 1
	data := make([]float32, batch*m.config.ContextLen*m.config.NumChannels)
	for t := range m.config.ContextLen {
		for c := range m.config.NumChannels {
			data[t*m.config.NumChannels+c] = float32(input[t][c])
		}
	}

	inputTensor, err := tensor.New[float32]([]int{batch, m.config.ContextLen, m.config.NumChannels}, data)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	ctx := context.Background()
	outputTensor, err := m.graph.Forward(ctx, inputTensor)
	if err != nil {
		return nil, fmt.Errorf("forward pass: %w", err)
	}

	// Convert output [1, forecast_len, channels] to [][]float64.
	outData := outputTensor.Data()
	result := make([][]float64, m.config.ForecastLen)
	for t := range m.config.ForecastLen {
		result[t] = make([]float64, m.config.NumChannels)
		for c := range m.config.NumChannels {
			result[t][c] = float64(outData[t*m.config.NumChannels+c])
		}
	}

	return result, nil
}

// validateInput checks that the input has the correct shape for the model.
func (m *FlowStateModel) validateInput(input [][]float64) error {
	if len(input) != m.config.ContextLen {
		return fmt.Errorf("input length must be %d, got %d", m.config.ContextLen, len(input))
	}
	if len(input) == 0 {
		return fmt.Errorf("input must not be empty")
	}
	if len(input[0]) != m.config.NumChannels {
		return fmt.Errorf("input channels must be %d, got %d", m.config.NumChannels, len(input[0]))
	}
	return nil
}

// copyGraphWeights copies parameter values from src to dst by matching
// parameter names.
func copyGraphWeights(src, dst *graph.Graph[float32]) {
	srcParams := make(map[string]*graph.Parameter[float32])
	for _, p := range src.Parameters() {
		srcParams[p.Name] = p
	}
	for _, p := range dst.Parameters() {
		if sp, ok := srcParams[p.Name]; ok {
			p.Value = sp.Value
		}
	}
}

// Config returns the FlowState configuration.
func (m *FlowStateModel) Config() *FlowStateConfig {
	return m.config
}
