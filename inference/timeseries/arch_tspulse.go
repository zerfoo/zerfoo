package timeseries

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/timeseries"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TSPulseConfig holds configuration for the TSPulse multi-task time series model.
type TSPulseConfig struct {
	ContextLen  int    // input context window length
	NumChannels int    // number of input channels (variables)
	PatchLen    int    // patch size for the dual-space encoder
	DModel      int    // model embedding dimension
	NumLayers   int    // number of transformer encoder layers per path
	MaskType    string // "hybrid" or "block"
	HeadType    string // "allhead" or "dualhead"
	NumClasses  int    // for classification head (0 if not used)
}

// validateTSPulseConfig checks that all required fields are set and consistent.
func validateTSPulseConfig(cfg *TSPulseConfig) error {
	if cfg.ContextLen <= 0 {
		return fmt.Errorf("ContextLen must be positive, got %d", cfg.ContextLen)
	}
	if cfg.NumChannels <= 0 {
		return fmt.Errorf("NumChannels must be positive, got %d", cfg.NumChannels)
	}
	if cfg.PatchLen <= 0 {
		return fmt.Errorf("PatchLen must be positive, got %d", cfg.PatchLen)
	}
	if cfg.DModel <= 0 {
		return fmt.Errorf("DModel must be positive, got %d", cfg.DModel)
	}
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("NumLayers must be positive, got %d", cfg.NumLayers)
	}
	if cfg.MaskType != "hybrid" && cfg.MaskType != "block" {
		return fmt.Errorf("MaskType must be %q or %q, got %q", "hybrid", "block", cfg.MaskType)
	}
	if cfg.HeadType != "allhead" && cfg.HeadType != "dualhead" {
		return fmt.Errorf("HeadType must be %q or %q, got %q", "allhead", "dualhead", cfg.HeadType)
	}
	if cfg.NumClasses < 0 {
		return fmt.Errorf("NumClasses must be non-negative, got %d", cfg.NumClasses)
	}
	return nil
}

// TSPulseTask identifies which inference task to run.
type TSPulseTask int

const (
	// TSPulseAnomalyDetect computes per-timestep anomaly scores via reconstruction error.
	TSPulseAnomalyDetect TSPulseTask = iota
	// TSPulseClassify produces class probabilities from semantic embeddings.
	TSPulseClassify
	// TSPulseImpute fills missing values using reconstruction from fine-grained embeddings.
	TSPulseImpute
	// TSPulseEmbed returns the semantic embedding vector for similarity search.
	TSPulseEmbed
)

// TSPulseModel wraps the TSPulse architecture for multi-task inference.
type TSPulseModel struct {
	encoder   *timeseries.DualSpaceEncoder[float32]
	reconHead *core.Linear[float32] // [dModel] -> [patchLen * numChannels]
	classHead *core.Linear[float32] // [dModel] -> [numClasses] (nil if numClasses == 0)
	config    *TSPulseConfig
	engine    compute.Engine[float32]
}

// LoadTSPulse loads a TSPulse model from a GGUF file and returns an
// inference-ready multi-task model.
func LoadTSPulse(path string, opts ...Option) (*TSPulseModel, error) {
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

	graniteCfg, err := LoadGraniteTimeSeriesConfig(gf.Metadata)
	if err != nil {
		return nil, fmt.Errorf("load granite config: %w", err)
	}

	numClasses := 0
	if v, ok := getMetaInt(gf.Metadata, "ts.signal.num_classes"); ok {
		numClasses = v
	}

	cfg := &TSPulseConfig{
		ContextLen:  graniteCfg.ContextLen,
		NumChannels: graniteCfg.InputFeatures,
		PatchLen:    graniteCfg.PatchLen,
		DModel:      graniteCfg.HiddenDim,
		NumLayers:   graniteCfg.NumMixerLayers,
		MaskType:    graniteCfg.MaskType,
		HeadType:    graniteCfg.HeadType,
		NumClasses:  numClasses,
	}

	// Apply defaults.
	if cfg.MaskType == "" {
		cfg.MaskType = "hybrid"
	}
	if cfg.HeadType == "" {
		cfg.HeadType = "allhead"
	}
	if cfg.PatchLen == 0 {
		cfg.PatchLen = 8
	}
	if cfg.DModel == 0 {
		cfg.DModel = 128
	}
	if cfg.NumLayers == 0 {
		cfg.NumLayers = 2
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	model, err := newTSPulseModel(cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TSPulse model: %w", err)
	}

	// Load weights from GGUF.
	tensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return nil, fmt.Errorf("load GGUF tensors: %w", err)
	}
	applyTSPulseWeights(model, tensors)

	return model, nil
}

// newTSPulseModel constructs a TSPulseModel from configuration.
func newTSPulseModel(cfg *TSPulseConfig, engine compute.Engine[float32]) (*TSPulseModel, error) {
	if err := validateTSPulseConfig(cfg); err != nil {
		return nil, fmt.Errorf("invalid TSPulse config: %w", err)
	}

	ops := engine.Ops()

	encoder, err := timeseries.NewDualSpaceEncoder[float32](
		engine, ops, cfg.DModel, cfg.PatchLen, cfg.NumLayers,
	)
	if err != nil {
		return nil, fmt.Errorf("create dual-space encoder: %w", err)
	}

	// Reconstruction head: project each patch embedding back to raw patch values.
	// [dModel] -> [patchLen * numChannels]
	reconOut := cfg.PatchLen * cfg.NumChannels
	reconHead, err := core.NewLinear[float32]("tspulse_recon_head", engine, ops, cfg.DModel, reconOut)
	if err != nil {
		return nil, fmt.Errorf("create reconstruction head: %w", err)
	}

	var classHead *core.Linear[float32]
	if cfg.NumClasses > 0 {
		classHead, err = core.NewLinear[float32]("tspulse_class_head", engine, ops, cfg.DModel, cfg.NumClasses)
		if err != nil {
			return nil, fmt.Errorf("create classification head: %w", err)
		}
	}

	return &TSPulseModel{
		encoder:   encoder,
		reconHead: reconHead,
		classHead: classHead,
		config:    cfg,
		engine:    engine,
	}, nil
}

// applyTSPulseWeights loads GGUF tensor weights into the model parameters.
func applyTSPulseWeights(m *TSPulseModel, tensors map[string]*tensor.TensorNumeric[float32]) {
	// Apply to encoder parameters.
	for _, p := range m.encoder.Parameters() {
		if w, ok := tensors[p.Name]; ok {
			p.Value = w
		}
	}
	// Apply to reconstruction head.
	for _, p := range m.reconHead.Parameters() {
		if w, ok := tensors[p.Name]; ok {
			p.Value = w
		}
	}
	// Apply to classification head.
	if m.classHead != nil {
		for _, p := range m.classHead.Parameters() {
			if w, ok := tensors[p.Name]; ok {
				p.Value = w
			}
		}
	}
}

// encode runs the dual-space encoder on the input and returns the encoder output.
// Input: [][]float64 [context_len][channels] -> flattened to [1, context_len] per channel.
// For simplicity, we average across channels before encoding to produce a single
// time series, matching the DualSpaceEncoder's [batch, seqLen] input contract.
func (m *TSPulseModel) encode(input [][]float64) (*timeseries.DualSpaceOutput[float32], error) {
	if len(input) != m.config.ContextLen {
		return nil, fmt.Errorf("input length must be %d, got %d", m.config.ContextLen, len(input))
	}
	if len(input[0]) != m.config.NumChannels {
		return nil, fmt.Errorf("input channels must be %d, got %d", m.config.NumChannels, len(input[0]))
	}

	// Average across channels for encoder input: [1, context_len].
	data := make([]float32, m.config.ContextLen)
	for t := range m.config.ContextLen {
		var sum float64
		for c := range m.config.NumChannels {
			sum += input[t][c]
		}
		data[t] = float32(sum / float64(m.config.NumChannels))
	}

	inputTensor, err := tensor.New[float32]([]int{1, m.config.ContextLen}, data)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	ctx := context.Background()
	return m.encoder.Forward(ctx, inputTensor)
}

// reconstruct takes fine-grained embeddings [1, numPatches, dModel] and
// produces reconstructed time series [context_len][channels].
func (m *TSPulseModel) reconstruct(fineGrained *tensor.TensorNumeric[float32]) ([][]float64, error) {
	ctx := context.Background()
	shape := fineGrained.Shape()
	numPatches := shape[1]

	// Flatten to [numPatches, dModel] for the linear head.
	flat, err := m.engine.Reshape(ctx, fineGrained, []int{numPatches, m.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("reshape fine-grained: %w", err)
	}

	// Apply reconstruction head: [numPatches, dModel] -> [numPatches, patchLen * numChannels].
	recon, err := m.reconHead.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("reconstruction head forward: %w", err)
	}

	// Reshape to [numPatches * patchLen, numChannels] so each row is one timestep.
	totalSteps := numPatches * m.config.PatchLen
	recon, err = m.engine.Reshape(ctx, recon, []int{totalSteps, m.config.NumChannels})
	if err != nil {
		return nil, fmt.Errorf("reshape recon output: %w", err)
	}

	// Convert to output shape [context_len][channels].
	contextLen := totalSteps
	if contextLen > m.config.ContextLen {
		contextLen = m.config.ContextLen
	}

	result := make([][]float64, contextLen)
	for t := range contextLen {
		result[t] = make([]float64, m.config.NumChannels)
		for c := range m.config.NumChannels {
			v, vErr := recon.At(t, c)
			if vErr != nil {
				return nil, fmt.Errorf("read recon value at (%d,%d): %w", t, c, vErr)
			}
			result[t][c] = float64(v)
		}
	}

	return result, nil
}

// AnomalyDetect returns anomaly scores per timestep.
// Input: [][]float64 [context_len][channels]
// Output: []float64 [context_len] (reconstruction error per timestep)
func (m *TSPulseModel) AnomalyDetect(input [][]float64) ([]float64, error) {
	out, err := m.encode(input)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	recon, err := m.reconstruct(out.FineGrained)
	if err != nil {
		return nil, fmt.Errorf("reconstruct: %w", err)
	}

	// Compute MSE between original and reconstructed per timestep.
	contextLen := len(recon)
	scores := make([]float64, contextLen)
	for t := range contextLen {
		var mse float64
		for c := range m.config.NumChannels {
			diff := input[t][c] - recon[t][c]
			mse += diff * diff
		}
		scores[t] = mse / float64(m.config.NumChannels)
	}

	return scores, nil
}

// Classify returns class probabilities.
// Input: [][]float64 [context_len][channels]
// Output: []float64 [num_classes]
func (m *TSPulseModel) Classify(input [][]float64) ([]float64, error) {
	if m.classHead == nil {
		return nil, fmt.Errorf("classification head not configured (NumClasses=0)")
	}

	out, err := m.encode(input)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	ctx := context.Background()

	// Semantic embedding: [1, dModel]. Apply classification head.
	logits, err := m.classHead.Forward(ctx, out.Semantic)
	if err != nil {
		return nil, fmt.Errorf("classification head forward: %w", err)
	}

	// Apply softmax to get probabilities.
	probs, err := m.engine.Softmax(ctx, logits, -1)
	if err != nil {
		return nil, fmt.Errorf("softmax: %w", err)
	}

	// Reshape to 1D [numClasses] and convert to []float64.
	probs, err = m.engine.Reshape(ctx, probs, []int{m.config.NumClasses})
	if err != nil {
		return nil, fmt.Errorf("reshape probs: %w", err)
	}
	result := make([]float64, m.config.NumClasses)
	for i := range m.config.NumClasses {
		v, vErr := probs.At(i)
		if vErr != nil {
			return nil, fmt.Errorf("read prob at %d: %w", i, vErr)
		}
		result[i] = float64(v)
	}

	return result, nil
}

// Impute fills missing values in the time series.
// Input: [][]float64 [context_len][channels], mask []bool [context_len] (true=missing)
// Output: [][]float64 [context_len][channels] (reconstructed at masked positions, original elsewhere)
func (m *TSPulseModel) Impute(input [][]float64, mask []bool) ([][]float64, error) {
	if len(mask) != m.config.ContextLen {
		return nil, fmt.Errorf("mask length must be %d, got %d", m.config.ContextLen, len(mask))
	}

	out, err := m.encode(input)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	recon, err := m.reconstruct(out.FineGrained)
	if err != nil {
		return nil, fmt.Errorf("reconstruct: %w", err)
	}

	// Replace masked positions with reconstructed values, keep original elsewhere.
	result := make([][]float64, m.config.ContextLen)
	for t := range m.config.ContextLen {
		result[t] = make([]float64, m.config.NumChannels)
		if t < len(recon) && mask[t] {
			copy(result[t], recon[t])
		} else {
			copy(result[t], input[t])
		}
	}

	return result, nil
}

// Embed returns the semantic embedding vector for similarity search.
// Input: [][]float64 [context_len][channels]
// Output: []float64 [d_model]
func (m *TSPulseModel) Embed(input [][]float64) ([]float64, error) {
	out, err := m.encode(input)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}

	// Reshape semantic [1, dModel] to 1D [dModel] and extract as []float64.
	ctx := context.Background()
	sem, err := m.engine.Reshape(ctx, out.Semantic, []int{m.config.DModel})
	if err != nil {
		return nil, fmt.Errorf("reshape semantic: %w", err)
	}
	result := make([]float64, m.config.DModel)
	for i := range m.config.DModel {
		v, vErr := sem.At(i)
		if vErr != nil {
			return nil, fmt.Errorf("read semantic at %d: %w", i, vErr)
		}
		result[i] = float64(v)
	}

	return result, nil
}

// Config returns the TSPulse configuration.
func (m *TSPulseModel) Config() *TSPulseConfig {
	return m.config
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
