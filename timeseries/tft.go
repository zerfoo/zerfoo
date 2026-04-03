package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// TFTConfig holds the configuration for a Temporal Fusion Transformer.
type TFTConfig struct {
	NumStaticFeatures int
	NumTimeFeatures   int
	DModel            int
	NHeads            int
	NHorizons         int
	Quantiles         []float64
}

// grnWeights holds the parameters for a Gated Residual Network.
type grnWeights struct {
	fc1    linearLayer // input -> d_model (ELU)
	fc2    linearLayer // d_model -> d_model
	gate   linearLayer // d_model -> d_model (sigmoid for GLU)
	proj   linearLayer // skip projection (only used when input dim != d_model)
	lnGain *tensor.TensorNumeric[float32]
	lnBias *tensor.TensorNumeric[float32]
}

// TFT implements the Temporal Fusion Transformer for multi-horizon
// probabilistic forecasting.
type TFT struct {
	config TFTConfig
	engine compute.Engine[float32]
	ops    numeric.Arithmetic[float32]

	// Variable selection networks.
	staticVSNWeights []linearLayer // per-variable transforms
	staticVSNSelect  linearLayer   // softmax selection weights
	timeVSNWeights   []linearLayer // per-variable transforms
	timeVSNSelect    linearLayer   // softmax selection weights

	// Static covariate encoder (GRN).
	staticEncoder grnWeights

	// Temporal processing: per-step GRN applied to time features.
	temporalGRN grnWeights

	// Temporal self-attention.
	attnQ linearLayer
	attnK linearLayer
	attnV linearLayer
	attnO linearLayer

	// Post-attention GRN and layer norm.
	postAttnGRN grnWeights

	// Output projection: d_model -> n_horizons * n_quantiles.
	outputProj linearLayer
}

// NewTFT creates a new Temporal Fusion Transformer with the given configuration.
func NewTFT(config TFTConfig, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*TFT, error) {
	if config.NumStaticFeatures <= 0 {
		return nil, fmt.Errorf("timeseries: NumStaticFeatures must be positive, got %d", config.NumStaticFeatures)
	}
	if config.NumTimeFeatures <= 0 {
		return nil, fmt.Errorf("timeseries: NumTimeFeatures must be positive, got %d", config.NumTimeFeatures)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("timeseries: DModel must be positive, got %d", config.DModel)
	}
	if config.NHeads <= 0 {
		return nil, fmt.Errorf("timeseries: NHeads must be positive, got %d", config.NHeads)
	}
	if config.DModel%config.NHeads != 0 {
		return nil, fmt.Errorf("timeseries: DModel (%d) must be divisible by NHeads (%d)", config.DModel, config.NHeads)
	}
	if config.NHorizons <= 0 {
		return nil, fmt.Errorf("timeseries: NHorizons must be positive, got %d", config.NHorizons)
	}
	if len(config.Quantiles) == 0 {
		return nil, fmt.Errorf("timeseries: Quantiles must have at least one element")
	}
	for i, q := range config.Quantiles {
		if q <= 0 || q >= 1 {
			return nil, fmt.Errorf("timeseries: Quantiles[%d] must be in (0, 1), got %f", i, q)
		}
	}

	m := &TFT{
		config: config,
		engine: engine,
		ops:    ops,
	}

	var err error

	// Static variable selection network: each static feature gets a linear transform to d_model.
	m.staticVSNWeights = make([]linearLayer, config.NumStaticFeatures)
	for i := 0; i < config.NumStaticFeatures; i++ {
		m.staticVSNWeights[i], err = newLinearLayer(1, config.DModel)
		if err != nil {
			return nil, fmt.Errorf("timeseries: static VSN weight %d: %w", i, err)
		}
	}
	m.staticVSNSelect, err = newLinearLayer(config.NumStaticFeatures*config.DModel, config.NumStaticFeatures)
	if err != nil {
		return nil, fmt.Errorf("timeseries: static VSN select: %w", err)
	}

	// Time variable selection network.
	m.timeVSNWeights = make([]linearLayer, config.NumTimeFeatures)
	for i := 0; i < config.NumTimeFeatures; i++ {
		m.timeVSNWeights[i], err = newLinearLayer(1, config.DModel)
		if err != nil {
			return nil, fmt.Errorf("timeseries: time VSN weight %d: %w", i, err)
		}
	}
	m.timeVSNSelect, err = newLinearLayer(config.NumTimeFeatures*config.DModel, config.NumTimeFeatures)
	if err != nil {
		return nil, fmt.Errorf("timeseries: time VSN select: %w", err)
	}

	// Static covariate encoder GRN.
	m.staticEncoder, err = newGRN(config.DModel, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("timeseries: static encoder GRN: %w", err)
	}

	// Temporal GRN.
	m.temporalGRN, err = newGRN(config.DModel, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("timeseries: temporal GRN: %w", err)
	}

	// Temporal self-attention projections.
	m.attnQ, err = newLinearLayer(config.DModel, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("timeseries: attn Q: %w", err)
	}
	m.attnK, err = newLinearLayer(config.DModel, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("timeseries: attn K: %w", err)
	}
	m.attnV, err = newLinearLayer(config.DModel, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("timeseries: attn V: %w", err)
	}
	m.attnO, err = newLinearLayer(config.DModel, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("timeseries: attn O: %w", err)
	}

	// Post-attention GRN.
	m.postAttnGRN, err = newGRN(config.DModel, config.DModel)
	if err != nil {
		return nil, fmt.Errorf("timeseries: post-attn GRN: %w", err)
	}

	// Output projection.
	nQuantiles := len(config.Quantiles)
	m.outputProj, err = newLinearLayer(config.DModel, config.NHorizons*nQuantiles)
	if err != nil {
		return nil, fmt.Errorf("timeseries: output proj: %w", err)
	}

	return m, nil
}

// Predict runs the TFT forward pass and returns multi-horizon quantile forecasts.
// staticFeatures has shape [numStaticFeatures].
// timeFeatures has shape [seqLen][numTimeFeatures].
// Returns a slice of shape [nHorizons][nQuantiles].
func (m *TFT) Predict(staticFeatures []float64, timeFeatures [][]float64) ([][]float64, error) {
	ctx := context.Background()
	nq := len(m.config.Quantiles)
	seqLen := len(timeFeatures)

	if len(staticFeatures) != m.config.NumStaticFeatures {
		return nil, fmt.Errorf("timeseries: expected %d static features, got %d",
			m.config.NumStaticFeatures, len(staticFeatures))
	}
	if seqLen == 0 {
		return nil, fmt.Errorf("timeseries: timeFeatures must have at least one time step")
	}
	for i, tf := range timeFeatures {
		if len(tf) != m.config.NumTimeFeatures {
			return nil, fmt.Errorf("timeseries: timeFeatures[%d] has %d features, expected %d",
				i, len(tf), m.config.NumTimeFeatures)
		}
	}

	// 1. Static variable selection.
	staticSelected, err := m.variableSelection(ctx, staticFeatures, m.staticVSNWeights, m.staticVSNSelect)
	if err != nil {
		return nil, fmt.Errorf("timeseries: static VSN: %w", err)
	}

	// 2. Static covariate encoding via GRN.
	staticContext, err := m.applyGRN(ctx, staticSelected, m.staticEncoder)
	if err != nil {
		return nil, fmt.Errorf("timeseries: static encoder: %w", err)
	}

	// 3. Process each time step: variable selection + add static context + temporal GRN.
	temporalOutputs := make([]*tensor.TensorNumeric[float32], seqLen)
	for t := 0; t < seqLen; t++ {
		timeSelected, err := m.variableSelection(ctx, timeFeatures[t], m.timeVSNWeights, m.timeVSNSelect)
		if err != nil {
			return nil, fmt.Errorf("timeseries: time VSN step %d: %w", t, err)
		}

		// Add static context as enrichment.
		enriched, err := m.engine.Add(ctx, timeSelected, staticContext)
		if err != nil {
			return nil, fmt.Errorf("timeseries: enrich step %d: %w", t, err)
		}

		// Apply temporal GRN.
		temporalOutputs[t], err = m.applyGRN(ctx, enriched, m.temporalGRN)
		if err != nil {
			return nil, fmt.Errorf("timeseries: temporal GRN step %d: %w", t, err)
		}
	}

	// 4. Stack temporal outputs into [seqLen, d_model] for attention.
	temporalStack, err := m.engine.Concat(ctx, temporalOutputs, 0)
	if err != nil {
		return nil, fmt.Errorf("timeseries: stack temporal: %w", err)
	}

	// 5. Temporal self-attention.
	attnOut, err := m.selfAttention(ctx, temporalStack)
	if err != nil {
		return nil, fmt.Errorf("timeseries: self-attention: %w", err)
	}

	// 6. Post-attention: residual + GRN over the last time step.
	// Use the last time step from attention output as summary.
	lastStep, err := m.sliceLastRow(ctx, attnOut, seqLen)
	if err != nil {
		return nil, fmt.Errorf("timeseries: slice last step: %w", err)
	}

	// Residual connection with the last temporal output.
	lastTemporal := temporalOutputs[seqLen-1]
	residual, err := m.engine.Add(ctx, lastStep, lastTemporal)
	if err != nil {
		return nil, fmt.Errorf("timeseries: post-attn residual: %w", err)
	}

	postAttn, err := m.applyGRN(ctx, residual, m.postAttnGRN)
	if err != nil {
		return nil, fmt.Errorf("timeseries: post-attn GRN: %w", err)
	}

	// 7. Output projection: [1, d_model] -> [1, n_horizons * n_quantiles].
	output, err := m.linear(ctx, postAttn, m.outputProj)
	if err != nil {
		return nil, fmt.Errorf("timeseries: output proj: %w", err)
	}

	// 8. Reshape output to [n_horizons][n_quantiles].
	outputData := output.Data()
	result := make([][]float64, m.config.NHorizons)
	for h := 0; h < m.config.NHorizons; h++ {
		result[h] = make([]float64, nq)
		for q := 0; q < nq; q++ {
			result[h][q] = float64(outputData[h*nq+q])
		}
	}

	return result, nil
}

// VariableSelectionWeights returns the softmax variable importance weights
// for the static or time variable selection network. This enables
// interpretability — a key feature of TFT.
// Pass "static" or "time" as featureType.
func (m *TFT) VariableSelectionWeights(featureType string, features []float64) ([]float64, error) {
	ctx := context.Background()

	var vsnWeights []linearLayer
	var vsnSelect linearLayer

	switch featureType {
	case "static":
		vsnWeights = m.staticVSNWeights
		vsnSelect = m.staticVSNSelect
	case "time":
		vsnWeights = m.timeVSNWeights
		vsnSelect = m.timeVSNSelect
	default:
		return nil, fmt.Errorf("timeseries: featureType must be \"static\" or \"time\", got %q", featureType)
	}

	nVars := len(vsnWeights)
	if len(features) != nVars {
		return nil, fmt.Errorf("timeseries: expected %d features, got %d", nVars, len(features))
	}

	// Transform each variable and concatenate.
	transformed := make([]*tensor.TensorNumeric[float32], nVars)
	for i := 0; i < nVars; i++ {
		feat, err := tensor.New[float32]([]int{1, 1}, []float32{float32(features[i])})
		if err != nil {
			return nil, err
		}
		transformed[i], err = m.linear(ctx, feat, vsnWeights[i])
		if err != nil {
			return nil, err
		}
	}

	// Flatten and concatenate all transformed features: [1, nVars * d_model].
	flat, err := m.engine.Concat(ctx, transformed, 1)
	if err != nil {
		return nil, err
	}

	// Compute selection weights via softmax.
	selectLogits, err := m.linear(ctx, flat, vsnSelect)
	if err != nil {
		return nil, err
	}
	selectWeights, err := m.engine.Softmax(ctx, selectLogits, -1)
	if err != nil {
		return nil, err
	}

	data := selectWeights.Data()
	result := make([]float64, nVars)
	for i, v := range data {
		result[i] = float64(v)
	}
	return result, nil
}

// variableSelection applies the variable selection network to a set of features.
// Returns a tensor of shape [1, d_model].
func (m *TFT) variableSelection(
	ctx context.Context,
	features []float64,
	vsnWeights []linearLayer,
	vsnSelect linearLayer,
) (*tensor.TensorNumeric[float32], error) {
	nVars := len(vsnWeights)

	// Transform each variable independently: scalar -> [1, d_model].
	transformed := make([]*tensor.TensorNumeric[float32], nVars)
	for i := 0; i < nVars; i++ {
		feat, err := tensor.New[float32]([]int{1, 1}, []float32{float32(features[i])})
		if err != nil {
			return nil, err
		}
		transformed[i], err = m.linear(ctx, feat, vsnWeights[i])
		if err != nil {
			return nil, err
		}
	}

	// Flatten all transformed features for selection: [1, nVars * d_model].
	flat, err := m.engine.Concat(ctx, transformed, 1)
	if err != nil {
		return nil, err
	}

	// Selection weights: [1, nVars * d_model] -> [1, nVars] -> softmax.
	selectLogits, err := m.linear(ctx, flat, vsnSelect)
	if err != nil {
		return nil, err
	}
	selectWeights, err := m.engine.Softmax(ctx, selectLogits, -1)
	if err != nil {
		return nil, err
	}

	// Weighted sum of transformed features.
	swData := selectWeights.Data()
	result, err := m.engine.MulScalar(ctx, transformed[0], swData[0])
	if err != nil {
		return nil, err
	}
	for i := 1; i < nVars; i++ {
		scaled, err := m.engine.MulScalar(ctx, transformed[i], swData[i])
		if err != nil {
			return nil, err
		}
		result, err = m.engine.Add(ctx, result, scaled)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

// applyGRN applies a Gated Residual Network: linear -> ELU -> linear -> gate (GLU) + residual + layer norm.
func (m *TFT) applyGRN(ctx context.Context, x *tensor.TensorNumeric[float32], grn grnWeights) (*tensor.TensorNumeric[float32], error) {
	// fc1: x -> hidden (with ELU activation).
	h, err := m.linear(ctx, x, grn.fc1)
	if err != nil {
		return nil, err
	}
	h, err = m.engine.UnaryOp(ctx, h, eluScalar)
	if err != nil {
		return nil, err
	}

	// fc2: hidden -> candidate.
	candidate, err := m.linear(ctx, h, grn.fc2)
	if err != nil {
		return nil, err
	}

	// Gate: sigmoid for GLU-style gating.
	gate, err := m.linear(ctx, h, grn.gate)
	if err != nil {
		return nil, err
	}
	gate, err = m.engine.UnaryOp(ctx, gate, sigmoidScalar)
	if err != nil {
		return nil, err
	}

	// GLU: candidate * gate.
	gated, err := m.engine.Mul(ctx, candidate, gate)
	if err != nil {
		return nil, err
	}

	// Skip connection: project input if dimensions differ.
	skip := x
	xShape := x.Shape()
	gShape := gated.Shape()
	if xShape[len(xShape)-1] != gShape[len(gShape)-1] {
		skip, err = m.linear(ctx, x, grn.proj)
		if err != nil {
			return nil, err
		}
	}

	// Residual add.
	out, err := m.engine.Add(ctx, gated, skip)
	if err != nil {
		return nil, err
	}

	// Layer normalization.
	out, err = m.layerNorm(ctx, out, grn.lnGain, grn.lnBias)
	if err != nil {
		return nil, err
	}

	return out, nil
}

// selfAttention applies multi-head self-attention over temporal steps.
// input shape: [seqLen, d_model].
func (m *TFT) selfAttention(ctx context.Context, x *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Project to Q, K, V: each [seqLen, d_model].
	q, err := m.linear(ctx, x, m.attnQ)
	if err != nil {
		return nil, err
	}
	k, err := m.linear(ctx, x, m.attnK)
	if err != nil {
		return nil, err
	}
	v, err := m.linear(ctx, x, m.attnV)
	if err != nil {
		return nil, err
	}

	// Multi-head scaled dot-product attention via layers/functional.
	attnOut, err := functional.MultiHeadAttention(ctx, m.engine, q, k, v, m.config.NHeads)
	if err != nil {
		return nil, err
	}

	// Output projection.
	return m.linear(ctx, attnOut, m.attnO)
}

// sliceLastRow extracts the last row from a [seqLen, d_model] tensor,
// returning [1, d_model].
func (m *TFT) sliceLastRow(ctx context.Context, x *tensor.TensorNumeric[float32], seqLen int) (*tensor.TensorNumeric[float32], error) {
	_ = ctx
	data := x.Data()
	dModel := m.config.DModel
	start := (seqLen - 1) * dModel
	lastData := make([]float32, dModel)
	copy(lastData, data[start:start+dModel])
	return tensor.New[float32]([]int{1, dModel}, lastData)
}

// linear computes x @ W + b.
func (m *TFT) linear(ctx context.Context, x *tensor.TensorNumeric[float32], l linearLayer) (*tensor.TensorNumeric[float32], error) {
	out, err := m.engine.MatMul(ctx, x, l.weights)
	if err != nil {
		return nil, err
	}
	return m.engine.Add(ctx, out, l.biases)
}

// layerNorm applies layer normalization via layers/functional.LayerNorm.
func (m *TFT) layerNorm(
	ctx context.Context,
	x *tensor.TensorNumeric[float32],
	gain, bias *tensor.TensorNumeric[float32],
) (*tensor.TensorNumeric[float32], error) {
	return functional.LayerNorm(ctx, m.engine, x, gain, bias, float32(1e-5))
}

// newGRN creates a Gated Residual Network with initialized weights.
func newGRN(inDim, dModel int) (grnWeights, error) {
	fc1, err := newLinearLayer(inDim, dModel)
	if err != nil {
		return grnWeights{}, err
	}
	fc2, err := newLinearLayer(dModel, dModel)
	if err != nil {
		return grnWeights{}, err
	}
	gate, err := newLinearLayer(dModel, dModel)
	if err != nil {
		return grnWeights{}, err
	}
	proj, err := newLinearLayer(inDim, dModel)
	if err != nil {
		return grnWeights{}, err
	}

	// Layer norm gain (ones) and bias (zeros).
	gainData := make([]float32, dModel)
	for i := range gainData {
		gainData[i] = 1.0
	}
	gain, err := tensor.New[float32]([]int{1, dModel}, gainData)
	if err != nil {
		return grnWeights{}, err
	}
	biasData := make([]float32, dModel)
	lnBias, err := tensor.New[float32]([]int{1, dModel}, biasData)
	if err != nil {
		return grnWeights{}, err
	}

	return grnWeights{
		fc1:    fc1,
		fc2:    fc2,
		gate:   gate,
		proj:   proj,
		lnGain: gain,
		lnBias: lnBias,
	}, nil
}

// eluScalar computes the ELU activation for a single float32 value.
func eluScalar(x float32) float32 {
	if x >= 0 {
		return x
	}
	return float32(math.Exp(float64(x)) - 1)
}

// sigmoidScalar computes the sigmoid activation for a single float32 value.
func sigmoidScalar(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// QuantileLoss computes the asymmetric quantile loss (pinball loss) for a set of predictions.
// predicted has shape [nHorizons][nQuantiles], targets has shape [nHorizons].
func QuantileLoss(predicted [][]float64, targets []float64, quantiles []float64) (float64, error) {
	if len(predicted) == 0 {
		return 0, fmt.Errorf("timeseries: predicted must not be empty")
	}
	if len(predicted) != len(targets) {
		return 0, fmt.Errorf("timeseries: predicted horizons (%d) != targets (%d)", len(predicted), len(targets))
	}
	nq := len(quantiles)
	for i, p := range predicted {
		if len(p) != nq {
			return 0, fmt.Errorf("timeseries: predicted[%d] has %d quantiles, expected %d", i, len(p), nq)
		}
	}

	var totalLoss float64
	count := 0
	for h := 0; h < len(targets); h++ {
		for q := 0; q < nq; q++ {
			diff := targets[h] - predicted[h][q]
			if diff >= 0 {
				totalLoss += quantiles[q] * diff
			} else {
				totalLoss += (quantiles[q] - 1) * diff
			}
			count++
		}
	}

	return totalLoss / float64(count), nil
}
