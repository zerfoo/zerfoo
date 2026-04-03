// Package timeseries implements time-series model builders.
package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/timeseries"
)

// FlowStateConfig holds configuration for the FlowState SSM-based time series
// forecasting model. FlowState uses an SSM encoder with a Functional Basis
// Decoder (Fourier basis) for continuous forecasting.
type FlowStateConfig struct {
	ContextLen   int     // input context window length
	ForecastLen  int     // prediction horizon
	NumChannels  int     // number of input/output variates
	PatchLen     int     // patch size for input patching
	DModel       int     // model dimension (SSM input/output)
	NumSSMLayers int     // number of stacked SSM layers
	DState       int     // SSM hidden state dimension
	NumBasis     int     // number of Fourier basis functions
	ScaleFactor  float32 // temporal scale for sampling rate adaptation
}

// BuildFlowState constructs a FlowState computation graph.
//
// Input:  [batch, context_len, num_channels]
// Output: [batch, forecast_len, num_channels]
//
// The architecture is:
//  1. Patch input per channel -> patch embedding
//  2. SSM encoder (N layers with residual + LayerNorm)
//  3. Mean pool over patches
//  4. Functional Basis Decoder (Fourier sin/cos)
func BuildFlowState[T tensor.Float](
	cfg *FlowStateConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], error) {
	if err := validateFlowStateConfig(cfg); err != nil {
		return nil, err
	}

	ops := engine.Ops()

	node, err := newFlowStateNode[T](cfg, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("create FlowState node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, cfg.ContextLen, cfg.NumChannels})
	builder.AddNode(node, input)

	return builder.Build(node)
}

func validateFlowStateConfig(cfg *FlowStateConfig) error {
	if cfg == nil {
		return fmt.Errorf("FlowStateConfig must not be nil")
	}
	if cfg.ContextLen <= 0 {
		return fmt.Errorf("ContextLen must be positive, got %d", cfg.ContextLen)
	}
	if cfg.ForecastLen <= 0 {
		return fmt.Errorf("ForecastLen must be positive, got %d", cfg.ForecastLen)
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
	if cfg.NumSSMLayers <= 0 {
		return fmt.Errorf("NumSSMLayers must be positive, got %d", cfg.NumSSMLayers)
	}
	if cfg.DState <= 0 {
		return fmt.Errorf("DState must be positive, got %d", cfg.DState)
	}
	if cfg.NumBasis <= 0 {
		return fmt.Errorf("NumBasis must be positive, got %d", cfg.NumBasis)
	}
	if cfg.ScaleFactor <= 0 {
		return fmt.Errorf("ScaleFactor must be positive, got %f", cfg.ScaleFactor)
	}
	return nil
}

// flowStateNode implements the full FlowState forward pass as a single graph node.
type flowStateNode[T tensor.Float] struct {
	cfg    *FlowStateConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Per-channel patch embedding (shared weights).
	patchEmbed *timeseries.PatchEmbed[T]

	// Stacked SSM layers with residual connections.
	ssmLayers []*timeseries.SSMLayer[T]
	layerNorms []*normalization.RMSNorm[T]

	// Final norm after SSM encoder.
	finalNorm *normalization.RMSNorm[T]

	// Functional Basis Decoder projection: [d_model] -> [num_basis * 2]
	// (sin coefficients + cos coefficients)
	basisProj *graph.Parameter[T]

	// Channel projection: [num_basis * 2] -> [num_channels] per time step,
	// applied after basis evaluation. Maps basis output to per-channel forecasts.
	channelProj *graph.Parameter[T]
}

func newFlowStateNode[T tensor.Float](cfg *FlowStateConfig, engine compute.Engine[T], ops numeric.Arithmetic[T]) (*flowStateNode[T], error) {
	pe, err := timeseries.NewPatchEmbed[T]("flowstate_patch_embed", engine, ops, cfg.PatchLen, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create patch embed: %w", err)
	}

	ssmLayers := make([]*timeseries.SSMLayer[T], cfg.NumSSMLayers)
	layerNorms := make([]*normalization.RMSNorm[T], cfg.NumSSMLayers)
	for i := range cfg.NumSSMLayers {
		ssm, sErr := timeseries.NewSSMLayer[T](engine, cfg.DState, cfg.DModel, cfg.DModel)
		if sErr != nil {
			return nil, fmt.Errorf("create SSM layer %d: %w", i, sErr)
		}
		// Rename SSM parameters to avoid collisions across layers.
		prefix := fmt.Sprintf("flowstate_ssm_%d", i)
		for _, p := range ssm.Parameters() {
			p.Name = fmt.Sprintf("%s_%s", prefix, p.Name)
		}

		ln, lnErr := normalization.NewRMSNorm[T](fmt.Sprintf("flowstate_norm_%d", i), engine, ops, cfg.DModel)
		if lnErr != nil {
			return nil, fmt.Errorf("create layer norm %d: %w", i, lnErr)
		}
		ssmLayers[i] = ssm
		layerNorms[i] = ln
	}

	finalNorm, err := normalization.NewRMSNorm[T]("flowstate_final_norm", engine, ops, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create final norm: %w", err)
	}

	// Basis projection: [d_model, num_basis * 2]
	basisDim := cfg.NumBasis * 2
	bpData := make([]T, cfg.DModel*basisDim)
	scale := T(math.Sqrt(2.0 / float64(cfg.DModel)))
	for i := range bpData {
		bpData[i] = ops.Mul(ops.FromFloat64(float64(i%7-3)*0.1), scale)
	}
	bpTensor, err := tensor.New[T]([]int{cfg.DModel, basisDim}, bpData)
	if err != nil {
		return nil, err
	}
	basisProj, err := graph.NewParameter[T]("flowstate_basis_proj", bpTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	// Channel projection: [1, num_channels] — broadcast across forecast_len.
	// This maps a single basis-decoded scalar to per-channel outputs.
	cpData := make([]T, cfg.NumChannels)
	cpScale := T(math.Sqrt(2.0 / float64(cfg.NumChannels)))
	for i := range cpData {
		cpData[i] = ops.Mul(ops.FromFloat64(float64(i%5-2)*0.1), cpScale)
	}
	cpTensor, err := tensor.New[T]([]int{1, cfg.NumChannels}, cpData)
	if err != nil {
		return nil, err
	}
	channelProj, err := graph.NewParameter[T]("flowstate_channel_proj", cpTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}

	return &flowStateNode[T]{
		cfg:         cfg,
		engine:      engine,
		ops:         ops,
		patchEmbed:  pe,
		ssmLayers:   ssmLayers,
		layerNorms:  layerNorms,
		finalNorm:   finalNorm,
		basisProj:   basisProj,
		channelProj: channelProj,
	}, nil
}

func (n *flowStateNode[T]) OpType() string { return "FlowState" }

func (n *flowStateNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"context_len":    n.cfg.ContextLen,
		"forecast_len":   n.cfg.ForecastLen,
		"num_channels":   n.cfg.NumChannels,
		"patch_len":      n.cfg.PatchLen,
		"d_model":        n.cfg.DModel,
		"num_ssm_layers": n.cfg.NumSSMLayers,
		"d_state":        n.cfg.DState,
		"num_basis":      n.cfg.NumBasis,
		"scale_factor":   n.cfg.ScaleFactor,
	}
}

func (n *flowStateNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.ForecastLen, n.cfg.NumChannels}
}

func (n *flowStateNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, n.patchEmbed.Parameters()...)
	for i, ssm := range n.ssmLayers {
		params = append(params, ssm.Parameters()...)
		params = append(params, n.layerNorms[i].Parameters()...)
	}
	params = append(params, n.finalNorm.Parameters()...)
	params = append(params, n.basisProj)
	params = append(params, n.channelProj)
	return params
}

// Forward processes [batch, context_len, num_channels] and produces [batch, forecast_len, num_channels].
//
// Pipeline:
//  1. Per-channel patching + embedding
//  2. SSM encoder with residual connections and layer norms
//  3. Mean pooling over patches
//  4. Functional Basis Decoder (Fourier sin/cos evaluation)
//  5. Channel projection to output shape
func (n *flowStateNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FlowState expects 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("FlowState input must be 3D [batch, context_len, num_channels], got shape %v", shape)
	}

	batch, seqLen, numCh := shape[0], shape[1], shape[2]

	// --- Step 1: Per-channel patch embedding ---
	// Process each channel independently through patch embed.
	// PatchEmbed expects [batch, seq_len] per channel.
	// Transpose [batch, seq_len, num_channels] -> [batch, num_channels, seq_len],
	// then split on axis 1 to get per-channel tensors.
	xT, err := n.engine.Transpose(ctx, x, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("transpose for channel split: %w", err)
	}
	channelSlices, err := n.engine.Split(ctx, xT, numCh, 1)
	if err != nil {
		return nil, fmt.Errorf("split channels: %w", err)
	}
	channelEmbeddings := make([]*tensor.TensorNumeric[T], numCh)
	for c := range numCh {
		// Reshape [batch, 1, seq_len] -> [batch, seq_len]
		varSlice, err := n.engine.Reshape(ctx, channelSlices[c], []int{batch, seqLen})
		if err != nil {
			return nil, fmt.Errorf("reshape channel %d: %w", c, err)
		}

		embedded, err := n.patchEmbed.Forward(ctx, varSlice)
		if err != nil {
			return nil, fmt.Errorf("patch embed channel %d: %w", c, err)
		}
		channelEmbeddings[c] = embedded
	}

	// Average channel embeddings: all are [batch, num_patches, d_model].
	// Sum then divide for mean across channels.
	hidden := channelEmbeddings[0]
	for c := 1; c < numCh; c++ {
		var err error
		hidden, err = n.engine.Add(ctx, hidden, channelEmbeddings[c])
		if err != nil {
			return nil, fmt.Errorf("sum channel embeddings: %w", err)
		}
	}
	if numCh > 1 {
		invCh := n.ops.FromFloat64(1.0 / float64(numCh))
		var err error
		hidden, err = n.engine.MulScalar(ctx, hidden, invCh)
		if err != nil {
			return nil, fmt.Errorf("mean channel embeddings: %w", err)
		}
	}

	// --- Step 2: SSM encoder with residual connections ---
	for i, ssm := range n.ssmLayers {
		// SSM forward: [batch, num_patches, d_model] -> [batch, num_patches, d_model]
		ssmOut, err := ssm.Forward(ctx, hidden)
		if err != nil {
			return nil, fmt.Errorf("SSM layer %d: %w", i, err)
		}

		// Residual connection.
		hidden, err = n.engine.Add(ctx, hidden, ssmOut)
		if err != nil {
			return nil, fmt.Errorf("residual add SSM %d: %w", i, err)
		}

		// Layer norm.
		hidden, err = n.layerNorms[i].Forward(ctx, hidden)
		if err != nil {
			return nil, fmt.Errorf("layer norm %d: %w", i, err)
		}
	}

	// Final norm.
	hidden, err = n.finalNorm.Forward(ctx, hidden)
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}

	// --- Step 3: Mean pooling over patches ---
	// hidden: [batch, num_patches, d_model] -> [batch, d_model]
	pooled, err := n.engine.ReduceMean(ctx, hidden, 1, false)
	if err != nil {
		return nil, fmt.Errorf("mean pool: %w", err)
	}

	// --- Step 4: Functional Basis Decoder ---
	// Project to basis coefficients: [batch, d_model] @ [d_model, num_basis*2] -> [batch, num_basis*2]
	coeffs, err := n.engine.MatMul(ctx, pooled, n.basisProj.Value)
	if err != nil {
		return nil, fmt.Errorf("basis projection: %w", err)
	}

	// Evaluate Fourier basis at target time points.
	// t[i] = i * scale_factor / forecast_len for i in [0, forecast_len)
	// forecast[t] = sum_k(a_k * sin(2*pi*k*t) + b_k * cos(2*pi*k*t))
	forecast, err := n.evaluateFourierBasis(coeffs, batch)
	if err != nil {
		return nil, fmt.Errorf("evaluate Fourier basis: %w", err)
	}

	// --- Step 5: Expand to channels ---
	// forecast: [batch, forecast_len] -> [batch, forecast_len, num_channels]
	// Reshape to [batch * forecast_len, 1] and matmul with [1, num_channels].
	flat, err := n.engine.Reshape(ctx, forecast, []int{batch * n.cfg.ForecastLen, 1})
	if err != nil {
		return nil, fmt.Errorf("reshape for channel proj: %w", err)
	}
	output, err := n.engine.MatMul(ctx, flat, n.channelProj.Value)
	if err != nil {
		return nil, fmt.Errorf("channel projection: %w", err)
	}
	output, err = n.engine.Reshape(ctx, output, []int{batch, n.cfg.ForecastLen, n.cfg.NumChannels})
	if err != nil {
		return nil, fmt.Errorf("reshape output: %w", err)
	}

	return output, nil
}

// evaluateFourierBasis evaluates the Fourier basis at discrete time points
// using the given coefficients.
//
// coeffs: [batch, num_basis * 2] where first num_basis are sin coefficients,
// next num_basis are cos coefficients.
//
// Output: [batch, forecast_len]
func (n *flowStateNode[T]) evaluateFourierBasis(coeffs *tensor.TensorNumeric[T], batch int) (*tensor.TensorNumeric[T], error) {
	numBasis := n.cfg.NumBasis
	forecastLen := n.cfg.ForecastLen
	scaleFactor := float64(n.cfg.ScaleFactor)

	coeffData := coeffs.Data()
	output := make([]T, batch*forecastLen)

	for b := range batch {
		for i := range forecastLen {
			// t = i * scale_factor / forecast_len (normalized to [0, scale_factor))
			t := float64(i) * scaleFactor / float64(forecastLen)

			var val float64
			for k := range numBasis {
				aK := float64(coeffData[b*numBasis*2+k])         // sin coefficient
				bK := float64(coeffData[b*numBasis*2+numBasis+k]) // cos coefficient
				freq := 2.0 * math.Pi * float64(k+1) * t
				val += aK*math.Sin(freq) + bK*math.Cos(freq)
			}
			output[b*forecastLen+i] = T(val)
		}
	}

	return tensor.New[T]([]int{batch, forecastLen}, output)
}

func (n *flowStateNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
