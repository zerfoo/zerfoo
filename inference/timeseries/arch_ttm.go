// Package timeseries implements time-series model builders.
package timeseries

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/timeseries"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TTMConfig holds configuration for building a TTM computation graph.
type TTMConfig struct {
	ContextLen     int  // input context window length
	ForecastLen    int  // output forecast horizon
	NumChannels    int  // number of input/output channels (variables)
	PatchLen       int  // patch size for adaptive patching
	NumPatches     int  // number of patches (ContextLen / PatchLen)
	DModel         int  // model hidden dimension
	NumMixerLayers int  // number of TSMixer blocks in the encoder backbone
	ChannelMixing  bool // if true, include feature-mixing MLP in TSMixer blocks
	Expansion      int  // MLP expansion factor, default 2
	NumExogenous   int  // number of future exogenous channels (0 = none)
	NumStatic      int  // number of static categorical features (0 = none)
}

// validateTTMConfig validates that the TTMConfig has all required fields set.
func validateTTMConfig(cfg *TTMConfig) error {
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
	if cfg.NumMixerLayers <= 0 {
		return fmt.Errorf("NumMixerLayers must be positive, got %d", cfg.NumMixerLayers)
	}
	if cfg.Expansion <= 0 {
		return fmt.Errorf("Expansion must be positive, got %d", cfg.Expansion)
	}
	// Derive NumPatches if not set.
	if cfg.NumPatches <= 0 {
		if cfg.ContextLen%cfg.PatchLen != 0 {
			return fmt.Errorf("ContextLen (%d) must be divisible by PatchLen (%d) when NumPatches is not set", cfg.ContextLen, cfg.PatchLen)
		}
		cfg.NumPatches = cfg.ContextLen / cfg.PatchLen
	}
	return nil
}

// BuildTTM constructs a TTM computation graph from GGUF tensor weights.
//
// The graph accepts input of shape [batch, context_len, channels] and
// produces output of shape [batch, forecast_len, channels].
//
// The TTM pipeline is:
//  1. Input normalization (standard scaling per channel)
//  2. Adaptive patching: reshape input into patches
//  3. Patch embedding: linear projection [patch_len] -> [d_model]
//  4. TSMixer backbone: N stacked TSMixer blocks
//  5. Decoder: project encoder output to forecast patches
//  6. Forecast head: linear [d_model] -> [forecast_len / num_forecast_patches * channels]
//  7. Output denormalization
func BuildTTM[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TTMConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], error) {
	g, _, err := buildTTMWithNode(tensors, cfg, engine)
	return g, err
}

// buildTTMWithNode constructs a TTM graph and returns both the graph and the
// internal node (needed for setting exogenous/static inputs before forward).
func buildTTMWithNode[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TTMConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], *ttmNode[T], error) {
	if err := validateTTMConfig(cfg); err != nil {
		return nil, nil, fmt.Errorf("invalid TTM config: %w", err)
	}

	ops := engine.Ops()
	node, err := newTTMNode[T](tensors, cfg, engine, ops)
	if err != nil {
		return nil, nil, fmt.Errorf("create TTM node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, cfg.ContextLen, cfg.NumChannels})
	builder.AddNode(node, input)

	g, err := builder.Build(node)
	if err != nil {
		return nil, nil, err
	}
	return g, node, nil
}

// ttmNode implements the full TTM forward pass as a single graph node.
type ttmNode[T tensor.Float] struct {
	cfg    *TTMConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Patch embedding: linear projection [patch_len] -> [d_model].
	patchEmbed *timeseries.PatchEmbed[T]

	// Encoder: stacked TSMixer blocks.
	encoderBlocks []*timeseries.TSMixerBlock[T]

	// Encoder output norm.
	encoderNorm *normalization.LayerNormalization[T]

	// Forecast head: linear [d_model (+ d_model if exog)] -> [forecast_patch_len].
	// forecast_patch_len = ForecastLen (projects directly to forecast length).
	forecastHead *core.Linear[T]

	// Exogenous variable projection: linear [num_exog] -> [d_model].
	// nil when NumExogenous == 0.
	exogProj *core.Linear[T]

	// Static categorical feature embedding: linear [num_static] -> [d_model].
	// nil when NumStatic == 0.
	staticEmbed *core.Linear[T]

	// exogInput holds exogenous data [batch, forecast_len, num_exog] set before
	// calling Forward. nil when not in use. This is set via SetExogenous and
	// cleared after each Forward call.
	exogInput *tensor.TensorNumeric[T]

	// staticInput holds static feature data [batch, num_static] set before
	// calling Forward. nil when not in use. Set via SetStatic and cleared
	// after each Forward call.
	staticInput *tensor.TensorNumeric[T]
}

func newTTMNode[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TTMConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*ttmNode[T], error) {
	// Patch embedding.
	pe, err := timeseries.NewPatchEmbed[T]("ttm_patch_embed", engine, ops, cfg.PatchLen, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create patch embed: %w", err)
	}

	// Load patch embedding weights from GGUF tensors if available.
	if w, ok := tensors["embedding.weight"]; ok {
		params := pe.Parameters()
		if len(params) > 0 {
			params[0].Value = w
		}
	}

	// Encoder blocks.
	blocks := make([]*timeseries.TSMixerBlock[T], cfg.NumMixerLayers)
	for i := range cfg.NumMixerLayers {
		block, bErr := timeseries.NewTSMixerBlock[T](
			engine, ops,
			cfg.NumPatches, cfg.DModel, cfg.Expansion,
			cfg.ChannelMixing,
		)
		if bErr != nil {
			return nil, fmt.Errorf("create encoder block %d: %w", i, bErr)
		}

		// Load block weights from GGUF tensors.
		loadBlockWeights(tensors, block, i)

		blocks[i] = block
	}

	// Encoder output LayerNorm.
	encNorm, err := normalization.NewLayerNormalization[T](engine, cfg.DModel)
	if err != nil {
		return nil, fmt.Errorf("create encoder norm: %w", err)
	}

	// Forecast head input dimension: d_model + d_model if exogenous features
	// are present (the exogenous projection outputs d_model and is concatenated).
	forecastInputDim := cfg.DModel
	if cfg.NumExogenous > 0 {
		forecastInputDim += cfg.DModel
	}

	// Forecast head: maps [forecastInputDim] -> [forecast_len].
	head, err := core.NewLinear[T]("ttm_forecast_head", engine, ops, forecastInputDim, cfg.ForecastLen)
	if err != nil {
		return nil, fmt.Errorf("create forecast head: %w", err)
	}

	// Load forecast head weights from GGUF.
	if w, ok := tensors["head.linear.weight"]; ok {
		params := head.Parameters()
		if len(params) > 0 {
			params[0].Value = w
		}
	}

	node := &ttmNode[T]{
		cfg:           cfg,
		engine:        engine,
		ops:           ops,
		patchEmbed:    pe,
		encoderBlocks: blocks,
		encoderNorm:   encNorm,
		forecastHead:  head,
	}

	// Exogenous variable projection: [num_exog] -> [d_model].
	if cfg.NumExogenous > 0 {
		exogProj, exErr := core.NewLinear[T]("ttm_exog_proj", engine, ops, cfg.NumExogenous, cfg.DModel)
		if exErr != nil {
			return nil, fmt.Errorf("create exog projection: %w", exErr)
		}
		if w, ok := tensors["exog_proj.weight"]; ok {
			params := exogProj.Parameters()
			if len(params) > 0 {
				params[0].Value = w
			}
		}
		node.exogProj = exogProj
	}

	// Static categorical feature embedding: [num_static] -> [d_model].
	if cfg.NumStatic > 0 {
		staticEmbed, sErr := core.NewLinear[T]("ttm_static_embed", engine, ops, cfg.NumStatic, cfg.DModel)
		if sErr != nil {
			return nil, fmt.Errorf("create static embedding: %w", sErr)
		}
		if w, ok := tensors["static_embed.weight"]; ok {
			params := staticEmbed.Parameters()
			if len(params) > 0 {
				params[0].Value = w
			}
		}
		node.staticEmbed = staticEmbed
	}

	return node, nil
}

// loadBlockWeights loads GGUF tensor weights into a TSMixer block.
// Weight naming convention: blk.{i}.mlp.fc1.weight, blk.{i}.norm.weight, etc.
func loadBlockWeights[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	block *timeseries.TSMixerBlock[T],
	layerIdx int,
) {
	params := block.Parameters()

	// TSMixer block parameters are returned in order:
	// timeMLP1.weights, timeMLP2.weights, timeNorm.params...,
	// [featMLP1.weights, featMLP2.weights, featNorm.params...] (if channel mixing)
	//
	// Map GGUF names to parameter indices.
	weightNames := []string{
		fmt.Sprintf("blk.%d.time_mlp.fc1.weight", layerIdx),
		fmt.Sprintf("blk.%d.time_mlp.fc2.weight", layerIdx),
		fmt.Sprintf("blk.%d.time_norm.weight", layerIdx),
		fmt.Sprintf("blk.%d.time_norm.bias", layerIdx),
		fmt.Sprintf("blk.%d.feat_mlp.fc1.weight", layerIdx),
		fmt.Sprintf("blk.%d.feat_mlp.fc2.weight", layerIdx),
		fmt.Sprintf("blk.%d.feat_norm.weight", layerIdx),
		fmt.Sprintf("blk.%d.feat_norm.bias", layerIdx),
	}

	for i, name := range weightNames {
		if w, ok := tensors[name]; ok && i < len(params) {
			params[i].Value = w
		}
	}
}

func (n *ttmNode[T]) OpType() string { return "TTM" }

func (n *ttmNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"context_len":      n.cfg.ContextLen,
		"forecast_len":     n.cfg.ForecastLen,
		"num_channels":     n.cfg.NumChannels,
		"patch_len":        n.cfg.PatchLen,
		"num_patches":      n.cfg.NumPatches,
		"d_model":          n.cfg.DModel,
		"num_mixer_layers": n.cfg.NumMixerLayers,
		"channel_mixing":   n.cfg.ChannelMixing,
		"expansion":        n.cfg.Expansion,
	}
}

func (n *ttmNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.ForecastLen, n.cfg.NumChannels}
}

func (n *ttmNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, n.patchEmbed.Parameters()...)
	for _, block := range n.encoderBlocks {
		params = append(params, block.Parameters()...)
	}
	params = append(params, n.encoderNorm.Parameters()...)
	params = append(params, n.forecastHead.Parameters()...)
	if n.exogProj != nil {
		params = append(params, n.exogProj.Parameters()...)
	}
	if n.staticEmbed != nil {
		params = append(params, n.staticEmbed.Parameters()...)
	}
	return params
}

// Forward processes [batch, context_len, channels] input and produces
// [batch, forecast_len, channels].
//
// Exogenous and static inputs are provided via SetExogenous/SetStatic
// before calling Forward. They are consumed (cleared) after each call.
func (n *ttmNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TTM expects 1 input, got %d", len(inputs))
	}

	// Capture and clear exogenous/static state for this forward pass.
	exogInput := n.exogInput
	staticInput := n.staticInput
	n.exogInput = nil
	n.staticInput = nil

	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("TTM input must be 3D [batch, context_len, channels], got shape %v", shape)
	}

	batch, _, numChannels := shape[0], shape[1], shape[2]

	// Step 1: Input normalization — compute per-channel mean and std.
	mean, std, err := n.channelStats(ctx, x, batch, numChannels)
	if err != nil {
		return nil, fmt.Errorf("compute channel stats: %w", err)
	}

	// Normalize: (x - mean) / std.
	x, err = n.normalizeInput(ctx, x, mean, std, batch, numChannels)
	if err != nil {
		return nil, fmt.Errorf("normalize input: %w", err)
	}

	// Compute static bias if static features are provided.
	// Static features [batch, num_static] -> projected to [batch, d_model]
	// and broadcast-added as bias to each mixer block's output.
	var staticBias *tensor.TensorNumeric[T]
	if n.staticEmbed != nil && staticInput != nil {
		sShape := staticInput.Shape()
		if len(sShape) != 2 || sShape[0] != batch || sShape[1] != n.cfg.NumStatic {
			return nil, fmt.Errorf("static input must be [%d, %d], got %v", batch, n.cfg.NumStatic, sShape)
		}
		var sErr error
		staticBias, sErr = n.staticEmbed.Forward(ctx, staticInput)
		if sErr != nil {
			return nil, fmt.Errorf("static embed: %w", sErr)
		}
		// staticBias is [batch, d_model]; expand to [batch, 1, d_model] for broadcasting.
		staticBias, sErr = n.engine.Reshape(ctx, staticBias, []int{batch, 1, n.cfg.DModel})
		if sErr != nil {
			return nil, fmt.Errorf("reshape static bias: %w", sErr)
		}
	}

	// Compute exogenous projection if exogenous variables are provided.
	// Exogenous [batch, forecast_len, num_exog] -> mean-pooled and projected
	// to [batch, d_model], then concatenated with encoder output before forecast head.
	var exogProjected *tensor.TensorNumeric[T]
	if n.exogProj != nil && exogInput != nil {
		eShape := exogInput.Shape()
		if len(eShape) != 3 || eShape[0] != batch || eShape[1] != n.cfg.ForecastLen || eShape[2] != n.cfg.NumExogenous {
			return nil, fmt.Errorf("exogenous input must be [%d, %d, %d], got %v", batch, n.cfg.ForecastLen, n.cfg.NumExogenous, eShape)
		}
		// Mean-pool over forecast_len: [batch, forecast_len, num_exog] -> [batch, num_exog]
		exogPooled, eErr := n.engine.ReduceMean(ctx, exogInput, 1, false)
		if eErr != nil {
			return nil, fmt.Errorf("exog mean pool: %w", eErr)
		}
		// Project: [batch, num_exog] -> [batch, d_model]
		exogProjected, eErr = n.exogProj.Forward(ctx, exogPooled)
		if eErr != nil {
			return nil, fmt.Errorf("exog projection: %w", eErr)
		}
	}

	// Process each channel independently through patch embed + encoder.
	// TTM is channel-independent: each channel gets its own patching and encoding.
	channelOutputs := make([]*tensor.TensorNumeric[T], numChannels)
	for c := range numChannels {
		// Extract channel c: [batch, context_len]
		chSlice, cErr := n.extractChannel(ctx, x, batch, c, numChannels)
		if cErr != nil {
			return nil, fmt.Errorf("extract channel %d: %w", c, cErr)
		}

		// Step 2-3: Patch + embed: [batch, context_len] -> [batch, num_patches, d_model]
		embedded, cErr := n.patchEmbed.Forward(ctx, chSlice)
		if cErr != nil {
			return nil, fmt.Errorf("patch embed channel %d: %w", c, cErr)
		}

		// Step 4: TSMixer backbone.
		hidden := embedded
		for i, block := range n.encoderBlocks {
			hidden, cErr = block.Forward(ctx, hidden)
			if cErr != nil {
				return nil, fmt.Errorf("encoder block %d channel %d: %w", i, c, cErr)
			}

			// Add static bias after each mixer block if available.
			// staticBias is [batch, 1, d_model], hidden is [batch, num_patches, d_model].
			if staticBias != nil {
				hidden, cErr = n.engine.Add(ctx, hidden, staticBias)
				if cErr != nil {
					return nil, fmt.Errorf("add static bias block %d channel %d: %w", i, c, cErr)
				}
			}
		}

		// Encoder output norm.
		hidden, cErr = n.encoderNorm.Forward(ctx, hidden)
		if cErr != nil {
			return nil, fmt.Errorf("encoder norm channel %d: %w", c, cErr)
		}

		// Mean pool over patches: [batch, num_patches, d_model] -> [batch, d_model]
		hidden, cErr = n.engine.ReduceMean(ctx, hidden, 1, false)
		if cErr != nil {
			return nil, fmt.Errorf("mean pool channel %d: %w", c, cErr)
		}

		// If exogenous features are configured, concatenate projected exogenous
		// with encoder output: [batch, d_model] + [batch, d_model] -> [batch, 2*d_model].
		// When exogenous data is not provided for this call, pad with zeros.
		if n.exogProj != nil {
			if exogProjected != nil {
				hidden, cErr = n.engine.Concat(ctx, []*tensor.TensorNumeric[T]{hidden, exogProjected}, 1)
				if cErr != nil {
					return nil, fmt.Errorf("concat exog channel %d: %w", c, cErr)
				}
			} else {
				// Pad with zeros for the exogenous portion.
				zeros := make([]T, batch*n.cfg.DModel)
				zeroTensor, zErr := tensor.New[T]([]int{batch, n.cfg.DModel}, zeros)
				if zErr != nil {
					return nil, fmt.Errorf("create exog zero pad channel %d: %w", c, zErr)
				}
				hidden, cErr = n.engine.Concat(ctx, []*tensor.TensorNumeric[T]{hidden, zeroTensor}, 1)
				if cErr != nil {
					return nil, fmt.Errorf("concat exog zeros channel %d: %w", c, cErr)
				}
			}
		}

		// Step 5-6: Forecast head: [batch, d_model (or 2*d_model)] -> [batch, forecast_len]
		hidden, cErr = n.forecastHead.Forward(ctx, hidden)
		if cErr != nil {
			return nil, fmt.Errorf("forecast head channel %d: %w", c, cErr)
		}

		channelOutputs[c] = hidden
	}

	// Stack channel outputs: each is [batch, forecast_len].
	// Concatenate to [batch, numChannels * forecast_len].
	stacked, err := n.engine.Concat(ctx, channelOutputs, 1)
	if err != nil {
		return nil, fmt.Errorf("concat channel outputs: %w", err)
	}

	// Reshape to [batch, numChannels, forecast_len].
	stacked, err = n.engine.Reshape(ctx, stacked, []int{batch, numChannels, n.cfg.ForecastLen})
	if err != nil {
		return nil, fmt.Errorf("reshape stacked: %w", err)
	}

	// Transpose to [batch, forecast_len, numChannels].
	output, err := n.engine.Transpose(ctx, stacked, []int{0, 2, 1})
	if err != nil {
		return nil, fmt.Errorf("transpose to [batch, forecast_len, channels]: %w", err)
	}

	// Step 7: Output denormalization — reverse the scaling.
	output, err = n.denormalizeOutput(ctx, output, mean, std, batch, numChannels)
	if err != nil {
		return nil, fmt.Errorf("denormalize output: %w", err)
	}

	return output, nil
}

// channelStats computes per-channel mean and standard deviation.
// Returns mean [numChannels] and std [numChannels].
func (n *ttmNode[T]) channelStats(ctx context.Context, x *tensor.TensorNumeric[T], batch, numChannels int) ([]T, []T, error) {
	_ = ctx
	data := x.Data()
	shape := x.Shape()
	contextLen := shape[1]

	mean := make([]T, numChannels)
	std := make([]T, numChannels)
	count := T(batch * contextLen)

	for c := range numChannels {
		var sum T
		for b := range batch {
			for t := range contextLen {
				sum += data[b*contextLen*numChannels+t*numChannels+c]
			}
		}
		mean[c] = sum / count

		var sumSq T
		for b := range batch {
			for t := range contextLen {
				diff := data[b*contextLen*numChannels+t*numChannels+c] - mean[c]
				sumSq += diff * diff
			}
		}
		variance := sumSq / count
		// std with epsilon for numerical stability.
		std[c] = T(1.0)
		if variance > 0 {
			std[c] = T(float64Sqrt(float64(variance)))
		}
	}

	return mean, std, nil
}

// float64Sqrt computes sqrt for float64.
func float64Sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method.
	z := x
	for i := 0; i < 20; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// normalizeInput normalizes input per channel: (x - mean) / std.
func (n *ttmNode[T]) normalizeInput(_ context.Context, x *tensor.TensorNumeric[T], mean, std []T, batch, numChannels int) (*tensor.TensorNumeric[T], error) {
	data := x.Data()
	shape := x.Shape()
	contextLen := shape[1]
	out := make([]T, len(data))

	for b := range batch {
		for t := range contextLen {
			for c := range numChannels {
				idx := b*contextLen*numChannels + t*numChannels + c
				out[idx] = (data[idx] - mean[c]) / std[c]
			}
		}
	}

	return tensor.New[T](shape, out)
}

// extractChannel extracts channel c from [batch, context_len, num_channels] as [batch, context_len].
func (n *ttmNode[T]) extractChannel(ctx context.Context, x *tensor.TensorNumeric[T], batch, c, _ int) (*tensor.TensorNumeric[T], error) {
	contextLen := x.Shape()[1]

	// Slice channel c along axis 2: [batch, context_len, num_channels] -> [batch, context_len, 1].
	slicer := core.NewSlice[T](n.engine, []int64{int64(c)}, []int64{int64(c + 1)}, []int64{2}, nil)
	sliced, err := slicer.Forward(ctx, x)
	if err != nil {
		return nil, fmt.Errorf("slice channel %d: %w", c, err)
	}

	// Reshape to [batch, context_len].
	out, err := n.engine.Reshape(ctx, sliced, []int{batch, contextLen})
	if err != nil {
		return nil, fmt.Errorf("reshape channel %d: %w", c, err)
	}

	return out, nil
}

// denormalizeOutput reverses the input normalization: output * std + mean.
func (n *ttmNode[T]) denormalizeOutput(_ context.Context, output *tensor.TensorNumeric[T], mean, std []T, batch, numChannels int) (*tensor.TensorNumeric[T], error) {
	data := output.Data()
	shape := output.Shape()
	forecastLen := shape[1]
	out := make([]T, len(data))

	for b := range batch {
		for t := range forecastLen {
			for c := range numChannels {
				idx := b*forecastLen*numChannels + t*numChannels + c
				out[idx] = data[idx]*std[c] + mean[c]
			}
		}
	}

	return tensor.New[T](shape, out)
}

func (n *ttmNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// TTMModel wraps a compiled TTM computation graph for inference.
type TTMModel struct {
	graph   *graph.Graph[float32]
	cfg     *TTMConfig
	engine  compute.Engine[float32]
	granite *GraniteTimeSeriesConfig
	node    *ttmNode[float32] // retained to set exogenous/static inputs
}

// LoadTTM loads a TTM model from a GGUF file and returns an inference-ready model.
func LoadTTM(path string, opts ...Option) (*TTMModel, error) {
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

	// Build TTMConfig from Granite config.
	cfg := &TTMConfig{
		ContextLen:     graniteCfg.ContextLen,
		ForecastLen:    graniteCfg.ForecastLen,
		NumChannels:    graniteCfg.InputFeatures,
		PatchLen:       graniteCfg.PatchLen,
		NumPatches:     graniteCfg.NumPatches,
		DModel:         graniteCfg.HiddenDim,
		NumMixerLayers: graniteCfg.NumMixerLayers,
		ChannelMixing:  graniteCfg.ChannelMixing,
		Expansion:      2,
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Load tensor weights from GGUF file using the standard loader.
	tensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return nil, fmt.Errorf("load GGUF tensors: %w", err)
	}

	g, node, err := buildTTMWithNode[float32](tensors, cfg, engine)
	if err != nil {
		return nil, fmt.Errorf("build TTM graph: %w", err)
	}

	return &TTMModel{
		graph:   g,
		cfg:     cfg,
		engine:  engine,
		granite: graniteCfg,
		node:    node,
	}, nil
}

// Forecast runs inference on the loaded TTM model.
// Input is [context_len][channels] and output is [forecast_len][channels].
func (m *TTMModel) Forecast(input [][]float64) ([][]float64, error) {
	if len(input) != m.cfg.ContextLen {
		return nil, fmt.Errorf("input length must be %d, got %d", m.cfg.ContextLen, len(input))
	}
	if len(input[0]) != m.cfg.NumChannels {
		return nil, fmt.Errorf("input channels must be %d, got %d", m.cfg.NumChannels, len(input[0]))
	}

	// Convert [][]float64 to flat float32 tensor [1, context_len, channels].
	batch := 1
	data := make([]float32, batch*m.cfg.ContextLen*m.cfg.NumChannels)
	for t := range m.cfg.ContextLen {
		for c := range m.cfg.NumChannels {
			data[t*m.cfg.NumChannels+c] = float32(input[t][c])
		}
	}

	inputTensor, err := tensor.New[float32]([]int{batch, m.cfg.ContextLen, m.cfg.NumChannels}, data)
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
	result := make([][]float64, m.cfg.ForecastLen)
	for t := range m.cfg.ForecastLen {
		result[t] = make([]float64, m.cfg.NumChannels)
		for c := range m.cfg.NumChannels {
			result[t][c] = float64(outData[t*m.cfg.NumChannels+c])
		}
	}

	return result, nil
}

// ForecastWithExogenous runs inference with future exogenous variables.
// Input is [context_len][channels], exogenous is [forecast_len][num_exog].
// Output is [forecast_len][channels].
func (m *TTMModel) ForecastWithExogenous(input [][]float64, exogenous [][]float64) ([][]float64, error) {
	if m.cfg.NumExogenous <= 0 {
		return nil, fmt.Errorf("model not configured for exogenous variables (NumExogenous=%d)", m.cfg.NumExogenous)
	}
	if len(exogenous) != m.cfg.ForecastLen {
		return nil, fmt.Errorf("exogenous length must be %d, got %d", m.cfg.ForecastLen, len(exogenous))
	}
	if len(exogenous[0]) != m.cfg.NumExogenous {
		return nil, fmt.Errorf("exogenous channels must be %d, got %d", m.cfg.NumExogenous, len(exogenous[0]))
	}

	// Build exogenous tensor [1, forecast_len, num_exog].
	batch := 1
	exogData := make([]float32, batch*m.cfg.ForecastLen*m.cfg.NumExogenous)
	for t := range m.cfg.ForecastLen {
		for e := range m.cfg.NumExogenous {
			exogData[t*m.cfg.NumExogenous+e] = float32(exogenous[t][e])
		}
	}
	exogTensor, err := tensor.New[float32]([]int{batch, m.cfg.ForecastLen, m.cfg.NumExogenous}, exogData)
	if err != nil {
		return nil, fmt.Errorf("create exogenous tensor: %w", err)
	}

	// Set exogenous on the node for this forward pass.
	m.node.exogInput = exogTensor

	// Delegate to standard Forecast (which calls the graph forward).
	return m.Forecast(input)
}

// Config returns the TTM configuration.
func (m *TTMModel) Config() *TTMConfig {
	return m.cfg
}

// Option configures TTM model loading.
type Option func(*options)

type options struct{}

func defaultOptions() *options {
	return &options{}
}
