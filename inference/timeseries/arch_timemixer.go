// Package timeseries implements time-series model builders.
package timeseries

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/functional"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TimeMixerConfig holds configuration for building a TimeMixer computation graph.
type TimeMixerConfig struct {
	// InputLen is the lookback window length.
	InputLen int
	// OutputLen is the forecast horizon.
	OutputLen int
	// NumVars is the number of input/output variates.
	NumVars int
	// NumScales is the number of decomposition scales (default 4).
	NumScales int
	// HiddenSize is the hidden dimension for mixing MLPs (default 256).
	HiddenSize int
	// NumLayers is the number of mixing layers (default 3).
	NumLayers int
}

// validateTimeMixerConfig validates that the TimeMixerConfig has all required fields.
func validateTimeMixerConfig(cfg *TimeMixerConfig) error {
	if cfg.InputLen <= 0 {
		return fmt.Errorf("InputLen must be positive, got %d", cfg.InputLen)
	}
	if cfg.OutputLen <= 0 {
		return fmt.Errorf("OutputLen must be positive, got %d", cfg.OutputLen)
	}
	if cfg.NumVars <= 0 {
		return fmt.Errorf("NumVars must be positive, got %d", cfg.NumVars)
	}
	if cfg.NumScales <= 0 {
		cfg.NumScales = 4
	}
	if cfg.HiddenSize <= 0 {
		cfg.HiddenSize = 256
	}
	if cfg.NumLayers <= 0 {
		cfg.NumLayers = 3
	}
	return nil
}

// BuildTimeMixer constructs a TimeMixer computation graph from GGUF tensor weights.
//
// The graph accepts input of shape [batch, input_len, num_vars] and produces
// output of shape [batch, output_len, num_vars].
//
// The TimeMixer pipeline is:
//  1. Multi-scale decomposition via learnable moving averages (trend/seasonal)
//  2. Past-decomposable mixing: MLPs mix components across scales per layer
//  3. Scale-specific linear projection heads for trend and seasonal
//  4. Softmax-gated combination of scale predictions
func BuildTimeMixer[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TimeMixerConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], error) {
	if err := validateTimeMixerConfig(cfg); err != nil {
		return nil, fmt.Errorf("invalid TimeMixer config: %w", err)
	}

	ops := engine.Ops()
	node, err := newTimeMixerNode[T](tensors, cfg, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("create TimeMixer node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, cfg.InputLen, cfg.NumVars})
	builder.AddNode(node, input)

	return builder.Build(node)
}

// timeMixerMLP holds the two-layer MLP weights for cross-scale mixing.
type timeMixerMLP[T tensor.Float] struct {
	fc1 *core.Linear[T] // [numScales, hiddenSize]
	fc2 *core.Linear[T] // [hiddenSize, numScales]
}

// timeMixerNode implements the full TimeMixer forward pass as a single graph node.
type timeMixerNode[T tensor.Float] struct {
	cfg    *TimeMixerConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Learnable moving average kernel weights per scale.
	// maWeights[s] has shape [kernelSize] where kernelSize = 2^(s+1).
	maWeights []*graph.Parameter[T]

	// Seasonal mixing MLPs, one per layer.
	seasonalMLPs []timeMixerMLP[T]

	// Trend mixing MLPs, one per layer.
	trendMLPs []timeMixerMLP[T]

	// Scale-specific projection heads for trend and seasonal.
	// trendHeads[s]: [inputLen, outputLen]
	trendHeads []*graph.Parameter[T]
	// seasonalHeads[s]: [inputLen, outputLen]
	seasonalHeads []*graph.Parameter[T]

	// Mixing weights (pre-softmax): [numScales]
	mixWeights *graph.Parameter[T]

	// Final layer norm on output.
	finalNorm *normalization.RMSNorm[T]
}

func newTimeMixerNode[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TimeMixerConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*timeMixerNode[T], error) {
	// Initialize learnable MA weights per scale.
	maWeights := make([]*graph.Parameter[T], cfg.NumScales)
	for s := range cfg.NumScales {
		kernelSize := 1 << (s + 1) // 2, 4, 8, 16, ...
		data := make([]T, kernelSize)
		uniform := ops.FromFloat64(1.0 / float64(kernelSize))
		for i := range data {
			data[i] = uniform
		}
		t, err := tensor.New[T]([]int{kernelSize}, data)
		if err != nil {
			return nil, fmt.Errorf("create MA weights scale %d: %w", s, err)
		}
		name := fmt.Sprintf("timemixer.ma_weights.%d", s)
		p, err := graph.NewParameter[T](name, t, tensor.New[T])
		if err != nil {
			return nil, fmt.Errorf("create MA weight param scale %d: %w", s, err)
		}
		if w, ok := tensors[name]; ok {
			p.Value = w
		}
		maWeights[s] = p
	}

	// Initialize mixing MLPs.
	seasonalMLPs := make([]timeMixerMLP[T], cfg.NumLayers)
	trendMLPs := make([]timeMixerMLP[T], cfg.NumLayers)
	for l := range cfg.NumLayers {
		var err error
		sPrefix := fmt.Sprintf("timemixer.layer.%d.seasonal_mlp", l)
		seasonalMLPs[l].fc1, err = core.NewLinear[T](sPrefix+".fc1", engine, ops, cfg.NumScales, cfg.HiddenSize)
		if err != nil {
			return nil, fmt.Errorf("create seasonal MLP fc1 layer %d: %w", l, err)
		}
		loadLinearWeights(tensors, seasonalMLPs[l].fc1, sPrefix+".fc1.weight")

		seasonalMLPs[l].fc2, err = core.NewLinear[T](sPrefix+".fc2", engine, ops, cfg.HiddenSize, cfg.NumScales)
		if err != nil {
			return nil, fmt.Errorf("create seasonal MLP fc2 layer %d: %w", l, err)
		}
		loadLinearWeights(tensors, seasonalMLPs[l].fc2, sPrefix+".fc2.weight")

		tPrefix := fmt.Sprintf("timemixer.layer.%d.trend_mlp", l)
		trendMLPs[l].fc1, err = core.NewLinear[T](tPrefix+".fc1", engine, ops, cfg.NumScales, cfg.HiddenSize)
		if err != nil {
			return nil, fmt.Errorf("create trend MLP fc1 layer %d: %w", l, err)
		}
		loadLinearWeights(tensors, trendMLPs[l].fc1, tPrefix+".fc1.weight")

		trendMLPs[l].fc2, err = core.NewLinear[T](tPrefix+".fc2", engine, ops, cfg.HiddenSize, cfg.NumScales)
		if err != nil {
			return nil, fmt.Errorf("create trend MLP fc2 layer %d: %w", l, err)
		}
		loadLinearWeights(tensors, trendMLPs[l].fc2, tPrefix+".fc2.weight")
	}

	// Initialize scale-specific projection heads.
	xavierBound := math.Sqrt(6.0 / float64(cfg.InputLen+cfg.OutputLen))
	trendHeads := make([]*graph.Parameter[T], cfg.NumScales)
	seasonalHeads := make([]*graph.Parameter[T], cfg.NumScales)
	for s := range cfg.NumScales {
		tData := make([]T, cfg.InputLen*cfg.OutputLen)
		sData := make([]T, cfg.InputLen*cfg.OutputLen)
		scale := ops.FromFloat64(xavierBound)
		for i := range tData {
			tData[i] = ops.Mul(ops.FromFloat64(float64(i%11-5)*0.1), scale)
			sData[i] = ops.Mul(ops.FromFloat64(float64(i%7-3)*0.1), scale)
		}

		tName := fmt.Sprintf("timemixer.trend_head.%d.weight", s)
		tt, err := tensor.New[T]([]int{cfg.InputLen, cfg.OutputLen}, tData)
		if err != nil {
			return nil, fmt.Errorf("create trend head %d: %w", s, err)
		}
		tp, err := graph.NewParameter[T](tName, tt, tensor.New[T])
		if err != nil {
			return nil, err
		}
		if w, ok := tensors[tName]; ok {
			tp.Value = w
		}
		trendHeads[s] = tp

		sName := fmt.Sprintf("timemixer.seasonal_head.%d.weight", s)
		st, err := tensor.New[T]([]int{cfg.InputLen, cfg.OutputLen}, sData)
		if err != nil {
			return nil, fmt.Errorf("create seasonal head %d: %w", s, err)
		}
		sp, err := graph.NewParameter[T](sName, st, tensor.New[T])
		if err != nil {
			return nil, err
		}
		if w, ok := tensors[sName]; ok {
			sp.Value = w
		}
		seasonalHeads[s] = sp
	}

	// Initialize mixing weights (pre-softmax) to zeros (uniform after softmax).
	mwData := make([]T, cfg.NumScales)
	mwTensor, err := tensor.New[T]([]int{cfg.NumScales}, mwData)
	if err != nil {
		return nil, fmt.Errorf("create mix weights: %w", err)
	}
	mixWeights, err := graph.NewParameter[T]("timemixer.mix_weights", mwTensor, tensor.New[T])
	if err != nil {
		return nil, err
	}
	if w, ok := tensors["timemixer.mix_weights"]; ok {
		mixWeights.Value = w
	}

	// Final norm on output.
	finalNorm, err := normalization.NewRMSNorm[T]("timemixer_final_norm", engine, ops, cfg.NumVars)
	if err != nil {
		return nil, fmt.Errorf("create final norm: %w", err)
	}

	return &timeMixerNode[T]{
		cfg:           cfg,
		engine:        engine,
		ops:           ops,
		maWeights:     maWeights,
		seasonalMLPs:  seasonalMLPs,
		trendMLPs:     trendMLPs,
		trendHeads:    trendHeads,
		seasonalHeads: seasonalHeads,
		mixWeights:    mixWeights,
		finalNorm:     finalNorm,
	}, nil
}

func (n *timeMixerNode[T]) OpType() string { return "TimeMixer" }

func (n *timeMixerNode[T]) Attributes() map[string]interface{} {
	return map[string]interface{}{
		"input_len":   n.cfg.InputLen,
		"output_len":  n.cfg.OutputLen,
		"num_vars":    n.cfg.NumVars,
		"num_scales":  n.cfg.NumScales,
		"hidden_size": n.cfg.HiddenSize,
		"num_layers":  n.cfg.NumLayers,
	}
}

func (n *timeMixerNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.OutputLen, n.cfg.NumVars}
}

func (n *timeMixerNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	for _, mw := range n.maWeights {
		params = append(params, mw)
	}
	for _, mlp := range n.seasonalMLPs {
		params = append(params, mlp.fc1.Parameters()...)
		params = append(params, mlp.fc2.Parameters()...)
	}
	for _, mlp := range n.trendMLPs {
		params = append(params, mlp.fc1.Parameters()...)
		params = append(params, mlp.fc2.Parameters()...)
	}
	for _, h := range n.trendHeads {
		params = append(params, h)
	}
	for _, h := range n.seasonalHeads {
		params = append(params, h)
	}
	params = append(params, n.mixWeights)
	params = append(params, n.finalNorm.Parameters()...)
	return params
}

// Forward processes [batch, input_len, num_vars] input and produces [batch, output_len, num_vars].
//
// The forward pass:
//  1. Decomposes input into trend and seasonal at multiple scales via learnable moving averages
//  2. Mixes components across scales using per-layer MLPs with bottom-up residuals
//  3. Projects each scale's components via linear heads
//  4. Combines scale predictions with softmax-gated mixing weights
func (n *timeMixerNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TimeMixer expects 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("TimeMixer input must be 3D [batch, input_len, num_vars], got shape %v", shape)
	}

	batch, inputLen, numVars := shape[0], shape[1], shape[2]
	if inputLen != n.cfg.InputLen {
		return nil, fmt.Errorf("TimeMixer input_len mismatch: got %d, want %d", inputLen, n.cfg.InputLen)
	}
	if numVars != n.cfg.NumVars {
		return nil, fmt.Errorf("TimeMixer num_vars mismatch: got %d, want %d", numVars, n.cfg.NumVars)
	}

	// Step 1: Multi-scale decomposition.
	// For each scale, compute trend via causal weighted moving average,
	// then seasonal = input - trend.
	// trends[s] and seasonals[s] are [batch, inputLen, numVars].
	//
	// NOTE: .Data() access is justified here — the causal moving average with
	// edge padding is a custom decomposition kernel with no engine equivalent.
	trends := make([]*tensor.TensorNumeric[T], n.cfg.NumScales)
	seasonals := make([]*tensor.TensorNumeric[T], n.cfg.NumScales)

	data := x.Data()
	for s := range n.cfg.NumScales {
		// Softmax normalize the kernel weights via engine.
		normKernel, err := n.engine.Softmax(ctx, n.maWeights[s].Value, 0)
		if err != nil {
			return nil, fmt.Errorf("softmax MA kernel scale %d: %w", s, err)
		}
		kernel := normKernel.Data()
		kernelSize := len(kernel)

		trendData := make([]T, batch*inputLen*numVars)
		seasonalData := make([]T, batch*inputLen*numVars)

		for b := range batch {
			for v := range numVars {
				for t := range inputLen {
					var sum T
					for j := range kernelSize {
						idx := t - j
						if idx < 0 {
							idx = 0 // edge padding
						}
						sum += kernel[j] * data[b*inputLen*numVars+idx*numVars+v]
					}
					pos := b*inputLen*numVars + t*numVars + v
					trendData[pos] = sum
					seasonalData[pos] = data[pos] - sum
				}
			}
		}

		trends[s], err = tensor.New[T]([]int{batch, inputLen, numVars}, trendData)
		if err != nil {
			return nil, fmt.Errorf("create trend scale %d: %w", s, err)
		}
		seasonals[s], err = tensor.New[T]([]int{batch, inputLen, numVars}, seasonalData)
		if err != nil {
			return nil, fmt.Errorf("create seasonal scale %d: %w", s, err)
		}
	}

	// Step 2: Past-decomposable mixing across scales.
	for l := range n.cfg.NumLayers {
		var err error
		seasonals, err = n.mixAcrossScales(ctx, seasonals, n.seasonalMLPs[l], batch, inputLen, numVars)
		if err != nil {
			return nil, fmt.Errorf("seasonal mixing layer %d: %w", l, err)
		}
		trends, err = n.mixAcrossScales(ctx, trends, n.trendMLPs[l], batch, inputLen, numVars)
		if err != nil {
			return nil, fmt.Errorf("trend mixing layer %d: %w", l, err)
		}
	}

	// Step 3: Compute softmax mixing weights via engine.
	smWeights, err := n.engine.Softmax(ctx, n.mixWeights.Value, 0)
	if err != nil {
		return nil, fmt.Errorf("softmax mix weights: %w", err)
	}

	// Step 4: Project each scale and combine with mixing weights.
	// Output: [batch, outputLen, numVars]
	var output *tensor.TensorNumeric[T]

	for s := range n.cfg.NumScales {
		// Reshape trend [batch, inputLen, numVars] -> [batch*numVars, inputLen]
		// via transpose to [batch, numVars, inputLen] then reshape.
		trendT, err := n.engine.Transpose(ctx, trends[s], []int{0, 2, 1})
		if err != nil {
			return nil, fmt.Errorf("transpose trend scale %d: %w", s, err)
		}
		trendFlat, err := n.engine.Reshape(ctx, trendT, []int{batch * numVars, inputLen})
		if err != nil {
			return nil, fmt.Errorf("reshape trend scale %d: %w", s, err)
		}

		// Same for seasonal.
		seasonalT, err := n.engine.Transpose(ctx, seasonals[s], []int{0, 2, 1})
		if err != nil {
			return nil, fmt.Errorf("transpose seasonal scale %d: %w", s, err)
		}
		seasonalFlat, err := n.engine.Reshape(ctx, seasonalT, []int{batch * numVars, inputLen})
		if err != nil {
			return nil, fmt.Errorf("reshape seasonal scale %d: %w", s, err)
		}

		// Project: trend [batch*numVars, inputLen] x trendHead [inputLen, outputLen]
		trendProj, err := n.engine.MatMul(ctx, trendFlat, n.trendHeads[s].Value)
		if err != nil {
			return nil, fmt.Errorf("trend projection scale %d: %w", s, err)
		}
		// Project: seasonal [batch*numVars, inputLen] x seasonalHead [inputLen, outputLen]
		seasonProj, err := n.engine.MatMul(ctx, seasonalFlat, n.seasonalHeads[s].Value)
		if err != nil {
			return nil, fmt.Errorf("seasonal projection scale %d: %w", s, err)
		}

		// trendProj + seasonProj: [batch*numVars, outputLen]
		scaleProj, err := n.engine.Add(ctx, trendProj, seasonProj)
		if err != nil {
			return nil, fmt.Errorf("add projections scale %d: %w", s, err)
		}

		// Reshape back to [batch, numVars, outputLen] then transpose to [batch, outputLen, numVars].
		reshaped, err := n.engine.Reshape(ctx, scaleProj, []int{batch, numVars, n.cfg.OutputLen})
		if err != nil {
			return nil, fmt.Errorf("reshape projection scale %d: %w", s, err)
		}
		scaleOut, err := n.engine.Transpose(ctx, reshaped, []int{0, 2, 1})
		if err != nil {
			return nil, fmt.Errorf("transpose projection scale %d: %w", s, err)
		}

		// Extract this scale's mixing weight as a scalar and multiply.
		sw := smWeights.Data()[s]
		weighted, err := n.engine.MulScalar(ctx, scaleOut, sw)
		if err != nil {
			return nil, fmt.Errorf("weight scale %d: %w", s, err)
		}

		if output == nil {
			output = weighted
		} else {
			output, err = n.engine.Add(ctx, output, weighted)
			if err != nil {
				return nil, fmt.Errorf("accumulate scale %d: %w", s, err)
			}
		}
	}

	// Apply final norm.
	output, err = n.finalNorm.Forward(ctx, output)
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}

	return output, nil
}

// mixAcrossScales applies a mixing MLP across scales for a set of components.
// Each component is [batch, inputLen, numVars]. The MLP takes a [numScales]
// vector at each (batch, time, var) position and produces a new [numScales] vector.
// Bottom-up residuals flow from coarser to finer scales.
func (n *timeMixerNode[T]) mixAcrossScales(
	ctx context.Context,
	components []*tensor.TensorNumeric[T],
	mlp timeMixerMLP[T],
	batch, inputLen, numVars int,
) ([]*tensor.TensorNumeric[T], error) {
	numScales := len(components)
	totalPositions := batch * inputLen * numVars

	// Stack components along a new last axis: each component [batch*inputLen*numVars]
	// becomes a column, producing [totalPositions, numScales].
	flatComponents := make([]*tensor.TensorNumeric[T], numScales)
	for s := range numScales {
		var err error
		flatComponents[s], err = n.engine.Reshape(ctx, components[s], []int{totalPositions, 1})
		if err != nil {
			return nil, fmt.Errorf("reshape component %d for gather: %w", s, err)
		}
	}
	gathered, err := n.engine.Concat(ctx, flatComponents, 1)
	if err != nil {
		return nil, fmt.Errorf("concat cross-scale: %w", err)
	}

	// MLP forward: fc1 -> ReLU -> fc2
	hidden, err := mlp.fc1.Forward(ctx, gathered)
	if err != nil {
		return nil, fmt.Errorf("mlp fc1: %w", err)
	}

	hidden, err = functional.ReLU(ctx, n.engine, n.ops, hidden)
	if err != nil {
		return nil, fmt.Errorf("relu: %w", err)
	}

	mixed, err := mlp.fc2.Forward(ctx, hidden)
	if err != nil {
		return nil, fmt.Errorf("mlp fc2: %w", err)
	}

	// Scatter back to per-scale tensors by transposing and reshaping.
	// mixed is [totalPositions, numScales]. Transpose to [numScales, totalPositions].
	mixedT, err := n.engine.Transpose(ctx, mixed, []int{1, 0})
	if err != nil {
		return nil, fmt.Errorf("transpose mixed: %w", err)
	}

	result := make([]*tensor.TensorNumeric[T], numScales)
	for s := range numScales {
		// Extract row s: reshape [numScales, totalPositions] -> slice row s.
		// Use Reshape on the transposed data to get individual scale vectors.
		rowData := mixedT.Data()[s*totalPositions : (s+1)*totalPositions]
		result[s], err = tensor.New[T]([]int{batch, inputLen, numVars}, rowData)
		if err != nil {
			return nil, fmt.Errorf("scatter scale %d: %w", s, err)
		}
	}

	// Bottom-up residual: coarse (higher index) adds to next finer.
	for s := numScales - 2; s >= 0; s-- {
		result[s], err = n.engine.Add(ctx, result[s], result[s+1])
		if err != nil {
			return nil, fmt.Errorf("bottom-up residual scale %d: %w", s, err)
		}
	}

	return result, nil
}

func (n *timeMixerNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
