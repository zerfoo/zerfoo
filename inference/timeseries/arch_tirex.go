// Package timeseries implements time-series model builders.
package timeseries

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/timeseries"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// TiRexConfig holds configuration for building a TiRex xLSTM graph.
type TiRexConfig struct {
	// NumLayers is the number of xLSTM blocks (alternating sLSTM/mLSTM).
	NumLayers int
	// InputDim is the number of input features per time step.
	InputDim int
	// HiddenDim is the hidden dimension of the xLSTM cells.
	HiddenDim int
	// Horizon is the prediction horizon.
	Horizon int
	// NumVars is the number of output variables (channels).
	NumVars int
	// BlockTypes specifies the type of each block: "slstm" or "mlstm".
	// Length must equal NumLayers. If nil, blocks alternate sLSTM/mLSTM.
	BlockTypes []string
}

// validateTiRexConfig validates that the TiRexConfig has all required fields.
func validateTiRexConfig(cfg *TiRexConfig) error {
	if cfg.NumLayers <= 0 {
		return fmt.Errorf("NumLayers must be positive, got %d", cfg.NumLayers)
	}
	if cfg.InputDim <= 0 {
		return fmt.Errorf("InputDim must be positive, got %d", cfg.InputDim)
	}
	if cfg.HiddenDim <= 0 {
		return fmt.Errorf("HiddenDim must be positive, got %d", cfg.HiddenDim)
	}
	if cfg.Horizon <= 0 {
		return fmt.Errorf("Horizon must be positive, got %d", cfg.Horizon)
	}
	if cfg.NumVars <= 0 {
		return fmt.Errorf("NumVars must be positive, got %d", cfg.NumVars)
	}
	if cfg.BlockTypes != nil {
		if len(cfg.BlockTypes) != cfg.NumLayers {
			return fmt.Errorf("BlockTypes length %d does not match NumLayers %d", len(cfg.BlockTypes), cfg.NumLayers)
		}
		for i, bt := range cfg.BlockTypes {
			if bt != "slstm" && bt != "mlstm" {
				return fmt.Errorf("BlockTypes[%d] must be \"slstm\" or \"mlstm\", got %q", i, bt)
			}
		}
	}
	return nil
}

// BuildTiRex constructs a TiRex xLSTM computation graph.
//
// The TiRex architecture stacks alternating sLSTM and mLSTM blocks with an
// input projection and output head. The pipeline is:
//
//  1. Input projection: [batch, seq_len, input_dim] -> [batch, seq_len, hidden_dim]
//  2. Sequence processing through alternating sLSTM/mLSTM blocks
//  3. Layer norm on final hidden state
//  4. Output head: [batch, hidden_dim] -> [batch, horizon * num_vars]
//  5. Reshape to [batch, horizon, num_vars]
//
// tensors is a map of GGUF tensor name -> tensor data for loading pre-trained weights.
func BuildTiRex[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TiRexConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], error) {
	if err := validateTiRexConfig(cfg); err != nil {
		return nil, fmt.Errorf("invalid TiRex config: %w", err)
	}

	ops := engine.Ops()
	node, err := newTiRexNode[T](tensors, cfg, engine, ops)
	if err != nil {
		return nil, fmt.Errorf("create TiRex node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, -1, cfg.InputDim})
	builder.AddNode(node, input)

	return builder.Build(node)
}

// tiRexBlock holds either an sLSTM or mLSTM cell for a single block.
type tiRexBlock[T tensor.Float] struct {
	blockType string
	slstm     *timeseries.SLSTM[T]
	mlstm     *timeseries.MLSTM[T]
}

// tiRexNode implements the full TiRex forward pass as a single graph node.
type tiRexNode[T tensor.Float] struct {
	cfg    *TiRexConfig
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]

	// Input projection: [input_dim, hidden_dim]
	inputProj *core.Linear[T]

	// Stack of xLSTM blocks.
	blocks []tiRexBlock[T]

	// Final layer norm on the hidden state.
	finalNorm *normalization.RMSNorm[T]

	// Output head: [hidden_dim, horizon * num_vars]
	outputHead *core.Linear[T]
}

func newTiRexNode[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TiRexConfig,
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
) (*tiRexNode[T], error) {
	// Input projection.
	inputProj, err := core.NewLinear[T]("tirex_input_proj", engine, ops, cfg.InputDim, cfg.HiddenDim)
	if err != nil {
		return nil, fmt.Errorf("create input projection: %w", err)
	}
	loadLinearWeights(tensors, inputProj, "tirex.input_proj.weight")

	// Build block type list.
	blockTypes := cfg.BlockTypes
	if blockTypes == nil {
		blockTypes = make([]string, cfg.NumLayers)
		for i := range cfg.NumLayers {
			if i%2 == 0 {
				blockTypes[i] = "slstm"
			} else {
				blockTypes[i] = "mlstm"
			}
		}
	}

	// Create xLSTM blocks. Each block takes hidden_dim input and produces hidden_dim output.
	blocks := make([]tiRexBlock[T], cfg.NumLayers)
	for i := range cfg.NumLayers {
		bt := blockTypes[i]
		switch bt {
		case "slstm":
			cell, sErr := timeseries.NewSLSTM[T](engine, cfg.HiddenDim, cfg.HiddenDim)
			if sErr != nil {
				return nil, fmt.Errorf("create sLSTM block %d: %w", i, sErr)
			}
			loadSLSTMWeights(tensors, cell, i)
			blocks[i] = tiRexBlock[T]{blockType: bt, slstm: cell}
		case "mlstm":
			cell, mErr := timeseries.NewMLSTM[T](engine, cfg.HiddenDim, cfg.HiddenDim)
			if mErr != nil {
				return nil, fmt.Errorf("create mLSTM block %d: %w", i, mErr)
			}
			loadMLSTMWeights(tensors, cell, i)
			blocks[i] = tiRexBlock[T]{blockType: bt, mlstm: cell}
		}
	}

	// Final layer norm.
	finalNorm, err := normalization.NewRMSNorm[T]("tirex_final_norm", engine, ops, cfg.HiddenDim)
	if err != nil {
		return nil, fmt.Errorf("create final norm: %w", err)
	}

	// Output head.
	outputHead, err := core.NewLinear[T]("tirex_output_head", engine, ops, cfg.HiddenDim, cfg.Horizon*cfg.NumVars)
	if err != nil {
		return nil, fmt.Errorf("create output head: %w", err)
	}
	loadLinearWeights(tensors, outputHead, "tirex.output_head.weight")

	return &tiRexNode[T]{
		cfg:        cfg,
		engine:     engine,
		ops:        ops,
		inputProj:  inputProj,
		blocks:     blocks,
		finalNorm:  finalNorm,
		outputHead: outputHead,
	}, nil
}

// loadLinearWeights loads a weight tensor into a Linear layer from the GGUF tensor map.
func loadLinearWeights[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	layer *core.Linear[T],
	name string,
) {
	if w, ok := tensors[name]; ok {
		params := layer.Parameters()
		if len(params) > 0 {
			params[0].Value = w
		}
	}
}

// loadSLSTMWeights loads sLSTM weights from GGUF tensors.
// GGUF naming: tirex.block.{i}.slstm.{param}
func loadSLSTMWeights[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cell *timeseries.SLSTM[T],
	layerIdx int,
) {
	prefix := fmt.Sprintf("tirex.block.%d.slstm.", layerIdx)
	paramMap := map[string]**graph.Parameter[T]{
		"Wi": &cell.Wi, "Wf": &cell.Wf, "Wz": &cell.Wz, "Wo": &cell.Wo,
		"Ri": &cell.Ri, "Rf": &cell.Rf, "Rz": &cell.Rz, "Ro": &cell.Ro,
		"bi": &cell.Bi, "bf": &cell.Bf, "bz": &cell.Bz, "bo": &cell.Bo,
	}
	for name, param := range paramMap {
		if w, ok := tensors[prefix+name]; ok {
			(*param).Value = w
		}
	}
}

// loadMLSTMWeights loads mLSTM weights from GGUF tensors.
// GGUF naming: tirex.block.{i}.mlstm.{param}
func loadMLSTMWeights[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cell *timeseries.MLSTM[T],
	layerIdx int,
) {
	prefix := fmt.Sprintf("tirex.block.%d.mlstm.", layerIdx)
	paramMap := map[string]**graph.Parameter[T]{
		"Wk": &cell.Wk, "Wv": &cell.Wv, "Wq": &cell.Wq,
		"Wi": &cell.Wi, "Wf": &cell.Wf, "Wo": &cell.Wo,
		"bi": &cell.Bi, "bf": &cell.Bf, "bo": &cell.Bo,
	}
	for name, param := range paramMap {
		if w, ok := tensors[prefix+name]; ok {
			(*param).Value = w
		}
	}
}

func (n *tiRexNode[T]) OpType() string { return "TiRex" }

func (n *tiRexNode[T]) Attributes() map[string]interface{} {
	blockTypes := make([]string, len(n.blocks))
	for i, b := range n.blocks {
		blockTypes[i] = b.blockType
	}
	return map[string]interface{}{
		"num_layers":  n.cfg.NumLayers,
		"input_dim":   n.cfg.InputDim,
		"hidden_dim":  n.cfg.HiddenDim,
		"horizon":     n.cfg.Horizon,
		"num_vars":    n.cfg.NumVars,
		"block_types": blockTypes,
	}
}

func (n *tiRexNode[T]) OutputShape() []int {
	return []int{-1, n.cfg.Horizon, n.cfg.NumVars}
}

func (n *tiRexNode[T]) Parameters() []*graph.Parameter[T] {
	var params []*graph.Parameter[T]
	params = append(params, n.inputProj.Parameters()...)
	for _, block := range n.blocks {
		switch block.blockType {
		case "slstm":
			params = append(params, block.slstm.Parameters()...)
		case "mlstm":
			params = append(params, block.mlstm.Parameters()...)
		}
	}
	params = append(params, n.finalNorm.Parameters()...)
	params = append(params, n.outputHead.Parameters()...)
	return params
}

// Forward processes [batch, seq_len, input_dim] input and produces [batch, horizon, num_vars].
//
// The forward pass:
//  1. Projects input to hidden_dim via a linear layer
//  2. Iterates over time steps, running each through the xLSTM block stack
//  3. Takes the final time step's hidden state
//  4. Applies layer norm and output head projection
//  5. Reshapes to [batch, horizon, num_vars]
func (n *tiRexNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("TiRex expects 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("TiRex input must be 3D [batch, seq_len, input_dim], got shape %v", shape)
	}

	batch, seqLen, inputDim := shape[0], shape[1], shape[2]
	if inputDim != n.cfg.InputDim {
		return nil, fmt.Errorf("TiRex input dim mismatch: got %d, want %d", inputDim, n.cfg.InputDim)
	}

	d := n.cfg.HiddenDim

	// Project input: [batch, seq_len, input_dim] -> [batch * seq_len, input_dim]
	// then linear -> [batch * seq_len, hidden_dim]
	flat, err := n.engine.Reshape(ctx, x, []int{batch * seqLen, inputDim})
	if err != nil {
		return nil, fmt.Errorf("reshape input: %w", err)
	}
	projected, err := n.inputProj.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("input projection: %w", err)
	}
	// Reshape back to [batch, seq_len, hidden_dim]
	projected, err = n.engine.Reshape(ctx, projected, []int{batch, seqLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape projected: %w", err)
	}

	// Initialize hidden states for each block.
	type slstmState struct {
		h, c, n, m *tensor.TensorNumeric[T]
	}
	type mlstmState struct {
		h    *tensor.TensorNumeric[T]
		cMat *tensor.TensorNumeric[T]
		n    *tensor.TensorNumeric[T]
		m    *tensor.TensorNumeric[T]
	}
	slstmStates := make([]slstmState, n.cfg.NumLayers)
	mlstmStates := make([]mlstmState, n.cfg.NumLayers)

	for i, block := range n.blocks {
		switch block.blockType {
		case "slstm":
			zeros := make([]T, batch*d)
			h, hErr := tensor.New[T]([]int{batch, d}, zeros)
			if hErr != nil {
				return nil, fmt.Errorf("init sLSTM h block %d: %w", i, hErr)
			}
			c, cErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if cErr != nil {
				return nil, fmt.Errorf("init sLSTM c block %d: %w", i, cErr)
			}
			nState, nErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if nErr != nil {
				return nil, fmt.Errorf("init sLSTM n block %d: %w", i, nErr)
			}
			slstmStates[i] = slstmState{h: h, c: c, n: nState}
		case "mlstm":
			h, hErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if hErr != nil {
				return nil, fmt.Errorf("init mLSTM h block %d: %w", i, hErr)
			}
			cMat, cErr := tensor.New[T]([]int{batch, d, d}, make([]T, batch*d*d))
			if cErr != nil {
				return nil, fmt.Errorf("init mLSTM C block %d: %w", i, cErr)
			}
			nState, nErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if nErr != nil {
				return nil, fmt.Errorf("init mLSTM n block %d: %w", i, nErr)
			}
			mlstmStates[i] = mlstmState{h: h, cMat: cMat, n: nState}
		}
	}

	// Split projected along the time axis: [batch, seq_len, hidden_dim] -> seq_len x [batch, 1, hidden_dim]
	timeSteps, err := n.engine.Split(ctx, projected, seqLen, 1)
	if err != nil {
		return nil, fmt.Errorf("split projected along time axis: %w", err)
	}

	// Process each time step through the block stack.
	var finalHidden *tensor.TensorNumeric[T]

	for t, ts := range timeSteps {
		// Reshape [batch, 1, hidden_dim] -> [batch, hidden_dim]
		stepInput, sErr := n.engine.Reshape(ctx, ts, []int{batch, d})
		if sErr != nil {
			return nil, fmt.Errorf("reshape time step %d: %w", t, sErr)
		}

		// Run through each block sequentially.
		blockInput := stepInput
		for i, block := range n.blocks {
			switch block.blockType {
			case "slstm":
				st := slstmStates[i]
				h, c, nState, mState, fErr := block.slstm.Forward(ctx, blockInput, st.h, st.c, st.n, st.m)
				if fErr != nil {
					return nil, fmt.Errorf("sLSTM block %d step %d: %w", i, t, fErr)
				}
				slstmStates[i] = slstmState{h: h, c: c, n: nState, m: mState}
				blockInput = h
			case "mlstm":
				st := mlstmStates[i]
				h, cMat, nState, mState, fErr := block.mlstm.Forward(ctx, blockInput, st.h, st.cMat, st.n, st.m)
				if fErr != nil {
					return nil, fmt.Errorf("mLSTM block %d step %d: %w", i, t, fErr)
				}
				mlstmStates[i] = mlstmState{h: h, cMat: cMat, n: nState, m: mState}
				blockInput = h
			}
		}

		finalHidden = blockInput
	}

	// Apply final layer norm: [batch, hidden_dim]
	// RMSNorm expects [batch, ..., dim], so reshape to [batch, 1, hidden_dim] then back.
	finalHidden, err = n.engine.Reshape(ctx, finalHidden, []int{batch, 1, d})
	if err != nil {
		return nil, fmt.Errorf("reshape for norm: %w", err)
	}
	finalHidden, err = n.finalNorm.Forward(ctx, finalHidden)
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}
	finalHidden, err = n.engine.Reshape(ctx, finalHidden, []int{batch, d})
	if err != nil {
		return nil, fmt.Errorf("reshape after norm: %w", err)
	}

	// Output head: [batch, hidden_dim] -> [batch, horizon * num_vars]
	output, err := n.outputHead.Forward(ctx, finalHidden)
	if err != nil {
		return nil, fmt.Errorf("output head: %w", err)
	}

	// Reshape to [batch, horizon, num_vars].
	output, err = n.engine.Reshape(ctx, output, []int{batch, n.cfg.Horizon, n.cfg.NumVars})
	if err != nil {
		return nil, fmt.Errorf("reshape output: %w", err)
	}

	return output, nil
}

func (n *tiRexNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// TiRexFeatureExtractor provides access to backbone features and output head
// parameters for fine-tuning. It runs the TiRex forward pass up to (but not
// including) the output head, returning the hidden representation.
type TiRexFeatureExtractor[T tensor.Float] struct {
	node *tiRexNode[T]
}

// ForwardFeatures runs the TiRex backbone (input projection, xLSTM blocks,
// layer norm) and returns the hidden representation [batch, hidden_dim].
func (e *TiRexFeatureExtractor[T]) ForwardFeatures(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.node.forwardFeatures(ctx, input)
}

// OutputHeadForward applies the output head linear layer to hidden features,
// returning [batch, horizon*num_vars].
func (e *TiRexFeatureExtractor[T]) OutputHeadForward(ctx context.Context, hidden *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	return e.node.outputHead.Forward(ctx, hidden)
}

// OutputHeadParams returns the trainable parameters of the output head.
func (e *TiRexFeatureExtractor[T]) OutputHeadParams() []*graph.Parameter[T] {
	return e.node.outputHead.Parameters()
}

// HiddenDim returns the hidden dimension of the backbone.
func (e *TiRexFeatureExtractor[T]) HiddenDim() int {
	return e.node.cfg.HiddenDim
}

// BuildTiRexWithExtractor constructs a TiRex computation graph and returns
// both the graph and a feature extractor for fine-tuning.
func BuildTiRexWithExtractor[T tensor.Float](
	tensors map[string]*tensor.TensorNumeric[T],
	cfg *TiRexConfig,
	engine compute.Engine[T],
) (*graph.Graph[T], *TiRexFeatureExtractor[T], error) {
	if err := validateTiRexConfig(cfg); err != nil {
		return nil, nil, fmt.Errorf("invalid TiRex config: %w", err)
	}

	ops := engine.Ops()
	node, err := newTiRexNode[T](tensors, cfg, engine, ops)
	if err != nil {
		return nil, nil, fmt.Errorf("create TiRex node: %w", err)
	}

	builder := graph.NewBuilder[T](engine)
	input := builder.Input([]int{-1, -1, cfg.InputDim})
	builder.AddNode(node, input)

	g, err := builder.Build(node)
	if err != nil {
		return nil, nil, err
	}

	return g, &TiRexFeatureExtractor[T]{node: node}, nil
}

// forwardFeatures runs the backbone up to the output head.
func (n *tiRexNode[T]) forwardFeatures(ctx context.Context, input *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("TiRex input must be 3D [batch, seq_len, input_dim], got shape %v", shape)
	}

	batch, seqLen, inputDim := shape[0], shape[1], shape[2]
	if inputDim != n.cfg.InputDim {
		return nil, fmt.Errorf("TiRex input dim mismatch: got %d, want %d", inputDim, n.cfg.InputDim)
	}

	d := n.cfg.HiddenDim

	flat, err := n.engine.Reshape(ctx, input, []int{batch * seqLen, inputDim})
	if err != nil {
		return nil, fmt.Errorf("reshape input: %w", err)
	}
	projected, err := n.inputProj.Forward(ctx, flat)
	if err != nil {
		return nil, fmt.Errorf("input projection: %w", err)
	}
	projected, err = n.engine.Reshape(ctx, projected, []int{batch, seqLen, d})
	if err != nil {
		return nil, fmt.Errorf("reshape projected: %w", err)
	}

	type slstmState struct {
		h, c, n, m *tensor.TensorNumeric[T]
	}
	type mlstmState struct {
		h    *tensor.TensorNumeric[T]
		cMat *tensor.TensorNumeric[T]
		n    *tensor.TensorNumeric[T]
		m    *tensor.TensorNumeric[T]
	}
	slstmStates := make([]slstmState, n.cfg.NumLayers)
	mlstmStates := make([]mlstmState, n.cfg.NumLayers)

	for i, block := range n.blocks {
		switch block.blockType {
		case "slstm":
			zeros := make([]T, batch*d)
			h, hErr := tensor.New[T]([]int{batch, d}, zeros)
			if hErr != nil {
				return nil, fmt.Errorf("init sLSTM h block %d: %w", i, hErr)
			}
			c, cErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if cErr != nil {
				return nil, fmt.Errorf("init sLSTM c block %d: %w", i, cErr)
			}
			nState, nErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if nErr != nil {
				return nil, fmt.Errorf("init sLSTM n block %d: %w", i, nErr)
			}
			slstmStates[i] = slstmState{h: h, c: c, n: nState}
		case "mlstm":
			h, hErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if hErr != nil {
				return nil, fmt.Errorf("init mLSTM h block %d: %w", i, hErr)
			}
			cMat, cErr := tensor.New[T]([]int{batch, d, d}, make([]T, batch*d*d))
			if cErr != nil {
				return nil, fmt.Errorf("init mLSTM C block %d: %w", i, cErr)
			}
			nState, nErr := tensor.New[T]([]int{batch, d}, make([]T, batch*d))
			if nErr != nil {
				return nil, fmt.Errorf("init mLSTM n block %d: %w", i, nErr)
			}
			mlstmStates[i] = mlstmState{h: h, cMat: cMat, n: nState}
		}
	}

	// Split projected along the time axis: [batch, seq_len, hidden_dim] -> seq_len x [batch, 1, hidden_dim]
	timeSteps, err := n.engine.Split(ctx, projected, seqLen, 1)
	if err != nil {
		return nil, fmt.Errorf("split projected along time axis: %w", err)
	}

	var finalHidden *tensor.TensorNumeric[T]

	for t, ts := range timeSteps {
		// Reshape [batch, 1, hidden_dim] -> [batch, hidden_dim]
		stepInput, sErr := n.engine.Reshape(ctx, ts, []int{batch, d})
		if sErr != nil {
			return nil, fmt.Errorf("reshape time step %d: %w", t, sErr)
		}

		blockInput := stepInput
		for i, block := range n.blocks {
			switch block.blockType {
			case "slstm":
				st := slstmStates[i]
				h, c, nState, mState, fErr := block.slstm.Forward(ctx, blockInput, st.h, st.c, st.n, st.m)
				if fErr != nil {
					return nil, fmt.Errorf("sLSTM block %d step %d: %w", i, t, fErr)
				}
				slstmStates[i] = slstmState{h: h, c: c, n: nState, m: mState}
				blockInput = h
			case "mlstm":
				st := mlstmStates[i]
				h, cMat, nState, mState, fErr := block.mlstm.Forward(ctx, blockInput, st.h, st.cMat, st.n, st.m)
				if fErr != nil {
					return nil, fmt.Errorf("mLSTM block %d step %d: %w", i, t, fErr)
				}
				mlstmStates[i] = mlstmState{h: h, cMat: cMat, n: nState, m: mState}
				blockInput = h
			}
		}

		finalHidden = blockInput
	}

	finalHidden, err = n.engine.Reshape(ctx, finalHidden, []int{batch, 1, d})
	if err != nil {
		return nil, fmt.Errorf("reshape for norm: %w", err)
	}
	finalHidden, err = n.finalNorm.Forward(ctx, finalHidden)
	if err != nil {
		return nil, fmt.Errorf("final norm: %w", err)
	}
	finalHidden, err = n.engine.Reshape(ctx, finalHidden, []int{batch, d})
	if err != nil {
		return nil, fmt.Errorf("reshape after norm: %w", err)
	}

	return finalHidden, nil
}
