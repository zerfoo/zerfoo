package timeseries

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

// MambaConfig holds the configuration for a Mamba model.
type MambaConfig struct {
	Channels     int // number of input channels
	InputLen     int // lookback window
	OutputLen    int // forecast horizon
	DModel       int // model dimension
	DState       int // SSM state dimension (typically 16)
	DConv        int // convolution kernel size (typically 4)
	ExpandFactor int // expansion factor (typically 2)
	NLayers      int // number of Mamba blocks
}

// mambaBlock holds weights for a single Mamba block.
type mambaBlock struct {
	// Input projection: dModel -> 2*dInner (split into x branch and gate branch).
	inProjW []float64 // [2*dInner * dModel]
	inProjB []float64 // [2*dInner]

	// Causal conv1d on x branch.
	convW []float64 // [dInner * dConv]
	convB []float64 // [dInner]

	// SSM parameters.
	dtProjW []float64 // [dInner * dInner] - delta projection
	dtProjB []float64 // [dInner]
	aLog    []float64 // [dInner * dState] - log of diagonal A matrix
	bProjW  []float64 // [dState * dInner] - input-dependent B projection
	bProjB  []float64 // [dState]
	cProjW  []float64 // [dState * dInner] - input-dependent C projection
	cProjB  []float64 // [dState]
	d       []float64 // [dInner] - skip connection

	// Output projection: dInner -> dModel.
	outProjW []float64 // [dModel * dInner]
	outProjB []float64 // [dModel]

	// Layer norm.
	lnGamma []float64 // [dModel]
	lnBeta  []float64 // [dModel]
}

// Mamba implements the Mamba selective state space model (NeurIPS 2023)
// for time-series forecasting.
type Mamba struct {
	config MambaConfig
	dInner int // expandFactor * dModel

	// Input embedding: channels -> dModel (applied per timestep).
	embedW []float64 // [dModel * channels]
	embedB []float64 // [dModel]

	blocks []mambaBlock

	// Output projection: dModel -> channels*outputLen.
	headW []float64 // [channels * outputLen * dModel]
	headB []float64 // [channels * outputLen]

	normMeans [][]float64
	normStds  [][]float64
}

// NewMamba creates a new Mamba model with the given configuration.
func NewMamba(config MambaConfig) (*Mamba, error) {
	if config.Channels <= 0 {
		return nil, fmt.Errorf("mamba: Channels must be positive, got %d", config.Channels)
	}
	if config.InputLen <= 0 {
		return nil, fmt.Errorf("mamba: InputLen must be positive, got %d", config.InputLen)
	}
	if config.OutputLen <= 0 {
		return nil, fmt.Errorf("mamba: OutputLen must be positive, got %d", config.OutputLen)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("mamba: DModel must be positive, got %d", config.DModel)
	}
	if config.DState <= 0 {
		return nil, fmt.Errorf("mamba: DState must be positive, got %d", config.DState)
	}
	if config.DConv <= 0 {
		return nil, fmt.Errorf("mamba: DConv must be positive, got %d", config.DConv)
	}
	if config.ExpandFactor <= 0 {
		return nil, fmt.Errorf("mamba: ExpandFactor must be positive, got %d", config.ExpandFactor)
	}
	if config.NLayers <= 0 {
		return nil, fmt.Errorf("mamba: NLayers must be positive, got %d", config.NLayers)
	}

	dInner := config.ExpandFactor * config.DModel

	m := &Mamba{
		config: config,
		dInner: dInner,
	}

	// Input embedding (per-timestep: channels -> dModel).
	embedScale := math.Sqrt(2.0 / float64(config.Channels+config.DModel))
	m.embedW = randNormSlice(config.DModel*config.Channels, embedScale)
	m.embedB = make([]float64, config.DModel)

	// Mamba blocks.
	m.blocks = make([]mambaBlock, config.NLayers)
	for l := 0; l < config.NLayers; l++ {
		b := &m.blocks[l]

		inScale := math.Sqrt(2.0 / float64(config.DModel+2*dInner))
		b.inProjW = randNormSlice(2*dInner*config.DModel, inScale)
		b.inProjB = make([]float64, 2*dInner)

		convScale := math.Sqrt(2.0 / float64(config.DConv+1))
		b.convW = randNormSlice(dInner*config.DConv, convScale)
		b.convB = make([]float64, dInner)

		dtScale := math.Sqrt(2.0 / float64(2*dInner))
		b.dtProjW = randNormSlice(dInner*dInner, dtScale)
		b.dtProjB = make([]float64, dInner)
		// Initialize dt bias to small positive values for stable softplus.
		for i := range b.dtProjB {
			b.dtProjB[i] = rand.Float64()*0.1 + 0.1
		}

		// A is parameterized as log values, initialized to small negative values.
		b.aLog = make([]float64, dInner*config.DState)
		for i := range b.aLog {
			b.aLog[i] = math.Log(rand.Float64()*0.5 + 0.5) // log of values in [0.5, 1.0]
		}

		bScale := math.Sqrt(2.0 / float64(dInner+config.DState))
		b.bProjW = randNormSlice(config.DState*dInner, bScale)
		b.bProjB = make([]float64, config.DState)
		b.cProjW = randNormSlice(config.DState*dInner, bScale)
		b.cProjB = make([]float64, config.DState)

		b.d = make([]float64, dInner)
		for i := range b.d {
			b.d[i] = 1.0 // Initialize skip connection to 1
		}

		outScale := math.Sqrt(2.0 / float64(dInner+config.DModel))
		b.outProjW = randNormSlice(config.DModel*dInner, outScale)
		b.outProjB = make([]float64, config.DModel)

		b.lnGamma = make([]float64, config.DModel)
		for i := range b.lnGamma {
			b.lnGamma[i] = 1.0
		}
		b.lnBeta = make([]float64, config.DModel)
	}

	// Output head.
	outDim := config.Channels * config.OutputLen
	headScale := math.Sqrt(2.0 / float64(config.DModel+outDim))
	m.headW = randNormSlice(outDim*config.DModel, headScale)
	m.headB = make([]float64, outDim)

	return m, nil
}

// softplus computes log(1 + exp(x)) with numerical stability.
func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return 0
	}
	return math.Log(1 + math.Exp(x))
}

// silu computes x * sigmoid(x).
func silu(x float64) float64 {
	return x / (1 + math.Exp(-x))
}

// layerNorm applies layer normalization to a vector.
func layerNorm(x, gamma, beta []float64) []float64 {
	n := len(x)
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(n)

	variance := 0.0
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float64(n)

	out := make([]float64, n)
	invStd := 1.0 / math.Sqrt(variance+1e-5)
	for i := range x {
		out[i] = gamma[i]*(x[i]-mean)*invStd + beta[i]
	}
	return out
}

// causalConv1d applies a causal 1D convolution along the time dimension.
// input: [seqLen][dInner], convW: [dInner][dConv], convB: [dInner].
// Returns [seqLen][dInner].
func causalConv1d(input [][]float64, convW []float64, convB []float64, dInner, dConv int) [][]float64 {
	seqLen := len(input)
	out := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		out[t] = make([]float64, dInner)
		for d := 0; d < dInner; d++ {
			val := convB[d]
			for k := 0; k < dConv; k++ {
				srcT := t - k
				if srcT < 0 {
					continue // zero padding for causal conv
				}
				val += convW[d*dConv+k] * input[srcT][d]
			}
			out[t][d] = val
		}
	}
	return out
}

// selectiveScan runs the selective scan (core SSM operation).
// x: [seqLen][dInner] - input after conv1d + SiLU.
// Returns [seqLen][dInner].
func (b *mambaBlock) selectiveScan(x [][]float64, dInner, dState int) [][]float64 {
	seqLen := len(x)
	out := make([][]float64, seqLen)

	// State: h[d][s] for each inner dimension and state dimension.
	h := make([][]float64, dInner)
	for d := 0; d < dInner; d++ {
		h[d] = make([]float64, dState)
	}

	for t := 0; t < seqLen; t++ {
		out[t] = make([]float64, dInner)

		// Compute delta = softplus(dtProj(x[t])).
		delta := make([]float64, dInner)
		for d := 0; d < dInner; d++ {
			val := b.dtProjB[d]
			for j := 0; j < dInner; j++ {
				val += b.dtProjW[d*dInner+j] * x[t][j]
			}
			delta[d] = softplus(val)
		}

		// Compute input-dependent B = bProj(x[t]).
		bVec := make([]float64, dState)
		for s := 0; s < dState; s++ {
			val := b.bProjB[s]
			for j := 0; j < dInner; j++ {
				val += b.bProjW[s*dInner+j] * x[t][j]
			}
			bVec[s] = val
		}

		// Compute input-dependent C = cProj(x[t]).
		cVec := make([]float64, dState)
		for s := 0; s < dState; s++ {
			val := b.cProjB[s]
			for j := 0; j < dInner; j++ {
				val += b.cProjW[s*dInner+j] * x[t][j]
			}
			cVec[s] = val
		}

		// For each inner dimension:
		for d := 0; d < dInner; d++ {
			// Discretize: Abar = exp(A * delta), Bbar = delta * B.
			y := 0.0
			for s := 0; s < dState; s++ {
				a := -math.Exp(b.aLog[d*dState+s]) // A is negative for stability
				aBar := math.Exp(a * delta[d])
				bBar := delta[d] * bVec[s]

				// State update: h[t] = Abar * h[t-1] + Bbar * x[t].
				h[d][s] = aBar*h[d][s] + bBar*x[t][d]

				// Output: y = C * h + D * x.
				y += cVec[s] * h[d][s]
			}
			out[t][d] = y + b.d[d]*x[t][d]
		}
	}

	return out
}

// forwardBlock runs a single Mamba block.
// input: [seqLen][dModel], returns [seqLen][dModel].
func (m *Mamba) forwardBlock(input [][]float64, blk *mambaBlock) [][]float64 {
	seqLen := len(input)
	dModel := m.config.DModel
	dInner := m.dInner

	// Layer norm.
	normed := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		normed[t] = layerNorm(input[t], blk.lnGamma, blk.lnBeta)
	}

	// Input projection: dModel -> 2*dInner, split into x and gate.
	xBranch := make([][]float64, seqLen)
	gate := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		proj := make([]float64, 2*dInner)
		for i := 0; i < 2*dInner; i++ {
			val := blk.inProjB[i]
			for j := 0; j < dModel; j++ {
				val += blk.inProjW[i*dModel+j] * normed[t][j]
			}
			proj[i] = val
		}
		xBranch[t] = make([]float64, dInner)
		copy(xBranch[t], proj[:dInner])
		gate[t] = make([]float64, dInner)
		copy(gate[t], proj[dInner:])
	}

	// Causal conv1d on x branch.
	xConv := causalConv1d(xBranch, blk.convW, blk.convB, dInner, m.config.DConv)

	// SiLU activation on convolved x.
	for t := 0; t < seqLen; t++ {
		for d := 0; d < dInner; d++ {
			xConv[t][d] = silu(xConv[t][d])
		}
	}

	// Selective scan (SSM).
	ssmOut := blk.selectiveScan(xConv, dInner, m.config.DState)

	// Output gate: elementwise multiply with SiLU-activated gate.
	gated := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		gated[t] = make([]float64, dInner)
		for d := 0; d < dInner; d++ {
			gated[t][d] = ssmOut[t][d] * silu(gate[t][d])
		}
	}

	// Output projection: dInner -> dModel.
	output := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		output[t] = make([]float64, dModel)
		for i := 0; i < dModel; i++ {
			val := blk.outProjB[i]
			for j := 0; j < dInner; j++ {
				val += blk.outProjW[i*dInner+j] * gated[t][j]
			}
			// Residual connection.
			output[t][i] = input[t][i] + val
		}
	}

	return output
}

// forward runs the full Mamba forward pass on a single sample.
// Input: [channels][inputLen], returns: [channels][outputLen].
func (m *Mamba) forward(input [][]float64) [][]float64 {
	dModel := m.config.DModel
	channels := m.config.Channels

	// Embed per timestep: input[c][t] -> seq[t][dModel].
	seqLen := m.config.InputLen
	seq := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		seq[t] = make([]float64, dModel)
		for i := 0; i < dModel; i++ {
			val := m.embedB[i]
			for c := 0; c < channels; c++ {
				val += m.embedW[i*channels+c] * input[c][t]
			}
			seq[t][i] = val
		}
	}

	// Process through Mamba blocks.
	for l := 0; l < m.config.NLayers; l++ {
		seq = m.forwardBlock(seq, &m.blocks[l])
	}

	// Use the last time step's output for prediction.
	lastHidden := seq[seqLen-1]

	// Output projection: dModel -> channels*outputLen.
	outDim := m.config.Channels * m.config.OutputLen
	outFlat := make([]float64, outDim)
	for i := 0; i < outDim; i++ {
		val := m.headB[i]
		for j := 0; j < dModel; j++ {
			val += m.headW[i*dModel+j] * lastHidden[j]
		}
		outFlat[i] = val
	}

	// Reshape to [channels][outputLen].
	output := make([][]float64, m.config.Channels)
	for c := 0; c < m.config.Channels; c++ {
		output[c] = outFlat[c*m.config.OutputLen : (c+1)*m.config.OutputLen]
	}
	return output
}

// paramCount returns the total number of trainable parameters.
func (m *Mamba) paramCount() int {
	dModel := m.config.DModel
	dInner := m.dInner
	dState := m.config.DState
	dConv := m.config.DConv
	channels := m.config.Channels
	outDim := channels * m.config.OutputLen

	// Embedding (per-timestep: channels -> dModel).
	n := dModel*channels + dModel

	// Per block.
	perBlock := 0
	perBlock += 2*dInner*dModel + 2*dInner         // inProj
	perBlock += dInner*dConv + dInner               // conv1d
	perBlock += dInner*dInner + dInner              // dtProj
	perBlock += dInner * dState                     // aLog
	perBlock += dState*dInner + dState              // bProj
	perBlock += dState*dInner + dState              // cProj
	perBlock += dInner                              // d
	perBlock += dModel*dInner + dModel              // outProj
	perBlock += dModel + dModel                     // layerNorm
	n += perBlock * m.config.NLayers

	// Output head.
	n += outDim*dModel + outDim

	return n
}

// flatParams returns pointers to all trainable parameters in a flat slice.
func (m *Mamba) flatParams() []*float64 {
	n := m.paramCount()
	params := make([]*float64, 0, n)

	appendSlice := func(s []float64) {
		for i := range s {
			params = append(params, &s[i])
		}
	}

	appendSlice(m.embedW)
	appendSlice(m.embedB)

	for l := range m.blocks {
		b := &m.blocks[l]
		appendSlice(b.inProjW)
		appendSlice(b.inProjB)
		appendSlice(b.convW)
		appendSlice(b.convB)
		appendSlice(b.dtProjW)
		appendSlice(b.dtProjB)
		appendSlice(b.aLog)
		appendSlice(b.bProjW)
		appendSlice(b.bProjB)
		appendSlice(b.cProjW)
		appendSlice(b.cProjB)
		appendSlice(b.d)
		appendSlice(b.outProjW)
		appendSlice(b.outProjB)
		appendSlice(b.lnGamma)
		appendSlice(b.lnBeta)
	}

	appendSlice(m.headW)
	appendSlice(m.headB)

	return params
}

// TrainWindowed trains the Mamba model on windowed data using AdamW with
// numerical gradient computation via finite differences.
func (m *Mamba) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("mamba: empty training set")
	}

	expectedLabels := nSamples * m.config.Channels * m.config.OutputLen
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("mamba: expected %d labels, got %d", expectedLabels, len(labels))
	}

	for i, w := range windows {
		if len(w) != m.config.Channels {
			return nil, fmt.Errorf("mamba: window %d has %d channels, expected %d", i, len(w), m.config.Channels)
		}
		for c, ch := range w {
			if len(ch) != m.config.InputLen {
				return nil, fmt.Errorf("mamba: window %d channel %d has length %d, expected %d", i, c, len(ch), m.config.InputLen)
			}
		}
	}

	if config.Epochs <= 0 {
		config.Epochs = 100
	}
	if config.LR <= 0 {
		config.LR = 1e-3
	}
	if config.Beta1 <= 0 {
		config.Beta1 = 0.9
	}
	if config.Beta2 <= 0 {
		config.Beta2 = 0.999
	}
	if config.Epsilon <= 0 {
		config.Epsilon = 1e-8
	}

	// Z-score normalize inputs.
	windows, m.normMeans, m.normStds = normalizeWindows(windows)

	nParams := m.paramCount()
	adam_m := make([]float64, nParams)
	adam_v := make([]float64, nParams)

	result := &TrainResult{
		LossHistory: make([]float64, config.Epochs),
	}

	batchSize := nSamples
	if config.BatchSize > 0 && config.BatchSize < nSamples {
		batchSize = config.BatchSize
	}

	// Compute loss for a batch.
	computeLoss := func(batchWindows [][][]float64, batchLabels []float64) float64 {
		bs := len(batchWindows)
		totalLoss := 0.0
		for s := 0; s < bs; s++ {
			pred := m.forward(batchWindows[s])
			for c := 0; c < m.config.Channels; c++ {
				for o := 0; o < m.config.OutputLen; o++ {
					labelIdx := s*m.config.Channels*m.config.OutputLen + c*m.config.OutputLen + o
					diff := pred[c][o] - batchLabels[labelIdx]
					totalLoss += diff * diff
				}
			}
		}
		return totalLoss / float64(bs*m.config.Channels*m.config.OutputLen)
	}

	const eps = 1e-5

	for epoch := 0; epoch < config.Epochs; epoch++ {
		epochLoss := 0.0
		nBatches := 0

		for start := 0; start < nSamples; start += batchSize {
			end := start + batchSize
			if end > nSamples {
				end = nSamples
			}
			batchWindows := windows[start:end]
			batchLabels := labels[start*m.config.Channels*m.config.OutputLen : end*m.config.Channels*m.config.OutputLen]

			batchLoss := computeLoss(batchWindows, batchLabels)
			epochLoss += batchLoss
			nBatches++

			// Numerical gradients via finite differences.
			params := m.flatParams()
			grads := make([]float64, nParams)
			for i, p := range params {
				orig := *p
				*p = orig + eps
				lossPlus := computeLoss(batchWindows, batchLabels)
				*p = orig - eps
				lossMinus := computeLoss(batchWindows, batchLabels)
				*p = orig
				grads[i] = (lossPlus - lossMinus) / (2 * eps)
				if !isFinite(grads[i]) {
					grads[i] = 0
				}
			}

			// Gradient clipping.
			if config.GradClip > 0 {
				norm := 0.0
				for _, g := range grads {
					norm += g * g
				}
				norm = math.Sqrt(norm)
				if norm > config.GradClip {
					scale := config.GradClip / norm
					for i := range grads {
						grads[i] *= scale
					}
				}
			}

			// AdamW update with LR warmup.
			lr := warmupLR(config.LR, epoch, config.WarmupEpochs)
			t := float64(epoch*((nSamples+batchSize-1)/batchSize) + nBatches)
			for i, p := range params {
				adam_m[i] = config.Beta1*adam_m[i] + (1-config.Beta1)*grads[i]
				adam_v[i] = config.Beta2*adam_v[i] + (1-config.Beta2)*grads[i]*grads[i]
				mHat := adam_m[i] / (1 - math.Pow(config.Beta1, t))
				vHat := adam_v[i] / (1 - math.Pow(config.Beta2, t))
				*p = *p - lr*(mHat/(math.Sqrt(vHat)+config.Epsilon)+config.WeightDecay*(*p))
			}
		}

		result.LossHistory[epoch] = epochLoss / float64(nBatches)
		result.FinalLoss = result.LossHistory[epoch]

		if !isFinite(result.FinalLoss) {
			return nil, fmt.Errorf("mamba: training diverged at epoch %d: loss=%v", epoch, result.FinalLoss)
		}
	}

	result.Metrics = map[string]float64{"mse": result.FinalLoss}
	return result, nil
}

// PredictWindowed runs inference on windowed data.
// windows: [nSamples][channels][inputLen].
// Returns flat predictions of length nSamples * channels * outputLen.
func (m *Mamba) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	if modelPath != "" {
		if err := m.loadWeights(modelPath); err != nil {
			return nil, fmt.Errorf("mamba: load weights: %w", err)
		}
	}

	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("mamba: empty input")
	}

	if m.normMeans != nil {
		windows = applyNormalization(windows, m.normMeans, m.normStds)
	}

	out := make([]float64, 0, nSamples*m.config.Channels*m.config.OutputLen)
	for _, w := range windows {
		if len(w) != m.config.Channels {
			return nil, fmt.Errorf("mamba: expected %d channels, got %d", m.config.Channels, len(w))
		}
		pred := m.forward(w)
		for c := 0; c < m.config.Channels; c++ {
			out = append(out, pred[c]...)
		}
	}
	return out, nil
}

// mambaBlockWeights is the JSON-serializable form of a Mamba block.
type mambaBlockWeights struct {
	InProjW  []float64 `json:"in_proj_w"`
	InProjB  []float64 `json:"in_proj_b"`
	ConvW    []float64 `json:"conv_w"`
	ConvB    []float64 `json:"conv_b"`
	DtProjW  []float64 `json:"dt_proj_w"`
	DtProjB  []float64 `json:"dt_proj_b"`
	ALog     []float64 `json:"a_log"`
	BProjW   []float64 `json:"b_proj_w"`
	BProjB   []float64 `json:"b_proj_b"`
	CProjW   []float64 `json:"c_proj_w"`
	CProjB   []float64 `json:"c_proj_b"`
	D        []float64 `json:"d"`
	OutProjW []float64 `json:"out_proj_w"`
	OutProjB []float64 `json:"out_proj_b"`
	LnGamma  []float64 `json:"ln_gamma"`
	LnBeta   []float64 `json:"ln_beta"`
}

// mambaWeights is the JSON-serializable form of Mamba parameters.
type mambaWeights struct {
	Config    MambaConfig         `json:"config"`
	EmbedW    []float64           `json:"embed_w"`
	EmbedB    []float64           `json:"embed_b"`
	Blocks    []mambaBlockWeights `json:"blocks"`
	HeadW     []float64           `json:"head_w"`
	HeadB     []float64           `json:"head_b"`
	NormMeans [][]float64         `json:"norm_means,omitempty"`
	NormStds  [][]float64         `json:"norm_stds,omitempty"`
}

// SaveWeights writes the model weights to a JSON file.
func (m *Mamba) SaveWeights(path string) error {
	w := mambaWeights{
		Config:    m.config,
		EmbedW:    m.embedW,
		EmbedB:    m.embedB,
		HeadW:     m.headW,
		HeadB:     m.headB,
		NormMeans: m.normMeans,
		NormStds:  m.normStds,
	}
	w.Blocks = make([]mambaBlockWeights, len(m.blocks))
	for i, b := range m.blocks {
		w.Blocks[i] = mambaBlockWeights{
			InProjW:  b.inProjW,
			InProjB:  b.inProjB,
			ConvW:    b.convW,
			ConvB:    b.convB,
			DtProjW:  b.dtProjW,
			DtProjB:  b.dtProjB,
			ALog:     b.aLog,
			BProjW:   b.bProjW,
			BProjB:   b.bProjB,
			CProjW:   b.cProjW,
			CProjB:   b.cProjB,
			D:        b.d,
			OutProjW: b.outProjW,
			OutProjB: b.outProjB,
			LnGamma:  b.lnGamma,
			LnBeta:   b.lnBeta,
		}
	}
	data, err := json.Marshal(w)
	if err != nil {
		return fmt.Errorf("mamba: marshal weights: %w", err)
	}
	return os.WriteFile(path, data, 0o644)
}

// loadWeights reads model weights from a JSON file.
func (m *Mamba) loadWeights(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var w mambaWeights
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	if w.Config != m.config {
		return fmt.Errorf("mamba: config mismatch: file has %+v, model has %+v", w.Config, m.config)
	}
	m.embedW = w.EmbedW
	m.embedB = w.EmbedB
	m.headW = w.HeadW
	m.headB = w.HeadB
	m.normMeans = w.NormMeans
	m.normStds = w.NormStds
	for i, bw := range w.Blocks {
		m.blocks[i].inProjW = bw.InProjW
		m.blocks[i].inProjB = bw.InProjB
		m.blocks[i].convW = bw.ConvW
		m.blocks[i].convB = bw.ConvB
		m.blocks[i].dtProjW = bw.DtProjW
		m.blocks[i].dtProjB = bw.DtProjB
		m.blocks[i].aLog = bw.ALog
		m.blocks[i].bProjW = bw.BProjW
		m.blocks[i].bProjB = bw.BProjB
		m.blocks[i].cProjW = bw.CProjW
		m.blocks[i].cProjB = bw.CProjB
		m.blocks[i].d = bw.D
		m.blocks[i].outProjW = bw.OutProjW
		m.blocks[i].outProjB = bw.OutProjB
		m.blocks[i].lnGamma = bw.LnGamma
		m.blocks[i].lnBeta = bw.LnBeta
	}
	return nil
}
