package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/model/gguf"
)

func init() {
	RegisterArchitecture("rwkv", buildRWKVGraph)
}

// RWKVConfig holds RWKV-specific model configuration.
type RWKVConfig struct {
	NumLayers  int
	HiddenSize int
	VocabSize  int
	HeadSize   int // WKV head size (default 64)
	NumHeads   int // HiddenSize / HeadSize
	LayerNormEps float32
}

// RWKVConfigFromGGUF extracts RWKV configuration from GGUF ModelConfig.
func RWKVConfigFromGGUF(cfg *gguf.ModelConfig) RWKVConfig {
	headSize := 64
	numHeads := cfg.HiddenSize / headSize
	if numHeads == 0 {
		numHeads = 1
	}
	eps := cfg.RMSNormEps
	if eps == 0 {
		eps = 1e-5
	}
	return RWKVConfig{
		NumLayers:    cfg.NumLayers,
		HiddenSize:   cfg.HiddenSize,
		VocabSize:    cfg.VocabSize,
		HeadSize:     headSize,
		NumHeads:     numHeads,
		LayerNormEps: eps,
	}
}

// buildRWKVGraph constructs a computation graph for the RWKV architecture.
func buildRWKVGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	rc := RWKVConfigFromGGUF(cfg)
	return BuildRWKV(rc, tensors, engine)
}

// BuildRWKV constructs a computation graph for the RWKV-6/7 architecture.
//
// Expected tensor names (GGUF RWKV convention):
//
//	token_embd.weight            — [vocab_size, hidden_size]
//	output.weight                — [vocab_size, hidden_size]
//	output_norm.weight           — [hidden_size]
//	output_norm.bias             — [hidden_size]
//	blocks.{i}.ln0.weight        — [hidden_size] (layer 0 only, pre-embedding norm)
//	blocks.{i}.ln0.bias          — [hidden_size] (layer 0 only)
//	blocks.{i}.ln1.weight        — [hidden_size] (time mixing norm)
//	blocks.{i}.ln1.bias          — [hidden_size]
//	blocks.{i}.ln2.weight        — [hidden_size] (channel mixing norm)
//	blocks.{i}.ln2.bias          — [hidden_size]
//	blocks.{i}.att.time_mix_r    — [1, 1, hidden_size]
//	blocks.{i}.att.time_mix_k    — [1, 1, hidden_size]
//	blocks.{i}.att.time_mix_v    — [1, 1, hidden_size]
//	blocks.{i}.att.time_mix_g    — [1, 1, hidden_size]
//	blocks.{i}.att.time_decay    — [num_heads, head_size]
//	blocks.{i}.att.time_faaaa    — [num_heads, head_size] (initial state)
//	blocks.{i}.att.receptance.weight — [hidden_size, hidden_size]
//	blocks.{i}.att.key.weight        — [hidden_size, hidden_size]
//	blocks.{i}.att.value.weight      — [hidden_size, hidden_size]
//	blocks.{i}.att.gate.weight       — [hidden_size, hidden_size]
//	blocks.{i}.att.output.weight     — [hidden_size, hidden_size]
//	blocks.{i}.att.ln_x.weight       — [hidden_size] (group norm)
//	blocks.{i}.att.ln_x.bias         — [hidden_size]
//	blocks.{i}.ffn.time_mix_k        — [1, 1, hidden_size]
//	blocks.{i}.ffn.time_mix_r        — [1, 1, hidden_size]
//	blocks.{i}.ffn.key.weight        — [ffn_size, hidden_size]
//	blocks.{i}.ffn.value.weight      — [hidden_size, ffn_size]
//	blocks.{i}.ffn.receptance.weight — [hidden_size, hidden_size]
func BuildRWKV(
	rc RWKVConfig,
	tensors map[string]*tensor.TensorNumeric[float32],
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	lookup := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	embedWeight, err := lookup("token_embd.weight")
	if err != nil {
		return nil, nil, err
	}

	lmHeadWeight, ok := tensors["output.weight"]
	if !ok {
		lmHeadWeight = embedWeight
	}

	outputNormWeight, err := lookup("output_norm.weight")
	if err != nil {
		return nil, nil, err
	}
	outputNormBias, ok := tensors["output_norm.bias"]
	if !ok {
		// Create zero bias if missing.
		zeroData := make([]float32, rc.HiddenSize)
		outputNormBias, _ = tensor.New([]int{rc.HiddenSize}, zeroData)
	}

	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)
	input := builder.Input([]int{1, 1})

	// Embedding lookup.
	embNode := &embeddingLookupNode[float32]{engine: proxy, weight: embedWeight}
	hidden := builder.AddNode(embNode, input)

	// Layer 0 pre-LN (ln0) applied to embeddings in RWKV.
	if ln0W, ok := tensors["blocks.0.ln0.weight"]; ok {
		ln0B, hasBias := tensors["blocks.0.ln0.bias"]
		if !hasBias {
			zeroData := make([]float32, rc.HiddenSize)
			ln0B, _ = tensor.New([]int{rc.HiddenSize}, zeroData)
		}
		ln0Node := &rwkvLayerNormNode[float32]{
			engine: proxy,
			ops:    ops,
			weight: param("blocks.0.ln0.weight", ln0W),
			bias:   param("blocks.0.ln0.bias", ln0B),
			eps:    rc.LayerNormEps,
		}
		hidden = builder.AddNode(ln0Node, hidden)
	}

	for i := 0; i < rc.NumLayers; i++ {
		prefix := fmt.Sprintf("blocks.%d.", i)

		// Time mixing (WKV attention).
		ln1W, err := lookup(prefix + "ln1.weight")
		if err != nil {
			return nil, nil, err
		}
		ln1B, ok := tensors[prefix+"ln1.bias"]
		if !ok {
			zeroData := make([]float32, rc.HiddenSize)
			ln1B, _ = tensor.New([]int{rc.HiddenSize}, zeroData)
		}

		ln1Node := &rwkvLayerNormNode[float32]{
			engine: proxy,
			ops:    ops,
			weight: param(prefix+"ln1.weight", ln1W),
			bias:   param(prefix+"ln1.bias", ln1B),
			eps:    rc.LayerNormEps,
		}
		normed1 := builder.AddNode(ln1Node, hidden)

		// Load time mixing weights.
		attWeights, err := loadRWKVTimeMixWeights(tensors, prefix, rc)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d time mixing: %w", i, err)
		}

		timeMixNode := &rwkvTimeMixNode[float32]{
			engine:     proxy,
			ops:        ops,
			hiddenSize: rc.HiddenSize,
			numHeads:   rc.NumHeads,
			headSize:   rc.HeadSize,
			weights:    attWeights,
		}
		timeMixOut := builder.AddNode(timeMixNode, normed1, hidden)

		// Residual add.
		resAdd1 := &mambaResidualAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd1, timeMixOut, hidden)

		// Channel mixing (FFN-like).
		ln2W, err := lookup(prefix + "ln2.weight")
		if err != nil {
			return nil, nil, err
		}
		ln2B, ok := tensors[prefix+"ln2.bias"]
		if !ok {
			zeroData := make([]float32, rc.HiddenSize)
			ln2B, _ = tensor.New([]int{rc.HiddenSize}, zeroData)
		}

		ln2Node := &rwkvLayerNormNode[float32]{
			engine: proxy,
			ops:    ops,
			weight: param(prefix+"ln2.weight", ln2W),
			bias:   param(prefix+"ln2.bias", ln2B),
			eps:    rc.LayerNormEps,
		}
		normed2 := builder.AddNode(ln2Node, hidden)

		// Load channel mixing weights.
		ffnWeights, err := loadRWKVChannelMixWeights(tensors, prefix)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d channel mixing: %w", i, err)
		}

		chanMixNode := &rwkvChannelMixNode[float32]{
			engine:  proxy,
			ops:     ops,
			weights: ffnWeights,
		}
		chanMixOut := builder.AddNode(chanMixNode, normed2, hidden)

		resAdd2 := &mambaResidualAddNode[float32]{engine: proxy}
		hidden = builder.AddNode(resAdd2, chanMixOut, hidden)
	}

	// Final LayerNorm.
	finalNormNode := &rwkvLayerNormNode[float32]{
		engine: proxy,
		ops:    ops,
		weight: param("output_norm.weight", outputNormWeight),
		bias:   param("output_norm.bias", outputNormBias),
		eps:    rc.LayerNormEps,
	}
	normedFinal := builder.AddNode(finalNormNode, hidden)

	// LM Head.
	lmHead := &lmHeadNode[float32]{engine: proxy, weight: lmHeadWeight}
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// rwkvTimeMixWeights holds the weight tensors for a RWKV time mixing block.
type rwkvTimeMixWeights[T tensor.Numeric] struct {
	timeMixR   *graph.Parameter[T] // [1, 1, hidden_size]
	timeMixK   *graph.Parameter[T]
	timeMixV   *graph.Parameter[T]
	timeMixG   *graph.Parameter[T]
	timeDecay  *graph.Parameter[T] // [num_heads, head_size]
	timeFaaaa  *graph.Parameter[T] // initial wkv state
	receptance *graph.Parameter[T] // [hidden_size, hidden_size]
	key        *graph.Parameter[T]
	value      *graph.Parameter[T]
	gate       *graph.Parameter[T]
	output     *graph.Parameter[T]
	lnXWeight  *graph.Parameter[T] // group norm weight
	lnXBias    *graph.Parameter[T]
}

// rwkvChannelMixWeights holds weights for a RWKV channel mixing block.
type rwkvChannelMixWeights[T tensor.Numeric] struct {
	timeMixK   *graph.Parameter[T]
	timeMixR   *graph.Parameter[T]
	key        *graph.Parameter[T] // [ffn_size, hidden_size]
	value      *graph.Parameter[T] // [hidden_size, ffn_size]
	receptance *graph.Parameter[T] // [hidden_size, hidden_size]
}

func loadRWKVTimeMixWeights(
	tensors map[string]*tensor.TensorNumeric[float32],
	prefix string,
	rc RWKVConfig,
) (*rwkvTimeMixWeights[float32], error) {
	p := prefix + "att."

	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	load := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	// Load required tensors.
	timeMixR, err := load(p + "time_mix_r")
	if err != nil {
		return nil, err
	}
	timeMixK, err := load(p + "time_mix_k")
	if err != nil {
		return nil, err
	}
	timeMixV, err := load(p + "time_mix_v")
	if err != nil {
		return nil, err
	}
	timeDecay, err := load(p + "time_decay")
	if err != nil {
		return nil, err
	}
	receptanceW, err := load(p + "receptance.weight")
	if err != nil {
		return nil, err
	}
	keyW, err := load(p + "key.weight")
	if err != nil {
		return nil, err
	}
	valueW, err := load(p + "value.weight")
	if err != nil {
		return nil, err
	}
	outputW, err := load(p + "output.weight")
	if err != nil {
		return nil, err
	}

	// Transpose projection weights from GGUF [out, in] to [in, out].
	receptanceWT, err := cpuTranspose2D(receptanceW)
	if err != nil {
		return nil, fmt.Errorf("transpose receptance: %w", err)
	}
	keyWT, err := cpuTranspose2D(keyW)
	if err != nil {
		return nil, fmt.Errorf("transpose key: %w", err)
	}
	valueWT, err := cpuTranspose2D(valueW)
	if err != nil {
		return nil, fmt.Errorf("transpose value: %w", err)
	}
	outputWT, err := cpuTranspose2D(outputW)
	if err != nil {
		return nil, fmt.Errorf("transpose output: %w", err)
	}

	w := &rwkvTimeMixWeights[float32]{
		timeMixR:   param(p+"time_mix_r", timeMixR),
		timeMixK:   param(p+"time_mix_k", timeMixK),
		timeMixV:   param(p+"time_mix_v", timeMixV),
		timeDecay:  param(p+"time_decay", timeDecay),
		receptance: param(p+"receptance.weight", receptanceWT),
		key:        param(p+"key.weight", keyWT),
		value:      param(p+"value.weight", valueWT),
		output:     param(p+"output.weight", outputWT),
	}

	// Optional: time_mix_g and gate (RWKV-6 gated variant).
	if tmG, ok := tensors[p+"time_mix_g"]; ok {
		w.timeMixG = param(p+"time_mix_g", tmG)
	}
	if gateW, ok := tensors[p+"gate.weight"]; ok {
		gateWT, tErr := cpuTranspose2D(gateW)
		if tErr != nil {
			return nil, fmt.Errorf("transpose gate: %w", tErr)
		}
		w.gate = param(p+"gate.weight", gateWT)
	}

	// time_faaaa: initial state (optional, RWKV-6).
	if tf, ok := tensors[p+"time_faaaa"]; ok {
		w.timeFaaaa = param(p+"time_faaaa", tf)
	}

	// ln_x: group norm inside time mixing output (RWKV-6).
	if lnXW, ok := tensors[p+"ln_x.weight"]; ok {
		w.lnXWeight = param(p+"ln_x.weight", lnXW)
	}
	if lnXB, ok := tensors[p+"ln_x.bias"]; ok {
		w.lnXBias = param(p+"ln_x.bias", lnXB)
	}

	return w, nil
}

func loadRWKVChannelMixWeights(
	tensors map[string]*tensor.TensorNumeric[float32],
	prefix string,
) (*rwkvChannelMixWeights[float32], error) {
	p := prefix + "ffn."

	param := func(name string, t *tensor.TensorNumeric[float32]) *graph.Parameter[float32] {
		return &graph.Parameter[float32]{Name: name, Value: t}
	}

	load := func(name string) (*tensor.TensorNumeric[float32], error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor %q", name)
		}
		return t, nil
	}

	timeMixK, err := load(p + "time_mix_k")
	if err != nil {
		return nil, err
	}
	timeMixR, err := load(p + "time_mix_r")
	if err != nil {
		return nil, err
	}
	keyW, err := load(p + "key.weight")
	if err != nil {
		return nil, err
	}
	valueW, err := load(p + "value.weight")
	if err != nil {
		return nil, err
	}
	receptanceW, err := load(p + "receptance.weight")
	if err != nil {
		return nil, err
	}

	// Transpose projection weights.
	keyWT, err := cpuTranspose2D(keyW)
	if err != nil {
		return nil, fmt.Errorf("transpose key: %w", err)
	}
	valueWT, err := cpuTranspose2D(valueW)
	if err != nil {
		return nil, fmt.Errorf("transpose value: %w", err)
	}
	receptanceWT, err := cpuTranspose2D(receptanceW)
	if err != nil {
		return nil, fmt.Errorf("transpose receptance: %w", err)
	}

	return &rwkvChannelMixWeights[float32]{
		timeMixK:   param(p+"time_mix_k", timeMixK),
		timeMixR:   param(p+"time_mix_r", timeMixR),
		key:        param(p+"key.weight", keyWT),
		value:      param(p+"value.weight", valueWT),
		receptance: param(p+"receptance.weight", receptanceWT),
	}, nil
}

// rwkvLayerNormNode applies standard LayerNorm (used instead of RMSNorm in RWKV).
// Output: (x - mean) / sqrt(var + eps) * weight + bias
type rwkvLayerNormNode[T tensor.Numeric] struct {
	engine compute.Engine[T]
	ops    numeric.Arithmetic[T]
	weight *graph.Parameter[T]
	bias   *graph.Parameter[T]
	eps    float32
}

func (n *rwkvLayerNormNode[T]) OpType() string { return "RWKVLayerNorm" }
func (n *rwkvLayerNormNode[T]) Attributes() map[string]any {
	return map[string]any{"eps": n.eps}
}
func (n *rwkvLayerNormNode[T]) OutputShape() []int { return nil }
func (n *rwkvLayerNormNode[T]) Parameters() []*graph.Parameter[T] {
	return []*graph.Parameter[T]{n.weight, n.bias}
}

func (n *rwkvLayerNormNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("RWKVLayerNorm requires 1 input, got %d", len(inputs))
	}
	x := inputs[0]
	shape := x.Shape()
	if len(shape) < 2 {
		return nil, fmt.Errorf("RWKVLayerNorm input must be at least 2D")
	}

	data := x.Data()
	wData := n.weight.Value.Data()
	bData := n.bias.Value.Data()
	hiddenSize := shape[len(shape)-1]
	n_ := 1
	for i := 0; i < len(shape)-1; i++ {
		n_ *= shape[i]
	}

	out := make([]T, len(data))
	eps := float64(n.eps)

	for i := 0; i < n_; i++ {
		off := i * hiddenSize
		row := data[off : off+hiddenSize]

		// Compute mean.
		var sum float64
		for _, v := range row {
			sum += float64(v)
		}
		mean := sum / float64(hiddenSize)

		// Compute variance.
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(hiddenSize)
		invStd := 1.0 / math.Sqrt(variance+eps)

		for j := 0; j < hiddenSize; j++ {
			norm := (float64(row[j]) - mean) * invStd
			out[off+j] = T(norm*float64(wData[j]) + float64(bData[j]))
		}
	}

	return tensor.New(shape, out)
}

func (n *rwkvLayerNormNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// rwkvTimeMixNode implements the RWKV-6 time mixing block (WKV linear attention).
//
// Token shifting: the current token's input is mixed with the previous token's
// hidden state using learned mix vectors (time_mix_r/k/v/g).
//
// WKV operator (linear complexity recurrence):
//
//	For each head h and position t:
//	  wkv[t] = Σ(i=0..t) exp(-(t-i)*decay_h + k_i[h]) * v_i[h]
//	         / Σ(i=0..t) exp(-(t-i)*decay_h + k_i[h])
//
// Implemented iteratively as:
//
//	state[h] = state[h] * exp(-decay_h) + exp(k_t[h]) * v_t[h]   (numerator)
//	norm[h]  = norm[h]  * exp(-decay_h) + exp(k_t[h])             (denominator)
//	wkv[t,h] = state[h] / norm[h]
//
// Output gated by sigmoid(receptance) and optionally by SiLU(gate).
type rwkvTimeMixNode[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	hiddenSize int
	numHeads   int
	headSize   int
	weights    *rwkvTimeMixWeights[T]
}

func (n *rwkvTimeMixNode[T]) OpType() string { return "RWKVTimeMix" }
func (n *rwkvTimeMixNode[T]) Attributes() map[string]any {
	return map[string]any{
		"hidden_size": n.hiddenSize,
		"num_heads":   n.numHeads,
		"head_size":   n.headSize,
	}
}
func (n *rwkvTimeMixNode[T]) OutputShape() []int { return nil }

func (n *rwkvTimeMixNode[T]) Parameters() []*graph.Parameter[T] {
	w := n.weights
	params := []*graph.Parameter[T]{
		w.timeMixR, w.timeMixK, w.timeMixV,
		w.timeDecay, w.receptance, w.key, w.value, w.output,
	}
	if w.timeMixG != nil {
		params = append(params, w.timeMixG)
	}
	if w.gate != nil {
		params = append(params, w.gate)
	}
	if w.timeFaaaa != nil {
		params = append(params, w.timeFaaaa)
	}
	if w.lnXWeight != nil {
		params = append(params, w.lnXWeight)
	}
	if w.lnXBias != nil {
		params = append(params, w.lnXBias)
	}
	return params
}

// Forward implements the time mixing block.
// inputs[0] = normed hidden [batch, seq, hidden_size]
// inputs[1] = pre-norm hidden (for token shift) [batch, seq, hidden_size]
func (n *rwkvTimeMixNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("RWKVTimeMix requires 2 inputs, got %d", len(inputs))
	}
	x := inputs[0] // normed: [batch, seq, hidden_size]
	xPrev := inputs[1] // for token shift

	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("RWKVTimeMix input must be 3D [batch, seq, hidden_size], got %v", shape)
	}
	batch, seqLen, H := shape[0], shape[1], shape[2]
	if H != n.hiddenSize {
		return nil, fmt.Errorf("RWKVTimeMix hidden size mismatch: got %d, want %d", H, n.hiddenSize)
	}

	xData := x.Data()
	xPrevData := xPrev.Data()

	tmrData := n.weights.timeMixR.Value.Data()
	tmkData := n.weights.timeMixK.Value.Data()
	tmvData := n.weights.timeMixV.Value.Data()

	// Token shift: shifted[t] = lerp(x[t], x[t-1], time_mix)
	// For t=0, x[t-1] = last token from previous call (use xPrev for simplicity — 0-padded).
	shiftedR := make([]T, batch*seqLen*H)
	shiftedK := make([]T, batch*seqLen*H)
	shiftedV := make([]T, batch*seqLen*H)
	var shiftedG []T
	hasMixG := n.weights.timeMixG != nil
	if hasMixG {
		shiftedG = make([]T, batch*seqLen*H)
	}

	var tmgData []T
	if hasMixG {
		tmgData = n.weights.timeMixG.Value.Data()
	}

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			var prevOff int
			if s == 0 {
				// Use xPrev (the pre-norm hidden state at position [b, 0]) as the
				// "previous token" state for the first token in the sequence.
				prevOff = b * seqLen * H
			} else {
				prevOff = (b*seqLen + s - 1) * H
			}
			curOff := (b*seqLen + s) * H

			// For s==0, previous = xPrev[b, 0]; otherwise xData[b, s-1].
			var prevSlice []T
			if s == 0 {
				prevSlice = xPrevData[prevOff : prevOff+H]
			} else {
				prevSlice = xData[prevOff : prevOff+H]
			}

			for d := 0; d < H; d++ {
				cur := float64(xData[curOff+d])
				prev := float64(prevSlice[d])

				mixR := float64(tmrData[d])
				mixK := float64(tmkData[d])
				mixV := float64(tmvData[d])

				shiftedR[curOff+d] = T(cur*mixR + prev*(1-mixR))
				shiftedK[curOff+d] = T(cur*mixK + prev*(1-mixK))
				shiftedV[curOff+d] = T(cur*mixV + prev*(1-mixV))

				if hasMixG {
					mixG := float64(tmgData[d])
					shiftedG[curOff+d] = T(cur*mixG + prev*(1-mixG))
				}
			}
		}
	}

	// Linear projections: r, k, v, (g) using matmul.
	rw := n.weights.receptance.Value.Data() // [H, H] (transposed)
	kw := n.weights.key.Value.Data()
	vw := n.weights.value.Value.Data()
	ow := n.weights.output.Value.Data()

	project := func(shifted []T, w []T) []T {
		// shifted: [batch*seqLen, H]; w: [H, H]
		out := make([]T, batch*seqLen*H)
		for row := 0; row < batch*seqLen; row++ {
			for col := 0; col < H; col++ {
				var sum T
				for k := 0; k < H; k++ {
					sum = n.ops.Add(sum, n.ops.Mul(shifted[row*H+k], w[k*H+col]))
				}
				out[row*H+col] = sum
			}
		}
		return out
	}

	rVec := project(shiftedR, rw)
	kVec := project(shiftedK, kw)
	vVec := project(shiftedV, vw)

	var gVec []T
	if hasMixG && n.weights.gate != nil {
		gw := n.weights.gate.Value.Data()
		gVec = project(shiftedG, gw)
	}

	// Apply sigmoid to r (receptance).
	for i, v := range rVec {
		rVec[i] = T(1.0 / (1.0 + math.Exp(-float64(v))))
	}

	// WKV linear attention (per head, iterative).
	// For each head h (size head_size), compute WKV recurrence.
	decayData := n.weights.timeDecay.Value.Data() // [num_heads, head_size]

	wkvOut := make([]T, batch*seqLen*H)

	for b := 0; b < batch; b++ {
		// State per head: numerator and denominator of WKV.
		// Shape: [num_heads, head_size] for num, [num_heads] for den.
		// RWKV-6 uses full state: [num_heads, head_size, head_size] for matrix-valued state.
		// Here we use the simplified scalar-per-head WKV (RWKV-4/5 style).
		wkvNum := make([]float64, n.numHeads*n.headSize) // Σ exp(w+k)*v
		wkvDen := make([]float64, n.numHeads*n.headSize) // Σ exp(w+k)

		// Initialize from time_faaaa if present (initial state).
		if n.weights.timeFaaaa != nil {
			for h := 0; h < n.numHeads; h++ {
				for j := 0; j < n.headSize; j++ {
					wkvDen[h*n.headSize+j] = float64(n.weights.timeFaaaa.Value.Data()[h*n.headSize+j])
				}
			}
		}

		for s := 0; s < seqLen; s++ {
			off := (b*seqLen + s) * H

			// Iterate over heads.
			for h := 0; h < n.numHeads; h++ {
				hOff := h * n.headSize
				decayOff := h * n.headSize

				for j := 0; j < n.headSize; j++ {
					idx := off + hOff + j
					k := float64(kVec[idx])
					v := float64(vVec[idx])
					decay := float64(decayData[decayOff+j])

					// decay is stored as log(-decay) in GGUF (negative log of decay rate).
					// Actual decay: exp(-exp(decay)) per step.
					actualDecay := math.Exp(-math.Exp(decay))

					stateIdx := h*n.headSize + j
					wkvNum[stateIdx] = wkvNum[stateIdx]*actualDecay + math.Exp(k)*v
					wkvDen[stateIdx] = wkvDen[stateIdx]*actualDecay + math.Exp(k)

					var wkvVal float64
					if math.Abs(wkvDen[stateIdx]) > 1e-30 {
						wkvVal = wkvNum[stateIdx] / wkvDen[stateIdx]
					}

					// Gate by receptance.
					r := float64(rVec[idx])
					wkvOut[idx] = T(r * wkvVal)
				}
			}
		}
	}

	// Apply group norm (ln_x) if present.
	if n.weights.lnXWeight != nil {
		lnXW := n.weights.lnXWeight.Value.Data()
		var lnXB []T
		if n.weights.lnXBias != nil {
			lnXB = n.weights.lnXBias.Value.Data()
		} else {
			lnXB = make([]T, H)
		}
		eps := 1e-5
		// Apply layer norm per token across all channels (group norm with 1 group = LN).
		for i := 0; i < batch*seqLen; i++ {
			off := i * H
			row := wkvOut[off : off+H]
			var sum float64
			for _, v := range row {
				sum += float64(v)
			}
			mean := sum / float64(H)
			var variance float64
			for _, v := range row {
				d := float64(v) - mean
				variance += d * d
			}
			variance /= float64(H)
			invStd := 1.0 / math.Sqrt(variance+eps)
			for j := 0; j < H; j++ {
				norm := (float64(row[j]) - mean) * invStd
				wkvOut[off+j] = T(norm*float64(lnXW[j]) + float64(lnXB[j]))
			}
		}
	}

	// Multiply by gate (SiLU) if present.
	if gVec != nil {
		for i := range wkvOut {
			g := float64(gVec[i])
			silu := g / (1.0 + math.Exp(-g))
			wkvOut[i] = n.ops.Mul(wkvOut[i], T(silu))
		}
	}

	// Output projection.
	outVec := make([]T, batch*seqLen*H)
	for row := 0; row < batch*seqLen; row++ {
		for col := 0; col < H; col++ {
			var sum T
			for k := 0; k < H; k++ {
				sum = n.ops.Add(sum, n.ops.Mul(wkvOut[row*H+k], ow[k*H+col]))
			}
			outVec[row*H+col] = sum
		}
	}

	return tensor.New([]int{batch, seqLen, H}, outVec)
}

func (n *rwkvTimeMixNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// rwkvChannelMixNode implements the RWKV channel mixing block.
//
// Channel mixing replaces the FFN in transformers:
//   - Token shifted input mixed with time_mix_k/r vectors
//   - Key projection followed by squared ReLU (activation)
//   - Value projection on activated keys
//   - Receptance gate: sigmoid(r_proj(shifted_r)) * value_output
type rwkvChannelMixNode[T tensor.Numeric] struct {
	engine  compute.Engine[T]
	ops     numeric.Arithmetic[T]
	weights *rwkvChannelMixWeights[T]
}

func (n *rwkvChannelMixNode[T]) OpType() string { return "RWKVChannelMix" }
func (n *rwkvChannelMixNode[T]) Attributes() map[string]any { return nil }
func (n *rwkvChannelMixNode[T]) OutputShape() []int          { return nil }

func (n *rwkvChannelMixNode[T]) Parameters() []*graph.Parameter[T] {
	w := n.weights
	return []*graph.Parameter[T]{w.timeMixK, w.timeMixR, w.key, w.value, w.receptance}
}

// Forward implements channel mixing.
// inputs[0] = normed hidden [batch, seq, hidden_size]
// inputs[1] = pre-norm hidden (for token shift)
func (n *rwkvChannelMixNode[T]) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("RWKVChannelMix requires 2 inputs, got %d", len(inputs))
	}
	x := inputs[0]
	xPrev := inputs[1]

	shape := x.Shape()
	if len(shape) != 3 {
		return nil, fmt.Errorf("RWKVChannelMix input must be 3D, got %v", shape)
	}
	batch, seqLen, H := shape[0], shape[1], shape[2]

	xData := x.Data()
	xPrevData := xPrev.Data()

	tmkData := n.weights.timeMixK.Value.Data()
	tmrData := n.weights.timeMixR.Value.Data()

	// Determine FFN hidden size from key weight shape: [H, ffnSize].
	kShape := n.weights.key.Value.Shape()
	ffnSize := kShape[1]

	// Token shift.
	shiftedK := make([]T, batch*seqLen*H)
	shiftedR := make([]T, batch*seqLen*H)

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			curOff := (b*seqLen + s) * H
			var prevSlice []T
			if s == 0 {
				prevOff := b * seqLen * H
				prevSlice = xPrevData[prevOff : prevOff+H]
			} else {
				prevOff := (b*seqLen + s - 1) * H
				prevSlice = xData[prevOff : prevOff+H]
			}

			for d := 0; d < H; d++ {
				cur := float64(xData[curOff+d])
				prev := float64(prevSlice[d])

				mixK := float64(tmkData[d])
				mixR := float64(tmrData[d])

				shiftedK[curOff+d] = T(cur*mixK + prev*(1-mixK))
				shiftedR[curOff+d] = T(cur*mixR + prev*(1-mixR))
			}
		}
	}

	kw := n.weights.key.Value.Data()        // [H, ffnSize]
	vw := n.weights.value.Value.Data()      // [ffnSize, H]
	rw := n.weights.receptance.Value.Data() // [H, H]

	// k projection: [batch*seqLen, H] x [H, ffnSize] -> [batch*seqLen, ffnSize].
	kVec := make([]T, batch*seqLen*ffnSize)
	for row := 0; row < batch*seqLen; row++ {
		for col := 0; col < ffnSize; col++ {
			var sum T
			for k := 0; k < H; k++ {
				sum = n.ops.Add(sum, n.ops.Mul(shiftedK[row*H+k], kw[k*ffnSize+col]))
			}
			kVec[row*ffnSize+col] = sum
		}
	}

	// Squared ReLU activation on k.
	for i, v := range kVec {
		f := float64(v)
		if f < 0 {
			f = 0
		}
		kVec[i] = T(f * f)
	}

	// v projection: [batch*seqLen, ffnSize] x [ffnSize, H] -> [batch*seqLen, H].
	vVec := make([]T, batch*seqLen*H)
	for row := 0; row < batch*seqLen; row++ {
		for col := 0; col < H; col++ {
			var sum T
			for k := 0; k < ffnSize; k++ {
				sum = n.ops.Add(sum, n.ops.Mul(kVec[row*ffnSize+k], vw[k*H+col]))
			}
			vVec[row*H+col] = sum
		}
	}

	// r projection: [batch*seqLen, H] x [H, H] -> [batch*seqLen, H].
	rVec := make([]T, batch*seqLen*H)
	for row := 0; row < batch*seqLen; row++ {
		for col := 0; col < H; col++ {
			var sum T
			for k := 0; k < H; k++ {
				sum = n.ops.Add(sum, n.ops.Mul(shiftedR[row*H+k], rw[k*H+col]))
			}
			rVec[row*H+col] = sum
		}
	}

	// Sigmoid on r.
	for i, v := range rVec {
		rVec[i] = T(1.0 / (1.0 + math.Exp(-float64(v))))
	}

	// Output: sigmoid(r) * v.
	out := make([]T, batch*seqLen*H)
	for i := range out {
		out[i] = n.ops.Mul(rVec[i], vVec[i])
	}

	return tensor.New([]int{batch, seqLen, H}, out)
}

func (n *rwkvChannelMixNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}
