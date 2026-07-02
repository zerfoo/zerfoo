package inference

import (
	"context"
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/layers/audio"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// VoxtralConfig holds Voxtral-specific model configuration.
type VoxtralConfig struct {
	// Audio encoder config (Whisper-large-v3 style).
	AudioHiddenDim        int
	AudioNumLayers        int
	AudioNumHeads         int
	AudioNumMels          int
	AudioIntermediateSize int
	AudioKernelSize       int

	// MLP adapter config.
	StackFactor int // number of consecutive encoder frames to concatenate (e.g. 4)

	// Text decoder config is stored in gguf.ModelConfig.
}

// VoxtralConfigFromGGUF extracts Voxtral configuration from GGUF ModelConfig.
func VoxtralConfigFromGGUF(cfg *gguf.ModelConfig) VoxtralConfig {
	audioHidden := 1280 // Whisper-large-v3 default
	audioLayers := 32   // Whisper-large-v3 default
	audioHeads := 20    // Whisper-large-v3 default
	audioMels := 128    // Voxtral uses 128 mel bins
	audioInter := 5120  // Whisper-large-v3 default
	stackFactor := 4    // Voxtral stacks 4 frames
	kernelSize := 3     // Whisper default

	if cfg.AudioHiddenSize > 0 {
		audioHidden = cfg.AudioHiddenSize
	}
	if cfg.AudioNumLayers > 0 {
		audioLayers = cfg.AudioNumLayers
	}
	if cfg.AudioNumHeads > 0 {
		audioHeads = cfg.AudioNumHeads
	}
	if cfg.AudioNumMels > 0 {
		audioMels = cfg.AudioNumMels
	}
	if cfg.AudioIntermediateSize > 0 {
		audioInter = cfg.AudioIntermediateSize
	}
	if cfg.AudioProjectorStackFactor > 0 {
		stackFactor = cfg.AudioProjectorStackFactor
	}

	return VoxtralConfig{
		AudioHiddenDim:        audioHidden,
		AudioNumLayers:        audioLayers,
		AudioNumHeads:         audioHeads,
		AudioNumMels:          audioMels,
		AudioIntermediateSize: audioInter,
		AudioKernelSize:       kernelSize,
		StackFactor:           stackFactor,
	}
}

// parseVoxtralConfig parses Voxtral-family config.json fields.
func parseVoxtralConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	// The text decoder is Llama-based.
	meta, err := parseLlamaConfig(raw)
	if err != nil {
		return nil, err
	}
	meta.Architecture = "voxtral"
	return meta, nil
}

// buildVoxtralGraph constructs a computation graph for the Voxtral architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// The Voxtral architecture is:
//
//	Audio -> Mel Spectrogram (128 bins) -> Whisper-large-v3 Encoder ->
//	Frame Stacking (4x) -> MLP Adapter -> Llama Text Decoder -> LM Head
//
// Audio encoder tensors use Voxtral mmproj naming:
//
//	a.conv1d.0.weight / bias         — Conv1D frontend layer 1
//	a.conv1d.1.weight / bias         — Conv1D frontend layer 2
//	a.blk.{i}.ln1.weight / bias      — Pre-attention LayerNorm
//	a.blk.{i}.attn_q.weight / bias   — Q projection (with bias)
//	a.blk.{i}.attn_k.weight / bias   — K projection (with bias)
//	a.blk.{i}.attn_v.weight / bias   — V projection (with bias)
//	a.blk.{i}.attn_o.weight          — Output projection
//	a.blk.{i}.ln2.weight / bias      — Pre-FFN LayerNorm
//	a.blk.{i}.ffn_up.weight / bias   — FFN up projection
//	a.blk.{i}.ffn_down.weight / bias — FFN down projection
//	a.post_ln.weight / bias           — Post LayerNorm
//	mm.a.mlp.0.weight / bias          — Adapter MLP layer 1
//	mm.a.mlp.2.weight / bias          — Adapter MLP layer 2
//
// Text decoder tensors follow the standard Llama naming convention.
func buildVoxtralGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	vc := VoxtralConfigFromGGUF(cfg)
	return BuildVoxtralModel(vc, tensors, cfg, engine)
}

// BuildVoxtralModel constructs the Voxtral computation graph from a weight map.
// Exported for benchmark and integration tests that construct synthetic weight maps.
func BuildVoxtralModel(
	vc VoxtralConfig,
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	// Build Whisper-style audio encoder with Voxtral-specific config.
	enc, err := audio.NewWhisperEncoder[float32](
		"voxtral_encoder", engine, ops,
		audio.WhisperEncoderConfig{
			NumMels:          vc.AudioNumMels,
			HiddenDim:        vc.AudioHiddenDim,
			NumHeads:         vc.AudioNumHeads,
			NumLayers:        vc.AudioNumLayers,
			KernelSize:       vc.AudioKernelSize,
			IntermediateSize: vc.AudioIntermediateSize,
			AttentionBias:    true, // Voxtral encoder always uses Q/K/V biases
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("create voxtral encoder: %w", err)
	}

	// Load audio encoder weights.
	if err := loadVoxtralEncoderWeights(enc, tensors, vc); err != nil {
		return nil, nil, fmt.Errorf("load voxtral encoder weights: %w", err)
	}

	tl := newTensorLookup(tensors)
	pw := newParamWrapper[float32]()

	// Load adapter MLP weights.
	adapterW0, err := tl.Lookup("mm.a.mlp.0.weight")
	if err != nil {
		return nil, nil, err
	}
	adapterB0, err := tl.Lookup("mm.a.mlp.0.bias")
	if err != nil {
		return nil, nil, err
	}
	adapterW2, err := tl.Lookup("mm.a.mlp.2.weight")
	if err != nil {
		return nil, nil, err
	}
	adapterB2, err := tl.Lookup("mm.a.mlp.2.bias")
	if err != nil {
		return nil, nil, err
	}

	// Get text model embedding weight.
	embedWeight, err := tl.Lookup("model.embed_tokens.weight")
	if err != nil {
		return nil, nil, err
	}

	lmHeadWeight, ok := tl.Optional("lm_head.weight")
	if !ok {
		lmHeadWeight = embedWeight
	}

	// Build the graph.
	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)

	// Input: mel spectrogram [1, num_mels, T_frames].
	melInput := builder.Input([]int{1, vc.AudioNumMels, -1})

	// Audio encoder: [1, num_mels, T] -> [T_downsampled, hidden_dim].
	encoderOut := builder.AddNode(enc, melInput)

	// Frame stacking + MLP adapter.
	adapterNode := &voxtralAdapterNode[float32]{
		engine:      proxy,
		ops:         ops,
		stackFactor: vc.StackFactor,
		w0:          adapterW0,
		b0:          adapterB0,
		w2:          adapterW2,
		b2:          adapterB2,
		textHidden:  cfg.HiddenSize,
	}
	adapterOut := builder.AddNode(adapterNode, encoderOut)

	// Text decoder (Llama-style transformer).
	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	hidden := adapterOut

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		inputNormW, err := tl.Lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := newVisionRMSNorm(proxy, rmsEps, pw.Wrap(prefix+"input_layernorm.weight", inputNormW))
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// GQA attention (same as Llama).
		qW, err := tl.Lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := tl.Lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := tl.Lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := tl.Lookup(prefix + "self_attn.o_proj.weight")
		if err != nil {
			return nil, nil, err
		}

		attnNode, err := newVisionGQA(
			proxy,
			cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, headDim, cfg.MaxSeqLen,
			cfg.RopeTheta, qW, kW, vW, oW, prefix, pw,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		attnOut := builder.AddNode(attnNode, normed)

		// Fused residual add + pre-FFN RMSNorm.
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// FFN (SwiGLU).
		gateW, err := tl.Lookup(prefix + "mlp.gate_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		upW, err := tl.Lookup(prefix + "mlp.up_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		downW, err := tl.Lookup(prefix + "mlp.down_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		ffnNode, err := newVisionSwiGLUFFN(proxy, cfg.HiddenSize, gateW, upW, downW, prefix)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}
		ffnOut := builder.AddNode(ffnNode, normed2)

		// Residual add.
		resAdd := &residualAddNode[float32]{engine: proxy, source: fusedNode}
		hidden = builder.AddNode(resAdd, ffnOut)
	}

	// Final RMSNorm.
	finalNormWeight, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNorm, err := newVisionRMSNorm(proxy, rmsEps, pw.Wrap("model.norm.weight", finalNormWeight))
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

	// LM Head.
	lmHead := newLMHeadNode(proxy, lmHeadWeight, 0)
	output := builder.AddNode(lmHead, normedFinal)

	g, err := builder.Build(output)
	if err != nil {
		return nil, nil, fmt.Errorf("build graph: %w", err)
	}

	g.SetEngineProxy(proxy)
	return g, embedWeight, nil
}

// loadVoxtralEncoderWeights maps Voxtral mmproj GGUF tensor data onto the
// WhisperEncoder parameters using positional indexing.
//
// Parameters() order with AttentionBias=true:
//
//	[0] conv1_weight, [1] conv1_bias, [2] conv2_weight, [3] conv2_bias,
//	then per block (15 params each):
//	  [0] ln1.gamma, [1] ln1.beta, [2] q_weights, [3] k_weights,
//	  [4] v_weights, [5] o_weights, [6] q_bias, [7] k_bias, [8] v_bias,
//	  [9] ln2.gamma, [10] ln2.beta,
//	  [11] ffn1_linear_weights, [12] ffn1_bias_biases,
//	  [13] ffn2_linear_weights, [14] ffn2_bias_biases,
//	then: [0] lnPost.gamma, [1] lnPost.beta.
func loadVoxtralEncoderWeights(
	enc *audio.WhisperEncoder[float32],
	tensors map[string]*tensor.TensorNumeric[float32],
	vc VoxtralConfig,
) error {
	params := enc.Parameters()

	// 4 (conv) + 15*numLayers (blocks with bias) + 2 (ln_post).
	paramsPerBlock := 15 // 12 base + 3 bias params
	expectedCount := 4 + paramsPerBlock*vc.AudioNumLayers + 2
	if len(params) != expectedCount {
		return fmt.Errorf("expected %d parameters, got %d", expectedCount, len(params))
	}

	ggufOrder := make([]string, 0, expectedCount)

	// Conv frontend.
	ggufOrder = append(ggufOrder,
		"a.conv1d.0.weight",
		"a.conv1d.0.bias",
		"a.conv1d.1.weight",
		"a.conv1d.1.bias",
	)

	// Per-block tensors.
	for i := 0; i < vc.AudioNumLayers; i++ {
		prefix := fmt.Sprintf("a.blk.%d.", i)
		ggufOrder = append(ggufOrder,
			prefix+"ln1.weight",
			prefix+"ln1.bias",
			prefix+"attn_q.weight",
			prefix+"attn_k.weight",
			prefix+"attn_v.weight",
			prefix+"attn_o.weight",
			prefix+"attn_q.bias",
			prefix+"attn_k.bias",
			prefix+"attn_v.bias",
			prefix+"ln2.weight",
			prefix+"ln2.bias",
			prefix+"ffn_up.weight",
			prefix+"ffn_up.bias",
			prefix+"ffn_down.weight",
			prefix+"ffn_down.bias",
		)
	}

	// Post layer norm.
	ggufOrder = append(ggufOrder,
		"a.post_ln.weight",
		"a.post_ln.bias",
	)

	for i, ggufName := range ggufOrder {
		t, ok := tensors[ggufName]
		if !ok {
			return fmt.Errorf("missing GGUF tensor %q", ggufName)
		}
		copy(params[i].Value.Data(), t.Data())
	}

	return nil
}

// voxtralAdapterNode implements frame stacking followed by a 2-layer MLP
// projection with GELU activation.
//
// Frame stacking: concatenates stackFactor consecutive encoder frames along
// the feature dimension, reducing temporal resolution by stackFactor.
//
// MLP: Linear(stackFactor*audioHidden, textHidden) -> GELU -> Linear(textHidden, textHidden)
type voxtralAdapterNode[T tensor.Numeric] struct {
	engine      compute.Engine[T]
	ops         numeric.Arithmetic[T]
	stackFactor int
	w0, w2      *tensor.TensorNumeric[T]
	b0, b2      *tensor.TensorNumeric[T]
	textHidden  int
}

func (a *voxtralAdapterNode[T]) OpType() string                    { return "VoxtralAdapter" }
func (a *voxtralAdapterNode[T]) Attributes() map[string]any        { return nil }
func (a *voxtralAdapterNode[T]) OutputShape() []int                { return nil }
func (a *voxtralAdapterNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (a *voxtralAdapterNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{a.w0, a.b0, a.w2, a.b2}
}

func (a *voxtralAdapterNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0] // [seqLen, audioHidden] from WhisperEncoder output

	shape := input.Shape()
	seqLen := shape[0]
	audioHidden := shape[1]

	// Frame stacking: group stackFactor consecutive frames.
	// Pad to multiple of stackFactor if needed.
	stackedLen := seqLen / a.stackFactor
	if seqLen%a.stackFactor != 0 {
		stackedLen++
	}
	paddedLen := stackedLen * a.stackFactor
	stackedDim := a.stackFactor * audioHidden

	inData := input.Data()

	// Pad input if necessary.
	padded := inData
	if paddedLen > seqLen {
		padded = make([]T, paddedLen*audioHidden)
		copy(padded, inData)
		// Zero-padding for remaining frames (already zero-valued).
	}

	// Stack frames: [stackedLen, stackFactor * audioHidden].
	stacked := make([]T, stackedLen*stackedDim)
	for i := 0; i < stackedLen; i++ {
		for f := 0; f < a.stackFactor; f++ {
			srcOffset := (i*a.stackFactor + f) * audioHidden
			dstOffset := i*stackedDim + f*audioHidden
			copy(stacked[dstOffset:dstOffset+audioHidden], padded[srcOffset:srcOffset+audioHidden])
		}
	}

	// MLP layer 1: [stackedLen, stackedDim] -> [stackedLen, textHidden]
	w0Data := a.w0.Data()
	b0Data := a.b0.Data()
	outDim := a.textHidden

	out1 := make([]T, stackedLen*outDim)
	for s := 0; s < stackedLen; s++ {
		for o := 0; o < outDim; o++ {
			var sum T
			for d := 0; d < stackedDim; d++ {
				sum = a.ops.Add(sum, a.ops.Mul(stacked[s*stackedDim+d], w0Data[o*stackedDim+d]))
			}
			out1[s*outDim+o] = a.ops.Add(sum, b0Data[o])
		}
	}

	// GELU activation.
	// TODO(T124.2.3): delegate to layers/activations.NewGelu once the
	// adapter operates on engine-resident tensors. The current MLP path
	// is a raw scalar Go loop on []T slices (no engine ops), so wrapping
	// each row into a tensor for the canonical Node would change storage
	// semantics; defer until the adapter itself is ported to engine ops.
	half := a.ops.FromFloat64(0.5)
	one := a.ops.One()
	coeff := a.ops.FromFloat64(0.044715)
	sqrtTwoOverPi := a.ops.FromFloat64(math.Sqrt(2.0 / math.Pi))
	for i, v := range out1 {
		x3 := a.ops.Mul(v, a.ops.Mul(v, v))
		inner := a.ops.Mul(sqrtTwoOverPi, a.ops.Add(v, a.ops.Mul(coeff, x3)))
		out1[i] = a.ops.Mul(half, a.ops.Mul(v, a.ops.Add(one, a.ops.Tanh(inner))))
	}

	// MLP layer 2: [stackedLen, textHidden] -> [stackedLen, textHidden]
	w2Data := a.w2.Data()
	b2Data := a.b2.Data()
	outDim2 := a.w2.Shape()[0]

	out2 := make([]T, stackedLen*outDim2)
	for s := 0; s < stackedLen; s++ {
		for o := 0; o < outDim2; o++ {
			var sum T
			for d := 0; d < outDim; d++ {
				sum = a.ops.Add(sum, a.ops.Mul(out1[s*outDim+d], w2Data[o*outDim+d]))
			}
			out2[s*outDim2+o] = a.ops.Add(sum, b2Data[o])
		}
	}

	// Return as [1, stackedLen, textHidden] for the Llama decoder.
	return tensor.New[T]([]int{1, stackedLen, outDim2}, out2)
}

func (a *voxtralAdapterNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Static interface assertions.
var _ graph.EmbeddedFrozenProvider[float32] = (*voxtralAdapterNode[float32])(nil)
