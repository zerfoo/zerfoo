package inference

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/vision"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func init() {
	RegisterArchitecture("llava", buildLLaVAGraph)
}

// LLaVAConfig holds LLaVA-specific model configuration.
type LLaVAConfig struct {
	// Vision encoder config.
	ImageSize       int
	PatchSize       int
	VisionHiddenDim int
	VisionNumHeads  int
	VisionNumLayers int
	NumChannels     int

	// Multi-modal projector config.
	ProjectorType string // "linear" or "mlp" (2-layer MLP is default for LLaVA 1.5+)

	// Text decoder config (stored in gguf.ModelConfig).
}

// LLaVAConfigFromGGUF extracts LLaVA configuration from GGUF ModelConfig.
func LLaVAConfigFromGGUF(cfg *gguf.ModelConfig) LLaVAConfig {
	imageSize := 336
	patchSize := 14
	visionHidden := 1024
	visionHeads := 16
	visionLayers := 24
	numChannels := 3

	if cfg.VisionImageSize > 0 {
		imageSize = cfg.VisionImageSize
	}
	if cfg.VisionPatchSize > 0 {
		patchSize = cfg.VisionPatchSize
	}
	if cfg.VisionHiddenSize > 0 {
		visionHidden = cfg.VisionHiddenSize
	}
	if cfg.VisionNumHeads > 0 {
		visionHeads = cfg.VisionNumHeads
	}
	if cfg.VisionNumLayers > 0 {
		visionLayers = cfg.VisionNumLayers
	}

	projType := "mlp"
	if cfg.ProjectorType != "" {
		projType = cfg.ProjectorType
	}

	return LLaVAConfig{
		ImageSize:       imageSize,
		PatchSize:       patchSize,
		VisionHiddenDim: visionHidden,
		VisionNumHeads:  visionHeads,
		VisionNumLayers: visionLayers,
		NumChannels:     numChannels,
		ProjectorType:   projType,
	}
}

// buildLLaVAGraph constructs a computation graph for the LLaVA architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// The LLaVA architecture is:
//
//	Image -> CLIP ViT Encoder -> Multi-Modal Projector -> Merged with Text Embeddings -> Llama Decoder -> LM Head
//
// For text-only inference, the graph degrades to a standard Llama model.
// For vision+language, image tokens are projected into the text embedding space
// and concatenated with text token embeddings before the decoder.
//
// Expected GGUF tensor names for the vision encoder:
//
//	vision.patch_embed.weight                       — [vision_hidden, C*P*P]
//	vision.patch_embed.bias                         — [vision_hidden]
//	vision.class_embedding                          — [vision_hidden]
//	vision.position_embedding                       — [num_patches+1, vision_hidden]
//	vision.blocks.{i}.ln1.weight                    — [vision_hidden]
//	vision.blocks.{i}.ln1.bias                      — [vision_hidden]
//	vision.blocks.{i}.attn.q_proj.weight            — [vision_hidden, vision_hidden]
//	vision.blocks.{i}.attn.k_proj.weight            — [vision_hidden, vision_hidden]
//	vision.blocks.{i}.attn.v_proj.weight            — [vision_hidden, vision_hidden]
//	vision.blocks.{i}.attn.o_proj.weight            — [vision_hidden, vision_hidden]
//	vision.blocks.{i}.ln2.weight                    — [vision_hidden]
//	vision.blocks.{i}.ln2.bias                      — [vision_hidden]
//	vision.blocks.{i}.mlp.fc1.weight                — [4*vision_hidden, vision_hidden]
//	vision.blocks.{i}.mlp.fc1.bias                  — [4*vision_hidden]
//	vision.blocks.{i}.mlp.fc2.weight                — [vision_hidden, 4*vision_hidden]
//	vision.blocks.{i}.mlp.fc2.bias                  — [vision_hidden]
//	vision.ln_post.weight                           — [vision_hidden]
//	vision.ln_post.bias                             — [vision_hidden]
//
// Multi-modal projector (MLP type):
//
//	mm_projector.0.weight                           — [text_hidden, vision_hidden]
//	mm_projector.0.bias                             — [text_hidden]
//	mm_projector.2.weight                           — [text_hidden, text_hidden]
//	mm_projector.2.bias                             — [text_hidden]
//
// Text decoder tensors follow the standard Llama naming convention.
func buildLLaVAGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	lc := LLaVAConfigFromGGUF(cfg)
	return BuildLLaVAModel(lc, tensors, cfg, engine)
}

// BuildLLaVAModel constructs the LLaVA computation graph from a weight map.
// Exported for benchmark and integration tests that construct synthetic weight maps.
func BuildLLaVAModel(
	lc LLaVAConfig,
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	// Build CLIP vision encoder.
	clipCfg := vision.CLIPEncoderConfig{
		ImageSize:   lc.ImageSize,
		PatchSize:   lc.PatchSize,
		HiddenDim:   lc.VisionHiddenDim,
		NumHeads:    lc.VisionNumHeads,
		NumLayers:   lc.VisionNumLayers,
		NumChannels: lc.NumChannels,
	}

	enc, err := vision.NewCLIPEncoder[float32]("clip", engine, ops, clipCfg)
	if err != nil {
		return nil, nil, fmt.Errorf("create clip encoder: %w", err)
	}

	// Load vision weights.
	if err := loadCLIPWeights(enc, tensors, clipCfg); err != nil {
		return nil, nil, fmt.Errorf("load clip weights: %w", err)
	}

	tl := newTensorLookup(tensors)
	pw := newParamWrapper[float32]()

	// Load multi-modal projector weights.
	projW0, err := tl.Lookup("mm_projector.0.weight")
	if err != nil {
		return nil, nil, err
	}
	projB0, err := tl.Lookup("mm_projector.0.bias")
	if err != nil {
		return nil, nil, err
	}

	var projW2, projB2 *tensor.TensorNumeric[float32]
	if lc.ProjectorType == "mlp" {
		var lookupErr error
		projW2, lookupErr = tl.Lookup("mm_projector.2.weight")
		if lookupErr != nil {
			return nil, nil, lookupErr
		}
		projB2, lookupErr = tl.Lookup("mm_projector.2.bias")
		if lookupErr != nil {
			return nil, nil, lookupErr
		}
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

	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	_, isGPUEngine := engine.(compute.WeightUploader)

	transposeWeight := func(name string, t *tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return transposeWeight2D(engine, isGPUEngine, name, t)
	}

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	// Build the graph.
	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)

	// Input: pixel values [1, C, H, W].
	imageInput := builder.Input([]int{1, lc.NumChannels, lc.ImageSize, lc.ImageSize})

	// Vision encoder: [1, C, H, W] -> [1, numPatches+1, visionHidden].
	visionOut := builder.AddNode(enc, imageInput)

	// Multi-modal projector: project vision tokens to text embedding space.
	projNode := &mmProjectorNode[float32]{
		engine:     proxy,
		ops:        ops,
		w0:         projW0,
		b0:         projB0,
		w2:         projW2,
		b2:         projB2,
		useMLP:     lc.ProjectorType == "mlp",
		textHidden: cfg.HiddenSize,
	}
	projectedVision := builder.AddNode(projNode, visionOut)

	hidden := projectedVision

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		// --- Input LayerNorm ---
		inputNormW, err := tl.Lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := normalization.NewRMSNormFromParam[float32](
			proxy, ops, rmsEps, pw.Wrap(prefix+"input_layernorm.weight", inputNormW),
		)
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// --- Self Attention (GQA with RoPE, composed from layers/) ---
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

		qWT, err := transposeWeight(prefix+"self_attn.q_proj.weight", qW)
		if err != nil {
			return nil, nil, err
		}
		kWT, err := transposeWeight(prefix+"self_attn.k_proj.weight", kW)
		if err != nil {
			return nil, nil, err
		}
		vWT, err := transposeWeight(prefix+"self_attn.v_proj.weight", vW)
		if err != nil {
			return nil, nil, err
		}
		oWT, err := transposeWeight(prefix+"self_attn.o_proj.weight", oW)
		if err != nil {
			return nil, nil, err
		}

		wq := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.q_proj.weight", qWT)), nil,
		)
		wk := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.k_proj.weight", kWT)), nil,
		)
		wv := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.v_proj.weight", vWT)), nil,
		)
		wo := core.NewDenseFromParams(
			core.NewLinearFromParam(proxy, pw.Wrap(prefix+"self_attn.o_proj.weight", oWT)), nil,
		)

		rope, err := embeddings.NewRotaryPositionalEmbedding[float32](
			context.Background(), proxy, headDim, cfg.MaxSeqLen,
			embeddings.WithRotaryBase(cfg.RopeTheta),
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d rope: %w", i, err)
		}

		gqa, err := attention.NewGroupedQueryAttentionFromParams[float32](
			proxy, ops, cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads,
			wq, wk, wv, wo, rope, headDim,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d gqa: %w", i, err)
		}
		gqa.LayerIndex = i
		attnOut := builder.AddNode(gqa, normed)

		// --- Fused Residual Add + Pre-FFN RMSNorm ---
		postNormW, err := tl.Lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// --- FFN (SwiGLU, composed from layers/) ---
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

		ffn, err := core.NewFFN[float32](
			prefix+"mlp", proxy, ops,
			cfg.HiddenSize, cfg.IntermediateSize, cfg.HiddenSize,
			core.WithSwiGLU[float32](),
			core.WithFFNNoBias[float32](),
		)
		if err != nil {
			return nil, nil, fmt.Errorf("layer %d ffn: %w", i, err)
		}

		gateWT, err := transposeWeight(prefix+"mlp.gate_proj.weight", gateW)
		if err != nil {
			return nil, nil, err
		}
		upWT, err := transposeWeight(prefix+"mlp.up_proj.weight", upW)
		if err != nil {
			return nil, nil, err
		}
		downWT, err := transposeWeight(prefix+"mlp.down_proj.weight", downW)
		if err != nil {
			return nil, nil, err
		}

		ffnParams := ffn.Parameters()
		ffnParams[0].Value = gateWT // w1 = gate_proj
		ffnParams[1].Value = downWT // w2 = down_proj
		ffnParams[2].Value = upWT   // w3 = up_proj

		ffnOut := builder.AddNode(ffn, normed2)

		// Residual add.
		resAdd := &residualAddNode[float32]{engine: proxy, source: fusedNode}
		hidden = builder.AddNode(resAdd, ffnOut)
	}

	// --- Final RMSNorm ---
	finalNormWeight, err := tl.Lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNorm, err := normalization.NewRMSNormFromParam[float32](
		proxy, ops, rmsEps, pw.Wrap("model.norm.weight", finalNormWeight),
	)
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

// mmProjectorNode implements the LLaVA multi-modal projector.
// For "linear" type: out = x * W0^T + b0
// For "mlp" type: out = GELU(x * W0^T + b0) * W2^T + b2
type mmProjectorNode[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	w0, w2     *tensor.TensorNumeric[T]
	b0, b2     *tensor.TensorNumeric[T]
	useMLP     bool
	textHidden int
}

func (p *mmProjectorNode[T]) OpType() string                    { return "MMProjector" }
func (p *mmProjectorNode[T]) Attributes() map[string]any        { return nil }
func (p *mmProjectorNode[T]) OutputShape() []int                { return nil }
func (p *mmProjectorNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (p *mmProjectorNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	frozen := []*tensor.TensorNumeric[T]{p.w0, p.b0}
	if p.useMLP && p.w2 != nil && p.b2 != nil {
		frozen = append(frozen, p.w2, p.b2)
	}
	return frozen
}

func (p *mmProjectorNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0] // [batch, seqLen, visionHidden]
	shape := input.Shape()
	batch := shape[0]
	seqLen := shape[1]
	inDim := shape[2]

	// First linear layer: [batch*seqLen, inDim] * [outDim, inDim]^T + bias.
	w0Data := p.w0.Data()
	b0Data := p.b0.Data()
	outDim := p.w0.Shape()[0] // w0 shape: [outDim, inDim]
	inData := input.Data()

	out := make([]T, batch*seqLen*outDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < outDim; o++ {
				var sum T
				for d := 0; d < inDim; d++ {
					xIdx := b*seqLen*inDim + s*inDim + d
					wIdx := o*inDim + d
					sum = p.ops.Add(sum, p.ops.Mul(inData[xIdx], w0Data[wIdx]))
				}
				oIdx := b*seqLen*outDim + s*outDim + o
				out[oIdx] = p.ops.Add(sum, b0Data[o])
			}
		}
	}

	if !p.useMLP {
		return tensor.New[T]([]int{batch, seqLen, outDim}, out)
	}

	// Apply GELU activation.
	// TODO(T124.2.3): delegate to layers/activations.NewGelu once the
	// projector operates on engine-resident tensors. Same situation as
	// arch_voxtral: this is a raw scalar Go loop on []T, not an engine
	// op chain; switching requires the projector to be ported first.
	half := p.ops.FromFloat64(0.5)
	one := p.ops.One()
	coeff := p.ops.FromFloat64(0.044715)
	sqrtTwoOverPi := p.ops.FromFloat64(1.1283791670955126) // sqrt(2/pi) precomputed
	for i, v := range out {
		x3 := p.ops.Mul(v, p.ops.Mul(v, v))
		inner := p.ops.Mul(sqrtTwoOverPi, p.ops.Add(v, p.ops.Mul(coeff, x3)))
		out[i] = p.ops.Mul(half, p.ops.Mul(v, p.ops.Add(one, p.ops.Tanh(inner))))
	}

	// Second linear layer.
	w2Data := p.w2.Data()
	b2Data := p.b2.Data()
	outDim2 := p.w2.Shape()[0]

	out2 := make([]T, batch*seqLen*outDim2)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < outDim2; o++ {
				var sum T
				for d := 0; d < outDim; d++ {
					xIdx := b*seqLen*outDim + s*outDim + d
					wIdx := o*outDim + d
					sum = p.ops.Add(sum, p.ops.Mul(out[xIdx], w2Data[wIdx]))
				}
				oIdx := b*seqLen*outDim2 + s*outDim2 + o
				out2[oIdx] = p.ops.Add(sum, b2Data[o])
			}
		}
	}

	return tensor.New[T]([]int{batch, seqLen, outDim2}, out2)
}

func (p *mmProjectorNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// loadCLIPWeights maps GGUF tensor data onto CLIPEncoder parameters.
func loadCLIPWeights(
	enc *vision.CLIPEncoder[float32],
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg vision.CLIPEncoderConfig,
) error {
	params := enc.Parameters()

	// Expected order from Parameters():
	// [0] patch_embed.weight, [1] patch_embed.bias,
	// [2] class_embedding, [3] position_embedding,
	// then per block (12 params each):
	//   [0] ln1.gamma, [1] ln1.beta, [2] q_weight, [3] k_weight,
	//   [4] v_weight, [5] o_weight, [6] ln2.gamma, [7] ln2.beta,
	//   [8] ffn1_weight, [9] ffn1_bias, [10] ffn2_weight, [11] ffn2_bias,
	// then: [0] lnPost.gamma, [1] lnPost.beta.

	expectedCount := 4 + 12*cfg.NumLayers + 2
	if len(params) != expectedCount {
		return fmt.Errorf("expected %d parameters, got %d", expectedCount, len(params))
	}

	ggufOrder := make([]string, 0, expectedCount)

	// Patch embedding.
	ggufOrder = append(ggufOrder,
		"vision.patch_embed.weight",
		"vision.patch_embed.bias",
		"vision.class_embedding",
		"vision.position_embedding",
	)

	// Per-block tensors.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("vision.blocks.%d.", i)
		ggufOrder = append(ggufOrder,
			prefix+"ln1.weight",
			prefix+"ln1.bias",
			prefix+"attn.q_proj.weight",
			prefix+"attn.k_proj.weight",
			prefix+"attn.v_proj.weight",
			prefix+"attn.o_proj.weight",
			prefix+"ln2.weight",
			prefix+"ln2.bias",
			prefix+"mlp.fc1.weight",
			prefix+"mlp.fc1.bias",
			prefix+"mlp.fc2.weight",
			prefix+"mlp.fc2.bias",
		)
	}

	// Post layer norm.
	ggufOrder = append(ggufOrder,
		"vision.ln_post.weight",
		"vision.ln_post.bias",
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

// Static interface assertions.
var _ graph.EmbeddedFrozenProvider[float32] = (*mmProjectorNode[float32])(nil)
