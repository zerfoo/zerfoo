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
	"github.com/zerfoo/zerfoo/layers/vision"
	"github.com/zerfoo/zerfoo/model/gguf"
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

	// Load multi-modal projector weights.
	projW0, ok := tensors["mm_projector.0.weight"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "mm_projector.0.weight")
	}
	projB0, ok := tensors["mm_projector.0.bias"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "mm_projector.0.bias")
	}

	var projW2, projB2 *tensor.TensorNumeric[float32]
	if lc.ProjectorType == "mlp" {
		var okW, okB bool
		projW2, okW = tensors["mm_projector.2.weight"]
		projB2, okB = tensors["mm_projector.2.bias"]
		if !okW {
			return nil, nil, fmt.Errorf("missing tensor %q", "mm_projector.2.weight")
		}
		if !okB {
			return nil, nil, fmt.Errorf("missing tensor %q", "mm_projector.2.bias")
		}
	}

	// Get text model embedding weight.
	embedWeight, ok := tensors["model.embed_tokens.weight"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "model.embed_tokens.weight")
	}

	lmHeadWeight, ok := tensors["lm_head.weight"]
	if !ok {
		lmHeadWeight = embedWeight
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

	// Text decoder: build the Llama transformer on the projected vision tokens.
	// The LLaVA decoder takes projected vision embeddings as input instead of
	// token embeddings. We use a passthrough embedding node.
	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	hidden := projectedVision

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

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("model.layers.%d.", i)

		inputNormW, err := lookup(prefix + "input_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		inputNorm, err := newRMSNormNode(proxy, ops, rmsEps, param(prefix+"input_layernorm.weight", inputNormW))
		if err != nil {
			return nil, nil, err
		}
		normed := builder.AddNode(inputNorm, hidden)

		// GQA attention (same as Llama).
		qW, err := lookup(prefix + "self_attn.q_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		kW, err := lookup(prefix + "self_attn.k_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		vW, err := lookup(prefix + "self_attn.v_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		oW, err := lookup(prefix + "self_attn.o_proj.weight")
		if err != nil {
			return nil, nil, err
		}

		attnNode := &llamaAttnNode[float32]{
			engine:    proxy,
			ops:       ops,
			qW:        qW,
			kW:        kW,
			vW:        vW,
			oW:        oW,
			numHeads:  cfg.NumHeads,
			numKVHeads: cfg.NumKVHeads,
			headDim:   headDim,
			ropeTheta: cfg.RopeTheta,
			maxSeqLen: cfg.MaxSeqLen,
		}
		attnOut := builder.AddNode(attnNode, normed)

		// Fused residual add + pre-FFN RMSNorm.
		postNormW, err := lookup(prefix + "post_attention_layernorm.weight")
		if err != nil {
			return nil, nil, err
		}
		fusedNode := &fusedAddRMSNormNode[float32]{engine: proxy, weight: postNormW, eps: rmsEps}
		normed2 := builder.AddNode(fusedNode, attnOut, hidden)

		// FFN (SwiGLU).
		gateW, err := lookup(prefix + "mlp.gate_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		upW, err := lookup(prefix + "mlp.up_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		downW, err := lookup(prefix + "mlp.down_proj.weight")
		if err != nil {
			return nil, nil, err
		}
		ffnNode := &llamaFFNNode[float32]{
			engine: proxy,
			ops:    ops,
			gateW:  gateW,
			upW:    upW,
			downW:  downW,
		}
		ffnOut := builder.AddNode(ffnNode, normed2)

		// Residual add.
		resAdd := &residualAddNode[float32]{engine: proxy, source: fusedNode}
		hidden = builder.AddNode(resAdd, ffnOut)
	}

	// Final RMSNorm.
	finalNormWeight, err := lookup("model.norm.weight")
	if err != nil {
		return nil, nil, err
	}
	finalNorm, err := newRMSNormNode(proxy, ops, rmsEps, param("model.norm.weight", finalNormWeight))
	if err != nil {
		return nil, nil, err
	}
	normedFinal := builder.AddNode(finalNorm, hidden)

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

func (p *mmProjectorNode[T]) OpType() string                  { return "MMProjector" }
func (p *mmProjectorNode[T]) Attributes() map[string]any       { return nil }
func (p *mmProjectorNode[T]) OutputShape() []int               { return nil }
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
	half := p.ops.FromFloat64(0.5)
	one := p.ops.One()
	coeff := p.ops.FromFloat64(0.044715)
	sqrtTwoOverPi := p.ops.FromFloat64(math.Sqrt(2.0 / math.Pi))
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

// llamaAttnNode implements grouped-query attention with RoPE for the LLaVA text decoder.
// This is a simplified attention node that operates on vision token sequences.
type llamaAttnNode[T tensor.Numeric] struct {
	engine     compute.Engine[T]
	ops        numeric.Arithmetic[T]
	qW, kW, vW, oW *tensor.TensorNumeric[T]
	numHeads   int
	numKVHeads int
	headDim    int
	ropeTheta  float64
	maxSeqLen  int
}

func (a *llamaAttnNode[T]) OpType() string                  { return "LLaVAAttn" }
func (a *llamaAttnNode[T]) Attributes() map[string]any       { return nil }
func (a *llamaAttnNode[T]) OutputShape() []int               { return nil }
func (a *llamaAttnNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (a *llamaAttnNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{a.qW, a.kW, a.vW, a.oW}
}

func (a *llamaAttnNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0] // [batch, seqLen, hiddenDim]
	shape := input.Shape()
	batch := shape[0]
	seqLen := shape[1]
	hiddenDim := shape[2]

	inData := input.Data()
	qWData := a.qW.Data()
	kWData := a.kW.Data()
	vWData := a.vW.Data()
	oWData := a.oW.Data()

	kvDim := a.numKVHeads * a.headDim
	qDim := a.numHeads * a.headDim

	// Project Q, K, V.
	q := make([]T, batch*seqLen*qDim)
	k := make([]T, batch*seqLen*kvDim)
	v := make([]T, batch*seqLen*kvDim)

	// Q = input * qW^T  (qW: [qDim, hiddenDim])
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < qDim; o++ {
				var sum T
				for d := 0; d < hiddenDim; d++ {
					sum = a.ops.Add(sum, a.ops.Mul(inData[b*seqLen*hiddenDim+s*hiddenDim+d], qWData[o*hiddenDim+d]))
				}
				q[b*seqLen*qDim+s*qDim+o] = sum
			}
			for o := 0; o < kvDim; o++ {
				var sumK, sumV T
				for d := 0; d < hiddenDim; d++ {
					xVal := inData[b*seqLen*hiddenDim+s*hiddenDim+d]
					sumK = a.ops.Add(sumK, a.ops.Mul(xVal, kWData[o*hiddenDim+d]))
					sumV = a.ops.Add(sumV, a.ops.Mul(xVal, vWData[o*hiddenDim+d]))
				}
				k[b*seqLen*kvDim+s*kvDim+o] = sumK
				v[b*seqLen*kvDim+s*kvDim+o] = sumV
			}
		}
	}

	// Apply RoPE to Q and K.
	applyRoPE(q, batch, seqLen, a.numHeads, a.headDim, a.ropeTheta, a.ops)
	applyRoPE(k, batch, seqLen, a.numKVHeads, a.headDim, a.ropeTheta, a.ops)

	// GQA: compute attention.
	scale := T(1.0 / math.Sqrt(float64(a.headDim)))
	kvGroupSize := a.numHeads / a.numKVHeads

	attnOut := make([]T, batch*seqLen*qDim)
	for b := 0; b < batch; b++ {
		for h := 0; h < a.numHeads; h++ {
			kvH := h / kvGroupSize
			scores := make([]T, seqLen*seqLen)
			for qi := 0; qi < seqLen; qi++ {
				for ki := 0; ki < seqLen; ki++ {
					var dot T
					for d := 0; d < a.headDim; d++ {
						qIdx := b*seqLen*qDim + qi*qDim + h*a.headDim + d
						kIdx := b*seqLen*kvDim + ki*kvDim + kvH*a.headDim + d
						dot = a.ops.Add(dot, a.ops.Mul(q[qIdx], k[kIdx]))
					}
					scores[qi*seqLen+ki] = a.ops.Mul(dot, scale)
				}
			}

			// Softmax.
			for qi := 0; qi < seqLen; qi++ {
				maxVal := scores[qi*seqLen]
				for ki := 1; ki < seqLen; ki++ {
					if a.ops.GreaterThan(scores[qi*seqLen+ki], maxVal) {
						maxVal = scores[qi*seqLen+ki]
					}
				}
				var sumExp T
				for ki := 0; ki < seqLen; ki++ {
					scores[qi*seqLen+ki] = a.ops.Exp(a.ops.Sub(scores[qi*seqLen+ki], maxVal))
					sumExp = a.ops.Add(sumExp, scores[qi*seqLen+ki])
				}
				for ki := 0; ki < seqLen; ki++ {
					scores[qi*seqLen+ki] = a.ops.Div(scores[qi*seqLen+ki], sumExp)
				}

				for d := 0; d < a.headDim; d++ {
					var val T
					for ki := 0; ki < seqLen; ki++ {
						vIdx := b*seqLen*kvDim + ki*kvDim + kvH*a.headDim + d
						val = a.ops.Add(val, a.ops.Mul(scores[qi*seqLen+ki], v[vIdx]))
					}
					attnOut[b*seqLen*qDim+qi*qDim+h*a.headDim+d] = val
				}
			}
		}
	}

	// Output projection: attnOut * oW^T  (oW: [hiddenDim, qDim])
	out := make([]T, batch*seqLen*hiddenDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < hiddenDim; o++ {
				var sum T
				for d := 0; d < qDim; d++ {
					sum = a.ops.Add(sum, a.ops.Mul(attnOut[b*seqLen*qDim+s*qDim+d], oWData[o*qDim+d]))
				}
				out[b*seqLen*hiddenDim+s*hiddenDim+o] = sum
			}
		}
	}

	return tensor.New[T]([]int{batch, seqLen, hiddenDim}, out)
}

func (a *llamaAttnNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// llamaFFNNode implements SwiGLU FFN for the LLaVA text decoder.
type llamaFFNNode[T tensor.Numeric] struct {
	engine                compute.Engine[T]
	ops                   numeric.Arithmetic[T]
	gateW, upW, downW     *tensor.TensorNumeric[T]
}

func (f *llamaFFNNode[T]) OpType() string                  { return "LLaVAFFN" }
func (f *llamaFFNNode[T]) Attributes() map[string]any       { return nil }
func (f *llamaFFNNode[T]) OutputShape() []int               { return nil }
func (f *llamaFFNNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (f *llamaFFNNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	return []*tensor.TensorNumeric[T]{f.gateW, f.upW, f.downW}
}

func (f *llamaFFNNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	input := inputs[0] // [batch, seqLen, hiddenDim]
	shape := input.Shape()
	batch := shape[0]
	seqLen := shape[1]
	hiddenDim := shape[2]
	interDim := f.gateW.Shape()[0]

	inData := input.Data()
	gateData := f.gateW.Data()
	upData := f.upW.Data()
	downData := f.downW.Data()

	// gate = input * gateW^T, up = input * upW^T
	gate := make([]T, batch*seqLen*interDim)
	up := make([]T, batch*seqLen*interDim)

	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < interDim; o++ {
				var sumG, sumU T
				for d := 0; d < hiddenDim; d++ {
					xVal := inData[b*seqLen*hiddenDim+s*hiddenDim+d]
					sumG = f.ops.Add(sumG, f.ops.Mul(xVal, gateData[o*hiddenDim+d]))
					sumU = f.ops.Add(sumU, f.ops.Mul(xVal, upData[o*hiddenDim+d]))
				}
				idx := b*seqLen*interDim + s*interDim + o
				// SiLU(gate) * up
				one := f.ops.One()
				negG := f.ops.Mul(f.ops.FromFloat64(-1.0), sumG)
				sigmoid := f.ops.Div(one, f.ops.Add(one, f.ops.Exp(negG)))
				gate[idx] = f.ops.Mul(f.ops.Mul(sumG, sigmoid), sumU)
				up[idx] = gate[idx] // reuse for down projection input
			}
		}
	}

	// down = gate * downW^T  (downW: [hiddenDim, interDim])
	out := make([]T, batch*seqLen*hiddenDim)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for o := 0; o < hiddenDim; o++ {
				var sum T
				for d := 0; d < interDim; d++ {
					sum = f.ops.Add(sum, f.ops.Mul(gate[b*seqLen*interDim+s*interDim+d], downData[o*interDim+d]))
				}
				out[b*seqLen*hiddenDim+s*hiddenDim+o] = sum
			}
		}
	}

	return tensor.New[T]([]int{batch, seqLen, hiddenDim}, out)
}

func (f *llamaFFNNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// applyRoPE applies rotary positional embeddings in-place.
func applyRoPE[T tensor.Numeric](data []T, batch, seqLen, numHeads, headDim int, theta float64, ops numeric.Arithmetic[T]) {
	dim := numHeads * headDim
	for b := 0; b < batch; b++ {
		for pos := 0; pos < seqLen; pos++ {
			for h := 0; h < numHeads; h++ {
				for d := 0; d < headDim/2; d++ {
					freq := 1.0 / math.Pow(theta, float64(2*d)/float64(headDim))
					angle := float64(pos) * freq
					cosVal := ops.FromFloat64(math.Cos(angle))
					sinVal := ops.FromFloat64(math.Sin(angle))

					idx0 := b*seqLen*dim + pos*dim + h*headDim + 2*d
					idx1 := idx0 + 1
					x0 := data[idx0]
					x1 := data[idx1]

					data[idx0] = ops.Sub(ops.Mul(x0, cosVal), ops.Mul(x1, sinVal))
					data[idx1] = ops.Add(ops.Mul(x1, cosVal), ops.Mul(x0, sinVal))
				}
			}
		}
	}
}

// newRMSNormNode creates an RMSNorm graph node using the normalization layer.
func newRMSNormNode(
	engine compute.Engine[float32],
	ops numeric.Float32Ops,
	eps float32,
	weightParam *graph.Parameter[float32],
) (*rmsNormWrapNode, error) {
	return &rmsNormWrapNode{
		engine: engine,
		ops:    ops,
		weight: weightParam.Value,
		eps:    eps,
	}, nil
}

// rmsNormWrapNode wraps RMSNorm as a graph node for LLaVA.
type rmsNormWrapNode struct {
	engine compute.Engine[float32]
	ops    numeric.Float32Ops
	weight *tensor.TensorNumeric[float32]
	eps    float32
}

func (r *rmsNormWrapNode) OpType() string                        { return "RMSNorm" }
func (r *rmsNormWrapNode) Attributes() map[string]any             { return nil }
func (r *rmsNormWrapNode) OutputShape() []int                     { return nil }
func (r *rmsNormWrapNode) Parameters() []*graph.Parameter[float32] { return nil }

func (r *rmsNormWrapNode) EmbeddedFrozen() []*tensor.TensorNumeric[float32] {
	return []*tensor.TensorNumeric[float32]{r.weight}
}

func (r *rmsNormWrapNode) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	input := inputs[0]
	shape := input.Shape()
	data := input.Data()
	wData := r.weight.Data()

	hiddenDim := shape[len(shape)-1]
	numTokens := len(data) / hiddenDim

	out := make([]float32, len(data))
	for t := 0; t < numTokens; t++ {
		offset := t * hiddenDim
		// Compute RMS.
		var sumSq float32
		for d := 0; d < hiddenDim; d++ {
			v := data[offset+d]
			sumSq += v * v
		}
		rms := float32(math.Sqrt(float64(sumSq/float32(hiddenDim)) + float64(r.eps)))
		invRMS := 1.0 / rms

		for d := 0; d < hiddenDim; d++ {
			out[offset+d] = data[offset+d] * invRMS * wData[d]
		}
	}

	return tensor.New(shape, out)
}

func (r *rmsNormWrapNode) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
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
var _ graph.EmbeddedFrozenProvider[float32] = (*llamaAttnNode[float32])(nil)
var _ graph.EmbeddedFrozenProvider[float32] = (*llamaFFNNode[float32])(nil)
var _ graph.EmbeddedFrozenProvider[float32] = (*rmsNormWrapNode)(nil)
