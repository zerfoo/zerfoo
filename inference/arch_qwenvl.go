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
	RegisterArchitecture("qwen_vl", buildQwenVLGraph)
}

// QwenVLConfig holds Qwen-VL-specific model configuration.
type QwenVLConfig struct {
	// Vision encoder config.
	ImageSize       int
	PatchSize       int
	VisionHiddenDim int
	VisionNumHeads  int
	VisionNumLayers int
	NumChannels     int

	// Multi-modal projector config.
	ProjectorType string // "linear" or "mlp"
}

// QwenVLConfigFromGGUF extracts Qwen-VL configuration from GGUF ModelConfig.
func QwenVLConfigFromGGUF(cfg *gguf.ModelConfig) QwenVLConfig {
	imageSize := 448
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

	return QwenVLConfig{
		ImageSize:       imageSize,
		PatchSize:       patchSize,
		VisionHiddenDim: visionHidden,
		VisionNumHeads:  visionHeads,
		VisionNumLayers: visionLayers,
		NumChannels:     numChannels,
		ProjectorType:   projType,
	}
}

// buildQwenVLGraph constructs a computation graph for the Qwen-VL architecture
// from pre-loaded GGUF tensors. It returns the graph and the embedding table
// tensor (needed by the generator for token lookup).
//
// The Qwen-VL architecture is:
//
//	Image -> CLIP ViT Encoder -> Multi-Modal Projector -> Merged with Text Embeddings -> Qwen2 Decoder -> LM Head
//
// For text-only inference, the graph degrades to a standard Qwen2 model.
// For vision+language, image tokens are projected into the text embedding space
// and concatenated with text token embeddings before the decoder.
//
// The text decoder uses Qwen2 conventions: attention bias on Q/K/V projections
// and RoPE theta defaulting to 1,000,000.
//
// Expected GGUF tensor names for the vision encoder:
//
//	vision.patch_embed.weight                       - [vision_hidden, C*P*P]
//	vision.patch_embed.bias                         - [vision_hidden]
//	vision.class_embedding                          - [vision_hidden]
//	vision.position_embedding                       - [num_patches+1, vision_hidden]
//	vision.blocks.{i}.ln1.weight                    - [vision_hidden]
//	vision.blocks.{i}.ln1.bias                      - [vision_hidden]
//	vision.blocks.{i}.attn.q_proj.weight            - [vision_hidden, vision_hidden]
//	vision.blocks.{i}.attn.k_proj.weight            - [vision_hidden, vision_hidden]
//	vision.blocks.{i}.attn.v_proj.weight            - [vision_hidden, vision_hidden]
//	vision.blocks.{i}.attn.o_proj.weight            - [vision_hidden, vision_hidden]
//	vision.blocks.{i}.ln2.weight                    - [vision_hidden]
//	vision.blocks.{i}.ln2.bias                      - [vision_hidden]
//	vision.blocks.{i}.mlp.fc1.weight                - [4*vision_hidden, vision_hidden]
//	vision.blocks.{i}.mlp.fc1.bias                  - [4*vision_hidden]
//	vision.blocks.{i}.mlp.fc2.weight                - [vision_hidden, 4*vision_hidden]
//	vision.blocks.{i}.mlp.fc2.bias                  - [vision_hidden]
//	vision.ln_post.weight                           - [vision_hidden]
//	vision.ln_post.bias                             - [vision_hidden]
//
// Multi-modal projector (MLP type):
//
//	mm_projector.0.weight                           - [text_hidden, vision_hidden]
//	mm_projector.0.bias                             - [text_hidden]
//	mm_projector.2.weight                           - [text_hidden, text_hidden]
//	mm_projector.2.bias                             - [text_hidden]
//
// Text decoder tensors follow the standard Qwen2 naming convention
// (same as Llama but with attention bias tensors).
func buildQwenVLGraph(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	qc := QwenVLConfigFromGGUF(cfg)
	return BuildQwenVLModel(qc, tensors, cfg, engine)
}

// BuildQwenVLModel constructs the Qwen-VL computation graph from a weight map.
// Exported for benchmark and integration tests that construct synthetic weight maps.
func BuildQwenVLModel(
	qc QwenVLConfig,
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	ops := numeric.Float32Ops{}

	// Build CLIP vision encoder.
	clipCfg := vision.CLIPEncoderConfig{
		ImageSize:   qc.ImageSize,
		PatchSize:   qc.PatchSize,
		HiddenDim:   qc.VisionHiddenDim,
		NumHeads:    qc.VisionNumHeads,
		NumLayers:   qc.VisionNumLayers,
		NumChannels: qc.NumChannels,
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
	if qc.ProjectorType == "mlp" {
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

	// Build the graph.
	proxy := compute.NewEngineProxy[float32](engine)
	builder := graph.NewBuilder[float32](proxy)

	// Input: pixel values [1, C, H, W].
	imageInput := builder.Input([]int{1, qc.NumChannels, qc.ImageSize, qc.ImageSize})

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
		useMLP:     qc.ProjectorType == "mlp",
		textHidden: cfg.HiddenSize,
	}
	projectedVision := builder.AddNode(projNode, visionOut)

	// Text decoder: Qwen2 transformer with attention bias.
	rmsEps := float32(1e-5)
	if cfg.RMSNormEps > 0 {
		rmsEps = cfg.RMSNormEps
	}

	hidden := projectedVision

	headDim := cfg.HiddenSize / cfg.NumHeads
	if cfg.HeadDim > 0 {
		headDim = cfg.HeadDim
	}

	ropeTheta := cfg.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 1000000 // Qwen default
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

		// GQA attention with bias (Qwen2 style).
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

		// Load attention biases (Qwen2 always has them).
		var qB, kB, vB *tensor.TensorNumeric[float32]
		if t, ok := tensors[prefix+"self_attn.q_proj.bias"]; ok {
			qB = t
		}
		if t, ok := tensors[prefix+"self_attn.k_proj.bias"]; ok {
			kB = t
		}
		if t, ok := tensors[prefix+"self_attn.v_proj.bias"]; ok {
			vB = t
		}

		attnNode := &qwenVLAttnNode[float32]{
			engine:     proxy,
			ops:        ops,
			qW:         qW,
			kW:         kW,
			vW:         vW,
			oW:         oW,
			qB:         qB,
			kB:         kB,
			vB:         vB,
			numHeads:   cfg.NumHeads,
			numKVHeads: cfg.NumKVHeads,
			headDim:    headDim,
			ropeTheta:  ropeTheta,
			maxSeqLen:  cfg.MaxSeqLen,
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

// qwenVLAttnNode implements grouped-query attention with RoPE and attention bias
// for the Qwen-VL text decoder. Extends the LLaVA attention node with optional
// Q/K/V bias terms (standard in Qwen2).
type qwenVLAttnNode[T tensor.Numeric] struct {
	engine            compute.Engine[T]
	ops               numeric.Arithmetic[T]
	qW, kW, vW, oW   *tensor.TensorNumeric[T]
	qB, kB, vB        *tensor.TensorNumeric[T] // optional attention biases
	numHeads          int
	numKVHeads        int
	headDim           int
	ropeTheta         float64
	maxSeqLen         int
}

func (a *qwenVLAttnNode[T]) OpType() string                  { return "QwenVLAttn" }
func (a *qwenVLAttnNode[T]) Attributes() map[string]any       { return nil }
func (a *qwenVLAttnNode[T]) OutputShape() []int               { return nil }
func (a *qwenVLAttnNode[T]) Parameters() []*graph.Parameter[T] { return nil }

func (a *qwenVLAttnNode[T]) EmbeddedFrozen() []*tensor.TensorNumeric[T] {
	frozen := []*tensor.TensorNumeric[T]{a.qW, a.kW, a.vW, a.oW}
	if a.qB != nil {
		frozen = append(frozen, a.qB)
	}
	if a.kB != nil {
		frozen = append(frozen, a.kB)
	}
	if a.vB != nil {
		frozen = append(frozen, a.vB)
	}
	return frozen
}

func (a *qwenVLAttnNode[T]) Forward(ctx context.Context, inputs ...*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
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

	// Add attention biases.
	if a.qB != nil {
		biasData := a.qB.Data()
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				for o := 0; o < qDim; o++ {
					idx := b*seqLen*qDim + s*qDim + o
					q[idx] = a.ops.Add(q[idx], biasData[o])
				}
			}
		}
	}
	if a.kB != nil {
		biasData := a.kB.Data()
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				for o := 0; o < kvDim; o++ {
					idx := b*seqLen*kvDim + s*kvDim + o
					k[idx] = a.ops.Add(k[idx], biasData[o])
				}
			}
		}
	}
	if a.vB != nil {
		biasData := a.vB.Data()
		for b := 0; b < batch; b++ {
			for s := 0; s < seqLen; s++ {
				for o := 0; o < kvDim; o++ {
					idx := b*seqLen*kvDim + s*kvDim + o
					v[idx] = a.ops.Add(v[idx], biasData[o])
				}
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

func (a *qwenVLAttnNode[T]) Backward(_ context.Context, _ types.BackwardMode, _ *tensor.TensorNumeric[T], _ ...*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	return nil, nil
}

// Static interface assertions.
var _ graph.EmbeddedFrozenProvider[float32] = (*qwenVLAttnNode[float32])(nil)
