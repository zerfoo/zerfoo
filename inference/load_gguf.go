package inference

import (
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// LoadFile loads a model from a local GGUF file and returns a ready-to-use Model.
func LoadFile(path string, opts ...Option) (*Model, error) {
	o := &loadOptions{device: "cpu", mmap: true}
	for _, opt := range opts {
		opt(o)
	}

	// Auto-disable mmap on CUDA devices. MmapStorage has alignment issues
	// on ARM64 (Grace Hopper) and the mmap dequantization path differs
	// numerically from the heap path. Heap loading is slightly slower to
	// start but produces correct results on GPU.
	if strings.HasPrefix(o.device, "cuda") && o.mmap {
		o.mmap = false
		slog.Info("auto-disabled mmap for CUDA device", "device", o.device)
	}

	// Load and parse the GGUF file.
	var gm *GGUFModel
	var mmapCloser io.Closer
	if o.mmap {
		var err error
		gm, mmapCloser, err = LoadGGUFMmap(path)
		if err != nil {
			return nil, fmt.Errorf("mmap load: %w", err)
		}
		slog.Info("loaded GGUF via mmap", "path", path, "tensors", len(gm.Tensors))
	} else {
		var err error
		gm, err = LoadGGUF(path)
		if err != nil {
			return nil, err
		}
	}

	// Extract tokenizer from GGUF metadata.
	tok, err := gguf.ExtractTokenizer(gm.File)
	if err != nil {
		return nil, fmt.Errorf("extract tokenizer: %w", err)
	}

	// Create compute engine.
	eng, err := createEngine(o.device)
	if err != nil {
		return nil, fmt.Errorf("create engine (%s): %w", o.device, err)
	}

	// Apply compute precision if requested.
	applyDType(eng, o.dtype)

	// Quantize weights to FP8 if requested. Must happen before buildArchGraph.
	if o.dtype == "fp8" {
		slog.Info("Quantizing weights to FP8 E4M3...")
		if _, err := gguf.QuantizeToFP8E4M3(gm.Tensors); err != nil {
			return nil, fmt.Errorf("FP8 quantization: %w", err)
		}
	}

	// Fuse QuaRot Hadamard rotation into weights if requested.
	// Must happen after FP8 quantization and before graph building so the
	// rotated weights are used directly — zero runtime cost.
	if o.quarot {
		if err := FuseQuaRotWeights(gm.Tensors, gm.Config.NumLayers); err != nil {
			return nil, fmt.Errorf("quarot weight fusion: %w", err)
		}
	}

	// Build architecture-specific graph.
	g, embWeight, err := buildArchGraph(gm.Config.Architecture, gm.Tensors, gm.Config, eng)
	if err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
	}

	// Derive VocabSize from embedding tensor shape if not set in metadata.
	if gm.Config.VocabSize == 0 {
		if emb, ok := gm.Tensors["model.embed_tokens.weight"]; ok {
			gm.Config.VocabSize = emb.Shape()[0]
		}
	}

	// Build metadata.
	meta := gm.ToModelMetadata()
	special := tok.SpecialTokens()
	meta.BOSTokenID = special.BOS
	meta.EOSTokenID = special.EOS

	// Upload model weights to GPU if the engine supports it.
	if uploader, ok := eng.(compute.WeightUploader); ok {
		tensors := g.ConstantTensors()
		if embWeight != nil {
			tensors = append(tensors, embWeight)
		}
		// Also upload layer parameters (e.g. RMSNorm gain weights).
		for _, p := range g.Parameters() {
			if p.Value != nil {
				tensors = append(tensors, p.Value)
			}
		}
		if err := uploader.UploadWeights(tensors); err != nil {
			return nil, fmt.Errorf("upload weights to GPU: %w", err)
		}
	}

	// Capture embedding weights for the Embed() method.
	var embData []float32
	var embHiddenSize int
	if embWeight != nil {
		embData = embWeight.Data()
		shape := embWeight.Shape()
		if len(shape) >= 2 {
			embHiddenSize = shape[1]
		}
	}

	maxSeqLen := meta.MaxPositionEmbeddings
	if o.maxSeqLen > 0 {
		maxSeqLen = o.maxSeqLen
	}

	var genOpts []generate.GeneratorOption
	if o.kvDtype == "fp16" {
		genOpts = append(genOpts, generate.WithGeneratorKVDtype("fp16"))
	}

	// PJRT compilation: when a plugin path is set, compile the graph via PJRT
	// instead of using the standard Engine compilation path.
	var pjrtPlan *graph.PJRTPlan[float32]
	if o.pjrtPlugin != "" {
		sampleInput, stErr := tensor.New[float32]([]int{1, 1}, []float32{1})
		if stErr != nil {
			return nil, fmt.Errorf("create PJRT sample input: %w", stErr)
		}
		plan, pErr := compilePJRT(o.pjrtPlugin, g, sampleInput)
		if pErr != nil {
			return nil, pErr
		}
		pjrtPlan = plan
		genOpts = append(genOpts, generate.WithPJRTPlan(plan))
	}

	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  meta.VocabSize,
		MaxSeqLen:  maxSeqLen,
		EOSTokenID: meta.EOSTokenID,
		BOSTokenID: meta.BOSTokenID,
		NumLayers:  meta.NumLayers,
	}, genOpts...)

	poolSize := o.sessionPoolSize
	if poolSize < 1 {
		poolSize = defaultSessionPoolSize
	}
	pool := make(chan *generate.InferenceSession[float32], poolSize)
	pool <- gen.NewSession() // Pre-warm with one session for CUDA graph address reuse.

	mdl := &Model{
		generator:   gen,
		tokenizer:   tok,
		engine:      eng,
		config:      *meta,
		closer:      mmapCloser, // nil when heap-loaded; releases mmap on Close()
		embWeights:  embData,
		hiddenSize:  embHiddenSize,
		sessionPool: pool,
		pjrtPlan:    pjrtPlan,
	}
	return mdl, nil
}

// BuildArchGraph dispatches to the appropriate architecture-specific graph
// builder. Exported for benchmark and integration tests that construct
// synthetic weight maps without loading from GGUF files.
func BuildArchGraph(
	arch string,
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	return buildArchGraph(arch, tensors, cfg, engine)
}

// buildArchGraph dispatches to the appropriate architecture-specific graph builder.
func buildArchGraph(
	arch string,
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	switch arch {
	case "llama":
		// Mistral models report arch="llama" in GGUF metadata but use
		// sliding window attention. Detect this by checking for a non-zero
		// SlidingWindow with SlidingWindowPattern==0 (pattern>0 indicates
		// Gemma 3, which has its own builder).
		if cfg.SlidingWindow > 0 && cfg.SlidingWindowPattern == 0 {
			return buildMistralGraph(tensors, cfg, engine)
		}
		return buildLlamaGraph(tensors, cfg, engine)
	case "gemma", "gemma3":
		return buildGemmaGraph(tensors, cfg, engine)
	case "qwen2":
		return buildQwenGraph(tensors, cfg, engine)
	case "mistral":
		return buildMistralGraph(tensors, cfg, engine)
	case "phi3", "phi":
		return buildPhiGraph(tensors, cfg, engine)
	case "deepseek_v3", "deepseek2":
		return buildDeepSeekGraph(tensors, cfg, engine)
	case "mamba":
		return buildMambaGraph(tensors, cfg, engine)
	case "mamba3":
		return buildMamba3Graph(tensors, cfg, engine)
	case "jamba":
		return buildJambaGraph(tensors, cfg, engine)
	case "whisper":
		return buildWhisperGraph(tensors, cfg, engine)
	case "bert":
		return buildBertGraph(tensors, cfg, engine)
	case "llama4":
		return buildLlama4Graph(tensors, cfg, engine)
	case "llava":
		return buildLLaVAGraph(tensors, cfg, engine)
	case "qwen_vl":
		return buildQwenVLGraph(tensors, cfg, engine)
	case "voxtral":
		return buildVoxtralGraph(tensors, cfg, engine)
	default:
		// Fall through to registry for dynamically registered architectures.
		builder, ok := GetArchitecture(arch)
		if !ok {
			return nil, nil, fmt.Errorf("unsupported architecture %q (registered: %v)", arch, ListArchitectures())
		}
		return builder(tensors, cfg, engine)
	}
}

