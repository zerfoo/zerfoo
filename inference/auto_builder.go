package inference

import (
	"fmt"
	"math"
	"strings"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// autoFeatures holds architecture-specific transformer features detected from
// GGUF metadata. These map directly to transformerGraphOpts fields.
type autoFeatures struct {
	embedScale          float32
	postNorm            bool
	qkNorm              bool
	logitSoftcap        float32
	slidingWindowSize   int
	attnBias            bool
	partialRotaryFactor float32
}

// detectFeatures examines GGUF ModelConfig metadata and determines which
// transformer features are needed for the architecture. Known architectures
// get their canonical feature set; unknown architectures get a plain
// decoder-only transformer with no special features.
func detectFeatures(cfg *gguf.ModelConfig) autoFeatures {
	var f autoFeatures
	arch := strings.ToLower(cfg.Architecture)

	switch arch {
	case "gemma", "gemma2":
		f.embedScale = float32(math.Sqrt(float64(cfg.HiddenSize)))

	case "gemma3", "gemma3n":
		f.embedScale = float32(math.Sqrt(float64(cfg.HiddenSize)))
		f.postNorm = true
		f.qkNorm = true
		f.logitSoftcap = cfg.LogitSoftcap

	case "qwen2":
		f.attnBias = true

	case "mistral":
		f.slidingWindowSize = cfg.SlidingWindow

	case "phi", "phi3":
		f.partialRotaryFactor = cfg.PartialRotaryFactor
	}

	// For architectures that report sliding window in metadata but use the
	// "llama" arch name (e.g. Mistral GGUF files), detect from config fields.
	if arch == "llama" && cfg.SlidingWindow > 0 && cfg.SlidingWindowPattern == 0 {
		f.slidingWindowSize = cfg.SlidingWindow
	}

	return f
}

// AutoBuild reads GGUF metadata from cfg and constructs the appropriate
// computation graph automatically, without requiring a hand-written
// per-model builder. It detects architecture features from metadata and
// delegates to the shared buildTransformerGraph for standard decoder-only
// transformer architectures.
//
// For non-transformer architectures (Mamba, Whisper, etc.) that have a
// registered ArchBuilder, AutoBuild falls back to that builder.
//
// For completely unknown architectures with standard decoder-only tensor
// names, AutoBuild constructs a plain transformer graph.
func AutoBuild(
	tensors map[string]*tensor.TensorNumeric[float32],
	cfg *gguf.ModelConfig,
	engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
	// Non-transformer architectures cannot be auto-built from metadata alone.
	// Delegate to their registered builder if one exists.
	if isNonTransformer(cfg.Architecture) {
		builder, ok := GetArchitecture(cfg.Architecture)
		if !ok {
			return nil, nil, fmt.Errorf("non-transformer architecture %q has no registered builder", cfg.Architecture)
		}
		return builder(tensors, cfg, engine)
	}

	embedWeight, ok := tensors["model.embed_tokens.weight"]
	if !ok {
		return nil, nil, fmt.Errorf("missing tensor %q", "model.embed_tokens.weight")
	}

	// Determine LM head weight: use separate weight if available, otherwise
	// tie to embedding table.
	lmHeadWeight, ok := tensors["lm_head.weight"]
	if !ok {
		lmHeadWeight = embedWeight
	}

	features := detectFeatures(cfg)
	opts := transformerGraphOpts{
		embedScale:          features.embedScale,
		postNorm:            features.postNorm,
		qkNorm:              features.qkNorm,
		logitSoftcap:        features.logitSoftcap,
		slidingWindowSize:   features.slidingWindowSize,
		attnBias:            features.attnBias,
		partialRotaryFactor: features.partialRotaryFactor,
	}

	g, err := buildTransformerGraph(tensors, cfg, engine, embedWeight, lmHeadWeight, opts)
	if err != nil {
		return nil, nil, err
	}

	return g, embedWeight, nil
}

// nonTransformerArchs lists architectures that use fundamentally different
// computation graphs (SSM, encoder-decoder, etc.) and cannot be auto-built
// from the standard decoder-only transformer template.
var nonTransformerArchs = map[string]bool{
	"mamba":   true,
	"mamba3":  true,
	"jamba":   true,
	"whisper": true,
	"rwkv":    true,
}

// isNonTransformer returns true if the architecture requires a specialized
// graph builder that cannot be generated from metadata alone.
func isNonTransformer(arch string) bool {
	return nonTransformerArchs[strings.ToLower(arch)]
}
