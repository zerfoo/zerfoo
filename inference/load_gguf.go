package inference

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/zerfoo/tensor"
)

// LoadFile loads a model from a local GGUF file and returns a ready-to-use Model.
func LoadFile(path string, opts ...Option) (*Model, error) {
	o := &loadOptions{device: "cpu"}
	for _, opt := range opts {
		opt(o)
	}

	// Load and parse the GGUF file.
	gm, err := LoadGGUF(path)
	if err != nil {
		return nil, err
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

	// Build architecture-specific graph.
	g, embWeight, err := buildArchGraph(gm.Config.Architecture, gm.Tensors, gm.Config, eng)
	if err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
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
		if err := uploader.UploadWeights(tensors); err != nil {
			return nil, fmt.Errorf("upload weights to GPU: %w", err)
		}
	}

	// Set embedding weight on the generator for token lookup.
	_ = embWeight

	maxSeqLen := meta.MaxPositionEmbeddings
	if o.maxSeqLen > 0 {
		maxSeqLen = o.maxSeqLen
	}

	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  meta.VocabSize,
		MaxSeqLen:  maxSeqLen,
		EOSTokenID: meta.EOSTokenID,
		BOSTokenID: meta.BOSTokenID,
		NumLayers:  meta.NumLayers,
	})

	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config:    *meta,
	}, nil
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
		return buildLlamaGraph(tensors, cfg, engine)
	case "gemma", "gemma3":
		return buildGemmaGraph(tensors, cfg, engine)
	default:
		return nil, nil, fmt.Errorf("unsupported architecture %q", arch)
	}
}

