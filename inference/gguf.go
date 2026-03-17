package inference

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/zerfoo/zerfoo/model/gguf"
	"github.com/zerfoo/ztensor/tensor"
)

// GGUFModel holds a loaded GGUF model's configuration and tensors.
// This is an intermediate representation; full inference requires an
// architecture-specific graph builder to convert these into a computation graph.
type GGUFModel struct {
	Config  *gguf.ModelConfig
	Tensors map[string]*tensor.TensorNumeric[float32]
	File    *gguf.File
}

// LoadGGUF loads a GGUF model file and returns its configuration and tensors.
// Tensor names are mapped from GGUF convention (blk.N.attn_q.weight) to
// Zerfoo canonical names (model.layers.N.self_attn.q_proj.weight).
func LoadGGUF(path string) (*GGUFModel, error) {
	f, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, fmt.Errorf("open GGUF file: %w", err)
	}
	defer func() { _ = f.Close() }()

	gf, err := gguf.Parse(f)
	if err != nil {
		return nil, fmt.Errorf("parse GGUF: %w", err)
	}

	cfg, err := gguf.ExtractModelConfig(gf)
	if err != nil {
		return nil, fmt.Errorf("extract model config: %w", err)
	}

	rawTensors, err := gguf.LoadTensors(gf, f)
	if err != nil {
		return nil, fmt.Errorf("load tensors: %w", err)
	}

	// Map GGUF tensor names to canonical Zerfoo names.
	mapped := make(map[string]*tensor.TensorNumeric[float32], len(rawTensors))
	for name, t := range rawTensors {
		canonical := gguf.MapTensorName(cfg.Architecture, name)
		mapped[canonical] = t
	}

	// Split merged QKV tensors (e.g., Phi attn_qkv.weight) into separate Q/K/V.
	if err := gguf.SplitMergedQKV(mapped, cfg); err != nil {
		return nil, fmt.Errorf("split merged QKV: %w", err)
	}

	// Split merged gate+up MLP tensors (e.g., Phi ffn_up with gate+up concatenated).
	if err := gguf.SplitMergedGateUp(mapped, cfg); err != nil {
		return nil, fmt.Errorf("split merged gate+up: %w", err)
	}

	return &GGUFModel{
		Config:  cfg,
		Tensors: mapped,
		File:    gf,
	}, nil
}

// ToModelMetadata converts a GGUF model config to inference.ModelMetadata.
func (m *GGUFModel) ToModelMetadata() *ModelMetadata {
	return &ModelMetadata{
		Architecture:          m.Config.Architecture,
		VocabSize:             m.Config.VocabSize,
		HiddenSize:            m.Config.HiddenSize,
		NumLayers:             m.Config.NumLayers,
		MaxPositionEmbeddings: m.Config.MaxSeqLen,
		NumQueryHeads:         m.Config.NumHeads,
		NumKeyValueHeads:      m.Config.NumKVHeads,
		IntermediateSize:      m.Config.IntermediateSize,
		RopeTheta:             m.Config.RopeTheta,
		ChatTemplate:          chatTemplateForArch(m.Config.Architecture),
	}
}

// chatTemplateForArch returns the chat template name for a GGUF architecture.
func chatTemplateForArch(arch string) string {
	switch arch {
	case "gemma", "gemma3":
		return "gemma"
	case "llama":
		return "llama"
	case "mistral":
		return "mistral"
	case "qwen2":
		return "qwen2"
	case "deepseek":
		return "deepseek"
	case "phi3":
		return "phi"
	default:
		return ""
	}
}
