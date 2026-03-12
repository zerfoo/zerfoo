package gguf

import (
	"fmt"
	"regexp"
	"strings"
)

// ModelConfig holds model configuration extracted from GGUF metadata.
type ModelConfig struct {
	Architecture     string
	Name             string
	VocabSize        int
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	NumKVHeads       int
	IntermediateSize int
	MaxSeqLen        int
	RopeTheta        float64
	HeadDim          int // explicit head dimension (0 = use HiddenSize/NumHeads)
}

// ExtractModelConfig reads GGUF metadata and returns a ModelConfig.
// The architecture field (general.architecture) determines which metadata
// key prefix to use (e.g., "llama." or "gemma.").
func ExtractModelConfig(f *File) (*ModelConfig, error) {
	arch, ok := f.GetString("general.architecture")
	if !ok || arch == "" {
		return nil, fmt.Errorf("missing general.architecture metadata")
	}

	cfg := &ModelConfig{Architecture: arch}

	if name, ok := f.GetString("general.name"); ok {
		cfg.Name = name
	}

	prefix := arch + "."

	if v, ok := f.GetUint32(prefix + "vocab_size"); ok {
		cfg.VocabSize = int(v)
	}
	if v, ok := f.GetUint32(prefix + "embedding_length"); ok {
		cfg.HiddenSize = int(v)
	}
	if v, ok := f.GetUint32(prefix + "block_count"); ok {
		cfg.NumLayers = int(v)
	}
	if v, ok := f.GetUint32(prefix + "attention.head_count"); ok {
		cfg.NumHeads = int(v)
	}
	if v, ok := f.GetUint32(prefix + "attention.head_count_kv"); ok {
		cfg.NumKVHeads = int(v)
	} else {
		// Default KV heads to query heads (MHA).
		cfg.NumKVHeads = cfg.NumHeads
	}
	if v, ok := f.GetUint32(prefix + "feed_forward_length"); ok {
		cfg.IntermediateSize = int(v)
	}
	if v, ok := f.GetUint32(prefix + "context_length"); ok {
		cfg.MaxSeqLen = int(v)
	}
	if v, ok := f.GetFloat32(prefix + "rope.freq_base"); ok {
		cfg.RopeTheta = float64(v)
	}
	// Gemma 3 uses separate global/local RoPE bases instead of a single rope.freq_base.
	// Fall back to the global base for the primary RopeTheta.
	if cfg.RopeTheta == 0 {
		if v, ok := f.GetFloat32(prefix + "rope.global.freq_base"); ok {
			cfg.RopeTheta = float64(v)
		}
	}
	// Extract explicit head dimension from attention.key_length if present.
	// Gemma 3 uses key_length=256 while hidden/heads=288.
	if v, ok := f.GetUint32(prefix + "attention.key_length"); ok {
		cfg.HeadDim = int(v)
	}

	return cfg, nil
}

// blkPattern matches "blk.N." prefix in GGUF tensor names.
var blkPattern = regexp.MustCompile(`^blk\.(\d+)\.(.+)$`)

// tensorNameMap maps GGUF tensor name suffixes (after blk.N.) to HuggingFace names.
var tensorNameMap = map[string]string{
	"attn_norm.weight":           "input_layernorm.weight",
	"attn_q.weight":              "self_attn.q_proj.weight",
	"attn_k.weight":              "self_attn.k_proj.weight",
	"attn_v.weight":              "self_attn.v_proj.weight",
	"attn_output.weight":         "self_attn.o_proj.weight",
	"attn_q_norm.weight":         "self_attn.q_norm.weight",
	"attn_k_norm.weight":         "self_attn.k_norm.weight",
	"ffn_norm.weight":            "post_attention_layernorm.weight",
	"post_attention_norm.weight": "post_attention_layernorm.weight",
	"post_ffw_norm.weight":       "post_feedforward_layernorm.weight",
	"ffn_gate.weight":            "mlp.gate_proj.weight",
	"ffn_up.weight":              "mlp.up_proj.weight",
	"ffn_down.weight":            "mlp.down_proj.weight",
}

// globalTensorMap maps global GGUF tensor names to HuggingFace names.
var globalTensorMap = map[string]string{
	"token_embd.weight":  "model.embed_tokens.weight",
	"output_norm.weight": "model.norm.weight",
	"output.weight":      "lm_head.weight",
}

// MapTensorName converts a GGUF tensor name to the Zerfoo/HuggingFace canonical name.
// Supports llama and gemma architectures (they share the same GGUF naming convention).
// Unknown names pass through unchanged.
func MapTensorName(_ string, ggufName string) string {
	// Check global names first.
	if mapped, ok := globalTensorMap[ggufName]; ok {
		return mapped
	}

	// Check block-level names.
	m := blkPattern.FindStringSubmatch(ggufName)
	if m == nil {
		return ggufName
	}

	layerNum := m[1]
	suffix := m[2]

	if mapped, ok := tensorNameMap[suffix]; ok {
		return "model.layers." + layerNum + "." + mapped
	}

	// Handle bias variants (e.g., attn_q.bias → self_attn.q_proj.bias).
	if strings.HasSuffix(suffix, ".bias") {
		weightSuffix := strings.TrimSuffix(suffix, ".bias") + ".weight"
		if mapped, ok := tensorNameMap[weightSuffix]; ok {
			biasMapped := strings.TrimSuffix(mapped, ".weight") + ".bias"
			return "model.layers." + layerNum + "." + biasMapped
		}
	}

	return ggufName
}
