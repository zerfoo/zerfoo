package gguf

import (
	"fmt"
	"log/slog"
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
	HeadDim              int     // explicit head dimension (0 = use HiddenSize/NumHeads)
	LogitSoftcap         float32 // if > 0, apply logit softcapping: cap * tanh(logit/cap)
	LocalRopeTheta       float64 // RoPE base for local/sliding-window layers (0 = use RopeTheta)
	SlidingWindow        int     // sliding window size for local attention layers
	SlidingWindowPattern int     // every Nth layer is global (0 = all global)
	RMSNormEps           float32 // RMSNorm epsilon (0 = use default 1e-5)
	PartialRotaryFactor  float32 // fraction of head dims to apply RoPE (0 = full rotation)

	// DeepSeek MLA (Multi-head Latent Attention) fields.
	KVLoRADim     int // KV compression rank (attention.kv_lora_rank)
	QLoRADim      int // Q compression rank (attention.q_lora_rank)
	QKRopeHeadDim int // RoPE head dimension for Q/K (attention.qk_rope_head_dim)

	// DeepSeek MoE (Mixture of Experts) fields.
	NumExperts         int // number of routed experts (expert_count)
	NumExpertsPerToken int // experts activated per token (expert_used_count)
	NumSharedExperts   int // number of shared experts (expert_shared_count)

	// TransMLA: converted MHA-to-MLA models (see ADR-069).
	TransMLAKVLoraDim int // KV LoRA rank from transmla.kv_lora_dim metadata (0 = not a TransMLA model)

	// Residual connection configuration.
	ResidualMode      string // "standard", "attnres", or "block_attnres" (default: "standard")
	AttnResNumBlocks  int    // number of blocks for block_attnres mode (default: 8)

	// BERT encoder-only fields.
	NumLabels    int     // number of output classes for sequence classification
	PoolerType   string  // pooling strategy ("cls" or "mean")
	LayerNormEps float32 // LayerNorm epsilon (0 = use default 1e-12)

	// Granite-specific fields.
	EmbeddingMultiplier float32 // multiply embeddings by this factor (0 = no scaling)
	ResidualMultiplier  float32 // multiply residual connections by this factor (0 = no scaling)

	// Vision encoder fields (LLaVA, multimodal models).
	VisionImageSize  int    // vision encoder input image size (e.g. 336)
	VisionPatchSize  int    // vision encoder patch size (e.g. 14)
	VisionHiddenSize int    // vision encoder hidden dimension
	VisionNumHeads   int    // vision encoder attention heads
	VisionNumLayers  int    // vision encoder transformer layers
	ProjectorType    string // multi-modal projector type ("linear" or "mlp")
}

// DetectActualArchitecture checks GGUF metadata to detect models that
// declare one architecture but are actually a different model family.
// For example, Mistral GGUF files declare general.architecture = "llama"
// but are identified by their model name, tokenizer pre-processor, or
// vocabulary size.
func DetectActualArchitecture(f *File, declared string) string {
	if declared != "llama" {
		return declared
	}

	// Check general.name for "mistral" (case-insensitive).
	if name, ok := f.GetString("general.name"); ok {
		lower := strings.ToLower(name)
		if strings.Contains(lower, "mistral") {
			slog.Info("detected Mistral model from general.name", "name", name, "declared", declared)
			return "mistral"
		}
	}

	// Check tokenizer.ggml.pre for "mistral" pre-tokenizer.
	if pre, ok := f.GetString("tokenizer.ggml.pre"); ok {
		lower := strings.ToLower(pre)
		if strings.Contains(lower, "mistral") {
			slog.Info("detected Mistral model from tokenizer.ggml.pre", "pre", pre, "declared", declared)
			return "mistral"
		}
	}

	return declared
}

// ExtractModelConfig reads GGUF metadata and returns a ModelConfig.
// The architecture field (general.architecture) determines which metadata
// key prefix to use (e.g., "llama." or "gemma."). After extracting config
// using the declared architecture's key prefix, the actual architecture is
// detected via [DetectActualArchitecture] to handle models like Mistral
// that declare "llama" but need different runtime behavior.
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
	// Extract logit softcapping value.
	if v, ok := f.GetFloat32(prefix + "final_logit_softcapping"); ok {
		cfg.LogitSoftcap = v
	}
	// Extract local RoPE base for alternating attention.
	if v, ok := f.GetFloat32(prefix + "rope.local.freq_base"); ok {
		cfg.LocalRopeTheta = float64(v)
	}
	// Extract sliding window size.
	if v, ok := f.GetUint32(prefix + "attention.sliding_window"); ok {
		cfg.SlidingWindow = int(v)
	}
	// Gemma 3 uses a fixed pattern of 6 for sliding window layers.
	if cfg.LocalRopeTheta > 0 {
		cfg.SlidingWindowPattern = 6
	}
	// Extract RMS norm epsilon.
	if v, ok := f.GetFloat32(prefix + "attention.layer_norm_rms_epsilon"); ok {
		cfg.RMSNormEps = v
	}
	// Extract partial rotary factor from rope.dimension_count.
	// Phi models apply RoPE to only a fraction of head dimensions.
	// factor = rope_dimension_count / head_dim.
	if v, ok := f.GetUint32(prefix + "rope.dimension_count"); ok {
		headDim := cfg.HiddenSize / cfg.NumHeads
		if cfg.HeadDim > 0 {
			headDim = cfg.HeadDim
		}
		if headDim > 0 {
			cfg.PartialRotaryFactor = float32(v) / float32(headDim)
		}
	}

	// Extract DeepSeek MLA fields.
	if v, ok := f.GetUint32(prefix + "attention.kv_lora_rank"); ok {
		cfg.KVLoRADim = int(v)
	}
	if v, ok := f.GetUint32(prefix + "attention.q_lora_rank"); ok {
		cfg.QLoRADim = int(v)
	}
	if v, ok := f.GetUint32(prefix + "attention.qk_rope_head_dim"); ok {
		cfg.QKRopeHeadDim = int(v)
	}

	// Extract DeepSeek MoE fields.
	if v, ok := f.GetUint32(prefix + "expert_count"); ok {
		cfg.NumExperts = int(v)
	}
	if v, ok := f.GetUint32(prefix + "expert_used_count"); ok {
		cfg.NumExpertsPerToken = int(v)
	}
	if v, ok := f.GetUint32(prefix + "expert_shared_count"); ok {
		cfg.NumSharedExperts = int(v)
	}

	// Extract Granite-specific fields.
	if v, ok := f.GetFloat32(prefix + "embedding_multiplier"); ok {
		cfg.EmbeddingMultiplier = v
	}
	if v, ok := f.GetFloat32(prefix + "residual_multiplier"); ok {
		cfg.ResidualMultiplier = v
	}
	// Granite uses "logit_scale" for softcapping (reuse LogitSoftcap field).
	if cfg.LogitSoftcap == 0 {
		if v, ok := f.GetFloat32(prefix + "logit_scale"); ok {
			cfg.LogitSoftcap = v
		}
	}

	// Extract BERT-specific fields.
	if v, ok := f.GetUint32(prefix + "num_labels"); ok {
		cfg.NumLabels = int(v)
	}
	if v, ok := f.GetString(prefix + "pooler_type"); ok {
		cfg.PoolerType = v
	}
	if v, ok := f.GetFloat32(prefix + "attention.layer_norm_epsilon"); ok {
		cfg.LayerNormEps = v
	}

	// Extract TransMLA fields (MHA-to-MLA converted models).
	if v, ok := f.GetUint32("transmla.kv_lora_dim"); ok {
		cfg.TransMLAKVLoraDim = int(v)
	}

	// Extract residual connection configuration.
	if v, ok := f.GetString("general.residual_mode"); ok {
		cfg.ResidualMode = v
	}
	if v, ok := f.GetUint32("general.attnres_blocks"); ok {
		cfg.AttnResNumBlocks = int(v)
	}

	// Extract vision encoder fields (LLaVA and multimodal models).
	if v, ok := f.GetUint32("clip.vision.image_size"); ok {
		cfg.VisionImageSize = int(v)
	}
	if v, ok := f.GetUint32("clip.vision.patch_size"); ok {
		cfg.VisionPatchSize = int(v)
	}
	if v, ok := f.GetUint32("clip.vision.embedding_length"); ok {
		cfg.VisionHiddenSize = int(v)
	}
	if v, ok := f.GetUint32("clip.vision.head_count"); ok {
		cfg.VisionNumHeads = int(v)
	}
	if v, ok := f.GetUint32("clip.vision.block_count"); ok {
		cfg.VisionNumLayers = int(v)
	}
	if v, ok := f.GetString("clip.vision.projector_type"); ok {
		cfg.ProjectorType = v
	}

	// Detect the actual architecture — e.g. Mistral models that declare "llama".
	cfg.Architecture = DetectActualArchitecture(f, cfg.Architecture)

	return cfg, nil
}

// blkPattern matches "blk.N." prefix in GGUF tensor names.
var blkPattern = regexp.MustCompile(`^blk\.(\d+)\.(.+)$`)

// tensorNameMap maps GGUF tensor name suffixes (after blk.N.) to HuggingFace names.
// For architectures like Llama/Gemma 2, ffn_norm is the pre-FFN norm and is called
// post_attention_layernorm in HuggingFace convention.
var tensorNameMap = map[string]string{
	"attn_norm.weight":   "input_layernorm.weight",
	"attn_q.weight":      "self_attn.q_proj.weight",
	"attn_k.weight":      "self_attn.k_proj.weight",
	"attn_v.weight":      "self_attn.v_proj.weight",
	"attn_qkv.weight":   "self_attn.qkv_proj.weight",
	"attn_output.weight": "self_attn.o_proj.weight",
	"attn_q_norm.weight": "self_attn.q_norm.weight",
	"attn_k_norm.weight": "self_attn.k_norm.weight",
	"ffn_norm.weight":    "post_attention_layernorm.weight",
	"ffn_gate.weight":    "mlp.gate_proj.weight",
	"ffn_up.weight":      "mlp.up_proj.weight",
	"ffn_down.weight":    "mlp.down_proj.weight",
}

// gemma3TensorNameMap overrides for Gemma 3 which has 4 norms per layer.
// In Gemma 3, ffn_norm is the pre-FFN norm (separate from post-attention norm).
var gemma3TensorNameMap = map[string]string{
	"attn_norm.weight":           "input_layernorm.weight",
	"attn_q.weight":              "self_attn.q_proj.weight",
	"attn_k.weight":              "self_attn.k_proj.weight",
	"attn_v.weight":              "self_attn.v_proj.weight",
	"attn_output.weight":         "self_attn.o_proj.weight",
	"attn_q_norm.weight":         "self_attn.q_norm.weight",
	"attn_k_norm.weight":         "self_attn.k_norm.weight",
	"post_attention_norm.weight": "post_attention_layernorm.weight",
	"ffn_norm.weight":            "pre_feedforward_layernorm.weight",
	"post_ffw_norm.weight":       "post_feedforward_layernorm.weight",
	"ffn_gate.weight":            "mlp.gate_proj.weight",
	"ffn_up.weight":              "mlp.up_proj.weight",
	"ffn_down.weight":            "mlp.down_proj.weight",
}

// bertTensorNameMap maps GGUF block-level tensor name suffixes to canonical names for BERT.
var bertTensorNameMap = map[string]string{
	"attn_norm.weight":   "attn_norm.weight",
	"attn_norm.bias":     "attn_norm.bias",
	"attn_q.weight":      "attn_q.weight",
	"attn_q.bias":        "attn_q.bias",
	"attn_k.weight":      "attn_k.weight",
	"attn_k.bias":        "attn_k.bias",
	"attn_v.weight":      "attn_v.weight",
	"attn_v.bias":        "attn_v.bias",
	"attn_output.weight": "attn_output.weight",
	"attn_output.bias":   "attn_output.bias",
	"ffn_up.weight":      "ffn_up.weight",
	"ffn_up.bias":        "ffn_up.bias",
	"ffn_down.weight":    "ffn_down.weight",
	"ffn_down.bias":      "ffn_down.bias",
	"ffn_norm.weight":    "ffn_norm.weight",
	"ffn_norm.bias":      "ffn_norm.bias",
}

// bertGlobalTensorMap maps global GGUF tensor names for BERT.
var bertGlobalTensorMap = map[string]string{
	"token_embd.weight":      "token_embd.weight",
	"position_embd.weight":   "position_embd.weight",
	"token_type_embd.weight": "token_type_embd.weight",
	"token_embd_norm.weight": "token_embd_norm.weight",
	"token_embd_norm.bias":   "token_embd_norm.bias",
	"cls_pooler.weight":      "cls_pooler.weight",
	"cls_pooler.bias":        "cls_pooler.bias",
	"cls.weight":             "cls.weight",
	"cls.bias":               "cls.bias",
}

// globalTensorMap maps global GGUF tensor names to HuggingFace names.
var globalTensorMap = map[string]string{
	"token_embd.weight":  "model.embed_tokens.weight",
	"output_norm.weight": "model.norm.weight",
	"output.weight":      "lm_head.weight",
}

// MapTensorName converts a GGUF tensor name to the Zerfoo/HuggingFace canonical name.
// The arch parameter selects architecture-specific name mappings (e.g., "gemma3"
// uses different norm names than "llama").
// Unknown names pass through unchanged.
func MapTensorName(arch string, ggufName string) string {
	// BERT uses its own global and block-level name maps that preserve
	// GGUF-style names (the BERT builder looks them up directly).
	if arch == "bert" {
		if mapped, ok := bertGlobalTensorMap[ggufName]; ok {
			return mapped
		}
		m := blkPattern.FindStringSubmatch(ggufName)
		if m == nil {
			return ggufName
		}
		layerNum := m[1]
		suffix := m[2]
		if mapped, ok := bertTensorNameMap[suffix]; ok {
			return "blk." + layerNum + "." + mapped
		}
		return ggufName
	}

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

	// Select architecture-specific name map.
	nameMap := tensorNameMap
	if arch == "gemma3" || arch == "gemma3n" {
		nameMap = gemma3TensorNameMap
	}

	if mapped, ok := nameMap[suffix]; ok {
		return "model.layers." + layerNum + "." + mapped
	}

	// Handle bias variants (e.g., attn_q.bias → self_attn.q_proj.bias).
	if strings.HasSuffix(suffix, ".bias") {
		weightSuffix := strings.TrimSuffix(suffix, ".bias") + ".weight"
		if mapped, ok := nameMap[weightSuffix]; ok {
			biasMapped := strings.TrimSuffix(mapped, ".weight") + ".bias"
			return "model.layers." + layerNum + "." + biasMapped
		}
	}

	return ggufName
}
