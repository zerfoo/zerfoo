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

	// Nemotron-H SSM (Mamba-2) fields.
	SSMStateSize  int // SSM state dimension (ssm.state_size)
	SSMConvKernel int // SSM convolution kernel width (ssm.conv_kernel)
	SSMNumHeads   int // SSM number of heads (ssm.num_heads)

	// Nemotron-H MoE shared expert count (expert_shared_count in nemotron_h_moe).
	ExpertSharedCount int
	// MoE expert gating configuration.
	ScoringFunc string // expert gating scoring function ("softmax" or "sigmoid"; default: "softmax")

	// Gemma 4 per-layer attention configuration.
	GlobalNumKVHeads          int     // KV head count for global attention layers (0 = use NumKVHeads)
	GlobalHeadDim             int     // head dimension for global attention layers (0 = use HeadDim)
	SlidingNumKVHeads         int     // KV head count for sliding attention layers (0 = use NumKVHeads)
	SlidingHeadDim            int     // head dimension for sliding attention layers (0 = use HeadDim)
	GlobalPartialRotaryFactor float32 // partial rotary factor for global layers (0 = full rotation)
	AttentionKEqV             bool    // if true, K and V share the same projection in global layers
	KVSharedLayers            int     // number of layers sharing KV projections (edge variants)
	PLEHiddenSize             int     // per-layer embedding hidden size (0 = disabled)
	DoubleWideMLP             bool    // if true, use double-width MLP (E2B variant)

	// Vision encoder fields (LLaVA, multimodal models).
	VisionImageSize  int    // vision encoder input image size (e.g. 336)
	VisionPatchSize  int    // vision encoder patch size (e.g. 14)
	VisionHiddenSize int    // vision encoder hidden dimension
	VisionNumHeads   int    // vision encoder attention heads
	VisionNumLayers  int    // vision encoder transformer layers
	ProjectorType    string // multi-modal projector type ("linear" or "mlp")

	// Audio encoder fields (Voxtral, multimodal speech-to-text models).
	AudioHiddenSize           int    // audio encoder hidden dimension
	AudioNumLayers            int    // audio encoder transformer layers
	AudioNumHeads             int    // audio encoder attention heads
	AudioNumMels              int    // number of mel spectrogram bins (e.g. 128)
	AudioIntermediateSize     int    // audio encoder FFN intermediate size
	AudioProjectorType        string // audio projector type (e.g. "mlp")
	AudioProjectorStackFactor int    // number of consecutive frames to stack (e.g. 4)
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

	// Extract expert gating scoring function (default: "softmax").
	cfg.ScoringFunc = "softmax"
	if v, ok := f.GetString(prefix + "expert_gating_func"); ok {
		cfg.ScoringFunc = v
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

	// Extract audio encoder fields (Voxtral mmproj models).
	if v, ok := f.GetUint32("audio.embedding_length"); ok {
		cfg.AudioHiddenSize = int(v)
	}
	if v, ok := f.GetUint32("audio.block_count"); ok {
		cfg.AudioNumLayers = int(v)
	}
	if v, ok := f.GetUint32("audio.head_count"); ok {
		cfg.AudioNumHeads = int(v)
	}
	if v, ok := f.GetUint32("audio.num_mel_bins"); ok {
		cfg.AudioNumMels = int(v)
	}
	if v, ok := f.GetUint32("audio.feed_forward_length"); ok {
		cfg.AudioIntermediateSize = int(v)
	}
	if v, ok := f.GetString("audio.projector_type"); ok {
		cfg.AudioProjectorType = v
	}
	if v, ok := f.GetUint32("audio.projector_stack_factor"); ok {
		cfg.AudioProjectorStackFactor = int(v)
	}

	// Extract Nemotron-H SSM fields. The SSM keys use the architecture prefix
	// (e.g. "nemotron_h_moe.ssm.state_size" or "nemotron_h.ssm.state_size").
	if v, ok := f.GetUint32(prefix + "ssm.state_size"); ok {
		cfg.SSMStateSize = int(v)
	}
	if v, ok := f.GetUint32(prefix + "ssm.conv_kernel"); ok {
		cfg.SSMConvKernel = int(v)
	}
	if v, ok := f.GetUint32(prefix + "ssm.num_heads"); ok {
		cfg.SSMNumHeads = int(v)
	}
	// Nemotron-H MoE shared expert count (distinct from DeepSeek NumSharedExperts
	// which uses the same GGUF key but is parsed above into NumSharedExperts).
	// ExpertSharedCount is populated for nemotron_h_moe models.
	if v, ok := f.GetUint32(prefix + "expert_shared_count"); ok {
		cfg.ExpertSharedCount = int(v)
	}

	// Extract Gemma 4 per-layer attention fields.
	// Prefer zerfoo-legacy keys, then fall back to canonical llama.cpp keys.
	if v, ok := f.GetUint32(prefix + "attention.global.head_count_kv"); ok {
		cfg.GlobalNumKVHeads = int(v)
	}
	if v, ok := f.GetUint32(prefix + "attention.global.key_length"); ok {
		cfg.GlobalHeadDim = int(v)
	}
	if v, ok := f.GetUint32(prefix + "attention.sliding.head_count_kv"); ok {
		cfg.SlidingNumKVHeads = int(v)
	}
	if v, ok := f.GetUint32(prefix + "attention.sliding.key_length"); ok {
		cfg.SlidingHeadDim = int(v)
	}
	// Canonical (llama.cpp) Gemma 4 keys.
	if cfg.GlobalHeadDim == 0 {
		if v, ok := f.GetUint32(prefix + "attention.key_length"); ok {
			cfg.GlobalHeadDim = int(v)
		}
	}
	if cfg.SlidingHeadDim == 0 {
		if v, ok := f.GetUint32(prefix + "attention.key_length_swa"); ok {
			cfg.SlidingHeadDim = int(v)
		}
	}
	// Derive AttentionKEqV from key_length == value_length when both present.
	if keyLen, ok := f.GetUint32(prefix + "attention.key_length"); ok {
		if valLen, ok2 := f.GetUint32(prefix + "attention.value_length"); ok2 && keyLen == valLen {
			cfg.AttentionKEqV = true
		}
	}
	if v, ok := f.GetFloat32(prefix + "rope.global.dimension_fraction"); ok {
		cfg.GlobalPartialRotaryFactor = v
	}
	if v, ok := f.GetString(prefix + "attention.k_eq_v"); ok && v == "true" {
		cfg.AttentionKEqV = true
	}
	if v, ok := f.GetUint32(prefix + "attention.k_eq_v"); ok && v == 1 {
		cfg.AttentionKEqV = true
	}
	if v, ok := f.GetUint32(prefix + "kv_shared_layers"); ok {
		cfg.KVSharedLayers = int(v)
	}
	if cfg.KVSharedLayers == 0 {
		if v, ok := f.GetUint32(prefix + "attention.shared_kv_layers"); ok {
			cfg.KVSharedLayers = int(v)
		}
	}
	if v, ok := f.GetUint32(prefix + "ple.hidden_size"); ok {
		cfg.PLEHiddenSize = int(v)
	}
	if cfg.PLEHiddenSize == 0 {
		if v, ok := f.GetUint32(prefix + "embedding_length_per_layer_input"); ok {
			cfg.PLEHiddenSize = int(v)
		}
	}
	if v, ok := f.GetString(prefix + "mlp.double_wide"); ok && v == "true" {
		cfg.DoubleWideMLP = true
	}
	if v, ok := f.GetUint32(prefix + "mlp.double_wide"); ok && v == 1 {
		cfg.DoubleWideMLP = true
	}
	// Gemma 4 canonical sliding window pattern and SWA RoPE base.
	if v, ok := f.GetUint32(prefix + "attention.sliding_window_pattern"); ok {
		cfg.SlidingWindowPattern = int(v)
	}
	if cfg.LocalRopeTheta == 0 {
		if v, ok := f.GetFloat32(prefix + "rope.freq_base_swa"); ok {
			cfg.LocalRopeTheta = float64(v)
		}
	}

	// Gemma 4 defaults and per-layer fallbacks.
	if strings.HasPrefix(arch, "gemma4") {
		if cfg.GlobalNumKVHeads == 0 {
			cfg.GlobalNumKVHeads = cfg.NumKVHeads
		}
		if cfg.SlidingNumKVHeads == 0 {
			cfg.SlidingNumKVHeads = cfg.NumKVHeads
		}
		if cfg.SlidingWindowPattern == 0 {
			cfg.SlidingWindowPattern = 6
		}
		if cfg.VocabSize == 0 {
			cfg.VocabSize = 262144
		}
	}

	// Route Gemma 4 sub-variant based on metadata fingerprint.
	// Canonical llama.cpp GGUFs declare arch=gemma4 for all variants; zerfoo
	// needs the specific builder ("gemma4" dense, "gemma4moe", "gemma4e" edge).
	if cfg.Architecture == "gemma4" {
		switch {
		case cfg.PLEHiddenSize > 0 || cfg.KVSharedLayers > 0:
			cfg.Architecture = "gemma4e"
			slog.Info("detected Gemma 4 edge variant from PLE/KV-sharing metadata",
				"ple_hidden_size", cfg.PLEHiddenSize, "kv_shared_layers", cfg.KVSharedLayers)
		case cfg.NumExperts > 0:
			cfg.Architecture = "gemma4moe"
			slog.Info("detected Gemma 4 MoE variant from expert_count metadata",
				"num_experts", cfg.NumExperts)
		}
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

// gpt2TensorNameMap maps GGUF block-level tensor name suffixes for GPT-2.
// GPT-2 uses fused QKV projections and biases on all linear layers.
var gpt2TensorNameMap = map[string]string{
	"attn_norm.weight":   "attn_norm.weight",
	"attn_norm.bias":     "attn_norm.bias",
	"attn_qkv.weight":    "attn_qkv.weight",
	"attn_qkv.bias":      "attn_qkv.bias",
	"attn_output.weight": "attn_output.weight",
	"attn_output.bias":   "attn_output.bias",
	"ffn_norm.weight":    "ffn_norm.weight",
	"ffn_norm.bias":      "ffn_norm.bias",
	"ffn_up.weight":      "ffn_up.weight",
	"ffn_up.bias":        "ffn_up.bias",
	"ffn_down.weight":    "ffn_down.weight",
	"ffn_down.bias":      "ffn_down.bias",
}

// gpt2GlobalTensorMap maps global GGUF tensor names for GPT-2.
var gpt2GlobalTensorMap = map[string]string{
	"token_embd.weight":    "token_embd.weight",
	"position_embd.weight": "position_embd.weight",
	"output_norm.weight":   "output_norm.weight",
	"output_norm.bias":     "output_norm.bias",
	"output.weight":        "output.weight",
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

// nemotronHTensorNameMap maps GGUF block-level tensor name suffixes for Nemotron-H.
// Nemotron-H is a hybrid architecture with attention, SSM (Mamba-2), and MoE layers.
// SSM and MoE tensor names are preserved as-is (the graph builder looks them up directly).
var nemotronHTensorNameMap = map[string]string{
	// Attention tensors.
	"attn_norm.weight":   "attn_norm.weight",
	"attn_q.weight":      "attn_q.weight",
	"attn_k.weight":      "attn_k.weight",
	"attn_v.weight":      "attn_v.weight",
	"attn_output.weight": "attn_output.weight",
	"ffn_norm.weight":    "ffn_norm.weight",
	"ffn_gate.weight":    "ffn_gate.weight",
	"ffn_up.weight":      "ffn_up.weight",
	"ffn_down.weight":    "ffn_down.weight",
	// SSM (Mamba-2) tensors.
	"ssm_in.weight":      "ssm_in.weight",
	"ssm_conv1d.weight":  "ssm_conv1d.weight",
	"ssm_dt.weight":      "ssm_dt.weight",
	"ssm_A.weight":       "ssm_A.weight",
	"ssm_D.weight":       "ssm_D.weight",
	"ssm_out.weight":     "ssm_out.weight",
	// MoE tensors.
	"ffn_gate_inp.weight":  "ffn_gate_inp.weight",
	"ffn_gate_exps.weight": "ffn_gate_exps.weight",
	"ffn_up_exps.weight":   "ffn_up_exps.weight",
	"ffn_down_exps.weight": "ffn_down_exps.weight",
}

// nemotronHGlobalTensorMap maps global GGUF tensor names for Nemotron-H.
var nemotronHGlobalTensorMap = map[string]string{
	"token_embd.weight":  "token_embd.weight",
	"output_norm.weight": "output_norm.weight",
	"output.weight":      "output.weight",
}

// minimaxM2TensorNameMap maps GGUF block-level tensor name suffixes for MiniMax-M2.
// MiniMax-M2 uses GQA attention with QK norms plus sigmoid MoE with routing bias.
// MoE tensors and routing bias are preserved as-is (the graph builder looks them up directly).
var minimaxM2TensorNameMap = map[string]string{
	// Attention norms and projections.
	"attn_norm.weight":   "attn_norm.weight",
	"attn_q.weight":      "attn_q.weight",
	"attn_k.weight":      "attn_k.weight",
	"attn_v.weight":      "attn_v.weight",
	"attn_output.weight": "attn_output.weight",
	// QK norms.
	"attn_q_norm.weight": "attn_q_norm.weight",
	"attn_k_norm.weight": "attn_k_norm.weight",
	// FFN norm.
	"ffn_norm.weight": "ffn_norm.weight",
	// MoE router.
	"ffn_gate_inp.weight": "ffn_gate_inp.weight",
	// MoE routing bias (unique to MiniMax-M2).
	"exp_probs_b": "exp_probs_b",
	// Stacked expert tensors.
	"ffn_gate_exps.weight": "ffn_gate_exps.weight",
	"ffn_up_exps.weight":   "ffn_up_exps.weight",
	"ffn_down_exps.weight": "ffn_down_exps.weight",
}

// minimaxM2GlobalTensorMap maps global GGUF tensor names for MiniMax-M2.
var minimaxM2GlobalTensorMap = map[string]string{
	"token_embd.weight":  "token_embd.weight",
	"output_norm.weight": "output_norm.weight",
	"output.weight":      "output.weight",
}

// voxtralAudioTensorMap maps Voxtral mmproj GGUF tensor name prefixes.
// Audio encoder tensors are prefixed with "a." and adapter tensors with "mm.a.mlp.".
// These are mapped to canonical names used by the Voxtral graph builder.
var voxtralAudioTensorMap = map[string]string{
	// Conv1D frontend.
	"a.conv1d.0.weight": "a.conv1d.0.weight",
	"a.conv1d.0.bias":   "a.conv1d.0.bias",
	"a.conv1d.1.weight": "a.conv1d.1.weight",
	"a.conv1d.1.bias":   "a.conv1d.1.bias",

	// Post layer norm.
	"a.post_ln.weight": "a.post_ln.weight",
	"a.post_ln.bias":   "a.post_ln.bias",
}

// voxtralAudioBlockTensorMap maps per-block tensor suffixes for Voxtral audio encoder.
var voxtralAudioBlockTensorMap = map[string]string{
	"ln1.weight":      "ln1.weight",
	"ln1.bias":        "ln1.bias",
	"ln2.weight":      "ln2.weight",
	"ln2.bias":        "ln2.bias",
	"attn_q.weight":   "attn_q.weight",
	"attn_q.bias":     "attn_q.bias",
	"attn_k.weight":   "attn_k.weight",
	"attn_k.bias":     "attn_k.bias",
	"attn_v.weight":   "attn_v.weight",
	"attn_v.bias":     "attn_v.bias",
	"attn_o.weight":   "attn_o.weight",
	"ffn_up.weight":   "ffn_up.weight",
	"ffn_up.bias":     "ffn_up.bias",
	"ffn_down.weight": "ffn_down.weight",
	"ffn_down.bias":   "ffn_down.bias",
}

// voxtralAdapterTensorMap maps Voxtral MLP adapter (projector) tensor names.
var voxtralAdapterTensorMap = map[string]string{
	"mm.a.mlp.0.weight": "mm.a.mlp.0.weight",
	"mm.a.mlp.0.bias":   "mm.a.mlp.0.bias",
	"mm.a.mlp.2.weight": "mm.a.mlp.2.weight",
	"mm.a.mlp.2.bias":   "mm.a.mlp.2.bias",
}

// globalTensorMap maps global GGUF tensor names to HuggingFace names.
var globalTensorMap = map[string]string{
	"token_embd.weight":  "model.embed_tokens.weight",
	"output_norm.weight": "model.norm.weight",
	"output.weight":      "lm_head.weight",
}

// gemma4eGlobalTensorMap adds the shared Per-Layer Embedding (PLE) tensors
// that Gemma 4 edge variants ship in addition to the standard globals.
// See docs/gemma4-edge-architecture.md and docs/adr/086-gemma4-edge-ple-architecture.md.
var gemma4eGlobalTensorMap = map[string]string{
	"token_embd.weight":            "model.embed_tokens.weight",
	"output_norm.weight":           "model.norm.weight",
	"output.weight":                "lm_head.weight",
	"per_layer_token_embd.weight":  "model.ple_embed_tokens.weight",
	"per_layer_model_proj.weight":  "model.ple_model_proj.weight",
	"per_layer_proj_norm.weight":   "model.ple_proj_norm.weight",
}

// gemma4eTensorNameMap extends gemma3TensorNameMap with the per-block
// tensors introduced by Gemma 4 edge: PLE per-layer projection, input gate,
// output scale, and the additional post_norm.
var gemma4eTensorNameMap = map[string]string{
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
	"post_norm.weight":           "post_layernorm.weight",
	"ffn_gate.weight":            "mlp.gate_proj.weight",
	"ffn_up.weight":              "mlp.up_proj.weight",
	"ffn_down.weight":            "mlp.down_proj.weight",
	"proj.weight":                "ple_layer_proj.weight",
	"inp_gate.weight":            "input_gate.weight",
	"layer_output_scale.weight":  "layer_output_scale.weight",
}

// MapTensorName converts a GGUF tensor name to the Zerfoo/HuggingFace canonical name.
// The arch parameter selects architecture-specific name mappings (e.g., "gemma3"
// uses different norm names than "llama").
// Unknown names pass through unchanged.
func MapTensorName(arch string, ggufName string) string {
	// Voxtral audio encoder tensors pass through unchanged (the builder looks them up directly).
	if arch == "voxtral" {
		// Global audio tensors.
		if _, ok := voxtralAudioTensorMap[ggufName]; ok {
			return ggufName
		}
		// Adapter tensors.
		if _, ok := voxtralAdapterTensorMap[ggufName]; ok {
			return ggufName
		}
		// Block-level audio tensors: "a.blk.N.suffix"
		if strings.HasPrefix(ggufName, "a.blk.") {
			return ggufName
		}
		// Fall through to standard tensor name mapping for text decoder tensors.
	}

	// GPT-2, BERT, and Nemotron-H use their own global and block-level name maps
	// that preserve GGUF-style names (their builders look them up directly).
	if arch == "gpt2" || arch == "bert" || arch == "nemotron_h" || arch == "nemotron_h_moe" || arch == "minimax-m2" {
		globalMap := gpt2GlobalTensorMap
		blockMap := gpt2TensorNameMap
		if arch == "bert" {
			globalMap = bertGlobalTensorMap
			blockMap = bertTensorNameMap
		} else if arch == "nemotron_h" || arch == "nemotron_h_moe" {
			globalMap = nemotronHGlobalTensorMap
			blockMap = nemotronHTensorNameMap
		} else if arch == "minimax-m2" {
			globalMap = minimaxM2GlobalTensorMap
			blockMap = minimaxM2TensorNameMap
		}
		if mapped, ok := globalMap[ggufName]; ok {
			return mapped
		}
		m := blkPattern.FindStringSubmatch(ggufName)
		if m == nil {
			return ggufName
		}
		layerNum := m[1]
		suffix := m[2]
		if mapped, ok := blockMap[suffix]; ok {
			return "blk." + layerNum + "." + mapped
		}
		return ggufName
	}

	// Check global names first. Gemma 4 edge has extra PLE globals.
	if arch == "gemma4e" {
		if mapped, ok := gemma4eGlobalTensorMap[ggufName]; ok {
			return mapped
		}
	}
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
	switch arch {
	case "gemma3", "gemma3n":
		nameMap = gemma3TensorNameMap
	case "gemma4e":
		nameMap = gemma4eTensorNameMap
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
