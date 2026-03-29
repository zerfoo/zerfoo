package inference

// RopeScalingConfig holds configuration for RoPE scaling methods (e.g., YaRN).
type RopeScalingConfig struct {
	Type                            string  `json:"type"`
	Factor                          float64 `json:"factor"`
	OriginalMaxPositionEmbeddings   int     `json:"original_max_position_embeddings"`
}

// ConfigParser parses a raw JSON map (from config.json) into ModelMetadata.
type ConfigParser func(raw map[string]interface{}) (*ModelMetadata, error)

// ArchConfigRegistry maps model_type strings to config parsers.
type ArchConfigRegistry struct {
	parsers map[string]ConfigParser
}

// newArchConfigRegistry creates an empty registry.
func newArchConfigRegistry() *ArchConfigRegistry {
	return &ArchConfigRegistry{
		parsers: make(map[string]ConfigParser),
	}
}

// Register adds a parser for the given model type.
func (r *ArchConfigRegistry) Register(modelType string, parser ConfigParser) {
	r.parsers[modelType] = parser
}

// Parse dispatches to the registered parser for the model_type in raw,
// or falls back to generic field extraction for unknown types.
func (r *ArchConfigRegistry) Parse(raw map[string]interface{}) (*ModelMetadata, error) {
	modelType, _ := raw["model_type"].(string)

	if parser, ok := r.parsers[modelType]; ok {
		return parser(raw)
	}
	return parseFallbackConfig(raw)
}

// DefaultArchConfigRegistry returns a registry with all built-in parsers registered.
func DefaultArchConfigRegistry() *ArchConfigRegistry {
	r := newArchConfigRegistry()
	r.Register("gemma", parseGemmaConfig)
	r.Register("gemma2", parseGemmaConfig)
	r.Register("gemma3", parseGemmaConfig)
	r.Register("gemma3n", parseGemmaConfig)
	r.Register("llama", parseLlamaConfig)
	r.Register("llama4", parseLlama4Config)
	r.Register("mistral", parseMistralConfig)
	r.Register("qwen2", parseQwenConfig)
	r.Register("phi3", parsePhiConfig)
	r.Register("phi", parsePhiConfig)
	r.Register("deepseek_v3", parseDeepSeekConfig)
	r.Register("mamba", parseMambaConfig)
	r.Register("mamba3", parseMamba3Config)
	r.Register("jamba", parseJambaConfig)
	r.Register("granite", parseGraniteConfig)
	r.Register("gpt2", parseGPT2Config)
	r.Register("llava", parseLLaVAConfig)
	r.Register("qwen_vl", parseQwenVLConfig)
	r.Register("voxtral", parseVoxtralConfig)
	return r
}

// parseGPT2Config parses GPT-2-family config.json fields.
// GPT-2 uses different naming conventions than modern HuggingFace models:
// n_embd, n_layer, n_head, n_positions, n_inner, layer_norm_epsilon.
// GPT-2 is MHA (not GQA), so NumQueryHeads == NumKeyValueHeads.
// TieWordEmbeddings is always true for GPT-2.
func parseGPT2Config(raw map[string]interface{}) (*ModelMetadata, error) {
	hiddenSize := getInt(raw, "n_embd")
	intermediateSize := getInt(raw, "n_inner")
	if intermediateSize == 0 {
		intermediateSize = 4 * hiddenSize
	}
	nHead := getInt(raw, "n_head")
	lnEps := getFloat(raw, "layer_norm_epsilon")
	if lnEps == 0 {
		lnEps = 1e-5
	}
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            hiddenSize,
		NumLayers:             getInt(raw, "n_layer"),
		NumQueryHeads:         nHead,
		NumKeyValueHeads:      nHead,
		IntermediateSize:      intermediateSize,
		MaxPositionEmbeddings: getInt(raw, "n_positions"),
		LayerNormEps:          lnEps,
		TieWordEmbeddings:     true,
	}
	return meta, nil
}

// parseGemmaConfig parses Gemma-family config.json fields.
func parseGemmaConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 10000 // Gemma default
	}
	return meta, nil
}

// parseLlamaConfig parses Llama-family config.json fields.
func parseLlamaConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 500000 // Llama 3 default
	}
	return meta, nil
}

// parseLlama4Config parses Llama 4-family config.json fields.
// Extends Llama with MoE fields (num_local_experts, num_experts_per_tok).
func parseLlama4Config(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		RopeScaling:           getRopeScaling(raw),
		NumExperts:            getInt(raw, "num_local_experts"),
		NumExpertsPerToken:    getInt(raw, "num_experts_per_tok"),
		NumSharedExperts:      getInt(raw, "num_shared_experts"),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 500000 // Llama 4 default
	}
	return meta, nil
}

// parseMistralConfig parses Mistral-family config.json fields.
// Nearly identical to Llama but adds sliding_window and defaults rope_theta to 10000.
func parseMistralConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		SlidingWindow:         getInt(raw, "sliding_window"),
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 10000 // Mistral default
	}
	return meta, nil
}

// parseQwenConfig parses Qwen2-family config.json fields.
// Qwen models always use attention bias and default rope_theta to 1000000.
func parseQwenConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		SlidingWindow:         getInt(raw, "sliding_window"),
		AttentionBias:         true, // Qwen always uses attention bias
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 1000000 // Qwen default
	}
	return meta, nil
}

// parsePhiConfig parses Phi-family config.json fields.
// Adds partial_rotary_factor (default 1.0) and tie_word_embeddings.
func parsePhiConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		SlidingWindow:         getInt(raw, "sliding_window"),
		PartialRotaryFactor:   getFloat(raw, "partial_rotary_factor"),
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 10000 // Phi default
	}
	if meta.PartialRotaryFactor == 0 {
		meta.PartialRotaryFactor = 1.0 // Full rotation by default
	}
	return meta, nil
}

// parseGraniteConfig parses IBM Granite-family config.json fields.
// Granite is similar to Llama but adds embedding_multiplier, residual_multiplier,
// and optional attention bias and logit softcapping.
func parseGraniteConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		AttentionBias:         getBool(raw, "attention_bias"),
		EmbeddingMultiplier:   getFloat(raw, "embedding_multiplier"),
		ResidualMultiplier:    getFloat(raw, "residual_multiplier"),
		LogitScale:            getFloat(raw, "logit_scale"),
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 10000 // Granite default
	}
	return meta, nil
}

// parseDeepSeekConfig parses DeepSeek V3-family config.json fields.
// Adds MLA fields (kv_lora_rank, q_lora_rank, qk_rope_head_dim) and
// MoE fields (n_routed_experts, num_experts_per_tok, n_shared_experts).
func parseDeepSeekConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		NumLayers:             getInt(raw, "num_hidden_layers"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		KVLoRADim:             getInt(raw, "kv_lora_rank"),
		QLoRADim:              getInt(raw, "q_lora_rank"),
		QKRopeHeadDim:         getInt(raw, "qk_rope_head_dim"),
		NumExperts:            getInt(raw, "n_routed_experts"),
		NumExpertsPerToken:    getInt(raw, "num_experts_per_tok"),
		NumSharedExperts:      getInt(raw, "n_shared_experts"),
		RopeScaling:           getRopeScaling(raw),
	}
	if meta.RopeTheta == 0 {
		meta.RopeTheta = 10000 // DeepSeek default
	}
	return meta, nil
}

// parseFallbackConfig extracts common fields using the most widespread
// HuggingFace naming conventions. Used for unknown model_type values.
func parseFallbackConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture:          getString(raw, "model_type"),
		VocabSize:             getInt(raw, "vocab_size"),
		HiddenSize:            getInt(raw, "hidden_size"),
		IntermediateSize:      getInt(raw, "intermediate_size"),
		MaxPositionEmbeddings: getInt(raw, "max_position_embeddings"),
		EOSTokenID:            getInt(raw, "eos_token_id"),
		BOSTokenID:            getInt(raw, "bos_token_id"),
		NumQueryHeads:         getInt(raw, "num_attention_heads"),
		NumKeyValueHeads:      getInt(raw, "num_key_value_heads"),
		RopeTheta:             getFloat(raw, "rope_theta"),
		TieWordEmbeddings:     getBool(raw, "tie_word_embeddings"),
		SlidingWindow:         getInt(raw, "sliding_window"),
		AttentionBias:         getBool(raw, "attention_bias"),
		RopeScaling:           getRopeScaling(raw),
	}

	// Try common alternative field names for num_layers.
	meta.NumLayers = getInt(raw, "num_hidden_layers")
	if meta.NumLayers == 0 {
		meta.NumLayers = getInt(raw, "num_layers")
	}

	return meta, nil
}

// --- Helper functions for extracting typed values from raw JSON maps ---

func getString(raw map[string]interface{}, key string) string {
	v, _ := raw[key].(string)
	return v
}

func getInt(raw map[string]interface{}, key string) int {
	switch v := raw[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

func getFloat(raw map[string]interface{}, key string) float64 {
	switch v := raw[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	default:
		return 0
	}
}

func getBool(raw map[string]interface{}, key string) bool {
	v, _ := raw[key].(bool)
	return v
}

// parseMambaConfig parses Mamba-family config.json fields.
func parseMambaConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta := &ModelMetadata{
		Architecture: getString(raw, "model_type"),
		VocabSize:    getInt(raw, "vocab_size"),
		HiddenSize:   getInt(raw, "d_model"),
		NumLayers:    getInt(raw, "num_hidden_layers"),
		EOSTokenID:   getInt(raw, "eos_token_id"),
		BOSTokenID:   getInt(raw, "bos_token_id"),
	}
	if meta.HiddenSize == 0 {
		meta.HiddenSize = getInt(raw, "hidden_size")
	}
	if meta.NumLayers == 0 {
		meta.NumLayers = getInt(raw, "num_layers")
	}
	return meta, nil
}

// parseLLaVAConfig parses LLaVA-family config.json fields.
// Extends Llama with vision encoder fields.
func parseLLaVAConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta, err := parseLlamaConfig(raw)
	if err != nil {
		return nil, err
	}
	meta.Architecture = "llava"
	return meta, nil
}

// parseQwenVLConfig parses Qwen-VL-family config.json fields.
// Extends Qwen2 with vision encoder fields.
func parseQwenVLConfig(raw map[string]interface{}) (*ModelMetadata, error) {
	meta, err := parseQwenConfig(raw)
	if err != nil {
		return nil, err
	}
	meta.Architecture = "qwen_vl"
	return meta, nil
}

func getRopeScaling(raw map[string]interface{}) *RopeScalingConfig {
	m, ok := raw["rope_scaling"].(map[string]interface{})
	if !ok {
		return nil
	}
	return &RopeScalingConfig{
		Type:                          getString(m, "type"),
		Factor:                        getFloat(m, "factor"),
		OriginalMaxPositionEmbeddings: getInt(m, "original_max_position_embeddings"),
	}
}
