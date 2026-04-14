package gguf

import "testing"

func TestMapTensorName_Llama(t *testing.T) {
	tests := []struct {
		gguf string
		want string
	}{
		{"token_embd.weight", "model.embed_tokens.weight"},
		{"output_norm.weight", "model.norm.weight"},
		{"output.weight", "lm_head.weight"},
		{"blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"},
		{"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight"},
		{"blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"},
		{"blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight"},
		{"blk.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight"},
		{"blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"},
		{"blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"},
		{"blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight"},
		{"blk.31.attn_q.weight", "model.layers.31.self_attn.q_proj.weight"},
		{"blk.15.ffn_gate.weight", "model.layers.15.mlp.gate_proj.weight"},
	}
	for _, tt := range tests {
		t.Run(tt.gguf, func(t *testing.T) {
			got := MapTensorName("llama", tt.gguf)
			if got != tt.want {
				t.Errorf("MapTensorName(llama, %q) = %q, want %q", tt.gguf, got, tt.want)
			}
		})
	}
}

func TestMapTensorName_Gemma(t *testing.T) {
	// Gemma uses the same tensor name mapping as Llama in GGUF.
	tests := []struct {
		gguf string
		want string
	}{
		{"token_embd.weight", "model.embed_tokens.weight"},
		{"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"},
	}
	for _, tt := range tests {
		t.Run(tt.gguf, func(t *testing.T) {
			got := MapTensorName("gemma", tt.gguf)
			if got != tt.want {
				t.Errorf("MapTensorName(gemma, %q) = %q, want %q", tt.gguf, got, tt.want)
			}
		})
	}
}

func TestMapTensorName_Phi3_QKV(t *testing.T) {
	tests := []struct {
		gguf string
		want string
	}{
		{"blk.0.attn_qkv.weight", "model.layers.0.self_attn.qkv_proj.weight"},
		{"blk.5.attn_qkv.bias", "model.layers.5.self_attn.qkv_proj.bias"},
		{"blk.31.attn_qkv.weight", "model.layers.31.self_attn.qkv_proj.weight"},
		// Ensure other mappings still work for phi3.
		{"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight"},
	}
	for _, tt := range tests {
		t.Run(tt.gguf, func(t *testing.T) {
			got := MapTensorName("phi3", tt.gguf)
			if got != tt.want {
				t.Errorf("MapTensorName(phi3, %q) = %q, want %q", tt.gguf, got, tt.want)
			}
		})
	}
}

func TestMapTensorName_GPT2(t *testing.T) {
	tests := []struct {
		gguf string
		want string
	}{
		// Global tensors.
		{"token_embd.weight", "token_embd.weight"},
		{"position_embd.weight", "position_embd.weight"},
		{"output_norm.weight", "output_norm.weight"},
		{"output_norm.bias", "output_norm.bias"},
		{"output.weight", "output.weight"},
		// Block-level tensors (layer 0).
		{"blk.0.attn_norm.weight", "blk.0.attn_norm.weight"},
		{"blk.0.attn_norm.bias", "blk.0.attn_norm.bias"},
		{"blk.0.attn_qkv.weight", "blk.0.attn_qkv.weight"},
		{"blk.0.attn_qkv.bias", "blk.0.attn_qkv.bias"},
		{"blk.0.attn_output.weight", "blk.0.attn_output.weight"},
		{"blk.0.attn_output.bias", "blk.0.attn_output.bias"},
		{"blk.0.ffn_norm.weight", "blk.0.ffn_norm.weight"},
		{"blk.0.ffn_norm.bias", "blk.0.ffn_norm.bias"},
		{"blk.0.ffn_up.weight", "blk.0.ffn_up.weight"},
		{"blk.0.ffn_up.bias", "blk.0.ffn_up.bias"},
		{"blk.0.ffn_down.weight", "blk.0.ffn_down.weight"},
		{"blk.0.ffn_down.bias", "blk.0.ffn_down.bias"},
		// Higher layer numbers.
		{"blk.11.attn_qkv.weight", "blk.11.attn_qkv.weight"},
		{"blk.11.ffn_down.bias", "blk.11.ffn_down.bias"},
		// Unknown tensor passes through.
		{"some.unknown.tensor", "some.unknown.tensor"},
	}
	for _, tt := range tests {
		t.Run(tt.gguf, func(t *testing.T) {
			got := MapTensorName("gpt2", tt.gguf)
			if got != tt.want {
				t.Errorf("MapTensorName(gpt2, %q) = %q, want %q", tt.gguf, got, tt.want)
			}
		})
	}
}

func TestMapTensorName_NemotronH(t *testing.T) {
	tests := []struct {
		arch string
		gguf string
		want string
	}{
		// Global tensors (nemotron_h).
		{"nemotron_h", "token_embd.weight", "token_embd.weight"},
		{"nemotron_h", "output_norm.weight", "output_norm.weight"},
		{"nemotron_h", "output.weight", "output.weight"},
		// Attention block tensors (nemotron_h).
		{"nemotron_h", "blk.0.attn_norm.weight", "blk.0.attn_norm.weight"},
		{"nemotron_h", "blk.0.attn_q.weight", "blk.0.attn_q.weight"},
		{"nemotron_h", "blk.0.attn_k.weight", "blk.0.attn_k.weight"},
		{"nemotron_h", "blk.0.attn_v.weight", "blk.0.attn_v.weight"},
		{"nemotron_h", "blk.0.attn_output.weight", "blk.0.attn_output.weight"},
		{"nemotron_h", "blk.0.ffn_norm.weight", "blk.0.ffn_norm.weight"},
		{"nemotron_h", "blk.0.ffn_gate.weight", "blk.0.ffn_gate.weight"},
		{"nemotron_h", "blk.0.ffn_up.weight", "blk.0.ffn_up.weight"},
		{"nemotron_h", "blk.0.ffn_down.weight", "blk.0.ffn_down.weight"},
		// SSM tensors (nemotron_h).
		{"nemotron_h", "blk.3.ssm_in.weight", "blk.3.ssm_in.weight"},
		{"nemotron_h", "blk.3.ssm_conv1d.weight", "blk.3.ssm_conv1d.weight"},
		{"nemotron_h", "blk.3.ssm_dt.weight", "blk.3.ssm_dt.weight"},
		{"nemotron_h", "blk.3.ssm_A.weight", "blk.3.ssm_A.weight"},
		{"nemotron_h", "blk.3.ssm_D.weight", "blk.3.ssm_D.weight"},
		{"nemotron_h", "blk.3.ssm_out.weight", "blk.3.ssm_out.weight"},
		// MoE tensors (nemotron_h_moe).
		{"nemotron_h_moe", "blk.5.ffn_gate_inp.weight", "blk.5.ffn_gate_inp.weight"},
		{"nemotron_h_moe", "blk.5.ffn_gate_exps.weight", "blk.5.ffn_gate_exps.weight"},
		{"nemotron_h_moe", "blk.5.ffn_up_exps.weight", "blk.5.ffn_up_exps.weight"},
		{"nemotron_h_moe", "blk.5.ffn_down_exps.weight", "blk.5.ffn_down_exps.weight"},
		// SSM tensors also work under nemotron_h_moe.
		{"nemotron_h_moe", "blk.2.ssm_in.weight", "blk.2.ssm_in.weight"},
		{"nemotron_h_moe", "blk.2.ssm_A.weight", "blk.2.ssm_A.weight"},
		// Global tensors (nemotron_h_moe).
		{"nemotron_h_moe", "token_embd.weight", "token_embd.weight"},
		{"nemotron_h_moe", "output_norm.weight", "output_norm.weight"},
		{"nemotron_h_moe", "output.weight", "output.weight"},
		// Unknown tensor passes through.
		{"nemotron_h", "some.unknown.tensor", "some.unknown.tensor"},
		{"nemotron_h_moe", "some.unknown.tensor", "some.unknown.tensor"},
	}
	for _, tt := range tests {
		t.Run(tt.arch+"/"+tt.gguf, func(t *testing.T) {
			got := MapTensorName(tt.arch, tt.gguf)
			if got != tt.want {
				t.Errorf("MapTensorName(%q, %q) = %q, want %q", tt.arch, tt.gguf, got, tt.want)
			}
		})
	}
}

func TestMapTensorName_Unknown(t *testing.T) {
	// Unknown names pass through unchanged.
	got := MapTensorName("llama", "some.unknown.tensor")
	if got != "some.unknown.tensor" {
		t.Errorf("unknown tensor name should pass through, got %q", got)
	}
}

func TestExtractModelConfig(t *testing.T) {
	meta := map[string]any{
		"general.architecture":           "llama",
		"general.name":                   "test-model",
		"llama.embedding_length":         uint32(2048),
		"llama.block_count":              uint32(22),
		"llama.attention.head_count":     uint32(32),
		"llama.attention.head_count_kv":  uint32(8),
		"llama.feed_forward_length":      uint32(5632),
		"llama.context_length":           uint32(8192),
		"llama.rope.freq_base":           float32(10000.0),
		"llama.vocab_size":               uint32(32000),
	}

	f := &File{Metadata: meta}
	cfg, err := ExtractModelConfig(f)
	if err != nil {
		t.Fatalf("ExtractModelConfig: %v", err)
	}

	if cfg.Architecture != "llama" {
		t.Errorf("Architecture = %q, want llama", cfg.Architecture)
	}
	if cfg.Name != "test-model" {
		t.Errorf("Name = %q, want test-model", cfg.Name)
	}
	if cfg.HiddenSize != 2048 {
		t.Errorf("HiddenSize = %d, want 2048", cfg.HiddenSize)
	}
	if cfg.NumLayers != 22 {
		t.Errorf("NumLayers = %d, want 22", cfg.NumLayers)
	}
	if cfg.NumHeads != 32 {
		t.Errorf("NumHeads = %d, want 32", cfg.NumHeads)
	}
	if cfg.NumKVHeads != 8 {
		t.Errorf("NumKVHeads = %d, want 8", cfg.NumKVHeads)
	}
	if cfg.IntermediateSize != 5632 {
		t.Errorf("IntermediateSize = %d, want 5632", cfg.IntermediateSize)
	}
	if cfg.MaxSeqLen != 8192 {
		t.Errorf("MaxSeqLen = %d, want 8192", cfg.MaxSeqLen)
	}
	if cfg.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want 32000", cfg.VocabSize)
	}
}

func TestExtractModelConfig_Gemma(t *testing.T) {
	meta := map[string]any{
		"general.architecture":          "gemma",
		"gemma.embedding_length":        uint32(2048),
		"gemma.block_count":             uint32(18),
		"gemma.attention.head_count":    uint32(8),
		"gemma.attention.head_count_kv": uint32(1),
		"gemma.feed_forward_length":     uint32(16384),
		"gemma.context_length":          uint32(8192),
	}

	f := &File{Metadata: meta}
	cfg, err := ExtractModelConfig(f)
	if err != nil {
		t.Fatalf("ExtractModelConfig: %v", err)
	}

	if cfg.Architecture != "gemma" {
		t.Errorf("Architecture = %q, want gemma", cfg.Architecture)
	}
	if cfg.HiddenSize != 2048 {
		t.Errorf("HiddenSize = %d, want 2048", cfg.HiddenSize)
	}
	if cfg.NumKVHeads != 1 {
		t.Errorf("NumKVHeads = %d, want 1", cfg.NumKVHeads)
	}
}

func TestExtractModelConfig_MissingArch(t *testing.T) {
	f := &File{Metadata: map[string]any{}}
	_, err := ExtractModelConfig(f)
	if err == nil {
		t.Error("expected error for missing architecture")
	}
}

func TestExtractModelConfig_ResidualConfig(t *testing.T) {
	t.Run("attnres mode from metadata", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":    "llama",
			"general.residual_mode":   "attnres",
			"llama.embedding_length":  uint32(2048),
			"llama.block_count":       uint32(22),
			"llama.attention.head_count": uint32(32),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.ResidualMode != "attnres" {
			t.Errorf("ResidualMode = %q, want %q", cfg.ResidualMode, "attnres")
		}
	})

	t.Run("block_attnres with custom blocks", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":    "llama",
			"general.residual_mode":   "block_attnres",
			"general.attnres_blocks":  uint32(4),
			"llama.embedding_length":  uint32(2048),
			"llama.block_count":       uint32(22),
			"llama.attention.head_count": uint32(32),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.ResidualMode != "block_attnres" {
			t.Errorf("ResidualMode = %q, want %q", cfg.ResidualMode, "block_attnres")
		}
		if cfg.AttnResNumBlocks != 4 {
			t.Errorf("AttnResNumBlocks = %d, want 4", cfg.AttnResNumBlocks)
		}
	})

	t.Run("defaults to empty when not present", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":    "llama",
			"llama.embedding_length":  uint32(2048),
			"llama.block_count":       uint32(22),
			"llama.attention.head_count": uint32(32),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.ResidualMode != "" {
			t.Errorf("ResidualMode = %q, want empty string", cfg.ResidualMode)
		}
		if cfg.AttnResNumBlocks != 0 {
			t.Errorf("AttnResNumBlocks = %d, want 0", cfg.AttnResNumBlocks)
		}
	})
}

func TestExtractModelConfig_NemotronSSM(t *testing.T) {
	t.Run("nemotron_h_moe with SSM and shared experts", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                      "nemotron_h_moe",
			"nemotron_h_moe.embedding_length":           uint32(4096),
			"nemotron_h_moe.block_count":                uint32(52),
			"nemotron_h_moe.attention.head_count":       uint32(32),
			"nemotron_h_moe.attention.head_count_kv":    uint32(8),
			"nemotron_h_moe.feed_forward_length":        uint32(8192),
			"nemotron_h_moe.context_length":             uint32(4096),
			"nemotron_h_moe.ssm.state_size":             uint32(128),
			"nemotron_h_moe.ssm.conv_kernel":            uint32(4),
			"nemotron_h_moe.ssm.num_heads":              uint32(64),
			"nemotron_h_moe.expert_count":               uint32(128),
			"nemotron_h_moe.expert_used_count":          uint32(6),
			"nemotron_h_moe.expert_shared_count":        uint32(2),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.Architecture != "nemotron_h_moe" {
			t.Errorf("Architecture = %q, want nemotron_h_moe", cfg.Architecture)
		}
		if cfg.SSMStateSize != 128 {
			t.Errorf("SSMStateSize = %d, want 128", cfg.SSMStateSize)
		}
		if cfg.SSMConvKernel != 4 {
			t.Errorf("SSMConvKernel = %d, want 4", cfg.SSMConvKernel)
		}
		if cfg.SSMNumHeads != 64 {
			t.Errorf("SSMNumHeads = %d, want 64", cfg.SSMNumHeads)
		}
		if cfg.ExpertSharedCount != 2 {
			t.Errorf("ExpertSharedCount = %d, want 2", cfg.ExpertSharedCount)
		}
		if cfg.NumExperts != 128 {
			t.Errorf("NumExperts = %d, want 128", cfg.NumExperts)
		}
		if cfg.NumExpertsPerToken != 6 {
			t.Errorf("NumExpertsPerToken = %d, want 6", cfg.NumExpertsPerToken)
		}
	})

	t.Run("nemotron_h dense with SSM", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                  "nemotron_h",
			"nemotron_h.embedding_length":           uint32(4096),
			"nemotron_h.block_count":                uint32(32),
			"nemotron_h.attention.head_count":       uint32(32),
			"nemotron_h.feed_forward_length":        uint32(16384),
			"nemotron_h.context_length":             uint32(8192),
			"nemotron_h.ssm.state_size":             uint32(64),
			"nemotron_h.ssm.conv_kernel":            uint32(4),
			"nemotron_h.ssm.num_heads":              uint32(32),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.Architecture != "nemotron_h" {
			t.Errorf("Architecture = %q, want nemotron_h", cfg.Architecture)
		}
		if cfg.SSMStateSize != 64 {
			t.Errorf("SSMStateSize = %d, want 64", cfg.SSMStateSize)
		}
		if cfg.SSMConvKernel != 4 {
			t.Errorf("SSMConvKernel = %d, want 4", cfg.SSMConvKernel)
		}
		if cfg.SSMNumHeads != 32 {
			t.Errorf("SSMNumHeads = %d, want 32", cfg.SSMNumHeads)
		}
		// No MoE fields for dense variant.
		if cfg.ExpertSharedCount != 0 {
			t.Errorf("ExpertSharedCount = %d, want 0", cfg.ExpertSharedCount)
		}
	})

	t.Run("non-SSM arch has zero SSM fields", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":       "llama",
			"llama.embedding_length":     uint32(4096),
			"llama.block_count":          uint32(32),
			"llama.attention.head_count": uint32(32),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.SSMStateSize != 0 || cfg.SSMConvKernel != 0 || cfg.SSMNumHeads != 0 {
			t.Errorf("expected zero SSM fields for llama, got state=%d conv=%d heads=%d",
				cfg.SSMStateSize, cfg.SSMConvKernel, cfg.SSMNumHeads)
		}
	})
}

func TestExtractModelConfig_ScoringFunc(t *testing.T) {
	t.Run("defaults to softmax", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":       "llama",
			"llama.embedding_length":     uint32(2048),
			"llama.block_count":          uint32(22),
			"llama.attention.head_count": uint32(32),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.SSMStateSize != 0 || cfg.SSMConvKernel != 0 || cfg.SSMNumHeads != 0 {
			t.Errorf("SSM fields should be zero for llama, got state=%d conv=%d heads=%d",
				cfg.SSMStateSize, cfg.SSMConvKernel, cfg.SSMNumHeads)
		}
		if cfg.ScoringFunc != "softmax" {
			t.Errorf("ScoringFunc = %q, want %q", cfg.ScoringFunc, "softmax")
		}
	})

	t.Run("sigmoid from metadata", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":             "minimax-m2",
			"minimax-m2.embedding_length":      uint32(3072),
			"minimax-m2.block_count":           uint32(32),
			"minimax-m2.attention.head_count":  uint32(24),
			"minimax-m2.expert_gating_func":    "sigmoid",
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.ScoringFunc != "sigmoid" {
			t.Errorf("ScoringFunc = %q, want %q", cfg.ScoringFunc, "sigmoid")
		}
	})
}

func TestMapTensorName_MiniMaxM2(t *testing.T) {
	tests := []struct {
		gguf string
		want string
	}{
		// Global tensors.
		{"token_embd.weight", "token_embd.weight"},
		{"output_norm.weight", "output_norm.weight"},
		{"output.weight", "output.weight"},
		// Block-level attention tensors.
		{"blk.0.attn_norm.weight", "blk.0.attn_norm.weight"},
		{"blk.0.attn_q.weight", "blk.0.attn_q.weight"},
		{"blk.0.attn_k.weight", "blk.0.attn_k.weight"},
		{"blk.0.attn_v.weight", "blk.0.attn_v.weight"},
		{"blk.0.attn_output.weight", "blk.0.attn_output.weight"},
		// QK norms.
		{"blk.0.attn_q_norm.weight", "blk.0.attn_q_norm.weight"},
		{"blk.0.attn_k_norm.weight", "blk.0.attn_k_norm.weight"},
		// FFN norm.
		{"blk.0.ffn_norm.weight", "blk.0.ffn_norm.weight"},
		// MoE router.
		{"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_inp.weight"},
		// MoE routing bias (unique to MiniMax-M2).
		{"blk.0.exp_probs_b", "blk.0.exp_probs_b"},
		// Stacked expert tensors.
		{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_gate_exps.weight"},
		{"blk.0.ffn_up_exps.weight", "blk.0.ffn_up_exps.weight"},
		{"blk.0.ffn_down_exps.weight", "blk.0.ffn_down_exps.weight"},
		// Higher layer numbers.
		{"blk.61.attn_q.weight", "blk.61.attn_q.weight"},
		{"blk.61.exp_probs_b", "blk.61.exp_probs_b"},
		{"blk.61.ffn_gate_exps.weight", "blk.61.ffn_gate_exps.weight"},
		// Unknown tensor passes through.
		{"some.unknown.tensor", "some.unknown.tensor"},
	}
	for _, tt := range tests {
		t.Run(tt.gguf, func(t *testing.T) {
			got := MapTensorName("minimax-m2", tt.gguf)
			if got != tt.want {
				t.Errorf("MapTensorName(minimax-m2, %q) = %q, want %q", tt.gguf, got, tt.want)
			}
		})
	}
}

func TestExtractModelConfig_KVHeadsDefaultsToHeads(t *testing.T) {
	meta := map[string]any{
		"general.architecture":       "llama",
		"llama.embedding_length":     uint32(4096),
		"llama.block_count":          uint32(32),
		"llama.attention.head_count": uint32(32),
		// No head_count_kv → should default to head_count.
	}

	f := &File{Metadata: meta}
	cfg, err := ExtractModelConfig(f)
	if err != nil {
		t.Fatalf("ExtractModelConfig: %v", err)
	}
	if cfg.NumKVHeads != 32 {
		t.Errorf("NumKVHeads = %d, want 32 (default to NumHeads)", cfg.NumKVHeads)
	}
}

func TestExtractModelConfig_Gemma4(t *testing.T) {
	t.Run("31B dense", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                          "gemma4",
			"general.name":                                  "Gemma 4 31B",
			"gemma4.embedding_length":                       uint32(4096),
			"gemma4.block_count":                            uint32(48),
			"gemma4.attention.head_count":                   uint32(32),
			"gemma4.attention.head_count_kv":                uint32(8),
			"gemma4.feed_forward_length":                    uint32(16384),
			"gemma4.context_length":                         uint32(131072),
			"gemma4.attention.global.head_count_kv":         uint32(4),
			"gemma4.attention.global.key_length":            uint32(256),
			"gemma4.attention.sliding.head_count_kv":        uint32(2),
			"gemma4.attention.sliding.key_length":           uint32(256),
			"gemma4.attention.k_eq_v":                       "true",
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.Architecture != "gemma4" {
			t.Errorf("Architecture = %q, want gemma4", cfg.Architecture)
		}
		if cfg.GlobalNumKVHeads != 4 {
			t.Errorf("GlobalNumKVHeads = %d, want 4", cfg.GlobalNumKVHeads)
		}
		if cfg.GlobalHeadDim != 256 {
			t.Errorf("GlobalHeadDim = %d, want 256", cfg.GlobalHeadDim)
		}
		if cfg.SlidingNumKVHeads != 2 {
			t.Errorf("SlidingNumKVHeads = %d, want 2", cfg.SlidingNumKVHeads)
		}
		if cfg.SlidingHeadDim != 256 {
			t.Errorf("SlidingHeadDim = %d, want 256", cfg.SlidingHeadDim)
		}
		if !cfg.AttentionKEqV {
			t.Error("AttentionKEqV = false, want true")
		}
		// Gemma 4 defaults.
		if cfg.SlidingWindowPattern != 6 {
			t.Errorf("SlidingWindowPattern = %d, want 6 (default)", cfg.SlidingWindowPattern)
		}
		if cfg.VocabSize != 262144 {
			t.Errorf("VocabSize = %d, want 262144 (default)", cfg.VocabSize)
		}
	})

	t.Run("26B-A4B MoE", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                          "gemma4",
			"gemma4.embedding_length":                       uint32(3072),
			"gemma4.block_count":                            uint32(36),
			"gemma4.attention.head_count":                   uint32(24),
			"gemma4.attention.head_count_kv":                uint32(8),
			"gemma4.feed_forward_length":                    uint32(12288),
			"gemma4.context_length":                         uint32(131072),
			"gemma4.attention.global.head_count_kv":         uint32(4),
			"gemma4.attention.global.key_length":            uint32(256),
			"gemma4.attention.sliding.head_count_kv":        uint32(2),
			"gemma4.attention.sliding.key_length":           uint32(256),
			"gemma4.attention.k_eq_v":                       "true",
			"gemma4.expert_count":                           uint32(128),
			"gemma4.expert_used_count":                      uint32(8),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.GlobalNumKVHeads != 4 {
			t.Errorf("GlobalNumKVHeads = %d, want 4", cfg.GlobalNumKVHeads)
		}
		if !cfg.AttentionKEqV {
			t.Error("AttentionKEqV = false, want true")
		}
		if cfg.NumExperts != 128 {
			t.Errorf("NumExperts = %d, want 128", cfg.NumExperts)
		}
		if cfg.NumExpertsPerToken != 8 {
			t.Errorf("NumExpertsPerToken = %d, want 8", cfg.NumExpertsPerToken)
		}
	})

	t.Run("E4B edge", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                   "gemma4",
			"gemma4.embedding_length":                uint32(2048),
			"gemma4.block_count":                     uint32(24),
			"gemma4.attention.head_count":            uint32(16),
			"gemma4.attention.head_count_kv":         uint32(4),
			"gemma4.feed_forward_length":             uint32(8192),
			"gemma4.context_length":                  uint32(8192),
			"gemma4.ple.hidden_size":                 uint32(256),
			"gemma4.kv_shared_layers":                uint32(4),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.PLEHiddenSize != 256 {
			t.Errorf("PLEHiddenSize = %d, want 256", cfg.PLEHiddenSize)
		}
		if cfg.KVSharedLayers != 4 {
			t.Errorf("KVSharedLayers = %d, want 4", cfg.KVSharedLayers)
		}
		if cfg.DoubleWideMLP {
			t.Error("DoubleWideMLP = true, want false for E4B")
		}
	})

	t.Run("E2B edge", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                   "gemma4",
			"gemma4.embedding_length":                uint32(1536),
			"gemma4.block_count":                     uint32(18),
			"gemma4.attention.head_count":            uint32(12),
			"gemma4.attention.head_count_kv":         uint32(4),
			"gemma4.feed_forward_length":             uint32(6144),
			"gemma4.context_length":                  uint32(8192),
			"gemma4.ple.hidden_size":                 uint32(128),
			"gemma4.kv_shared_layers":                uint32(4),
			"gemma4.mlp.double_wide":                 "true",
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.PLEHiddenSize != 128 {
			t.Errorf("PLEHiddenSize = %d, want 128", cfg.PLEHiddenSize)
		}
		if cfg.KVSharedLayers != 4 {
			t.Errorf("KVSharedLayers = %d, want 4", cfg.KVSharedLayers)
		}
		if !cfg.DoubleWideMLP {
			t.Error("DoubleWideMLP = false, want true for E2B")
		}
		// Verify defaults still apply.
		if cfg.SlidingWindowPattern != 6 {
			t.Errorf("SlidingWindowPattern = %d, want 6 (default)", cfg.SlidingWindowPattern)
		}
		if cfg.VocabSize != 262144 {
			t.Errorf("VocabSize = %d, want 262144 (default)", cfg.VocabSize)
		}
	})

	t.Run("k_eq_v as uint32", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":         "gemma4",
			"gemma4.embedding_length":      uint32(4096),
			"gemma4.block_count":           uint32(48),
			"gemma4.attention.head_count":  uint32(32),
			"gemma4.attention.k_eq_v":      uint32(1),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if !cfg.AttentionKEqV {
			t.Error("AttentionKEqV = false, want true (from uint32)")
		}
	})

	t.Run("double_wide as uint32", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":         "gemma4",
			"gemma4.embedding_length":      uint32(1536),
			"gemma4.block_count":           uint32(18),
			"gemma4.attention.head_count":  uint32(12),
			"gemma4.mlp.double_wide":       uint32(1),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if !cfg.DoubleWideMLP {
			t.Error("DoubleWideMLP = false, want true (from uint32)")
		}
	})

	t.Run("canonical llama.cpp keys (unsloth E2B)", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                         "gemma4",
			"gemma4.embedding_length":                      uint32(1536),
			"gemma4.block_count":                           uint32(35),
			"gemma4.attention.head_count":                  uint32(8),
			"gemma4.attention.head_count_kv":               uint32(1),
			"gemma4.feed_forward_length":                   uint32(8192),
			"gemma4.context_length":                        uint32(32768),
			"gemma4.attention.key_length":                  uint32(256),
			"gemma4.attention.value_length":                uint32(256),
			"gemma4.attention.key_length_swa":              uint32(256),
			"gemma4.attention.value_length_swa":            uint32(256),
			"gemma4.attention.shared_kv_layers":            uint32(10),
			"gemma4.attention.sliding_window_pattern":      uint32(5),
			"gemma4.embedding_length_per_layer_input":      uint32(256),
			"gemma4.rope.freq_base":                        float32(1000000.0),
			"gemma4.rope.freq_base_swa":                    float32(10000.0),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.GlobalHeadDim != 256 {
			t.Errorf("GlobalHeadDim = %d, want 256 (from attention.key_length)", cfg.GlobalHeadDim)
		}
		if cfg.SlidingHeadDim != 256 {
			t.Errorf("SlidingHeadDim = %d, want 256 (from attention.key_length_swa)", cfg.SlidingHeadDim)
		}
		if !cfg.AttentionKEqV {
			t.Error("AttentionKEqV = false, want true (derived from key_length == value_length)")
		}
		if cfg.KVSharedLayers != 10 {
			t.Errorf("KVSharedLayers = %d, want 10 (from attention.shared_kv_layers)", cfg.KVSharedLayers)
		}
		if cfg.PLEHiddenSize != 256 {
			t.Errorf("PLEHiddenSize = %d, want 256 (from embedding_length_per_layer_input)", cfg.PLEHiddenSize)
		}
		if cfg.SlidingWindowPattern != 5 {
			t.Errorf("SlidingWindowPattern = %d, want 5 (from attention.sliding_window_pattern)", cfg.SlidingWindowPattern)
		}
		if cfg.LocalRopeTheta != 10000.0 {
			t.Errorf("LocalRopeTheta = %v, want 10000 (from rope.freq_base_swa)", cfg.LocalRopeTheta)
		}
		if cfg.GlobalNumKVHeads != 1 {
			t.Errorf("GlobalNumKVHeads = %d, want 1 (fallback to NumKVHeads)", cfg.GlobalNumKVHeads)
		}
		if cfg.SlidingNumKVHeads != 1 {
			t.Errorf("SlidingNumKVHeads = %d, want 1 (fallback to NumKVHeads)", cfg.SlidingNumKVHeads)
		}
	})

	t.Run("routes dense to gemma4", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":        "gemma4",
			"gemma4.embedding_length":     uint32(4096),
			"gemma4.block_count":          uint32(48),
			"gemma4.attention.head_count": uint32(32),
		}
		cfg, err := ExtractModelConfig(&File{Metadata: meta})
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Architecture != "gemma4" {
			t.Errorf("Architecture = %q, want gemma4 (dense)", cfg.Architecture)
		}
	})

	t.Run("routes edge variant via PLE metadata", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":                    "gemma4",
			"gemma4.embedding_length":                 uint32(1536),
			"gemma4.block_count":                      uint32(35),
			"gemma4.attention.head_count":             uint32(8),
			"gemma4.embedding_length_per_layer_input": uint32(256),
			"gemma4.attention.shared_kv_layers":       uint32(10),
		}
		cfg, err := ExtractModelConfig(&File{Metadata: meta})
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Architecture != "gemma4e" {
			t.Errorf("Architecture = %q, want gemma4e", cfg.Architecture)
		}
	})

	t.Run("routes MoE variant via expert_count", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":        "gemma4",
			"gemma4.embedding_length":     uint32(3072),
			"gemma4.block_count":          uint32(36),
			"gemma4.attention.head_count": uint32(24),
			"gemma4.expert_count":         uint32(128),
			"gemma4.expert_used_count":    uint32(8),
		}
		cfg, err := ExtractModelConfig(&File{Metadata: meta})
		if err != nil {
			t.Fatal(err)
		}
		if cfg.Architecture != "gemma4moe" {
			t.Errorf("Architecture = %q, want gemma4moe", cfg.Architecture)
		}
	})

	t.Run("explicit vocab overrides default", func(t *testing.T) {
		meta := map[string]any{
			"general.architecture":         "gemma4",
			"gemma4.embedding_length":      uint32(4096),
			"gemma4.block_count":           uint32(48),
			"gemma4.attention.head_count":  uint32(32),
			"gemma4.vocab_size":            uint32(300000),
		}
		f := &File{Metadata: meta}
		cfg, err := ExtractModelConfig(f)
		if err != nil {
			t.Fatalf("ExtractModelConfig: %v", err)
		}
		if cfg.VocabSize != 300000 {
			t.Errorf("VocabSize = %d, want 300000 (explicit overrides default)", cfg.VocabSize)
		}
	})
}
