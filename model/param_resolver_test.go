package model

import "testing"

func TestNewParamResolver_ArchitectureSelection(t *testing.T) {
	tests := []struct {
		arch     string
		wantType string
	}{
		{"llama", "identity"},
		{"gemma", "identity"},
		{"gemma2", "identity"},
		{"gemma3", "identity"},
		{"mistral", "identity"},
		{"qwen2", "identity"},
		{"phi", "regex"},
		{"phi3", "regex"},
		{"deepseek_v3", "identity"},
		{"unknown_arch", "identity"},
		{"", "identity"},
	}
	for _, tt := range tests {
		t.Run(tt.arch, func(t *testing.T) {
			r := NewParamResolver(tt.arch)
			if r == nil {
				t.Fatal("NewParamResolver returned nil")
			}
			switch tt.wantType {
			case "identity":
				if _, ok := r.(*identityResolver); !ok {
					t.Errorf("expected *identityResolver for arch %q, got %T", tt.arch, r)
				}
			case "regex":
				if _, ok := r.(*regexResolver); !ok {
					t.Errorf("expected *regexResolver for arch %q, got %T", tt.arch, r)
				}
			}
		})
	}
}

func TestIdentityResolver_PassesThrough(t *testing.T) {
	r := NewParamResolver("llama")
	tests := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.15.mlp.gate_proj.weight",
		"model.layers.15.mlp.up_proj.weight",
		"model.layers.15.mlp.down_proj.weight",
		"model.layers.3.input_layernorm.weight",
		"model.layers.3.post_attention_layernorm.weight",
		"model.embed_tokens.weight",
		"lm_head.weight",
		"model.norm.weight",
		"",
	}
	for _, name := range tests {
		t.Run(name, func(t *testing.T) {
			got := r.Resolve(name)
			if got != name {
				t.Errorf("Resolve(%q) = %q, want %q", name, got, name)
			}
		})
	}
}

func TestPhiResolver_RenamesDenseProj(t *testing.T) {
	r := NewParamResolver("phi")
	tests := []struct {
		input string
		want  string
	}{
		// Phi-specific: dense_proj → o_proj
		{
			"model.layers.0.self_attn.dense_proj.weight",
			"model.layers.0.self_attn.o_proj.weight",
		},
		{
			"model.layers.31.self_attn.dense_proj.weight",
			"model.layers.31.self_attn.o_proj.weight",
		},
		// Other attention projections unchanged
		{
			"model.layers.0.self_attn.q_proj.weight",
			"model.layers.0.self_attn.q_proj.weight",
		},
		{
			"model.layers.0.self_attn.k_proj.weight",
			"model.layers.0.self_attn.k_proj.weight",
		},
		{
			"model.layers.0.self_attn.v_proj.weight",
			"model.layers.0.self_attn.v_proj.weight",
		},
		// FFN names unchanged
		{
			"model.layers.0.mlp.gate_proj.weight",
			"model.layers.0.mlp.gate_proj.weight",
		},
		// Norm names unchanged
		{
			"model.layers.0.input_layernorm.weight",
			"model.layers.0.input_layernorm.weight",
		},
		// Embedding/head unchanged
		{
			"model.embed_tokens.weight",
			"model.embed_tokens.weight",
		},
		{
			"lm_head.weight",
			"lm_head.weight",
		},
		// Empty string
		{"", ""},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := r.Resolve(tt.input)
			if got != tt.want {
				t.Errorf("Resolve(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestPhiResolver_Phi3Alias(t *testing.T) {
	r := NewParamResolver("phi3")
	got := r.Resolve("model.layers.5.self_attn.dense_proj.weight")
	want := "model.layers.5.self_attn.o_proj.weight"
	if got != want {
		t.Errorf("Resolve for phi3 = %q, want %q", got, want)
	}
}

func TestGemmaResolver_Identity(t *testing.T) {
	for _, arch := range []string{"gemma", "gemma2", "gemma3"} {
		t.Run(arch, func(t *testing.T) {
			r := NewParamResolver(arch)
			name := "model.layers.0.self_attn.o_proj.weight"
			got := r.Resolve(name)
			if got != name {
				t.Errorf("Resolve(%q) = %q, want identity", name, got)
			}
		})
	}
}

func TestMistralResolver_Identity(t *testing.T) {
	r := NewParamResolver("mistral")
	name := "model.layers.0.self_attn.o_proj.weight"
	got := r.Resolve(name)
	if got != name {
		t.Errorf("Resolve(%q) = %q, want identity", name, got)
	}
}

func TestQwenResolver_Identity(t *testing.T) {
	r := NewParamResolver("qwen2")
	tests := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.q_proj.bias",
		"model.layers.0.self_attn.k_proj.bias",
		"model.layers.0.self_attn.v_proj.bias",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.self_attn.o_proj.bias",
	}
	for _, name := range tests {
		t.Run(name, func(t *testing.T) {
			got := r.Resolve(name)
			if got != name {
				t.Errorf("Resolve(%q) = %q, want identity", name, got)
			}
		})
	}
}

func TestDeepSeekResolver_MLANames(t *testing.T) {
	r := NewParamResolver("deepseek_v3")
	// MLA-specific names should pass through unchanged
	tests := []string{
		"model.layers.0.self_attn.kv_a_proj.weight",
		"model.layers.0.self_attn.kv_b_proj.weight",
		"model.layers.0.self_attn.q_a_proj.weight",
		"model.layers.0.self_attn.q_b_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
	}
	for _, name := range tests {
		t.Run(name, func(t *testing.T) {
			got := r.Resolve(name)
			if got != name {
				t.Errorf("Resolve(%q) = %q, want identity", name, got)
			}
		})
	}
}

func TestResolveAll(t *testing.T) {
	r := NewParamResolver("phi")

	input := map[string]string{
		"model.layers.0.self_attn.dense_proj.weight": "tensor_a",
		"model.layers.0.self_attn.q_proj.weight":     "tensor_b",
		"model.embed_tokens.weight":                   "tensor_c",
	}

	result := ResolveAll(r, input)

	// Original names preserved
	if result["model.layers.0.self_attn.dense_proj.weight"] != "tensor_a" {
		t.Error("original name 'dense_proj' should be preserved")
	}
	if result["model.layers.0.self_attn.q_proj.weight"] != "tensor_b" {
		t.Error("original name 'q_proj' should be preserved")
	}
	if result["model.embed_tokens.weight"] != "tensor_c" {
		t.Error("original name 'embed_tokens' should be preserved")
	}

	// Canonical alias added for renamed param
	if result["model.layers.0.self_attn.o_proj.weight"] != "tensor_a" {
		t.Error("canonical alias 'o_proj' should map to tensor_a")
	}

	// No spurious aliases for identity-mapped names
	if len(result) != 4 {
		t.Errorf("expected 4 entries (3 original + 1 alias), got %d", len(result))
	}
}

func TestResolveAll_NoAliasesForIdentity(t *testing.T) {
	r := NewParamResolver("llama")

	input := map[string]string{
		"model.layers.0.self_attn.q_proj.weight": "tensor_a",
		"model.layers.0.self_attn.o_proj.weight": "tensor_b",
	}

	result := ResolveAll(r, input)

	if len(result) != 2 {
		t.Errorf("expected 2 entries (no aliases for identity), got %d", len(result))
	}
}
