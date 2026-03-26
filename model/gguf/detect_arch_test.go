package gguf

import (
	"testing"
)

func TestDetectActualArchitecture(t *testing.T) {
	tests := []struct {
		name     string
		declared string
		meta     map[string]any
		want     string
	}{
		{
			name:     "llama stays llama when no mistral signals",
			declared: "llama",
			meta: map[string]any{
				"general.name": "Meta Llama 3.1 8B",
			},
			want: "llama",
		},
		{
			name:     "detected from general.name containing Mistral",
			declared: "llama",
			meta: map[string]any{
				"general.name": "Mistral-7B-Instruct-v0.3",
			},
			want: "mistral",
		},
		{
			name:     "detected from general.name case insensitive",
			declared: "llama",
			meta: map[string]any{
				"general.name": "mistral-nemo-12b",
			},
			want: "mistral",
		},
		{
			name:     "detected from tokenizer.ggml.pre",
			declared: "llama",
			meta: map[string]any{
				"general.name":        "some-model",
				"tokenizer.ggml.pre": "mistral-v3",
			},
			want: "mistral",
		},
		{
			name:     "non-llama declared architecture passes through",
			declared: "gemma3",
			meta: map[string]any{
				"general.name": "Mistral-7B",
			},
			want: "gemma3",
		},
		{
			name:     "no metadata still returns declared",
			declared: "llama",
			meta:     map[string]any{},
			want:     "llama",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &File{Metadata: tt.meta}
			got := DetectActualArchitecture(f, tt.declared)
			if got != tt.want {
				t.Errorf("DetectActualArchitecture(%q) = %q, want %q", tt.declared, got, tt.want)
			}
		})
	}
}

func TestExtractModelConfig_MistralDetection(t *testing.T) {
	// Simulate a Mistral GGUF file that declares arch="llama".
	f := &File{
		Metadata: map[string]any{
			"general.architecture":       "llama",
			"general.name":               "Mistral-7B-Instruct-v0.3",
			"llama.vocab_size":           uint32(32768),
			"llama.embedding_length":     uint32(4096),
			"llama.block_count":          uint32(32),
			"llama.attention.head_count":    uint32(32),
			"llama.attention.head_count_kv": uint32(8),
			"llama.feed_forward_length":  uint32(14336),
			"llama.context_length":       uint32(32768),
			"llama.rope.freq_base":       float32(10000),
			"llama.attention.sliding_window": uint32(4096),
		},
	}

	cfg, err := ExtractModelConfig(f)
	if err != nil {
		t.Fatalf("ExtractModelConfig: %v", err)
	}

	if cfg.Architecture != "mistral" {
		t.Errorf("Architecture = %q, want %q", cfg.Architecture, "mistral")
	}
	if cfg.VocabSize != 32768 {
		t.Errorf("VocabSize = %d, want 32768", cfg.VocabSize)
	}
	if cfg.SlidingWindow != 4096 {
		t.Errorf("SlidingWindow = %d, want 4096", cfg.SlidingWindow)
	}
}

func TestExtractTokenizer_MistralSpecialTokens(t *testing.T) {
	// Mistral uses BOS=1, EOS=2 — verify tokenizer reads these from GGUF.
	tokens := make([]any, 5)
	tokens[0] = "<unk>"
	tokens[1] = "<s>"
	tokens[2] = "</s>"
	tokens[3] = "[INST]"
	tokens[4] = "[/INST]"

	f := &File{
		Metadata: map[string]any{
			"tokenizer.ggml.model":            "llama",
			"tokenizer.ggml.tokens":           tokens,
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.eos_token_id":     uint32(2),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}

	tok, err := ExtractTokenizer(f)
	if err != nil {
		t.Fatalf("ExtractTokenizer: %v", err)
	}

	special := tok.SpecialTokens()
	if special.BOS != 1 {
		t.Errorf("BOS = %d, want 1", special.BOS)
	}
	if special.EOS != 2 {
		t.Errorf("EOS = %d, want 2", special.EOS)
	}
}
