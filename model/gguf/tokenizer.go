package gguf

import (
	"fmt"
	"strings"

	tokenizer "github.com/zerfoo/ztoken"
)

// ExtractTokenizer builds a BPETokenizer from GGUF metadata. GGUF files store
// tokenizer data under the "tokenizer.ggml.*" metadata keys.
func ExtractTokenizer(f *File) (*tokenizer.BPETokenizer, error) {
	// Extract token vocabulary.
	tokensRaw, ok := f.Metadata["tokenizer.ggml.tokens"]
	if !ok {
		return nil, fmt.Errorf("missing tokenizer.ggml.tokens metadata")
	}
	tokensArr, ok := tokensRaw.([]any)
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens: expected array, got %T", tokensRaw)
	}

	vocab := make(map[string]int, len(tokensArr))
	for i, v := range tokensArr {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("tokenizer.ggml.tokens[%d]: expected string, got %T", i, v)
		}
		vocab[s] = i
	}

	// Extract merges (optional -- some models have no merges).
	var merges []tokenizer.MergePair
	if mergesRaw, ok := f.Metadata["tokenizer.ggml.merges"]; ok {
		mergesArr, ok := mergesRaw.([]any)
		if !ok {
			return nil, fmt.Errorf("tokenizer.ggml.merges: expected array, got %T", mergesRaw)
		}
		merges = make([]tokenizer.MergePair, 0, len(mergesArr))
		for i, v := range mergesArr {
			s, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("tokenizer.ggml.merges[%d]: expected string, got %T", i, v)
			}
			left, right, found := strings.Cut(s, " ")
			if !found {
				return nil, fmt.Errorf("tokenizer.ggml.merges[%d]: invalid merge %q", i, s)
			}
			merges = append(merges, tokenizer.MergePair{Left: left, Right: right})
		}
	}

	// Extract special token IDs.
	special := tokenizer.SpecialTokens{}
	if v, ok := getUint32Meta(f, "tokenizer.ggml.bos_token_id"); ok {
		special.BOS = int(v)
	}
	if v, ok := getUint32Meta(f, "tokenizer.ggml.eos_token_id"); ok {
		special.EOS = int(v)
	}
	if v, ok := getUint32Meta(f, "tokenizer.ggml.unknown_token_id"); ok {
		special.UNK = int(v)
	}
	if v, ok := getUint32Meta(f, "tokenizer.ggml.padding_token_id"); ok {
		special.PAD = int(v)
	}

	// GPT-2 style models use byte-level BPE encoding.
	var byteLevelBPE bool
	if model, ok := f.GetString("tokenizer.ggml.model"); ok && model == "gpt2" {
		byteLevelBPE = true
	}
	tok := tokenizer.NewBPETokenizer(vocab, merges, special, byteLevelBPE)

	// Extract token scores for SentencePiece unigram models (e.g., Mistral, Llama).
	// These models have no merges; they use scores for greedy encoding.
	if scoresRaw, ok := f.Metadata["tokenizer.ggml.scores"]; ok {
		if scoresArr, ok := scoresRaw.([]any); ok {
			scores := make([]float32, len(scoresArr))
			for i, v := range scoresArr {
				switch s := v.(type) {
				case float32:
					scores[i] = s
				case float64:
					scores[i] = float32(s)
				}
			}
			tok.SetScores(scores)
		}
	}

	// SentencePiece models (tokenizer.ggml.model = "llama") use ▁ (U+2581)
	// as a space marker. Enable SentencePiece pre-tokenization for these.
	if model, ok := f.GetString("tokenizer.ggml.model"); ok && model == "llama" {
		tok.SetSentencePiece(true)
	}

	// Extract control/special tokens (token_type == 3) for exact matching
	// during encoding. Without this, tokens like <start_of_turn> would be
	// split into characters by BPE.
	if typesRaw, ok := f.Metadata["tokenizer.ggml.token_type"]; ok {
		if typesArr, ok := typesRaw.([]any); ok {
			specialTokens := make(map[string]int)
			for i, typeVal := range typesArr {
				var tokenType int32
				switch v := typeVal.(type) {
				case int32:
					tokenType = v
				case uint32:
					tokenType = int32(v)
				default:
					continue
				}
				// Type 3 = control/special token.
				if tokenType == 3 && i < len(tokensArr) {
					if s, ok := tokensArr[i].(string); ok && s != "" {
						specialTokens[s] = i
					}
				}
			}
			if len(specialTokens) > 0 {
				tok.SetSpecialTokenStrings(specialTokens)
			}
		}
	}

	return tok, nil
}

// getUint32Meta extracts a uint32 metadata value, handling both uint32 and int32.
func getUint32Meta(f *File, key string) (uint32, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	switch val := v.(type) {
	case uint32:
		return val, true
	case int32:
		return uint32(val), true
	default:
		return 0, false
	}
}
