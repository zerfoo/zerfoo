package gguf

import (
	"testing"
)

func TestExtractTokenizer_Basic(t *testing.T) {
	// Build a minimal GGUF File with tokenizer metadata.
	tokens := make([]any, 10)
	tokens[0] = "<unk>"
	tokens[1] = "<s>"
	tokens[2] = "</s>"
	tokens[3] = "<pad>"
	tokens[4] = "h"
	tokens[5] = "e"
	tokens[6] = "l"
	tokens[7] = "o"
	tokens[8] = "he"
	tokens[9] = "lo"

	merges := []any{"h e", "l o"}

	f := &File{
		Metadata: map[string]any{
			"tokenizer.ggml.model":            "gpt2",
			"tokenizer.ggml.tokens":           tokens,
			"tokenizer.ggml.merges":           merges,
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.eos_token_id":     uint32(2),
			"tokenizer.ggml.unknown_token_id": uint32(0),
			"tokenizer.ggml.padding_token_id": uint32(3),
		},
	}

	tok, err := ExtractTokenizer(f)
	if err != nil {
		t.Fatalf("ExtractTokenizer: %v", err)
	}

	if tok.VocabSize() != 10 {
		t.Errorf("VocabSize = %d, want 10", tok.VocabSize())
	}

	special := tok.SpecialTokens()
	if special.BOS != 1 {
		t.Errorf("BOS = %d, want 1", special.BOS)
	}
	if special.EOS != 2 {
		t.Errorf("EOS = %d, want 2", special.EOS)
	}
	if special.UNK != 0 {
		t.Errorf("UNK = %d, want 0", special.UNK)
	}
	if special.PAD != 3 {
		t.Errorf("PAD = %d, want 3", special.PAD)
	}
}

func TestExtractTokenizer_EncodeDecode(t *testing.T) {
	// Build vocab: characters plus merged tokens.
	tokens := make([]any, 10)
	tokens[0] = "<unk>"
	tokens[1] = "<s>"
	tokens[2] = "</s>"
	tokens[3] = "h"
	tokens[4] = "e"
	tokens[5] = "l"
	tokens[6] = "o"
	tokens[7] = "he"
	tokens[8] = "ll"
	tokens[9] = "hello"

	merges := []any{"h e", "l l", "he ll", "hell o"}

	f := &File{
		Metadata: map[string]any{
			"tokenizer.ggml.model":            "gpt2",
			"tokenizer.ggml.tokens":           tokens,
			"tokenizer.ggml.merges":           merges,
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.eos_token_id":     uint32(2),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}

	tok, err := ExtractTokenizer(f)
	if err != nil {
		t.Fatalf("ExtractTokenizer: %v", err)
	}

	// Encode "hello" -- BPE should merge to single token.
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(ids) != 1 || ids[0] != 9 {
		t.Errorf("Encode(\"hello\") = %v, want [9]", ids)
	}

	// Decode back.
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "hello" {
		t.Errorf("Decode(%v) = %q, want %q", ids, decoded, "hello")
	}
}

func TestExtractTokenizer_MissingTokens(t *testing.T) {
	f := &File{
		Metadata: map[string]any{
			"tokenizer.ggml.model": "gpt2",
			// No tokens array.
		},
	}

	_, err := ExtractTokenizer(f)
	if err == nil {
		t.Fatal("expected error for missing tokens")
	}
}

func TestExtractTokenizer_ByteLevelBPE(t *testing.T) {
	// GPT-2 models should enable byte-level BPE. With byte-level BPE,
	// printable ASCII bytes map to themselves so a vocab of ASCII characters
	// works for encode/decode round-trips.
	tokens := make([]any, 5)
	tokens[0] = "<unk>"
	tokens[1] = "<s>"
	tokens[2] = "h"
	tokens[3] = "i"
	tokens[4] = "hi"

	merges := []any{"h i"}

	t.Run("gpt2", func(t *testing.T) {
		f := &File{
			Metadata: map[string]any{
				"tokenizer.ggml.model":            "gpt2",
				"tokenizer.ggml.tokens":           tokens,
				"tokenizer.ggml.merges":           merges,
				"tokenizer.ggml.bos_token_id":     uint32(1),
				"tokenizer.ggml.eos_token_id":     uint32(1),
				"tokenizer.ggml.unknown_token_id": uint32(0),
			},
		}

		tok, err := ExtractTokenizer(f)
		if err != nil {
			t.Fatalf("ExtractTokenizer: %v", err)
		}

		// Byte-level BPE encode/decode round-trip should work for ASCII.
		ids, err := tok.Encode("hi")
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if len(ids) != 1 || ids[0] != 4 {
			t.Errorf("Encode(\"hi\") = %v, want [4]", ids)
		}

		decoded, err := tok.Decode(ids)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		if decoded != "hi" {
			t.Errorf("Decode(%v) = %q, want %q", ids, decoded, "hi")
		}
	})

	t.Run("llama", func(t *testing.T) {
		// SentencePiece (llama) models should NOT use byte-level BPE.
		// They use ▁ (U+2581) as a space prefix instead.
		spTokens := make([]any, 7)
		spTokens[0] = "<unk>"
		spTokens[1] = "<s>"
		spTokens[2] = "\u2581" // ▁
		spTokens[3] = "h"
		spTokens[4] = "i"
		spTokens[5] = "\u2581h" // ▁h
		spTokens[6] = "\u2581hi" // ▁hi

		spMerges := []any{"\u2581 h", "\u2581h i"}

		f := &File{
			Metadata: map[string]any{
				"tokenizer.ggml.model":            "llama",
				"tokenizer.ggml.tokens":           spTokens,
				"tokenizer.ggml.merges":           spMerges,
				"tokenizer.ggml.bos_token_id":     uint32(1),
				"tokenizer.ggml.eos_token_id":     uint32(1),
				"tokenizer.ggml.unknown_token_id": uint32(0),
			},
		}

		tok, err := ExtractTokenizer(f)
		if err != nil {
			t.Fatalf("ExtractTokenizer: %v", err)
		}

		// SentencePiece prepends ▁ to text, so "hi" becomes "▁hi" -> token 6.
		ids, err := tok.Encode("hi")
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if len(ids) != 1 || ids[0] != 6 {
			t.Errorf("Encode(\"hi\") = %v, want [6]", ids)
		}
	})
}

func TestExtractTokenizer_SentencePieceScores(t *testing.T) {
	// SentencePiece unigram models have scores but no merges.
	// The tokenizer uses greedy longest-match with score-based selection.
	tokens := make([]any, 8)
	tokens[0] = "<unk>"
	tokens[1] = "<s>"
	tokens[2] = "</s>"
	tokens[3] = "\u2581"     // ▁
	tokens[4] = "\u2581he"   // ▁he
	tokens[5] = "llo"
	tokens[6] = "\u2581hello" // ▁hello (longest match, best score)
	tokens[7] = "o"

	// Scores: negative log probs. Higher (closer to 0) = more likely.
	// ▁hello should be preferred over ▁he + llo because it has a better score.
	scores := make([]any, 8)
	scores[0] = float32(0)     // <unk>
	scores[1] = float32(0)     // <s>
	scores[2] = float32(0)     // </s>
	scores[3] = float32(-3.0)  // ▁
	scores[4] = float32(-5.0)  // ▁he
	scores[5] = float32(-5.0)  // llo
	scores[6] = float32(-2.0)  // ▁hello (best score for this text)
	scores[7] = float32(-6.0)  // o

	f := &File{
		Metadata: map[string]any{
			"tokenizer.ggml.model":            "llama",
			"tokenizer.ggml.tokens":           tokens,
			"tokenizer.ggml.scores":           scores,
			"tokenizer.ggml.bos_token_id":     uint32(1),
			"tokenizer.ggml.eos_token_id":     uint32(2),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}

	tok, err := ExtractTokenizer(f)
	if err != nil {
		t.Fatalf("ExtractTokenizer: %v", err)
	}

	// Encode "hello" -- SentencePiece prepends ▁, so it becomes "▁hello".
	// With scores, the tokenizer should select token 6 (▁hello) as the
	// longest match.
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("Encode(\"hello\") returned empty")
	}

	// Decode back and verify round-trip produces readable output.
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if decoded != "hello" {
		t.Errorf("Decode(%v) = %q, want %q", ids, decoded, "hello")
	}
}

func TestExtractTokenizer_NoMerges(t *testing.T) {
	// A tokenizer with tokens but no merges should still work (character-level).
	tokens := make([]any, 4)
	tokens[0] = "<unk>"
	tokens[1] = "a"
	tokens[2] = "b"
	tokens[3] = "c"

	f := &File{
		Metadata: map[string]any{
			"tokenizer.ggml.model":            "gpt2",
			"tokenizer.ggml.tokens":           tokens,
			"tokenizer.ggml.bos_token_id":     uint32(0),
			"tokenizer.ggml.eos_token_id":     uint32(0),
			"tokenizer.ggml.unknown_token_id": uint32(0),
		},
	}

	tok, err := ExtractTokenizer(f)
	if err != nil {
		t.Fatalf("ExtractTokenizer: %v", err)
	}

	ids, err := tok.Encode("abc")
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	// Should encode as individual characters.
	if len(ids) != 3 {
		t.Errorf("Encode(\"abc\") = %v, want 3 tokens", ids)
	}
}
