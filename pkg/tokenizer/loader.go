package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// tokenizerJSON represents the HuggingFace tokenizer.json schema.
type tokenizerJSON struct {
	Model        modelJSON         `json:"model"`
	AddedTokens  []addedTokenJSON  `json:"added_tokens"`
	PreTokenizer *preTokenizerJSON `json:"pre_tokenizer"`
	Normalizer   *normalizerJSON   `json:"normalizer"`
	Decoder      *decoderJSON      `json:"decoder"`
}

type modelJSON struct {
	Type      string          `json:"type"`
	Vocab     map[string]int  `json:"vocab"`
	RawMerges json.RawMessage `json:"merges"`
}

type addedTokenJSON struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type preTokenizerJSON struct {
	Type          string             `json:"type"`
	PreTokenizers []preTokenizerJSON `json:"pretokenizers"`
}

type normalizerJSON struct {
	Type        string           `json:"type"`
	Normalizers []normalizerJSON `json:"normalizers"`
}

type decoderJSON struct {
	Type     string             `json:"type"`
	Pattern  *decoderPatternJSON `json:"pattern"`
	Content  string             `json:"content"`
	Decoders []decoderJSON      `json:"decoders"`
}

type decoderPatternJSON struct {
	String string `json:"String"`
}

// LoadFromJSON reads a HuggingFace tokenizer.json file and returns a BPETokenizer.
func LoadFromJSON(path string) (*BPETokenizer, error) {
	data, err := os.ReadFile(path) //nolint:gosec // user-provided path
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}

	if tj.Model.Type != "" && tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("unsupported model type: %q (only BPE supported)", tj.Model.Type)
	}

	// Parse merges — supports both ["a b", …] and [["a","b"], …] formats.
	merges, err := parseMerges(tj.Model.RawMerges)
	if err != nil {
		return nil, fmt.Errorf("parse merges: %w", err)
	}

	// Detect byte-level BPE from pre-tokenizer config.
	byteLevelBPE := isByteLevelPreTokenizer(tj.PreTokenizer)

	// Extract special tokens.
	special := extractSpecialTokens(tj.AddedTokens)

	// Build normalizer function if present.
	normalizer := buildNormalizer(tj.Normalizer)

	tok := NewBPETokenizer(tj.Model.Vocab, merges, special, byteLevelBPE)
	tok.normalizer = normalizer

	// Detect SentencePiece mode from the decoder config.
	if isSentencePieceDecoder(tj.Decoder) {
		tok.SetSentencePiece(true)
	}

	return tok, nil
}

// isByteLevelPreTokenizer returns true if the pre-tokenizer config uses ByteLevel.
func isByteLevelPreTokenizer(pt *preTokenizerJSON) bool {
	if pt == nil {
		return false
	}
	if pt.Type == "ByteLevel" {
		return true
	}
	if pt.Type == "Sequence" {
		for _, child := range pt.PreTokenizers {
			if child.Type == "ByteLevel" {
				return true
			}
		}
	}
	return false
}

// isSentencePieceDecoder returns true if the decoder config indicates a
// SentencePiece tokenizer. This is detected by a Metaspace decoder type or
// a Replace rule that converts U+2581 (▁) to a space.
func isSentencePieceDecoder(d *decoderJSON) bool {
	if d == nil {
		return false
	}
	if d.Type == "Metaspace" {
		return true
	}
	if d.Type == "Replace" && d.Pattern != nil && d.Pattern.String == "\u2581" {
		return true
	}
	if d.Type == "Sequence" {
		for i := range d.Decoders {
			if isSentencePieceDecoder(&d.Decoders[i]) {
				return true
			}
		}
	}
	return false
}

// extractSpecialTokens finds BOS, EOS, PAD, UNK from added_tokens.
func extractSpecialTokens(tokens []addedTokenJSON) SpecialTokens {
	special := SpecialTokens{}
	for _, t := range tokens {
		if !t.Special {
			continue
		}
		switch {
		case strings.Contains(t.Content, "bos") || t.Content == "<s>":
			special.BOS = t.ID
		case strings.Contains(t.Content, "eos") || t.Content == "</s>":
			special.EOS = t.ID
		case strings.Contains(t.Content, "pad") || t.Content == "<pad>":
			special.PAD = t.ID
		case strings.Contains(t.Content, "unk") || t.Content == "<unk>":
			special.UNK = t.ID
		}
	}
	return special
}

// NormalizerFunc transforms text before tokenization.
type NormalizerFunc func(string) string

// buildNormalizer creates a normalizer function from the JSON config.
func buildNormalizer(n *normalizerJSON) NormalizerFunc {
	if n == nil {
		return nil
	}
	switch n.Type {
	case "NFC":
		return func(s string) string { return norm.NFC.String(s) }
	case "NFD":
		return func(s string) string { return norm.NFD.String(s) }
	case "Lowercase":
		return strings.ToLower
	case "Strip":
		return func(s string) string { return strings.TrimFunc(s, unicode.IsSpace) }
	case "Sequence":
		var chain []NormalizerFunc
		for i := range n.Normalizers {
			if fn := buildNormalizer(&n.Normalizers[i]); fn != nil {
				chain = append(chain, fn)
			}
		}
		if len(chain) == 0 {
			return nil
		}
		return func(s string) string {
			for _, fn := range chain {
				s = fn(s)
			}
			return s
		}
	default:
		return nil
	}
}

// parseMerges decodes merges from JSON, accepting either space-separated
// strings (["a b", …]) or two-element arrays ([["a","b"], …]).
func parseMerges(raw json.RawMessage) ([]MergePair, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	// Try []string first (most common).
	var stringMerges []string
	if err := json.Unmarshal(raw, &stringMerges); err == nil {
		merges := make([]MergePair, 0, len(stringMerges))
		for i, m := range stringMerges {
			left, right, ok := strings.Cut(m, " ")
			if !ok {
				return nil, fmt.Errorf("invalid merge at index %d: %q", i, m)
			}
			merges = append(merges, MergePair{Left: left, Right: right})
		}
		return merges, nil
	}

	// Try [][]string (Gemma 3 format).
	var arrayMerges [][]string
	if err := json.Unmarshal(raw, &arrayMerges); err != nil {
		return nil, fmt.Errorf("unsupported merges format: %w", err)
	}
	merges := make([]MergePair, 0, len(arrayMerges))
	for i, pair := range arrayMerges {
		if len(pair) != 2 {
			return nil, fmt.Errorf("invalid merge at index %d: expected 2 elements, got %d", i, len(pair))
		}
		merges = append(merges, MergePair{Left: pair[0], Right: pair[1]})
	}
	return merges, nil
}
