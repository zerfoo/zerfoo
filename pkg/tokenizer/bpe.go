package tokenizer

import (
	"fmt"
	"strings"
)

// MergePair represents an adjacent token pair used in BPE merging.
type MergePair struct {
	Left  string
	Right string
}

// BPETokenizer implements the Tokenizer interface using byte-pair encoding.
// It loads vocabulary and merge rules from HuggingFace tokenizer.json format.
type BPETokenizer struct {
	vocab        map[string]int
	reverseVocab map[int]string
	mergeRanks   map[MergePair]int
	special      SpecialTokens
	// byteEncoder maps each byte (0-255) to a printable Unicode character,
	// following the GPT-2 byte-level BPE convention.
	byteEncoder map[byte]rune
	// byteDecoder is the inverse of byteEncoder.
	byteDecoder map[rune]byte
	// preTokenize controls how text is split before BPE merging.
	// If true, byte-level pre-tokenization is used (GPT-2 style).
	byteLevelBPE bool
}

// NewBPETokenizer creates a BPETokenizer from vocabulary, merge rules, and special tokens.
func NewBPETokenizer(vocab map[string]int, merges []MergePair, special SpecialTokens, byteLevelBPE bool) *BPETokenizer {
	reverseVocab := make(map[int]string, len(vocab))
	for k, v := range vocab {
		reverseVocab[v] = k
	}
	mergeRanks := make(map[MergePair]int, len(merges))
	for i, m := range merges {
		mergeRanks[m] = i
	}
	t := &BPETokenizer{
		vocab:        vocab,
		reverseVocab: reverseVocab,
		mergeRanks:   mergeRanks,
		special:      special,
		byteLevelBPE: byteLevelBPE,
	}
	if byteLevelBPE {
		t.byteEncoder, t.byteDecoder = buildByteEncoderDecoder()
	}
	return t
}

// Encode tokenizes text into a sequence of token IDs using BPE.
func (t *BPETokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return []int{}, nil
	}
	words := t.preTokenize(text)
	var ids []int
	for _, word := range words {
		wordIDs, err := t.encodeWord(word)
		if err != nil {
			return nil, err
		}
		ids = append(ids, wordIDs...)
	}
	return ids, nil
}

// EncodeWithSpecialTokens wraps Encode and optionally prepends BOS / appends EOS.
func (t *BPETokenizer) EncodeWithSpecialTokens(text string, addBOS bool, addEOS bool) ([]int, error) {
	ids, err := t.Encode(text)
	if err != nil {
		return nil, err
	}
	if addBOS {
		ids = append([]int{t.special.BOS}, ids...)
	}
	if addEOS {
		ids = append(ids, t.special.EOS)
	}
	return ids, nil
}

// Decode converts token IDs back to text.
func (t *BPETokenizer) Decode(ids []int) (string, error) {
	var sb strings.Builder
	for _, id := range ids {
		tok, ok := t.reverseVocab[id]
		if !ok {
			return "", fmt.Errorf("unknown token ID: %d", id)
		}
		sb.WriteString(tok)
	}
	result := sb.String()
	if t.byteLevelBPE {
		decoded, err := t.decodeByteLevelBPE(result)
		if err != nil {
			return "", err
		}
		return decoded, nil
	}
	return result, nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (t *BPETokenizer) VocabSize() int {
	return len(t.vocab)
}

// GetToken returns the string for a given token ID.
func (t *BPETokenizer) GetToken(id int) (string, bool) {
	tok, ok := t.reverseVocab[id]
	return tok, ok
}

// GetID returns the ID for a given token string.
func (t *BPETokenizer) GetID(token string) (int, bool) {
	id, ok := t.vocab[token]
	return id, ok
}

// SpecialTokens returns the special token configuration.
func (t *BPETokenizer) SpecialTokens() SpecialTokens {
	return t.special
}

// preTokenize splits text into words for BPE processing.
func (t *BPETokenizer) preTokenize(text string) []string {
	if t.byteLevelBPE {
		return t.byteLevelPreTokenize(text)
	}
	return strings.Fields(text)
}

// byteLevelPreTokenize converts text to byte-level BPE tokens.
// Each byte of the UTF-8 encoding is mapped to a printable Unicode character.
// Whitespace is preserved as part of tokens (prefixed to the following word).
func (t *BPETokenizer) byteLevelPreTokenize(text string) []string {
	// Split on whitespace boundaries, preserving the space as prefix of next word.
	var words []string
	var current strings.Builder
	for i, r := range text {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			// Prefix space to next word token.
			if i == 0 || (i > 0 && (text[i-1] == ' ' || text[i-1] == '\t' || text[i-1] == '\n' || text[i-1] == '\r')) {
				// Leading/consecutive space becomes its own token.
				encoded := t.encodeBytesToChars([]byte{byte(r)})
				words = append(words, encoded)
			} else {
				current.WriteString(t.encodeBytesToChars([]byte{byte(r)}))
			}
		} else {
			current.WriteString(t.encodeBytesToChars([]byte(string(r))))
		}
	}
	if current.Len() > 0 {
		words = append(words, current.String())
	}
	return words
}

// encodeBytesToChars maps raw bytes to their BPE character representation.
func (t *BPETokenizer) encodeBytesToChars(b []byte) string {
	var sb strings.Builder
	for _, c := range b {
		sb.WriteRune(t.byteEncoder[c])
	}
	return sb.String()
}

// decodeByteLevelBPE reverses byte-level encoding back to UTF-8 text.
func (t *BPETokenizer) decodeByteLevelBPE(text string) (string, error) {
	var bytes []byte
	for _, r := range text {
		b, ok := t.byteDecoder[r]
		if !ok {
			return "", fmt.Errorf("unknown byte-level BPE character: %c (U+%04X)", r, r)
		}
		bytes = append(bytes, b)
	}
	return string(bytes), nil
}

// encodeWord applies BPE merging to a single pre-tokenized word.
func (t *BPETokenizer) encodeWord(word string) ([]int, error) {
	if word == "" {
		return nil, nil
	}

	// Split into individual characters as initial tokens.
	chars := []rune(word)
	symbols := make([]string, len(chars))
	for i, c := range chars {
		symbols[i] = string(c)
	}

	// Iteratively merge the highest-priority adjacent pair.
	for len(symbols) > 1 {
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			pair := MergePair{Left: symbols[i], Right: symbols[i+1]}
			if rank, ok := t.mergeRanks[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			break // No more merges possible.
		}
		// Merge the pair at bestIdx.
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	// Look up token IDs for the final symbols.
	ids := make([]int, len(symbols))
	for i, sym := range symbols {
		id, ok := t.vocab[sym]
		if !ok {
			id = t.special.UNK
		}
		ids[i] = id
	}
	return ids, nil
}

// buildByteEncoderDecoder creates the GPT-2 byte-to-character mapping.
// Printable ASCII characters map to themselves. Other bytes map to
// Unicode characters starting at U+0100 (Latin Extended-B).
func buildByteEncoderDecoder() (map[byte]rune, map[rune]byte) {
	enc := make(map[byte]rune, 256)
	dec := make(map[rune]byte, 256)

	// Printable ASCII ranges that map to themselves:
	// '!' (33) to '~' (126), plus non-breaking characters.
	n := rune(256) // Next available Unicode codepoint for non-printable bytes.
	for i := 0; i < 256; i++ {
		b := byte(i)
		if isPrintableGPT2Byte(b) {
			enc[b] = rune(b)
			dec[rune(b)] = b
		} else {
			enc[b] = n
			dec[n] = b
			n++
		}
	}
	return enc, dec
}

// isPrintableGPT2Byte returns true if the byte maps to itself in GPT-2 encoding.
func isPrintableGPT2Byte(b byte) bool {
	// '!' (33) through '~' (126)
	if b >= 33 && b <= 126 {
		return true
	}
	// Latin-1 supplement: 161-172, 174-255
	if b >= 161 && b <= 172 {
		return true
	}
	if b >= 174 {
		return true
	}
	return false
}

// Statically assert BPETokenizer implements Tokenizer.
var _ Tokenizer = (*BPETokenizer)(nil)
