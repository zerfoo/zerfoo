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
	// sentencePiece enables SentencePiece-style pre-tokenization where spaces
	// are replaced with ▁ (U+2581) and words are split at ▁ boundaries.
	sentencePiece bool
	// specialTokens maps special token strings to their IDs for exact matching
	// during encoding (e.g., "<start_of_turn>" -> 105).
	specialTokens map[string]int
	// normalizer is an optional text normalization function applied before tokenization.
	normalizer NormalizerFunc
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
	if t.normalizer != nil {
		text = t.normalizer(text)
	}

	// If special tokens are registered, split around them first.
	if len(t.specialTokens) > 0 {
		return t.encodeWithSpecials(text)
	}

	return t.encodeSegment(text, true)
}

// encodeSegment tokenizes a text segment that contains no special tokens.
// addLeadingSpace controls whether SentencePiece mode prepends ▁ to the text.
func (t *BPETokenizer) encodeSegment(text string, addLeadingSpace bool) ([]int, error) {
	if text == "" {
		return nil, nil
	}
	var words []string
	if t.byteLevelBPE {
		words = t.byteLevelPreTokenize(text)
	} else if t.sentencePiece {
		words = t.sentencePiecePreTokenize(text, addLeadingSpace)
	} else {
		words = strings.Fields(text)
	}
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

// encodeWithSpecials splits text around special token strings, encoding each
// special token as its single ID and encoding text between them with BPE.
// In SentencePiece mode, only the very first text segment (before any special
// token) gets the leading ▁ prefix. Text after special tokens does not.
func (t *BPETokenizer) encodeWithSpecials(text string) ([]int, error) {
	var ids []int
	isFirst := true
	for len(text) > 0 {
		// Find the earliest special token in the remaining text.
		bestIdx := -1
		bestLen := 0
		bestID := 0
		for tok, id := range t.specialTokens {
			idx := strings.Index(text, tok)
			if idx >= 0 && (bestIdx == -1 || idx < bestIdx || (idx == bestIdx && len(tok) > bestLen)) {
				bestIdx = idx
				bestLen = len(tok)
				bestID = id
			}
		}
		if bestIdx == -1 {
			// No more special tokens; encode the rest.
			segIDs, err := t.encodeSegment(text, isFirst)
			if err != nil {
				return nil, err
			}
			ids = append(ids, segIDs...)
			break
		}
		// Encode text before the special token.
		if bestIdx > 0 {
			segIDs, err := t.encodeSegment(text[:bestIdx], isFirst)
			if err != nil {
				return nil, err
			}
			ids = append(ids, segIDs...)
		}
		// Add the special token ID.
		ids = append(ids, bestID)
		isFirst = false
		text = text[bestIdx+bestLen:]
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
	if t.sentencePiece {
		// Replace ▁ with space and trim leading space.
		result = strings.ReplaceAll(result, "\u2581", " ")
		result = strings.TrimPrefix(result, " ")
		return result, nil
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

// SetSentencePiece enables SentencePiece-style pre-tokenization where spaces
// are replaced with ▁ (U+2581) and the text is split at ▁ boundaries.
func (t *BPETokenizer) SetSentencePiece(enabled bool) {
	t.sentencePiece = enabled
}

// SetSpecialTokenStrings registers token strings that should be matched
// as single tokens during encoding instead of being split by BPE.
func (t *BPETokenizer) SetSpecialTokenStrings(tokens map[string]int) {
	t.specialTokens = tokens
}

// sentencePiecePreTokenize implements SentencePiece-style pre-tokenization.
// Text is split on whitespace boundaries. Words that follow a space get ▁
// (U+2581) prepended. Newlines are emitted as separate tokens.
// If addLeadingSpace is true, the very first word also gets ▁ prepended.
func (t *BPETokenizer) sentencePiecePreTokenize(text string, addLeadingSpace bool) []string {
	var words []string
	isFirstWord := true
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if i > 0 {
			words = append(words, "\n")
			isFirstWord = true // newline resets: next word has no ▁ prefix
		}
		if line == "" {
			continue
		}
		lineWords := strings.SplitAfter(line, " ")
		for _, w := range lineWords {
			w = strings.TrimRight(w, " ")
			if w == "" {
				continue
			}
			if isFirstWord && addLeadingSpace {
				words = append(words, "\u2581"+w)
			} else if !isFirstWord {
				words = append(words, "\u2581"+w)
			} else {
				words = append(words, w)
			}
			isFirstWord = false
		}
	}
	return words
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
	for i := range 256 {
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
