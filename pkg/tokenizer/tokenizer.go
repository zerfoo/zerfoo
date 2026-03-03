// Package tokenizer provides text tokenization for ML model inference.
//
// The Tokenizer interface abstracts over different tokenization algorithms
// (whitespace, BPE, SentencePiece). Implementations include WhitespaceTokenizer
// for testing and BPETokenizer for production use with HuggingFace models.
package tokenizer

import (
	"strings"
)

// SpecialTokens holds IDs for commonly used special tokens.
type SpecialTokens struct {
	BOS int // Beginning of sequence
	EOS int // End of sequence
	PAD int // Padding
	UNK int // Unknown token
}

// Tokenizer is the interface for all tokenizer implementations.
type Tokenizer interface {
	// Encode converts text into a sequence of token IDs.
	Encode(text string) ([]int, error)

	// Decode converts a sequence of token IDs back into text.
	Decode(ids []int) (string, error)

	// VocabSize returns the total number of tokens in the vocabulary.
	VocabSize() int

	// GetToken returns the string token for a given ID and whether it exists.
	GetToken(id int) (string, bool)

	// GetID returns the token ID for a given string and whether it exists.
	GetID(token string) (int, bool)

	// SpecialTokens returns the special token IDs for this tokenizer.
	SpecialTokens() SpecialTokens
}

// WhitespaceTokenizer provides simple whitespace-based tokenization.
// It splits text on whitespace boundaries and maps words to integer IDs.
// Useful for testing and non-production scenarios.
type WhitespaceTokenizer struct {
	vocab        map[string]int
	reverseVocab map[int]string
	nextID       int
	special      SpecialTokens
}

// NewWhitespaceTokenizer creates a WhitespaceTokenizer pre-loaded with
// standard special tokens: <unk> (0), <s> (1), </s> (2), <pad> (3).
func NewWhitespaceTokenizer() *WhitespaceTokenizer {
	t := &WhitespaceTokenizer{
		vocab:        make(map[string]int),
		reverseVocab: make(map[int]string),
		nextID:       0,
	}
	unkID := t.AddToken("<unk>")
	bosID := t.AddToken("<s>")
	eosID := t.AddToken("</s>")
	padID := t.AddToken("<pad>")
	t.special = SpecialTokens{
		BOS: bosID,
		EOS: eosID,
		PAD: padID,
		UNK: unkID,
	}
	return t
}

// AddToken adds a token to the vocabulary if it does not already exist.
// Returns the token's ID.
func (t *WhitespaceTokenizer) AddToken(token string) int {
	if id, ok := t.vocab[token]; ok {
		return id
	}
	id := t.nextID
	t.vocab[token] = id
	t.reverseVocab[id] = token
	t.nextID++
	return id
}

// Encode splits text on whitespace and returns token IDs.
// Unknown words map to the UNK token ID.
func (t *WhitespaceTokenizer) Encode(text string) ([]int, error) {
	words := strings.Fields(text)
	tokenIDs := make([]int, len(words))
	for i, word := range words {
		if id, ok := t.vocab[word]; ok {
			tokenIDs[i] = id
		} else {
			tokenIDs[i] = t.special.UNK
		}
	}
	return tokenIDs, nil
}

// Decode converts token IDs back to a space-separated string.
func (t *WhitespaceTokenizer) Decode(ids []int) (string, error) {
	words := make([]string, len(ids))
	for i, id := range ids {
		if word, ok := t.reverseVocab[id]; ok {
			words[i] = word
		} else {
			words[i] = "<unk>"
		}
	}
	return strings.Join(words, " "), nil
}

// VocabSize returns the number of tokens in the vocabulary.
func (t *WhitespaceTokenizer) VocabSize() int {
	return len(t.vocab)
}

// GetToken returns the string token for a given ID.
func (t *WhitespaceTokenizer) GetToken(id int) (string, bool) {
	word, ok := t.reverseVocab[id]
	return word, ok
}

// GetID returns the token ID for a given string.
func (t *WhitespaceTokenizer) GetID(token string) (int, bool) {
	id, ok := t.vocab[token]
	return id, ok
}

// SpecialTokens returns the special token IDs.
func (t *WhitespaceTokenizer) SpecialTokens() SpecialTokens {
	return t.special
}

// Statically assert WhitespaceTokenizer implements Tokenizer.
var _ Tokenizer = (*WhitespaceTokenizer)(nil)
