// tokenizer/tokenizer.go
package tokenizer

import (
	"strings"
)

// Tokenizer provides basic text tokenization functionality.
// This is a highly simplified example (whitespace tokenization).
// A feature-complete tokenizer would implement subword algorithms (BPE, WordPiece, SentencePiece).
type Tokenizer struct {
	// In a real tokenizer, this would hold the vocabulary, merges, etc.
	vocab map[string]int
	reverseVocab map[int]string
	nextID int
}

// NewTokenizer creates a new simple Tokenizer.
func NewTokenizer() *Tokenizer {
	t := &Tokenizer{
		vocab: make(map[string]int),
		reverseVocab: make(map[int]string),
		nextID: 0,
	}
	// Add some basic special tokens
	t.AddToken("<unk>") // Unknown token
	t.AddToken("<s>")   // Start of sequence
	t.AddToken("</s>")  // End of sequence
	return t
}

// AddToken adds a token to the vocabulary if it doesn't exist.
func (t *Tokenizer) AddToken(token string) int {
	if id, ok := t.vocab[token]; ok {
		return id
	}
	id := t.nextID
	t.vocab[token] = id
	t.reverseVocab[id] = token
	t.nextID++
	return id
}

// Encode converts a text string into a slice of token IDs.
// This uses simple whitespace tokenization.
func (t *Tokenizer) Encode(text string) []int {
	words := strings.Fields(text) // Split by whitespace
	tokenIDs := make([]int, len(words))
	for i, word := range words {
		if id, ok := t.vocab[word]; ok {
			tokenIDs[i] = id
		} else {
			tokenIDs[i] = t.vocab["<unk>"] // Use unknown token for OOV words
		}
	}
	return tokenIDs
}

// Decode converts a slice of token IDs back into a text string.
func (t *Tokenizer) Decode(tokenIDs []int) string {
	words := make([]string, len(tokenIDs))
	for i, id := range tokenIDs {
		if word, ok := t.reverseVocab[id]; ok {
			words[i] = word
		} else {
			words[i] = "<unk>" // Should not happen if encoding uses <unk>
		}
	}
	return strings.Join(words, " ")
}
