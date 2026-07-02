package generate

import (
	"math"

	"github.com/zerfoo/zerfoo/generate/grammar"
)

// applyTokenMask sets logits[i] = -Inf for all i where mask[i] == false.
// This constrains sampling to only tokens allowed by a grammar.
func applyTokenMask(logits []float64, mask []bool) {
	for i := range logits {
		if i < len(mask) && !mask[i] {
			logits[i] = math.Inf(-1)
		}
	}
}

// advanceGrammar advances the grammar state through all bytes of the sampled
// token. If any byte is rejected (should not happen since the token was masked),
// the grammar state is returned unchanged.
func advanceGrammar(g *grammar.Grammar, tokenID int, vocab []string) *grammar.Grammar {
	if tokenID < 0 || tokenID >= len(vocab) {
		return g
	}
	tok := vocab[tokenID]
	cur := g
	for _, b := range []byte(tok) {
		next, ok := cur.Advance(b)
		if !ok {
			return g
		}
		cur = next
	}
	return cur
}
