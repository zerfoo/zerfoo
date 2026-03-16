package grammar

// TokenMask returns a boolean mask over vocab.
// mask[i] is true if vocab token i is a valid next token at the current grammar state.
//
// For each token, every byte must advance the grammar successfully.
// If any byte is rejected by Grammar.Advance, the token is invalid.
func TokenMask(g *Grammar, vocab []string) []bool {
	mask := make([]bool, len(vocab))
	for i, tok := range vocab {
		if tok == "" {
			continue
		}
		cur := g
		valid := true
		for _, b := range []byte(tok) {
			next, ok := cur.Advance(b)
			if !ok {
				valid = false
				break
			}
			cur = next
		}
		mask[i] = valid
	}
	return mask
}
