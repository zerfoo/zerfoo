package grammar

// Grammar is an immutable state machine that tracks the current parse position
// within a JSON value constrained by a JSON Schema. Each call to Advance
// consumes one byte and returns a new Grammar (or reports the byte as invalid).
type Grammar struct {
	node node
}

// Advance consumes byte b and returns the resulting grammar state.
// If b is not a valid next byte, ok is false.
func (g *Grammar) Advance(b byte) (next *Grammar, ok bool) {
	n, valid := g.node.advance(b)
	if !valid {
		return nil, false
	}
	return &Grammar{node: n}, true
}

// ValidBytes returns every byte that is a valid next character in the current
// state. This is used by constrained decoding to mask logits.
func (g *Grammar) ValidBytes() []byte {
	return g.node.validBytes()
}

// IsComplete returns true when the grammar is in an accepting state — i.e. a
// complete, valid JSON value has been consumed.
func (g *Grammar) IsComplete() bool {
	return g.node.isComplete()
}

// node is the internal interface implemented by each grammar state kind.
type node interface {
	advance(b byte) (node, bool)
	validBytes() []byte
	isComplete() bool
}
