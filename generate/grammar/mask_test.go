package grammar

import "testing"

func TestTokenMaskNumberSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "number"})
	if err != nil {
		t.Fatal(err)
	}
	vocab := []string{"1", "42", "-3", "a", "{", "0.5", "true"}
	mask := TokenMask(g, vocab)

	want := map[string]bool{
		"1":    true,
		"42":   true,
		"-3":   true,
		"a":    false,
		"{":    false,
		"0.5":  true,
		"true": false,
	}
	for i, tok := range vocab {
		if mask[i] != want[tok] {
			t.Errorf("TokenMask[%q] = %v, want %v", tok, mask[i], want[tok])
		}
	}
}

func TestTokenMaskBooleanSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "boolean"})
	if err != nil {
		t.Fatal(err)
	}
	vocab := []string{"true", "false", "maybe", "tru", "fals"}
	mask := TokenMask(g, vocab)

	want := map[string]bool{
		"true":  true,
		"false": true,
		"maybe": false,
		"tru":   true,  // partial prefix is valid (all bytes accepted)
		"fals":  true,  // partial prefix is valid
	}
	for i, tok := range vocab {
		if mask[i] != want[tok] {
			t.Errorf("TokenMask[%q] = %v, want %v", tok, mask[i], want[tok])
		}
	}
}

func TestTokenMaskEmptyVocab(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "integer"})
	if err != nil {
		t.Fatal(err)
	}
	mask := TokenMask(g, nil)
	if len(mask) != 0 {
		t.Errorf("expected empty mask, got len=%d", len(mask))
	}
	mask = TokenMask(g, []string{})
	if len(mask) != 0 {
		t.Errorf("expected empty mask, got len=%d", len(mask))
	}
}

func TestTokenMaskEmptyToken(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "integer"})
	if err != nil {
		t.Fatal(err)
	}
	mask := TokenMask(g, []string{"", "1", ""})
	if mask[0] {
		t.Error("empty token should be invalid")
	}
	if !mask[1] {
		t.Error("\"1\" should be valid for integer")
	}
	if mask[2] {
		t.Error("empty token should be invalid")
	}
}

func TestTokenMaskWhitespaceToken(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "integer"})
	if err != nil {
		t.Fatal(err)
	}
	// Space is not a valid byte for integer grammar start.
	mask := TokenMask(g, []string{" ", "\t", "\n"})
	for i, tok := range []string{" ", "\t", "\n"} {
		if mask[i] {
			t.Errorf("whitespace token %q should be invalid for integer schema", tok)
		}
	}
}

func TestTokenMaskMixedValidInvalidBytes(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "integer"})
	if err != nil {
		t.Fatal(err)
	}
	// "1a" — first byte valid, second invalid.
	mask := TokenMask(g, []string{"1a", "1", "12", "abc"})
	want := []bool{false, true, true, false}
	for i, w := range want {
		if mask[i] != w {
			t.Errorf("mask[%d] = %v, want %v", i, mask[i], w)
		}
	}
}

func TestTokenMaskCompleteGrammar(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "null"})
	if err != nil {
		t.Fatal(err)
	}
	// Advance to complete state.
	final, ok := feedString(g, "null")
	if !ok || !final.IsComplete() {
		t.Fatal("failed to reach complete state")
	}
	// At complete state, no token should be valid (no valid continuations).
	mask := TokenMask(final, []string{"a", "1", "{", "null", " "})
	for i, tok := range []string{"a", "1", "{", "null", " "} {
		if mask[i] {
			t.Errorf("token %q should be invalid after complete grammar", tok)
		}
	}
}

func TestTokenMaskObjectPartialState(t *testing.T) {
	g, err := Convert(&JSONSchema{
		Type: "object",
		Properties: map[string]*JSONSchema{
			"name": {Type: "string"},
		},
		Required: []string{"name"},
	})
	if err != nil {
		t.Fatal(err)
	}
	// After consuming "{", valid next is '"' (start of key).
	g2, ok := g.Advance('{')
	if !ok {
		t.Fatal("expected { to be accepted")
	}
	vocab := []string{`"name"`, `"`, "}", "a", "{", "1"}
	mask := TokenMask(g2, vocab)
	// '"name"' should be valid (full key).
	if !mask[0] {
		t.Error(`expected "name" to be valid`)
	}
	// '"' should be valid (start of key).
	if !mask[1] {
		t.Error(`expected " to be valid`)
	}
	// '}' should be invalid (required field missing).
	if mask[2] {
		t.Error("expected } to be invalid (required field missing)")
	}
	// 'a' should be invalid.
	if mask[3] {
		t.Error("expected a to be invalid")
	}
}

func TestTokenMaskStringSchema(t *testing.T) {
	g, err := Convert(&JSONSchema{Type: "string"})
	if err != nil {
		t.Fatal(err)
	}
	// Initial state: only '"' starts a string.
	vocab := []string{`"hello"`, `"`, "hello", "1", "{"}
	mask := TokenMask(g, vocab)
	if !mask[0] {
		t.Error(`expected "hello" to be valid`)
	}
	if !mask[1] {
		t.Error(`expected " to be valid`)
	}
	if mask[2] {
		t.Error("expected hello (no quote) to be invalid")
	}
	if mask[3] {
		t.Error("expected 1 to be invalid")
	}
	if mask[4] {
		t.Error("expected { to be invalid")
	}
}

func TestTokenMaskSingleByteMatchesValidBytes(t *testing.T) {
	// For single-byte tokens, TokenMask should agree with ValidBytes.
	g, err := Convert(&JSONSchema{Type: "integer"})
	if err != nil {
		t.Fatal(err)
	}
	// Build single-byte vocab from all ASCII bytes.
	vocab := make([]string, 256)
	for i := range vocab {
		vocab[i] = string([]byte{byte(i)})
	}
	mask := TokenMask(g, vocab)
	validSet := make(map[byte]bool)
	for _, b := range g.ValidBytes() {
		validSet[b] = true
	}
	for i := 0; i < 256; i++ {
		if mask[i] != validSet[byte(i)] {
			t.Errorf("byte 0x%02x: TokenMask=%v, ValidBytes=%v", i, mask[i], validSet[byte(i)])
		}
	}
}
