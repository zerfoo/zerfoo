package tokenizer

import (
	"testing"
)

func TestWhitespaceTokenizer_Interface(t *testing.T) {
	// Verify WhitespaceTokenizer satisfies Tokenizer interface at runtime.
	var tok Tokenizer = NewWhitespaceTokenizer()
	if tok.VocabSize() < 4 {
		t.Errorf("VocabSize() = %d, want >= 4 (special tokens)", tok.VocabSize())
	}
}

func TestWhitespaceTokenizer_SpecialTokens(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	sp := tok.SpecialTokens()

	tests := []struct {
		name string
		id   int
		want string
	}{
		{"UNK", sp.UNK, "<unk>"},
		{"BOS", sp.BOS, "<s>"},
		{"EOS", sp.EOS, "</s>"},
		{"PAD", sp.PAD, "<pad>"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tok.GetToken(tc.id)
			if !ok {
				t.Fatalf("GetToken(%d) returned ok=false", tc.id)
			}
			if got != tc.want {
				t.Errorf("GetToken(%d) = %q, want %q", tc.id, got, tc.want)
			}
		})
	}
}

func TestWhitespaceTokenizer_AddToken(t *testing.T) {
	tok := NewWhitespaceTokenizer()

	tests := []struct {
		name   string
		token  string
		wantID int
	}{
		{"new token", "hello", 4},
		{"another new token", "world", 5},
		{"duplicate token", "hello", 4},
		{"duplicate special", "<unk>", 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			id := tok.AddToken(tc.token)
			if id != tc.wantID {
				t.Errorf("AddToken(%q) = %d, want %d", tc.token, id, tc.wantID)
			}
		})
	}

	// Verify reverse vocab consistency.
	for token, id := range tok.vocab {
		if got := tok.reverseVocab[id]; got != token {
			t.Errorf("reverseVocab[%d] = %q, want %q", id, got, token)
		}
	}
}

func TestWhitespaceTokenizer_Encode(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"known words", "hello world", []int{4, 5}},
		{"unknown word", "foo", []int{0}},
		{"mixed known and unknown", "hello foo world", []int{4, 0, 5}},
		{"empty string", "", []int{}},
		{"all unknown", "a b c", []int{0, 0, 0}},
		{"extra whitespace", "  hello   world  ", []int{4, 5}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(got) != len(tc.wantIDs) {
				t.Fatalf("Encode(%q) returned %d tokens, want %d", tc.input, len(got), len(tc.wantIDs))
			}
			for i, id := range got {
				if id != tc.wantIDs[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tc.input, i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

func TestWhitespaceTokenizer_Decode(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")

	tests := []struct {
		name     string
		input    []int
		wantText string
	}{
		{"valid IDs", []int{4, 5}, "hello world"},
		{"special tokens", []int{0, 1, 2}, "<unk> <s> </s>"},
		{"out of range ID", []int{999}, "<unk>"},
		{"empty slice", []int{}, ""},
		{"mixed valid and invalid", []int{4, 999, 5}, "hello <unk> world"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.Decode(tc.input)
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", tc.input, err)
			}
			if got != tc.wantText {
				t.Errorf("Decode(%v) = %q, want %q", tc.input, got, tc.wantText)
			}
		})
	}
}

func TestWhitespaceTokenizer_GetToken(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	tok.AddToken("hello")

	tests := []struct {
		name   string
		id     int
		want   string
		wantOK bool
	}{
		{"special token", 0, "<unk>", true},
		{"user token", 4, "hello", true},
		{"out of range", 999, "", false},
		{"negative ID", -1, "", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := tok.GetToken(tc.id)
			if ok != tc.wantOK {
				t.Errorf("GetToken(%d) ok = %v, want %v", tc.id, ok, tc.wantOK)
			}
			if got != tc.want {
				t.Errorf("GetToken(%d) = %q, want %q", tc.id, got, tc.want)
			}
		})
	}
}

func TestWhitespaceTokenizer_GetID(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	tok.AddToken("hello")

	tests := []struct {
		name   string
		token  string
		wantID int
		wantOK bool
	}{
		{"existing token", "hello", 4, true},
		{"special token", "<unk>", 0, true},
		{"missing token", "missing", 0, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			id, ok := tok.GetID(tc.token)
			if ok != tc.wantOK {
				t.Errorf("GetID(%q) ok = %v, want %v", tc.token, ok, tc.wantOK)
			}
			if id != tc.wantID {
				t.Errorf("GetID(%q) = %d, want %d", tc.token, id, tc.wantID)
			}
		})
	}
}

func TestWhitespaceTokenizer_VocabSize(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	if got := tok.VocabSize(); got != 4 {
		t.Errorf("VocabSize() = %d, want 4 (4 special tokens)", got)
	}
	tok.AddToken("hello")
	if got := tok.VocabSize(); got != 5 {
		t.Errorf("VocabSize() after AddToken = %d, want 5", got)
	}
}

func TestWhitespaceTokenizer_RoundTrip(t *testing.T) {
	tok := NewWhitespaceTokenizer()
	tok.AddToken("the")
	tok.AddToken("quick")
	tok.AddToken("brown")
	tok.AddToken("fox")

	text := "the quick brown fox"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode(%q) error: %v", text, err)
	}
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode(%v) error: %v", ids, err)
	}
	if decoded != text {
		t.Errorf("round-trip failed: Encode(%q) = %v, Decode = %q", text, ids, decoded)
	}
}
