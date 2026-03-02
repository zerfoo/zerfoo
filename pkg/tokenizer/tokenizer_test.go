package tokenizer

import (
	"testing"
)

func TestNewTokenizer(t *testing.T) {
	tok := NewTokenizer()

	// Should have 3 special tokens: <unk>, <s>, </s>
	tests := []struct {
		token  string
		wantID int
	}{
		{"<unk>", 0},
		{"<s>", 1},
		{"</s>", 2},
	}

	for _, tc := range tests {
		id, ok := tok.vocab[tc.token]
		if !ok {
			t.Errorf("NewTokenizer() missing special token %q", tc.token)
			continue
		}
		if id != tc.wantID {
			t.Errorf("NewTokenizer() token %q id = %d, want %d", tc.token, id, tc.wantID)
		}
	}

	if tok.nextID != 3 {
		t.Errorf("NewTokenizer() nextID = %d, want 3", tok.nextID)
	}
}

func TestAddToken(t *testing.T) {
	tok := NewTokenizer()

	tests := []struct {
		name   string
		token  string
		wantID int
	}{
		{"new token", "hello", 3},
		{"another new token", "world", 4},
		{"duplicate token", "hello", 3},
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

	// Verify reverse vocab consistency
	for token, id := range tok.vocab {
		if got := tok.reverseVocab[id]; got != token {
			t.Errorf("reverseVocab[%d] = %q, want %q", id, got, token)
		}
	}
}

func TestEncode(t *testing.T) {
	tok := NewTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"known words", "hello world", []int{3, 4}},
		{"unknown word", "foo", []int{0}},
		{"mixed known and unknown", "hello foo world", []int{3, 0, 4}},
		{"empty string", "", []int{}},
		{"all unknown", "a b c", []int{0, 0, 0}},
		{"extra whitespace", "  hello   world  ", []int{3, 4}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tok.Encode(tc.input)
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

func TestDecode(t *testing.T) {
	tok := NewTokenizer()
	tok.AddToken("hello")
	tok.AddToken("world")

	tests := []struct {
		name     string
		input    []int
		wantText string
	}{
		{"valid IDs", []int{3, 4}, "hello world"},
		{"special tokens", []int{0, 1, 2}, "<unk> <s> </s>"},
		{"out of range ID", []int{999}, "<unk>"},
		{"empty slice", []int{}, ""},
		{"mixed valid and invalid", []int{3, 999, 4}, "hello <unk> world"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tok.Decode(tc.input)
			if got != tc.wantText {
				t.Errorf("Decode(%v) = %q, want %q", tc.input, got, tc.wantText)
			}
		})
	}
}

func TestGetToken(t *testing.T) {
	tok := NewTokenizer()
	tok.AddToken("hello")

	tests := []struct {
		name string
		id   int
		want string
	}{
		{"special token", 0, "<unk>"},
		{"user token", 3, "hello"},
		{"out of range", 999, "<unk>"},
		{"negative ID", -1, "<unk>"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tok.GetToken(tc.id)
			if got != tc.want {
				t.Errorf("GetToken(%d) = %q, want %q", tc.id, got, tc.want)
			}
		})
	}
}

func TestRoundTrip(t *testing.T) {
	tok := NewTokenizer()
	tok.AddToken("the")
	tok.AddToken("quick")
	tok.AddToken("brown")
	tok.AddToken("fox")

	text := "the quick brown fox"
	ids := tok.Encode(text)
	decoded := tok.Decode(ids)

	if decoded != text {
		t.Errorf("round-trip failed: Encode(%q) = %v, Decode = %q", text, ids, decoded)
	}
}
