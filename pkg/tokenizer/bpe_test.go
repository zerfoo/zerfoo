package tokenizer

import (
	"testing"
)

func makeTestBPE() *BPETokenizer {
	// Small vocabulary for testing.
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
		"<pad>": 3,
		"h":     4,
		"e":     5,
		"l":     6,
		"o":     7,
		"w":     8,
		"r":     9,
		"d":     10,
		" ":     11,
		"he":    12,
		"ll":    13,
		"lo":    14,
		"hel":   15,
		"hell":  16,
		"hello": 17,
		"wo":    18,
		"wor":   19,
		"worl":  20,
		"world": 21,
	}
	// Merge order matters: BPE always picks the globally lowest-rank pair.
	// For "hello": h,e,l,l,o -> he,l,l,o -> hel,l,o -> hel,lo -> hello
	// For "world": w,o,r,l,d -> wo,r,l,d -> wor,l,d -> worl,d -> world
	merges := []MergePair{
		{Left: "h", Right: "e"},     // rank 0: h+e -> he
		{Left: "he", Right: "l"},    // rank 1: he+l -> hel (before l+l!)
		{Left: "l", Right: "o"},     // rank 2: l+o -> lo
		{Left: "hel", Right: "lo"},  // rank 3: hel+lo -> hello
		{Left: "w", Right: "o"},     // rank 4: w+o -> wo
		{Left: "wo", Right: "r"},    // rank 5: wo+r -> wor
		{Left: "wor", Right: "l"},   // rank 6: wor+l -> worl
		{Left: "worl", Right: "d"},  // rank 7: worl+d -> world
		{Left: "l", Right: "l"},     // rank 8: l+l -> ll (low priority)
		{Left: "hel", Right: "l"},   // rank 9: hel+l -> hell (alt path)
		{Left: "hell", Right: "o"},  // rank 10: hell+o -> hello (alt path)
	}
	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 3, UNK: 0}
	return NewBPETokenizer(vocab, merges, special, false)
}

func TestBPETokenizer_Encode(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{"single word", "hello", []int{17}},
		{"two words whitespace split", "hello world", []int{17, 21}},
		{"empty string", "", []int{}},
		{"unknown characters", "xyz", []int{0, 0, 0}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.Encode(tc.input)
			if err != nil {
				t.Fatalf("Encode(%q) error: %v", tc.input, err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("Encode(%q) len = %d, want %d; got %v", tc.input, len(ids), len(tc.wantIDs), ids)
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("Encode(%q)[%d] = %d, want %d", tc.input, i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

func TestBPETokenizer_Decode(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name     string
		ids      []int
		wantText string
		wantErr  bool
	}{
		{"single token", []int{17}, "hello", false},
		{"multiple tokens", []int{17, 11, 21}, "hello world", false},
		{"empty", []int{}, "", false},
		{"unknown ID", []int{999}, "", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tok.Decode(tc.ids)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Decode(%v) expected error, got %q", tc.ids, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("Decode(%v) error: %v", tc.ids, err)
			}
			if got != tc.wantText {
				t.Errorf("Decode(%v) = %q, want %q", tc.ids, got, tc.wantText)
			}
		})
	}
}

func TestBPETokenizer_RoundTrip(t *testing.T) {
	tok := makeTestBPE()
	// Add space token for round-trip
	text := "hello"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode(%q) error: %v", text, err)
	}
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode(%v) error: %v", ids, err)
	}
	if decoded != text {
		t.Errorf("round-trip: Encode(%q) = %v, Decode = %q", text, ids, decoded)
	}
}

func TestBPETokenizer_EncodeWithSpecialTokens(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		name    string
		text    string
		addBOS  bool
		addEOS  bool
		wantIDs []int
	}{
		{"no special tokens", "hello", false, false, []int{17}},
		{"with BOS", "hello", true, false, []int{1, 17}},
		{"with EOS", "hello", false, true, []int{17, 2}},
		{"with both", "hello", true, true, []int{1, 17, 2}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ids, err := tok.EncodeWithSpecialTokens(tc.text, tc.addBOS, tc.addEOS)
			if err != nil {
				t.Fatalf("EncodeWithSpecialTokens error: %v", err)
			}
			if len(ids) != len(tc.wantIDs) {
				t.Fatalf("got %v (len=%d), want %v (len=%d)", ids, len(ids), tc.wantIDs, len(tc.wantIDs))
			}
			for i, id := range ids {
				if id != tc.wantIDs[i] {
					t.Errorf("[%d] = %d, want %d", i, id, tc.wantIDs[i])
				}
			}
		})
	}
}

func TestBPETokenizer_VocabSize(t *testing.T) {
	tok := makeTestBPE()
	if got := tok.VocabSize(); got != 22 {
		t.Errorf("VocabSize() = %d, want 22", got)
	}
}

func TestBPETokenizer_GetToken(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		id     int
		want   string
		wantOK bool
	}{
		{17, "hello", true},
		{0, "<unk>", true},
		{999, "", false},
	}

	for _, tc := range tests {
		got, ok := tok.GetToken(tc.id)
		if ok != tc.wantOK || got != tc.want {
			t.Errorf("GetToken(%d) = (%q, %v), want (%q, %v)", tc.id, got, ok, tc.want, tc.wantOK)
		}
	}
}

func TestBPETokenizer_GetID(t *testing.T) {
	tok := makeTestBPE()

	tests := []struct {
		token  string
		wantID int
		wantOK bool
	}{
		{"hello", 17, true},
		{"<unk>", 0, true},
		{"missing", 0, false},
	}

	for _, tc := range tests {
		id, ok := tok.GetID(tc.token)
		if ok != tc.wantOK || id != tc.wantID {
			t.Errorf("GetID(%q) = (%d, %v), want (%d, %v)", tc.token, id, ok, tc.wantID, tc.wantOK)
		}
	}
}

func TestBPETokenizer_MergeOrder(t *testing.T) {
	// Verify that BPE merges are applied in priority order.
	// With merges: h+e (0), l+l (1), l+o (2), he+l (3), hel+l (4), hell+o (5)
	// "hello" should merge as: h,e,l,l,o -> he,l,l,o -> hel,l,o -> hell,o -> hello
	tok := makeTestBPE()
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 17 {
		t.Errorf("Encode(\"hello\") = %v, want [17]", ids)
	}
}

func TestByteLevelBPEEncoderDecoder(t *testing.T) {
	enc, dec := buildByteEncoderDecoder()

	// All 256 bytes must be mapped.
	if len(enc) != 256 {
		t.Errorf("encoder size = %d, want 256", len(enc))
	}
	if len(dec) != 256 {
		t.Errorf("decoder size = %d, want 256", len(dec))
	}

	// Verify round-trip for every byte.
	for i := range 256 {
		b := byte(i)
		r := enc[b]
		got, ok := dec[r]
		if !ok {
			t.Errorf("byte %d: encoded to %c (U+%04X), decoder missing", i, r, r)
			continue
		}
		if got != b {
			t.Errorf("byte %d: encoded to %c, decoded to %d", i, r, got)
		}
	}

	// Printable ASCII should map to themselves.
	for b := byte(33); b <= 126; b++ {
		if enc[b] != rune(b) {
			t.Errorf("byte %d (%c): expected self-mapping, got U+%04X", b, b, enc[b])
		}
	}
}

func TestBPETokenizer_ByteLevelBPE(t *testing.T) {
	// Build a tiny byte-level BPE tokenizer.
	enc, _ := buildByteEncoderDecoder()

	// Map individual byte characters to IDs.
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
	}
	nextID := 3
	for i := range 256 {
		s := string(enc[byte(i)])
		vocab[s] = nextID
		nextID++
	}

	// Add a merge for "h" + "i" = "hi"
	hChar := string(enc['h'])
	iChar := string(enc['i'])
	hiToken := hChar + iChar
	vocab[hiToken] = nextID

	merges := []MergePair{
		{Left: hChar, Right: iChar},
	}
	special := SpecialTokens{BOS: 1, EOS: 2, PAD: 0, UNK: 0}
	tok := NewBPETokenizer(vocab, merges, special, true)

	// Encode "hi" -- should produce the merged token.
	ids, err := tok.Encode("hi")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != nextID {
		t.Errorf("Encode(\"hi\") = %v, want [%d]", ids, nextID)
	}

	// Decode should round-trip.
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "hi" {
		t.Errorf("Decode(%v) = %q, want \"hi\"", ids, decoded)
	}
}
