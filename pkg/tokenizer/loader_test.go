package tokenizer

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadFromJSON(t *testing.T) {
	tok, err := LoadFromJSON(filepath.Join("testdata", "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}

	// Verify vocabulary was loaded.
	if got := tok.VocabSize(); got != 22 {
		t.Errorf("VocabSize() = %d, want 22", got)
	}

	// Verify encoding.
	ids, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 17 {
		t.Errorf("Encode(\"hello\") = %v, want [17]", ids)
	}

	// Verify multi-word encoding.
	ids, err = tok.Encode("hello world")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 2 || ids[0] != 17 || ids[1] != 21 {
		t.Errorf("Encode(\"hello world\") = %v, want [17 21]", ids)
	}

	// Verify special tokens.
	special := tok.SpecialTokens()
	if special.BOS != 1 {
		t.Errorf("BOS = %d, want 1", special.BOS)
	}
	if special.EOS != 2 {
		t.Errorf("EOS = %d, want 2", special.EOS)
	}
	if special.PAD != 3 {
		t.Errorf("PAD = %d, want 3", special.PAD)
	}
	if special.UNK != 0 {
		t.Errorf("UNK = %d, want 0", special.UNK)
	}
}

func TestLoadFromJSON_ByteLevel(t *testing.T) {
	// Create a byte-level pre-tokenizer fixture.
	fixture := `{
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "ab": 5},
    "merges": ["a b"]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ],
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "ByteLevel"}
    ]
  }
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	tok, err := LoadFromJSON(path)
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}
	if !tok.byteLevelBPE {
		t.Error("expected byteLevelBPE to be true for ByteLevel pre-tokenizer")
	}
}

func TestLoadFromJSON_Normalizer(t *testing.T) {
	fixture := `{
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "hello": 1, "h": 2, "e": 3, "l": 4, "o": 5},
    "merges": ["h e", "he l", "hel l", "hell o"]
  },
  "added_tokens": [],
  "normalizer": {
    "type": "Sequence",
    "normalizers": [
      {"type": "Lowercase"},
      {"type": "NFC"}
    ]
  }
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	tok, err := LoadFromJSON(path)
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}

	// "HELLO" should be lowercased then tokenized.
	ids, err := tok.Encode("HELLO")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 1 {
		t.Errorf("Encode(\"HELLO\") with Lowercase normalizer = %v, want [1]", ids)
	}
}

func TestLoadFromJSON_ArrayMerges(t *testing.T) {
	// Gemma 3 style: merges as [["a","b"], …] instead of ["a b", …].
	fixture := `{
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "a": 3, "b": 4, "ab": 5},
    "merges": [["a", "b"]]
  },
  "added_tokens": [
    {"id": 0, "content": "<unk>", "special": true},
    {"id": 1, "content": "<s>", "special": true},
    {"id": 2, "content": "</s>", "special": true}
  ]
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	tok, err := LoadFromJSON(path)
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}
	if got := tok.VocabSize(); got != 6 {
		t.Errorf("VocabSize() = %d, want 6", got)
	}

	ids, err := tok.Encode("ab")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(ids) != 1 || ids[0] != 5 {
		t.Errorf("Encode(\"ab\") = %v, want [5]", ids)
	}
}

func TestLoadFromJSON_InvalidModel(t *testing.T) {
	fixture := `{
  "model": {"type": "WordPiece", "vocab": {}, "merges": []},
  "added_tokens": []
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFromJSON(path)
	if err == nil {
		t.Fatal("expected error for WordPiece model type")
	}
}

func TestLoadFromJSON_InvalidMerge(t *testing.T) {
	fixture := `{
  "model": {"type": "BPE", "vocab": {}, "merges": ["nospace"]},
  "added_tokens": []
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFromJSON(path)
	if err == nil {
		t.Fatal("expected error for invalid merge format")
	}
}

func TestLoadFromJSON_FileNotFound(t *testing.T) {
	_, err := LoadFromJSON("/nonexistent/tokenizer.json")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadFromJSON_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := LoadFromJSON(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestLoadFromJSON_RoundTrip(t *testing.T) {
	tok, err := LoadFromJSON(filepath.Join("testdata", "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}

	text := "hello"
	ids, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	decoded, err := tok.Decode(ids)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != text {
		t.Errorf("round-trip: %q -> %v -> %q", text, ids, decoded)
	}
}

func TestBuildNormalizer_AllTypes(t *testing.T) {
	tests := []struct {
		name     string
		norm     *normalizerJSON
		input    string
		expected string
		isNil    bool
	}{
		{"nil", nil, "", "", true},
		{"NFC", &normalizerJSON{Type: "NFC"}, "\u00e9", "\u00e9", false},
		{"NFD", &normalizerJSON{Type: "NFD"}, "\u00e9", "e\u0301", false},
		{"Lowercase", &normalizerJSON{Type: "Lowercase"}, "HELLO", "hello", false},
		{"Strip", &normalizerJSON{Type: "Strip"}, "  hello  ", "hello", false},
		{"Unknown", &normalizerJSON{Type: "FooBar"}, "", "", true},
		{
			"Sequence",
			&normalizerJSON{
				Type: "Sequence",
				Normalizers: []normalizerJSON{
					{Type: "Lowercase"},
					{Type: "Strip"},
				},
			},
			"  HELLO  ",
			"hello",
			false,
		},
		{
			"Sequence empty",
			&normalizerJSON{
				Type:        "Sequence",
				Normalizers: []normalizerJSON{},
			},
			"",
			"",
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fn := buildNormalizer(tt.norm)
			if tt.isNil {
				if fn != nil {
					t.Error("expected nil normalizer")
				}
				return
			}
			if fn == nil {
				t.Fatal("expected non-nil normalizer")
			}
			got := fn(tt.input)
			if got != tt.expected {
				t.Errorf("normalizer(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestByteLevelPreTokenize(t *testing.T) {
	// Create a minimal byte-level BPE tokenizer.
	fixture := `{
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2},
    "merges": []
  },
  "added_tokens": [],
  "pre_tokenizer": {
    "type": "ByteLevel"
  }
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	tok, err := LoadFromJSON(path)
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}

	tests := []struct {
		name  string
		input string
		want  int // minimum number of tokens
	}{
		{"single word", "hello", 1},
		{"two words", "hello world", 2},
		{"leading space", " hello", 2},
		{"multiple spaces", "a  b", 3},
		{"tab and newline", "a\tb\nc", 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			words := tok.byteLevelPreTokenize(tt.input)
			if len(words) < tt.want {
				t.Errorf("byteLevelPreTokenize(%q) produced %d tokens, want >= %d: %v",
					tt.input, len(words), tt.want, words)
			}
		})
	}
}

func TestLoadFromJSON_SentencePieceDecoder(t *testing.T) {
	tests := []struct {
		name          string
		decoder       string
		wantSP        bool
	}{
		{
			name: "Metaspace decoder",
			decoder: `{"type": "Metaspace"}`,
			wantSP: true,
		},
		{
			name: "Sequence with Replace U+2581",
			decoder: `{
				"type": "Sequence",
				"decoders": [
					{"type": "Replace", "pattern": {"String": "\u2581"}, "content": " "},
					{"type": "Fuse"}
				]
			}`,
			wantSP: true,
		},
		{
			name: "no decoder",
			decoder: "",
			wantSP: false,
		},
		{
			name: "non-SentencePiece decoder",
			decoder: `{"type": "ByteLevel"}`,
			wantSP: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decoderField := ""
			if tt.decoder != "" {
				decoderField = `,"decoder": ` + tt.decoder
			}
			fixture := `{
  "model": {
    "type": "BPE",
    "vocab": {"\u2581hello": 0, "\u2581world": 1, "<s>": 2, "</s>": 3},
    "merges": []
  },
  "added_tokens": [
    {"id": 2, "content": "<s>", "special": true},
    {"id": 3, "content": "</s>", "special": true}
  ]` + decoderField + `}`

			dir := t.TempDir()
			path := filepath.Join(dir, "tokenizer.json")
			if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
				t.Fatal(err)
			}

			tok, err := LoadFromJSON(path)
			if err != nil {
				t.Fatalf("LoadFromJSON error: %v", err)
			}

			if tok.sentencePiece != tt.wantSP {
				t.Errorf("sentencePiece = %v, want %v", tok.sentencePiece, tt.wantSP)
			}
		})
	}
}

func TestLoadFromJSON_SentencePieceDecode(t *testing.T) {
	// Test that tokens with U+2581 prefix are decoded with spaces.
	fixture := `{
  "model": {
    "type": "BPE",
    "vocab": {"\u2581hello": 0, "\u2581world": 1, "foo": 2},
    "merges": []
  },
  "added_tokens": [],
  "decoder": {"type": "Metaspace"}
}`
	dir := t.TempDir()
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, []byte(fixture), 0o600); err != nil {
		t.Fatal(err)
	}

	tok, err := LoadFromJSON(path)
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}

	// Decode tokens containing U+2581 prefix — should produce spaces.
	decoded, err := tok.Decode([]int{0, 1})
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "hello world" {
		t.Errorf("Decode([0,1]) = %q, want %q", decoded, "hello world")
	}

	// Decode tokens without U+2581 — no spurious spaces.
	decoded, err = tok.Decode([]int{2})
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if decoded != "foo" {
		t.Errorf("Decode([2]) = %q, want %q", decoded, "foo")
	}
}

func TestLoadFromJSON_EncodeWithSpecialTokens(t *testing.T) {
	tok, err := LoadFromJSON(filepath.Join("testdata", "tokenizer.json"))
	if err != nil {
		t.Fatalf("LoadFromJSON error: %v", err)
	}

	ids, err := tok.EncodeWithSpecialTokens("hello", true, true)
	if err != nil {
		t.Fatalf("EncodeWithSpecialTokens error: %v", err)
	}
	// BOS=1, hello=17, EOS=2
	want := []int{1, 17, 2}
	if len(ids) != len(want) {
		t.Fatalf("got %v (len=%d), want %v (len=%d)", ids, len(ids), want, len(want))
	}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("[%d] = %d, want %d", i, id, want[i])
		}
	}
}
