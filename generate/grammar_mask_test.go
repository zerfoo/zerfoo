package generate

import (
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/generate/grammar"
)

func TestApplyTokenMask(t *testing.T) {
	tests := []struct {
		name   string
		logits []float64
		mask   []bool
		want   []float64
	}{
		{
			name:   "all allowed",
			logits: []float64{1.0, 2.0, 3.0},
			mask:   []bool{true, true, true},
			want:   []float64{1.0, 2.0, 3.0},
		},
		{
			name:   "all blocked",
			logits: []float64{1.0, 2.0, 3.0},
			mask:   []bool{false, false, false},
			want:   []float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)},
		},
		{
			name:   "partial mask",
			logits: []float64{1.0, 2.0, 3.0, 4.0},
			mask:   []bool{true, false, true, false},
			want:   []float64{1.0, math.Inf(-1), 3.0, math.Inf(-1)},
		},
		{
			name:   "mask shorter than logits",
			logits: []float64{1.0, 2.0, 3.0},
			mask:   []bool{true},
			want:   []float64{1.0, 2.0, 3.0},
		},
		{
			name:   "empty",
			logits: []float64{},
			mask:   []bool{},
			want:   []float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			applyTokenMask(tt.logits, tt.mask)
			for i, got := range tt.logits {
				if math.IsInf(tt.want[i], -1) {
					if !math.IsInf(got, -1) {
						t.Errorf("logits[%d] = %v, want -Inf", i, got)
					}
				} else if got != tt.want[i] {
					t.Errorf("logits[%d] = %v, want %v", i, got, tt.want[i])
				}
			}
		})
	}
}

func TestAdvanceGrammar(t *testing.T) {
	// Build a grammar for {"type":"integer"} — accepts JSON integers.
	schema := &grammar.JSONSchema{Type: "integer"}
	g, err := grammar.Convert(schema)
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	vocab := []string{"1", "2", "3", "abc", ""}

	t.Run("valid token advances", func(t *testing.T) {
		next := advanceGrammar(g, 0, vocab) // token "1"
		if next == g {
			t.Error("grammar should have advanced")
		}
		if !next.IsComplete() {
			t.Error("integer grammar should be complete after digit")
		}
	})

	t.Run("invalid token returns original", func(t *testing.T) {
		next := advanceGrammar(g, 3, vocab) // token "abc"
		if next != g {
			t.Error("grammar should not advance on invalid token")
		}
	})

	t.Run("out of range token returns original", func(t *testing.T) {
		next := advanceGrammar(g, 100, vocab)
		if next != g {
			t.Error("grammar should not advance on out-of-range token")
		}
	})

	t.Run("empty token returns original", func(t *testing.T) {
		next := advanceGrammar(g, 4, vocab) // token ""
		if next != g {
			t.Error("grammar should not advance on empty token")
		}
	})
}

func TestGrammarMaskIntegration(t *testing.T) {
	// Build a grammar for a simple object: {"name":"string","age":"integer"}
	schema := &grammar.JSONSchema{
		Type: "object",
		Properties: map[string]*grammar.JSONSchema{
			"name": {Type: "string"},
			"age":  {Type: "integer"},
		},
		Required: []string{"name", "age"},
	}
	g, err := grammar.Convert(schema)
	if err != nil {
		t.Fatalf("Convert: %v", err)
	}

	// Simulate a vocab where token 0 = "{", token 1 = "hello", token 2 = "}"
	vocab := []string{"{", "hello", "}"}

	mask := grammar.TokenMask(g, vocab)

	// Only "{" should be valid at the start of an object.
	if !mask[0] {
		t.Error(`token "{" should be valid at start`)
	}
	if mask[1] {
		t.Error(`token "hello" should be invalid at start`)
	}
	if mask[2] {
		t.Error(`token "}" should be invalid at start (required fields missing)`)
	}

	// Advance through "{" and check next valid bytes.
	g2 := advanceGrammar(g, 0, vocab)
	mask2 := grammar.TokenMask(g2, vocab)

	// After "{", we need a key (starts with '"'), so none of our simple tokens should match.
	if mask2[0] {
		t.Error(`token "{" should be invalid after opening brace`)
	}
}
