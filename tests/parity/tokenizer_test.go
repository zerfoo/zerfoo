package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/helpers"
	"github.com/zerfoo/zerfoo/tests/internal/testutil"
)

func TestTokenizerParity(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	helpers.ImplZerfoo.SetSeed(123)
	prompts := testutil.LoadPrompts("tests/testdata/prompts.txt", 500, 123)
	for i, p := range prompts {
		a, err := helpers.ImplZerfoo.Tokenize(p)
		if err != nil {
			t.Fatalf("tokenize err: %v", err)
		}
		b, err := helpers.ImplZerfoo.RefTokenize(p)
		if err != nil {
			t.Fatalf("ref tokenize err: %v", err)
		}
		if len(a) != len(b) {
			t.Fatalf("mismatch at %d: len %d vs %d", i, len(a), len(b))
		}
		for j := range a {
			if a[j] != b[j] {
				t.Fatalf("token mismatch at %d:%d got %d want %d", i, j, a[j], b[j])
			}
		}
	}
}
