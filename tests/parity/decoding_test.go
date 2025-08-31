package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/helpers"
)

func TestDecodingParityGreedy(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	helpers.ImplZerfoo.SetSeed(123)
	
	// Test deterministic greedy decoding
	tokenSequences := [][]int{
		{1, 2, 3, 4, 5},
		{10, 20, 30},
		{100, 200, 300, 400},
		{1, 1, 1, 1, 1, 1},
		{42},
	}
	
	for i, tokens := range tokenSequences {
		// Test that greedy decoding is deterministic
		decoded1, err := helpers.ImplZerfoo.DecodeGreedy(tokens)
		if err != nil {
			t.Fatalf("decode greedy 1 at %d: %v", i, err)
		}
		
		decoded2, err := helpers.ImplZerfoo.DecodeGreedy(tokens)
		if err != nil {
			t.Fatalf("decode greedy 2 at %d: %v", i, err)
		}
		
		if decoded1 != decoded2 {
			t.Fatalf("greedy decoding not deterministic at %d: %q vs %q", i, decoded1, decoded2)
		}
		
		// Verify round-trip consistency if encoding is available
		if decoded1 != "" {
			reEncoded, err := helpers.ImplZerfoo.Tokenize(decoded1)
			if err == nil {
				// Round-trip should preserve meaning (though exact tokens may differ due to normalization)
				reDecoded, err := helpers.ImplZerfoo.DecodeGreedy(reEncoded)
				if err != nil {
					t.Logf("round-trip decode failed at %d: %v", i, err)
				} else if reDecoded == "" {
					t.Logf("round-trip produced empty result at %d", i)
				}
			}
		}
	}
}

func TestDecodingParityTopP(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	helpers.ImplZerfoo.SetSeed(123)
	
	// Test deterministic top-p decoding with deterministic flag
	tokenSequences := [][]int{
		{1, 2, 3, 4, 5},
		{10, 20, 30},
		{100, 200, 300, 400},
	}
	
	pValues := []float64{0.9, 0.95, 0.99}
	
	for _, p := range pValues {
		for i, tokens := range tokenSequences {
			// Set deterministic mode (S1.4.2 requirement)
			helpers.ImplZerfoo.SetSeed(123) // Reset seed for consistency
			
			decoded1, err := helpers.ImplZerfoo.DecodeTopP(tokens, p)
			if err != nil {
				t.Fatalf("decode top-p 1 at %d (p=%.2f): %v", i, p, err)
			}
			
			helpers.ImplZerfoo.SetSeed(123) // Reset seed for deterministic check
			decoded2, err := helpers.ImplZerfoo.DecodeTopP(tokens, p)
			if err != nil {
				t.Fatalf("decode top-p 2 at %d (p=%.2f): %v", i, p, err)
			}
			
			if decoded1 != decoded2 {
				t.Fatalf("top-p decoding not deterministic at %d (p=%.2f): %q vs %q", i, p, decoded1, decoded2)
			}
		}
	}
}

func TestDecodingParityEdgeCases(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	helpers.ImplZerfoo.SetSeed(123)
	
	tests := []struct {
		name   string
		tokens []int
		expectError bool
	}{
		{
			name:   "empty_tokens",
			tokens: []int{},
			expectError: false, // Should return empty string
		},
		{
			name:   "single_token", 
			tokens: []int{1},
			expectError: false,
		},
		{
			name:   "negative_token",
			tokens: []int{-1},
			expectError: true, // Invalid token ID
		},
		{
			name:   "very_large_token",
			tokens: []int{999999999},
			expectError: true, // Out of vocabulary
		},
		{
			name:   "mixed_valid_invalid",
			tokens: []int{1, 2, -1, 4},
			expectError: true, // Contains invalid token
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decoded, err := helpers.ImplZerfoo.DecodeGreedy(tt.tokens)
			
			if tt.expectError {
				if err == nil {
					t.Errorf("expected error for %s, got decoded: %q", tt.name, decoded)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error for %s: %v", tt.name, err)
				}
			}
		})
	}
}

func TestDecodingDeterministicFlag(t *testing.T) {
	if helpers.ImplZerfoo == nil {
		t.Skip("wire ImplZerfoo in tests/helpers/wire.go")
	}
	
	// Test S1.4.2: Add a flag to force deterministic sampling
	// This test verifies that when deterministic mode is enabled,
	// sampling operations produce identical results
	
	tokens := []int{1, 2, 3, 4, 5}
	
	// Test that multiple runs with same seed produce identical results
	const numRuns = 5
	var results []string
	
	for i := 0; i < numRuns; i++ {
		helpers.ImplZerfoo.SetSeed(42) // Same seed each time
		
		// Test both greedy and top-p for determinism
		greedyResult, err := helpers.ImplZerfoo.DecodeGreedy(tokens)
		if err != nil {
			t.Fatalf("greedy decode failed on run %d: %v", i, err)
		}
		
		helpers.ImplZerfoo.SetSeed(42) // Reset seed before top-p
		topPResult, err := helpers.ImplZerfoo.DecodeTopP(tokens, 0.9)
		if err != nil {
			t.Fatalf("top-p decode failed on run %d: %v", i, err)
		}
		
		// Greedy should always be the same
		if i == 0 {
			results = append(results, greedyResult, topPResult)
		} else {
			if greedyResult != results[0] {
				t.Fatalf("greedy decoding not deterministic: run 0=%q, run %d=%q", results[0], i, greedyResult)
			}
			if topPResult != results[1] {
				t.Fatalf("top-p decoding not deterministic: run 0=%q, run %d=%q", results[1], i, topPResult)
			}
		}
	}
	
	t.Logf("Deterministic decoding verified across %d runs", numRuns)
}