package main

import "testing"

func TestContainsRepeatedChar(t *testing.T) {
	tests := []struct {
		name   string
		s      string
		runLen int
		want   bool
	}{
		{"empty", "", 5, false},
		{"short run below threshold", "aaab", 5, false},
		{"long run at threshold", "aaaaa", 5, true},
		{"long run beyond threshold", "xaaaaaax", 5, true},
		{"whitespace resets run", "aa aa aa", 5, false},
		{"mixed content safe", "Hello world", 5, false},
		{"degenerate punctuation", "!!!!!!!", 5, true},
		{"runLen=1 disabled", "ab", 1, false},
		{"runLen=0 disabled", "aaaaaaaa", 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := containsRepeatedChar(tc.s, tc.runLen)
			if got != tc.want {
				t.Fatalf("containsRepeatedChar(%q, %d) = %v, want %v", tc.s, tc.runLen, got, tc.want)
			}
		})
	}
}
