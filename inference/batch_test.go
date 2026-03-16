package inference

import (
	"context"
	"testing"
)

func TestGenerateBatch(t *testing.T) {
	tests := []struct {
		name       string
		prompts    []string
		tokenSeq   []int // token sequence the fixedLogitsNode cycles through
		wantLen    int
		wantNilErr bool
	}{
		{
			name:       "empty slice",
			prompts:    nil,
			tokenSeq:   []int{6, 2},
			wantLen:    0,
			wantNilErr: true,
		},
		{
			name:       "single prompt",
			prompts:    []string{"hello"},
			tokenSeq:   []int{6, 2}, // produces token 6 ("foo") then EOS
			wantLen:    1,
			wantNilErr: true,
		},
		{
			name:       "two prompts",
			prompts:    []string{"hello", "world"},
			tokenSeq:   []int{6, 2},
			wantLen:    2,
			wantNilErr: true,
		},
		{
			name:       "four prompts",
			prompts:    []string{"hello", "world", "foo", "bar"},
			tokenSeq:   []int{6, 2},
			wantLen:    4,
			wantNilErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := buildTestModel(t, 8, tt.tokenSeq)
			results, err := m.GenerateBatch(context.Background(), tt.prompts, WithTemperature(0), WithMaxTokens(10))
			if tt.wantNilErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(results) != tt.wantLen {
				t.Fatalf("got %d results, want %d", len(results), tt.wantLen)
			}
			for i, r := range results {
				if r == "" && tt.wantLen > 0 {
					t.Errorf("results[%d] is empty", i)
				}
			}
		})
	}
}

func TestGenerateBatch_ContextCancellation(t *testing.T) {
	m := buildTestModel(t, 8, []int{6, 7, 6, 7, 6, 7}) // no EOS — relies on maxTokens or cancellation

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	results, _ := m.GenerateBatch(ctx, []string{"hello", "world"}, WithTemperature(0), WithMaxTokens(5))
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
}

func TestGenerateBatch_SinglePromptOutput(t *testing.T) {
	// Token 6 = "foo", token 2 = EOS. Sequence: produce "foo" then stop.
	m := buildTestModel(t, 8, []int{6, 2})

	results, err := m.GenerateBatch(context.Background(), []string{"hello"}, WithTemperature(0), WithMaxTokens(10))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0] != "foo" {
		t.Errorf("got %q, want %q", results[0], "foo")
	}
}
