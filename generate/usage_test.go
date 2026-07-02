package generate

import (
	"context"
	"testing"
)

func TestTokenUsage(t *testing.T) {
	u := &TokenUsage{}
	if u.PromptTokens() != 0 {
		t.Errorf("PromptTokens() = %d, want 0", u.PromptTokens())
	}
	if u.CompletionTokens() != 0 {
		t.Errorf("CompletionTokens() = %d, want 0", u.CompletionTokens())
	}

	u.SetPromptTokens(42)
	u.SetCompletionTokens(7)
	if u.PromptTokens() != 42 {
		t.Errorf("PromptTokens() = %d, want 42", u.PromptTokens())
	}
	if u.CompletionTokens() != 7 {
		t.Errorf("CompletionTokens() = %d, want 7", u.CompletionTokens())
	}
}

func TestTokenUsageContext(t *testing.T) {
	ctx := context.Background()

	// No usage in empty context.
	if u := TokenUsageFromContext(ctx); u != nil {
		t.Error("expected nil usage from empty context")
	}

	// Store and retrieve usage.
	usage := &TokenUsage{}
	usage.SetPromptTokens(100)
	usage.SetCompletionTokens(25)

	ctx = WithTokenUsage(ctx, usage)
	got := TokenUsageFromContext(ctx)
	if got == nil {
		t.Fatal("expected non-nil usage from context")
	}
	if got.PromptTokens() != 100 {
		t.Errorf("PromptTokens() = %d, want 100", got.PromptTokens())
	}
	if got.CompletionTokens() != 25 {
		t.Errorf("CompletionTokens() = %d, want 25", got.CompletionTokens())
	}
}
