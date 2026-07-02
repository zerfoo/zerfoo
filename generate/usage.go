package generate

import (
	"context"
	"sync/atomic"
)

// TokenUsage tracks prompt and completion token counts during generation.
// It is safe for concurrent reads via the atomic loads, and is written by
// the generation session after prefill and decode complete.
type TokenUsage struct {
	promptTokens     atomic.Int64
	completionTokens atomic.Int64
}

// SetPromptTokens stores the prompt token count.
func (u *TokenUsage) SetPromptTokens(n int) {
	u.promptTokens.Store(int64(n))
}

// SetCompletionTokens stores the completion token count.
func (u *TokenUsage) SetCompletionTokens(n int) {
	u.completionTokens.Store(int64(n))
}

// PromptTokens returns the prompt token count.
func (u *TokenUsage) PromptTokens() int {
	return int(u.promptTokens.Load())
}

// CompletionTokens returns the completion token count.
func (u *TokenUsage) CompletionTokens() int {
	return int(u.completionTokens.Load())
}

type usageKey struct{}

// WithTokenUsage returns a new context carrying the given TokenUsage.
// Billing middleware should call this before dispatching to the handler,
// then read back the counts after the handler returns.
func WithTokenUsage(ctx context.Context, usage *TokenUsage) context.Context {
	return context.WithValue(ctx, usageKey{}, usage)
}

// TokenUsageFromContext extracts the TokenUsage from the context, or nil
// if none is present.
func TokenUsageFromContext(ctx context.Context) *TokenUsage {
	u, _ := ctx.Value(usageKey{}).(*TokenUsage)
	return u
}
