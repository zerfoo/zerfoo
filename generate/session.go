package generate

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/tensor"
)

// InferenceSession holds per-request state for autoregressive generation.
// Multiple sessions can share the same Generator (and its compiled graph,
// tokenizer, and engine) while maintaining independent KV caches, position
// counters, and sampling state. This enables concurrent request handling
// without the Generator-level mutex serializing all generation.
type InferenceSession[T tensor.Numeric] struct {
	gen      *Generator[T]       // shared: graph, tokenizer, engine, compiled plan
	cache    CacheProvider[T]    // per-session KV cache
	pos      int                 // current position in the sequence
	sampling SamplingConfig      // per-session sampling defaults
	mu       sync.Mutex          // serializes Generate/GenerateStream within this session
}

// NewSession creates an InferenceSession that shares the Generator's compiled
// graph, tokenizer, and engine but allocates its own KV cache and position
// counter. The returned session is safe for use by a single goroutine at a
// time (its own mutex protects per-session state).
func (gen *Generator[T]) NewSession() *InferenceSession[T] {
	var cache CacheProvider[T]
	if gen.blockPool != nil {
		cache = NewPagedKVCache[T](gen.blockPool, gen.config.NumLayers)
	} else {
		cache = NewKVCache[T](gen.config.NumLayers, gen.config.MaxSeqLen)
	}

	return &InferenceSession[T]{
		gen:      gen,
		cache:    cache,
		pos:      0,
		sampling: DefaultSamplingConfig(),
	}
}

// Generate produces text from a prompt using the session's KV cache and the
// given sampling configuration. The session's position counter and cache are
// updated so that subsequent calls can continue the conversation.
func (s *InferenceSession[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return "", errors.New("generate: InferenceSession.Generate not yet implemented")
}

// GenerateStream produces text from a prompt, delivering each token to the
// stream as it is generated. Like Generate, the session's KV cache and
// position counter are updated across calls.
func (s *InferenceSession[T]) GenerateStream(ctx context.Context, prompt string, sc SamplingConfig, stream TokenStream) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	return errors.New("generate: InferenceSession.GenerateStream not yet implemented")
}

// Position returns the current sequence position (number of tokens processed).
func (s *InferenceSession[T]) Position() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pos
}

// Reset clears the session's KV cache and resets the position counter to zero,
// allowing the session to be reused for a new conversation.
func (s *InferenceSession[T]) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cache.Reset()
	s.pos = 0
}

// Cache returns the session's KV cache provider.
func (s *InferenceSession[T]) Cache() CacheProvider[T] {
	return s.cache
}

// String returns a human-readable description of the session state.
func (s *InferenceSession[T]) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return fmt.Sprintf("InferenceSession{pos=%d, cacheSeqLen=%d}", s.pos, s.cache.SeqLen())
}
