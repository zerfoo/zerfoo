package generate

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	tokenizer "github.com/zerfoo/ztoken"

	"github.com/zerfoo/zerfoo/generate/grammar"
)

// InferenceSession holds per-session state for independent, concurrent
// inference. Each session owns its own KV cache and position tracking,
// allowing multiple sessions to generate simultaneously without data races.
type InferenceSession[T tensor.Numeric] struct {
	graph     *graph.Graph[T]
	tokenizer tokenizer.Tokenizer
	engine    compute.Engine[T]
	config    ModelConfig
	cache     CacheProvider[T]
	graphMu   *sync.Mutex // shared mutex for graph Forward (graph is not concurrent-safe)
	mu        sync.Mutex  // serializes Generate calls within this session
}

// NewSession creates a new InferenceSession with its own KV cache.
// The session shares the Generator's graph, tokenizer, and engine but
// maintains independent KV cache state for isolation.
func (gen *Generator[T]) NewSession() *InferenceSession[T] {
	var cache CacheProvider[T]
	if gen.blockPool != nil {
		cache = NewPagedKVCache[T](gen.blockPool, gen.config.NumLayers)
	} else if _, ok := any(gen.engine).(compute.WeightUploader); ok {
		cache = gen.newTensorCache()
	} else {
		cache = NewKVCache[T](gen.config.NumLayers, gen.config.MaxSeqLen)
	}

	return &InferenceSession[T]{
		graph:     gen.graph,
		tokenizer: gen.tokenizer,
		engine:    gen.engine,
		config:    gen.config,
		cache:     cache,
		graphMu:   &gen.mu,
	}
}

// Cache returns the session's KV cache provider.
func (s *InferenceSession[T]) Cache() CacheProvider[T] {
	return s.cache
}

// Generate produces text from a prompt using the session's own KV cache.
// Multiple sessions can Generate concurrently without data races, though
// calls within a single session are serialized.
func (s *InferenceSession[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 256
	}

	promptIDs, err := s.tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}

	if s.config.BOSTokenID > 0 {
		promptIDs = append([]int{s.config.BOSTokenID}, promptIDs...)
	}

	// Build grammar vocab cache if grammar-constrained decoding is active.
	if sc.GrammarState != nil {
		vocabSize := s.tokenizer.VocabSize()
		sc.grammarVocab = make([]string, vocabSize)
		for i := range vocabSize {
			if tok, ok := s.tokenizer.GetToken(i); ok {
				sc.grammarVocab[i] = tok
			}
		}
	}

	// Reset the cache for a fresh generation.
	s.cache.Reset()
	genCtx := WithCache(ctx, s.cache)

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[s.config.EOSTokenID] = true

	generatedIDs := make([]int, 0, sc.MaxNewTokens)

	// Prefill: run the full prompt through the graph.
	prefillTensor, err := s.idsToTensor(promptIDs)
	if err != nil {
		return "", fmt.Errorf("create prefill tensor: %w", err)
	}

	logits, err := s.graphForward(genCtx, prefillTensor, true)
	if err != nil {
		return "", fmt.Errorf("prefill forward: %w", err)
	}

	nextToken, err := s.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return "", fmt.Errorf("sample after prefill: %w", err)
	}

	// Advance grammar state after sampling.
	if sc.GrammarState != nil {
		sc.GrammarState = advanceGrammar(sc.GrammarState, nextToken, sc.grammarVocab)
	}

	if stopSet[nextToken] {
		return "", nil
	}
	generatedIDs = append(generatedIDs, nextToken)

	// Pre-allocate a [1,1] tensor for the decode loop.
	decodeBuf := []T{T(nextToken)}
	tokenTensor, err := tensor.New([]int{1, 1}, decodeBuf)
	if err != nil {
		return "", fmt.Errorf("create decode tensor: %w", err)
	}

	// Autoregressive decode loop.
	for range sc.MaxNewTokens - 1 {
		if err := ctx.Err(); err != nil {
			break
		}
		// Stop early if grammar is complete.
		if sc.GrammarState != nil && sc.GrammarState.IsComplete() {
			break
		}

		decodeBuf[0] = T(nextToken)

		logits, err = s.graphForward(genCtx, tokenTensor, false)
		if err != nil {
			return "", fmt.Errorf("decode forward: %w", err)
		}

		nextToken, err = s.sampleFromLogits(logits, sc, generatedIDs)
		if err != nil {
			return "", fmt.Errorf("sample: %w", err)
		}

		// Advance grammar state after sampling.
		if sc.GrammarState != nil {
			sc.GrammarState = advanceGrammar(sc.GrammarState, nextToken, sc.grammarVocab)
		}

		if stopSet[nextToken] {
			break
		}
		generatedIDs = append(generatedIDs, nextToken)
	}

	if len(generatedIDs) == 0 {
		return "", nil
	}

	result, err := s.tokenizer.Decode(generatedIDs)
	if err != nil {
		return "", fmt.Errorf("decode output: %w", err)
	}
	return result, nil
}

// GenerateStream produces text from a prompt using the session's own KV cache,
// delivering each token to the stream as it is generated.
func (s *InferenceSession[T]) GenerateStream(ctx context.Context, prompt string, sc SamplingConfig, stream TokenStream) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 256
	}

	promptIDs, err := s.tokenizer.Encode(prompt)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return fmt.Errorf("prompt produced no tokens")
	}

	if s.config.BOSTokenID > 0 {
		promptIDs = append([]int{s.config.BOSTokenID}, promptIDs...)
	}

	// Build grammar vocab cache if grammar-constrained decoding is active.
	if sc.GrammarState != nil {
		vocabSize := s.tokenizer.VocabSize()
		sc.grammarVocab = make([]string, vocabSize)
		for i := range vocabSize {
			if tok, ok := s.tokenizer.GetToken(i); ok {
				sc.grammarVocab[i] = tok
			}
		}
	}

	s.cache.Reset()
	genCtx := WithCache(ctx, s.cache)

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[s.config.EOSTokenID] = true

	generatedIDs := make([]int, 0, sc.MaxNewTokens)
	prevDecoded := ""

	// Prefill.
	prefillTensor, err := s.idsToTensor(promptIDs)
	if err != nil {
		return fmt.Errorf("create prefill tensor: %w", err)
	}

	logits, err := s.graphForward(genCtx, prefillTensor, true)
	if err != nil {
		return fmt.Errorf("prefill forward: %w", err)
	}

	nextToken, err := s.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return fmt.Errorf("sample after prefill: %w", err)
	}

	if stopSet[nextToken] {
		return stream.OnToken("", true)
	}
	generatedIDs = append(generatedIDs, nextToken)

	if emitErr := s.emitToken(generatedIDs, &prevDecoded, sc.StopStrings, stream); emitErr != nil {
		if errors.Is(emitErr, errStopString) {
			return nil
		}
		return emitErr
	}

	decodeBuf := []T{T(nextToken)}
	tokenTensor, tErr := tensor.New([]int{1, 1}, decodeBuf)
	if tErr != nil {
		return fmt.Errorf("create decode tensor: %w", tErr)
	}

	for range sc.MaxNewTokens - 1 {
		if err := ctx.Err(); err != nil {
			break
		}

		decodeBuf[0] = T(nextToken)

		logits, err = s.graphForward(genCtx, tokenTensor, false)
		if err != nil {
			return fmt.Errorf("decode forward: %w", err)
		}

		nextToken, err = s.sampleFromLogits(logits, sc, generatedIDs)
		if err != nil {
			return fmt.Errorf("sample: %w", err)
		}

		if stopSet[nextToken] {
			break
		}
		generatedIDs = append(generatedIDs, nextToken)

		if emitErr := s.emitToken(generatedIDs, &prevDecoded, sc.StopStrings, stream); emitErr != nil {
			if errors.Is(emitErr, errStopString) {
				return nil
			}
			return emitErr
		}
	}

	return stream.OnToken("", true)
}

// emitToken decodes the full generated sequence, computes the incremental
// difference from the previous decoding, and emits it to the stream.
func (s *InferenceSession[T]) emitToken(
	generatedIDs []int,
	prevDecoded *string,
	stopStrings []string,
	stream TokenStream,
) error {
	decoded, err := s.tokenizer.Decode(generatedIDs)
	if err != nil {
		return fmt.Errorf("decode token: %w", err)
	}

	for _, ss := range stopStrings {
		if idx := strings.Index(decoded, ss); idx >= 0 {
			remaining := decoded[:idx]
			if len(remaining) > len(*prevDecoded) {
				delta := remaining[len(*prevDecoded):]
				if err := stream.OnToken(delta, false); err != nil {
					return err
				}
			}
			if err := stream.OnToken("", true); err != nil {
				return err
			}
			return errStopString
		}
	}

	if len(decoded) > len(*prevDecoded) {
		delta := decoded[len(*prevDecoded):]
		*prevDecoded = decoded
		return stream.OnToken(delta, false)
	}
	*prevDecoded = decoded
	return nil
}

// graphForward runs a graph forward pass under the shared graph mutex.
// It uses defer to ensure the mutex is released even if Forward panics.
func (s *InferenceSession[T]) graphForward(ctx context.Context, input *tensor.TensorNumeric[T], _ bool) (*tensor.TensorNumeric[T], error) {
	s.graphMu.Lock()
	defer s.graphMu.Unlock()
	// Always reset stateful nodes before forward. The graph is shared across
	// sessions, so any prior session's forward may have left stale state.
	// Each session's KV cache is passed via context, providing isolation.
	s.graph.ResetStatefulNodes()
	return s.graph.Forward(ctx, input)
}

// idsToTensor converts token IDs to a [1, seqLen] input tensor.
func (s *InferenceSession[T]) idsToTensor(ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}

// sampleFromLogits extracts the last-position logits from a [1, seqLen, vocabSize]
// tensor and samples a token using greedy decoding.
func (s *InferenceSession[T]) sampleFromLogits(
	logits *tensor.TensorNumeric[T],
	sc SamplingConfig,
	generatedTokens []int,
) (int, error) {
	shape := logits.Shape()
	if len(shape) != 3 {
		return 0, fmt.Errorf("expected 3D logits [batch, seq, vocab], got shape %v", shape)
	}
	vocabSize := shape[2]
	seqLen := shape[1]

	totalElems := seqLen * vocabSize
	data := make([]T, totalElems)

	if gs, ok := logits.GetStorage().(*tensor.GPUStorage[T]); ok {
		if err := gs.CopyTo(data); err != nil {
			return 0, fmt.Errorf("copy logits from GPU: %w", err)
		}
	} else {
		copy(data, logits.Data())
	}

	lastStart := (seqLen - 1) * vocabSize
	if lastStart+vocabSize > len(data) {
		return 0, fmt.Errorf("logits data too short: %d < %d", len(data), lastStart+vocabSize)
	}

	// Grammar masking: if a grammar state is active, apply token mask before sampling.
	if sc.GrammarState != nil && len(sc.grammarVocab) > 0 {
		logitsF64 := make([]float64, vocabSize)
		for i := range vocabSize {
			logitsF64[i] = float64(data[lastStart+i])
		}

		mask := grammar.TokenMask(sc.GrammarState, sc.grammarVocab)
		applyTokenMask(logitsF64, mask)

		if sc.RepetitionPenalty > 0 && sc.RepetitionPenalty != 1.0 {
			applyRepetitionPenalty(logitsF64, generatedTokens, sc.RepetitionPenalty)
		}

		if sc.Temperature > 0 {
			applyTemperature(logitsF64, sc.Temperature)
			if sc.TopK > 0 && sc.TopK < vocabSize {
				applyTopK(logitsF64, sc.TopK)
			}
			if sc.TopP > 0 && sc.TopP < 1.0 {
				applyTopP(logitsF64, sc.TopP)
			}
			return sampleFromDistribution(logitsF64), nil
		}
		return argmax(logitsF64), nil
	}

	// Greedy fast path (no grammar).
	if sc.Temperature <= 0 && (sc.RepetitionPenalty <= 0 || sc.RepetitionPenalty == 1.0) {
		best := 0
		bestVal := data[lastStart]
		for i := 1; i < vocabSize; i++ {
			if data[lastStart+i] > bestVal {
				bestVal = data[lastStart+i]
				best = i
			}
		}
		return best, nil
	}

	logitsF64 := make([]float64, vocabSize)
	for i := range vocabSize {
		logitsF64[i] = float64(data[lastStart+i])
	}

	if sc.RepetitionPenalty > 0 && sc.RepetitionPenalty != 1.0 {
		applyRepetitionPenalty(logitsF64, generatedTokens, sc.RepetitionPenalty)
	}

	if sc.Temperature <= 0 {
		return argmax(logitsF64), nil
	}

	applyTemperature(logitsF64, sc.Temperature)

	if sc.TopK > 0 && sc.TopK < vocabSize {
		applyTopK(logitsF64, sc.TopK)
	}

	if sc.TopP > 0 && sc.TopP < 1.0 {
		applyTopP(logitsF64, sc.TopP)
	}

	return sampleFromDistribution(logitsF64), nil
}
