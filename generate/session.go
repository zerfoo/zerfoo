package generate

import (
	"context"
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	tokenizer "github.com/zerfoo/ztoken"
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
	mu        sync.Mutex // serializes Generate calls within this session
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

	// Reset the cache for a fresh generation.
	s.cache.Reset()
	genCtx := WithCache(ctx, s.cache)

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[s.config.EOSTokenID] = true

	generatedIDs := make([]int, 0, sc.MaxNewTokens)

	// Reset stateful auto-input nodes for this generation.
	s.graph.ResetStatefulNodes()

	// Prefill: run the full prompt through the graph.
	prefillTensor, err := s.idsToTensor(promptIDs)
	if err != nil {
		return "", fmt.Errorf("create prefill tensor: %w", err)
	}

	logits, err := s.graph.Forward(genCtx, prefillTensor)
	if err != nil {
		return "", fmt.Errorf("prefill forward: %w", err)
	}

	nextToken, err := s.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return "", fmt.Errorf("sample after prefill: %w", err)
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

		decodeBuf[0] = T(nextToken)

		logits, err = s.graph.Forward(genCtx, tokenTensor)
		if err != nil {
			return "", fmt.Errorf("decode forward: %w", err)
		}

		nextToken, err = s.sampleFromLogits(logits, sc, generatedIDs)
		if err != nil {
			return "", fmt.Errorf("sample: %w", err)
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

	// Greedy fast path.
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
