package generate

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

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
	graph        *graph.Graph[T]
	tokenizer    tokenizer.Tokenizer
	engine       compute.Engine[T]
	config       ModelConfig
	cache        CacheProvider[T]
	graphMu      *sync.Mutex // shared mutex for graph Forward (graph is not concurrent-safe)
	mu           sync.Mutex  // serializes Generate calls within this session
	compileOnce  func(ctx context.Context, input *tensor.TensorNumeric[T]) // triggers graph compilation + CUDA graph capture
	planRef      *atomic.Pointer[graph.ExecutionPlan[T]]                   // shared reference to compiled execution plan
	poolResetter compute.PoolResetter                                      // cached type assertion; nil if engine doesn't implement it
	stopSet      map[int]bool                                              // reusable stop-token set, cleared and repopulated each call
	generatedIDs []int                                                     // reusable slice for generated token IDs
	prefixCache  *PrefixCache[T]                                           // shared prefix cache for KV block reuse; nil if disabled
	pjrtPlan     *graph.PJRTPlan[T]                                        // when non-nil, use PJRT backend; KV cache managed by PJRTPlan
}

// NewSession creates a new InferenceSession with its own KV cache.
// The session shares the Generator's graph, tokenizer, and engine but
// maintains independent KV cache state for isolation.
func (gen *Generator[T]) NewSession() *InferenceSession[T] {
	var cache CacheProvider[T]
	if gen.compressedKVChunkSize > 0 {
		cache = NewCompressedKVCache[T](gen.engine, gen.config.NumLayers, 0, 0, gen.compressedKVChunkSize)
	} else if gen.blockPool != nil {
		cache = NewPagedKVCache[T](gen.blockPool, gen.config.NumLayers)
	} else if qc, ok := gen.newQuantizedCache(); ok {
		cache = qc
	} else if _, ok := any(gen.engine).(compute.WeightUploader); ok {
		cache = gen.newTensorCache()
	} else {
		cache = NewKVCache[T](gen.config.NumLayers, gen.config.MaxSeqLen)
	}

	var poolResetter compute.PoolResetter
	if pr, ok := any(gen.engine).(compute.PoolResetter); ok {
		poolResetter = pr
	}

	return &InferenceSession[T]{
		graph:        gen.graph,
		tokenizer:    gen.tokenizer,
		engine:       gen.engine,
		config:       gen.config,
		cache:        cache,
		graphMu:      &gen.mu,
		compileOnce:  gen.compileGraph,
		planRef:      &gen.plan,
		poolResetter: poolResetter,
		prefixCache:  gen.prefixCache,
		pjrtPlan:     gen.pjrtPlan,
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

	// Hold the graph mutex for the entire generation to avoid per-step
	// lock/unlock overhead. CUDA graph replay is ~0.5ms per step; even
	// microseconds of mutex overhead per step compounds over 256+ steps.
	s.graphMu.Lock()
	defer s.graphMu.Unlock()

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

	// PJRT path: use RunPrefill/RunDecode instead of graph Forward.
	if s.pjrtPlan != nil {
		return s.pjrtGenerate(ctx, s.pjrtPlan, promptIDs, sc)
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

	s.prepareStopSet(sc.StopTokenIDs)
	stopSet := s.stopSet

	s.prepareGeneratedIDs(sc.MaxNewTokens)
	generatedIDs := s.generatedIDs[:0]

	// Check prefix cache for a matching KV block prefix to avoid redundant prefill.
	prefillIDs := promptIDs
	if s.prefixCache != nil {
		if pagedCache, ok := s.cache.(*PagedKVCache[T]); ok {
			promptIDs32 := intsToInt32(promptIDs)
			cachedBlocks, matchedLen := s.prefixCache.Match(promptIDs32)
			if matchedLen > 0 && matchedLen <= len(promptIDs) {
				// Inject cached blocks into the PagedKVCache.
				pagedCache.InjectBlocks(cachedBlocks, matchedLen)
				prefillIDs = promptIDs[matchedLen:]
			}
		}
	}

	var logits *tensor.TensorNumeric[T]
	if len(prefillIDs) > 0 {
		// Prefill: run the (possibly shortened) prompt through the graph.
		prefillTensor, ptErr := s.idsToTensor(prefillIDs)
		if ptErr != nil {
			return "", fmt.Errorf("create prefill tensor: %w", ptErr)
		}

		var fwdErr error
		logits, fwdErr = s.graphForward(genCtx, prefillTensor, true)
		if fwdErr != nil {
			return "", fmt.Errorf("prefill forward: %w", fwdErr)
		}
	} else {
		// Full prefix was cached — run a single-token forward to get logits
		// for the last prompt position.
		lastTok := promptIDs[len(promptIDs)-1]
		lastTensor, ltErr := s.idsToTensor([]int{lastTok})
		if ltErr != nil {
			return "", fmt.Errorf("create last-token tensor: %w", ltErr)
		}

		var fwdErr error
		logits, fwdErr = s.graphForward(genCtx, lastTensor, true)
		if fwdErr != nil {
			return "", fmt.Errorf("last-token forward: %w", fwdErr)
		}
	}

	// Store the prompt's blocks in the prefix cache for future sessions.
	if s.prefixCache != nil {
		if pagedCache, ok := s.cache.(*PagedKVCache[T]); ok {
			blocks := pagedCache.BlockTable()
			if len(blocks) > 0 {
				promptIDs32 := intsToInt32(promptIDs)
				s.prefixCache.Insert(promptIDs32, blocks)
			}
		}
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

		// Reset arena pool between tokens so intermediates are reclaimed.
		// Without this, the GPU arena grows monotonically, fragmenting memory
		// and preventing CUDA graph replay from reusing buffer addresses.
		if s.poolResetter != nil {
			s.poolResetter.ResetPool()
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

	// Record token usage for billing middleware.
	if usage := TokenUsageFromContext(ctx); usage != nil {
		usage.SetPromptTokens(len(promptIDs))
		usage.SetCompletionTokens(len(generatedIDs))
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

	s.graphMu.Lock()
	defer s.graphMu.Unlock()

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

	// PJRT path: use RunPrefill/RunDecode instead of graph Forward.
	if s.pjrtPlan != nil {
		return s.pjrtGenerateStream(ctx, s.pjrtPlan, promptIDs, sc, stream)
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

	s.prepareStopSet(sc.StopTokenIDs)
	stopSet := s.stopSet

	s.prepareGeneratedIDs(sc.MaxNewTokens)
	generatedIDs := s.generatedIDs[:0]
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

		// Reset arena pool between tokens so intermediates are reclaimed.
		if s.poolResetter != nil {
			s.poolResetter.ResetPool()
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

	// Record token usage for billing middleware.
	if usage := TokenUsageFromContext(ctx); usage != nil {
		usage.SetPromptTokens(len(promptIDs))
		usage.SetCompletionTokens(len(generatedIDs))
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
// graphForward runs a graph forward pass. The caller (Generate/GenerateStream)
// must already hold s.graphMu.
func (s *InferenceSession[T]) graphForward(ctx context.Context, input *tensor.TensorNumeric[T], reset bool) (*tensor.TensorNumeric[T], error) {
	// Only reset on prefill (first forward of a generation). Resetting on every
	// decode step destroys CUDA graph capture, causing a 50%+ throughput regression.
	if reset {
		s.graph.ResetStatefulNodes()
	}

	// Use compiled execution plan (CUDA graph) if available.
	if s.planRef != nil {
		if p := s.planRef.Load(); p != nil {
			return p.Run(ctx, input)
		}
	}

	// No compiled plan yet — run uncompiled forward and trigger compilation.
	result, err := s.graph.Forward(ctx, input)
	if err == nil && !reset && s.compileOnce != nil {
		s.compileOnce(ctx, input)
	}
	return result, err
}

// idsToTensor converts token IDs to a [1, seqLen] input tensor.
func (s *InferenceSession[T]) idsToTensor(ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}

// prepareStopSet clears and repopulates the session's reusable stop-token set.
// If the map has not been allocated yet, it is created with sufficient capacity.
func (s *InferenceSession[T]) prepareStopSet(stopTokenIDs []int) {
	needed := len(stopTokenIDs) + 1 // +1 for EOS
	if s.stopSet == nil {
		s.stopSet = make(map[int]bool, needed)
	} else {
		clear(s.stopSet)
	}
	for _, id := range stopTokenIDs {
		s.stopSet[id] = true
	}
	s.stopSet[s.config.EOSTokenID] = true
}

// prepareGeneratedIDs ensures the session's reusable generatedIDs slice has
// at least the requested capacity. The caller resets length via [:0].
func (s *InferenceSession[T]) prepareGeneratedIDs(minCap int) {
	if cap(s.generatedIDs) < minCap {
		s.generatedIDs = make([]int, 0, minCap)
	}
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

	// GPU argmax fast path: copies 4 bytes instead of ~1MB of logits.
	if idx, ok := tryGPUArgmax(logits, s.engine, sc); ok {
		return idx, nil
	}

	data, err := copyLogitsToCPU(logits, seqLen, vocabSize)
	if err != nil {
		return 0, err
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

		return applyTemperatureAndTopP(logitsF64, sc, vocabSize), nil
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

	return applyTemperatureAndTopP(logitsF64, sc, vocabSize), nil
}

// intsToInt32 converts a slice of int to int32 for use with the radix tree.
func intsToInt32(ids []int) []int32 {
	out := make([]int32, len(ids))
	for i, id := range ids {
		out[i] = int32(id)
	}
	return out
}
