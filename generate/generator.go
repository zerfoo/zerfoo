package generate

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/tensor"
)

// ModelConfig holds model architecture parameters needed for generation.
type ModelConfig struct {
	VocabSize  int // Total tokens in vocabulary
	MaxSeqLen  int // Maximum sequence length the model supports
	EOSTokenID int // End-of-sequence token ID
	BOSTokenID int // Beginning-of-sequence token ID
	NumLayers  int // Number of transformer layers (for KV cache sizing)
}

// SamplingConfig controls how tokens are selected during generation.
type SamplingConfig struct {
	Temperature       float64  // Divide logits by this value; 0 = greedy
	TopK              int      // Keep only top K tokens; 0 = disabled
	TopP              float64  // Keep tokens with cumulative prob >= P; 1.0 = disabled
	RepetitionPenalty float64  // Penalize repeated tokens; 1.0 = disabled
	MaxNewTokens      int      // Maximum number of tokens to generate
	StopTokenIDs      []int    // Stop when any of these token IDs are generated
	StopStrings       []string // Stop when output contains any of these strings
}

// DefaultSamplingConfig returns a SamplingConfig with sensible defaults.
func DefaultSamplingConfig() SamplingConfig {
	return SamplingConfig{
		Temperature:       1.0,
		TopK:              0,
		TopP:              1.0,
		RepetitionPenalty: 1.0,
		MaxNewTokens:      256,
	}
}

// GeneratorOption configures a Generator.
type GeneratorOption func(*generatorOptions)

type generatorOptions struct {
	pagedKVMaxMB int // when > 0, use PagedKVCache with this memory budget
	headDim      int // required for paged KV (auto-detected from model if 0)
}

// WithPagedKV enables paged KV caching with the given memory budget in MB.
// When enabled, the Generator allocates blocks from a shared BlockPool
// instead of pre-allocating the full maxSeqLen per sequence. headDim is the
// per-position storage size: for GQA models pass numKVHeads * actualHeadDim
// so the pool can store all KV heads per position.
func WithPagedKV(maxMemoryMB, headDim int) GeneratorOption {
	return func(o *generatorOptions) {
		o.pagedKVMaxMB = maxMemoryMB
		o.headDim = headDim
	}
}

// Generator produces text autoregressively using a loaded model graph.
type Generator[T tensor.Numeric] struct {
	graph     *graph.Graph[T]
	tokenizer tokenizer.Tokenizer
	engine    compute.Engine[T]
	config    ModelConfig
	pool      *compute.TensorPool[T]                    // reusable intermediate buffers
	blockPool *BlockPool[T]                              // nil when using pre-allocated KV cache
	headDim   int                                        // per-head dim for paged KV
	plan      atomic.Pointer[graph.ExecutionPlan[T]]     // compiled decode plan (nil until first decode)
	planOnce  sync.Once                                  // ensures compile happens once
}

// NewGenerator creates a Generator from a model graph, tokenizer, engine, and config.
func NewGenerator[T tensor.Numeric](
	g *graph.Graph[T],
	tok tokenizer.Tokenizer,
	eng compute.Engine[T],
	cfg ModelConfig,
	opts ...GeneratorOption,
) *Generator[T] {
	var gopts generatorOptions
	for _, o := range opts {
		o(&gopts)
	}

	gen := &Generator[T]{
		graph:     g,
		tokenizer: tok,
		engine:    eng,
		config:    cfg,
	}

	if g != nil {
		gen.pool = compute.NewTensorPool[T]()
		g.WithPool(gen.pool)
	}

	if gopts.pagedKVMaxMB > 0 && gopts.headDim > 0 {
		pool, err := NewBlockPool[T](cfg.NumLayers, 16, gopts.headDim, gopts.pagedKVMaxMB)
		if err == nil {
			gen.blockPool = pool
			gen.headDim = gopts.headDim
		}
	}

	return gen
}

// Graph returns the underlying computation graph.
func (gen *Generator[T]) Graph() *graph.Graph[T] { return gen.graph }

// Tokenizer returns the tokenizer.
func (gen *Generator[T]) Tokenizer() tokenizer.Tokenizer { return gen.tokenizer }

// Engine returns the compute engine.
func (gen *Generator[T]) Engine() compute.Engine[T] { return gen.engine }

// Config returns the model configuration.
func (gen *Generator[T]) Config() ModelConfig { return gen.config }

// compileGraph tries CompileTraced when an EngineProxy is available, with
// graceful fallback to Compile on error or plan validation failure.
func (gen *Generator[T]) compileGraph(ctx context.Context, tokenTensor *tensor.TensorNumeric[T]) {
	gen.planOnce.Do(func() {
		var compiled *graph.ExecutionPlan[T]
		var cErr error
		if proxy := gen.graph.EngineProxy(); proxy != nil {
			compiled, cErr = gen.graph.CompileTraced(ctx, tokenTensor)
			if cErr == nil {
				// Validate traced plan with a test run.
				if _, vErr := compiled.Run(ctx, tokenTensor); vErr != nil {
					log.Printf("generate: CompileTraced plan validation failed, falling back to Compile: %v", vErr)
					compiled, cErr = gen.graph.Compile(ctx, tokenTensor)
				}
			} else {
				log.Printf("generate: CompileTraced failed, falling back to Compile: %v", cErr)
				compiled, cErr = gen.graph.Compile(ctx, tokenTensor)
			}
		} else {
			compiled, cErr = gen.graph.Compile(ctx, tokenTensor)
		}
		if cErr == nil {
			gen.plan.Store(compiled)
			go tryCompileMegakernel(compiled, nil)
		}
	})
}

// Generate produces text from a prompt using the given sampling configuration.
// It tokenizes the prompt, runs the autoregressive loop with KV caching, and
// returns the generated text (excluding the prompt).
func (gen *Generator[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error) {
	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 256
	}

	promptIDs, err := gen.tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}

	var cacheProvider CacheProvider[T]
	if gen.blockPool != nil {
		cacheProvider = NewPagedKVCache[T](gen.blockPool, gen.config.NumLayers)
	} else {
		cacheProvider = NewKVCache[T](gen.config.NumLayers, gen.config.MaxSeqLen)
	}
	genCtx := WithCache(ctx, cacheProvider)

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[gen.config.EOSTokenID] = true

	generatedIDs := make([]int, 0, sc.MaxNewTokens)

	// Prefill: run the full prompt through the graph.
	prefillTensor, err := gen.idsToTensor(promptIDs)
	if err != nil {
		return "", fmt.Errorf("create prefill tensor: %w", err)
	}

	logits, err := gen.graph.Forward(genCtx, prefillTensor)
	if err != nil {
		return "", fmt.Errorf("prefill forward: %w", err)
	}

	nextToken, err := gen.sampleFromLogits(logits, sc, generatedIDs)
	if err != nil {
		return "", fmt.Errorf("sample after prefill: %w", err)
	}

	if stopSet[nextToken] {
		return "", nil
	}
	generatedIDs = append(generatedIDs, nextToken)

	if stopped, text := gen.checkStop(generatedIDs, sc.StopStrings); stopped {
		return text, nil
	}

	// Autoregressive decode loop.
	for range sc.MaxNewTokens - 1 {
		if err := ctx.Err(); err != nil {
			break
		}

		tokenTensor, tErr := gen.idsToTensor([]int{nextToken})
		if tErr != nil {
			return "", fmt.Errorf("create token tensor: %w", tErr)
		}

		if p := gen.plan.Load(); p != nil {
			logits, err = p.Run(genCtx, tokenTensor)
		} else {
			logits, err = gen.graph.Forward(genCtx, tokenTensor)
			// After the first decode Forward(), compile the graph.
			// The graph's memo from this Forward() provides shapes
			// without re-executing (avoids corrupting model state).
			if err == nil {
				gen.compileGraph(genCtx, tokenTensor)
			}
		}
		if err != nil {
			return "", fmt.Errorf("decode forward: %w", err)
		}

		nextToken, err = gen.sampleFromLogits(logits, sc, generatedIDs)
		if err != nil {
			return "", fmt.Errorf("sample: %w", err)
		}

		if stopSet[nextToken] {
			break
		}
		generatedIDs = append(generatedIDs, nextToken)

		if stopped, text := gen.checkStop(generatedIDs, sc.StopStrings); stopped {
			return text, nil
		}
	}

	if len(generatedIDs) == 0 {
		return "", nil
	}

	result, err := gen.tokenizer.Decode(generatedIDs)
	if err != nil {
		return "", fmt.Errorf("decode output: %w", err)
	}
	return result, nil
}

// sampleFromLogits extracts the last-position logits from a [1, seqLen, vocabSize]
// tensor and samples a token.
func (gen *Generator[T]) sampleFromLogits(
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

	data := logits.Data()
	lastStart := (seqLen - 1) * vocabSize
	if lastStart+vocabSize > len(data) {
		return 0, fmt.Errorf("logits data too short: %d < %d", len(data), lastStart+vocabSize)
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

// idsToTensor converts token IDs to a [1, seqLen] input tensor.
func (gen *Generator[T]) idsToTensor(ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}

// checkStop checks if the decoded generated tokens contain any stop string.
func (gen *Generator[T]) checkStop(generatedIDs []int, stopStrings []string) (bool, string) {
	if len(stopStrings) == 0 {
		return false, ""
	}
	decoded, err := gen.tokenizer.Decode(generatedIDs)
	if err != nil {
		return false, ""
	}
	for _, ss := range stopStrings {
		if idx := strings.Index(decoded, ss); idx >= 0 {
			return true, decoded[:idx]
		}
	}
	return false, ""
}
