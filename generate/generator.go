package generate

import (
	"context"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/metrics/runtime"
	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/internal/cuda"
	tokenizer "github.com/zerfoo/ztoken"
	"github.com/zerfoo/ztensor/tensor"
)

// debugOnnx caches the ZERFOO_DEBUG_ONNX environment variable check at init time.
var debugOnnx = os.Getenv("ZERFOO_DEBUG_ONNX") == "1"

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
	StopStrings       []string         // Stop when output contains any of these strings
	GrammarState      *grammar.Grammar // Optional grammar for constrained decoding
	grammarVocab      []string         // Cached token strings for grammar masking (built lazily)
	AdapterName       string           // Optional LoRA adapter name for per-request selection
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
	pagedKVMaxMB          int    // when > 0, use PagedKVCache with this memory budget
	headDim               int    // required for paged KV (auto-detected from model if 0)
	kvDtype               string // KV cache storage dtype: "fp32" (default) or "fp16"
	specDraft             *specDraftConfig // when non-nil, use speculative decoding
	prefixCacheBlocks     int    // when > 0, enable prefix caching with this many cached blocks
	metricsCollector      runtime.Collector // optional metrics collector
	compressedKVChunkSize int    // when > 0, use CompressedKVCache with this chunk size
	eagleWeightsPath      string // when non-empty, enable EAGLE speculative decoding with weights from this GGUF path
	tieredKVCfg           *TieredKVStoreConfig // when non-nil, use TieredKVStore
	pjrtPlan              any    // *graph.PJRTPlan[T], stored as any to avoid type param on generatorOptions
}

// specDraftConfig holds configuration for speculative decoding via a draft model.
type specDraftConfig struct {
	draftGraph     interface{} // *graph.Graph[T], stored as interface to avoid type param on generatorOptions
	draftCfg       ModelConfig
	draftLen       int     // K lookahead tokens per speculative step
	fallbackAlpha  float64 // fall back to standard decode if rolling alpha < this
}

// WithMetrics attaches a metrics collector to the generator. When speculative
// decoding is active, the generator updates a "speculative_acceptance_rate"
// gauge after each verify step.
func WithMetrics(c runtime.Collector) GeneratorOption {
	return func(o *generatorOptions) {
		o.metricsCollector = c
	}
}

// WithGeneratorKVDtype sets the KV cache storage dtype.
// Supported: "fp32" (default), "fp16", "q4", "q3".
func WithGeneratorKVDtype(dtype string) GeneratorOption {
	return func(o *generatorOptions) {
		o.kvDtype = dtype
	}
}

// WithSpeculativeDraft enables speculative decoding using a separate draft
// model graph. The draft model proposes draftLen tokens greedily per step,
// then the target model verifies them in a single batched forward pass.
// If the rolling acceptance rate drops below 0.4, generation falls back
// to standard autoregressive decoding for the remainder.
func WithSpeculativeDraft[T tensor.Numeric](draftGraph *graph.Graph[T], draftCfg ModelConfig, draftLen int) GeneratorOption {
	if draftLen <= 0 {
		draftLen = 4
	}
	return func(o *generatorOptions) {
		o.specDraft = &specDraftConfig{
			draftGraph:    draftGraph,
			draftCfg:      draftCfg,
			draftLen:      draftLen,
			fallbackAlpha: 0.4,
		}
	}
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

// WithPrefixCache enables prefix caching with the given capacity in blocks.
// When enabled and paged KV is active, sessions that share the same system
// prompt prefix reuse cached KV blocks instead of re-running prefill.
func WithPrefixCache(capacityBlocks int) GeneratorOption {
	return func(o *generatorOptions) {
		o.prefixCacheBlocks = capacityBlocks
	}
}

// WithCompressedKV enables compressed KV caching using chunk-wise mean pooling.
// When a chunk of chunkSize tokens fills up, it is compressed into a single
// vector by averaging. If chunkSize <= 0, it defaults to 64.
func WithCompressedKV(chunkSize int) GeneratorOption {
	if chunkSize <= 0 {
		chunkSize = 64
	}
	return func(o *generatorOptions) {
		o.compressedKVChunkSize = chunkSize
	}
}

// WithEAGLE enables EAGLE-style self-speculative decoding. headWeightsPath
// points to a GGUF file containing the EAGLE head weights. When the file
// exists at generation time the generator uses the EAGLE decode loop; if the
// file is missing or the path is empty, it falls back to vanilla
// autoregressive decoding.
func WithEAGLE(headWeightsPath string) GeneratorOption {
	return func(o *generatorOptions) {
		o.eagleWeightsPath = headWeightsPath
	}
}

// WithTieredKV enables tiered KV caching with hot/warm/cold storage tiers.
// When enabled, a TieredKVStore is created per generation call using the
// provided configuration. NumLayers and MaxSeqLen are filled from the model
// config if left at zero.
func WithTieredKV(cfg TieredKVStoreConfig) GeneratorOption {
	return func(o *generatorOptions) {
		o.tieredKVCfg = &cfg
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
	kvDtype   string                                     // KV cache storage dtype ("fp32" or "fp16")
	plan      atomic.Pointer[graph.ExecutionPlan[T]]     // compiled decode plan (nil until first decode)
	planOnce  sync.Once                                  // ensures compile happens once
	mu              sync.Mutex                                 // serializes Generate/GenerateStream calls (graph state is not concurrent-safe)
	specDraft             *specDraftConfig                           // nil unless speculative decoding is enabled
	prefixCache           *PrefixCache[T]                            // nil unless prefix caching is enabled
	specAcceptRate        runtime.GaugeMetric                        // speculative acceptance rate gauge
	compressedKVChunkSize int                                        // when > 0, use CompressedKVCache
	eagleWeightsPath      string                                     // when non-empty, EAGLE decode is preferred
	tieredKVCfg           *TieredKVStoreConfig                       // when non-nil, use TieredKVStore per generation call
	pjrtPlan              *graph.PJRTPlan[T]                         // when non-nil, use PJRT backend for inference
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

	mc := gopts.metricsCollector
	if mc == nil {
		mc = runtime.Nop()
	}

	var pjrtPlan *graph.PJRTPlan[T]
	if gopts.pjrtPlan != nil {
		pjrtPlan, _ = gopts.pjrtPlan.(*graph.PJRTPlan[T])
	}

	gen := &Generator[T]{
		graph:                 g,
		tokenizer:             tok,
		engine:                eng,
		config:                cfg,
		kvDtype:               gopts.kvDtype,
		specDraft:             gopts.specDraft,
		specAcceptRate:        mc.Gauge("speculative_acceptance_rate"),
		compressedKVChunkSize: gopts.compressedKVChunkSize,
		eagleWeightsPath:      gopts.eagleWeightsPath,
		tieredKVCfg:           gopts.tieredKVCfg,
		pjrtPlan:              pjrtPlan,
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

	if gopts.prefixCacheBlocks > 0 && gen.blockPool != nil {
		gen.prefixCache = NewPrefixCache[T](gopts.prefixCacheBlocks, gen.blockPool)
	}

	return gen
}

// Graph returns the underlying computation graph.
func (gen *Generator[T]) Graph() *graph.Graph[T] { return gen.graph }

// LockGraph acquires the generator's graph mutex. The graph is not
// concurrency-safe, so any caller that runs Forward on gen.Graph() outside of
// Generate/GenerateStream (e.g. a speculative-decode path that builds its own
// generator over this Generator's graph) must hold this lock for the
// duration of that work to serialize with normal generation.
func (gen *Generator[T]) LockGraph() { gen.mu.Lock() }

// UnlockGraph releases the generator's graph mutex acquired by LockGraph.
func (gen *Generator[T]) UnlockGraph() { gen.mu.Unlock() }

// PJRTPlan returns the PJRT plan, or nil if PJRT is not enabled.
func (gen *Generator[T]) PJRTPlan() *graph.PJRTPlan[T] { return gen.pjrtPlan }

// Tokenizer returns the tokenizer.
func (gen *Generator[T]) Tokenizer() tokenizer.Tokenizer { return gen.tokenizer }

// Engine returns the compute engine.
func (gen *Generator[T]) Engine() compute.Engine[T] { return gen.engine }

// Config returns the model configuration.
func (gen *Generator[T]) Config() ModelConfig { return gen.config }

// GetPrefixCache returns the prefix cache, or nil if prefix caching is disabled.
func (gen *Generator[T]) GetPrefixCache() *PrefixCache[T] { return gen.prefixCache }

// EAGLEWeightsPath returns the configured EAGLE head weights path, or empty
// if EAGLE was not requested.
func (gen *Generator[T]) EAGLEWeightsPath() string { return gen.eagleWeightsPath }

// EAGLEEnabled reports whether EAGLE speculative decoding should be used.
// It returns true only when a weights path was configured and the file exists.
func (gen *Generator[T]) EAGLEEnabled() bool {
	if gen.eagleWeightsPath == "" {
		return false
	}
	_, err := os.Stat(gen.eagleWeightsPath)
	return err == nil
}

// compileGraph tries CompileTraced when an EngineProxy is available, with
// graceful fallback to Compile on error or plan validation failure.
//
// Compilation runs Forward on graph nodes to trace operations. To prevent
// these extra forward passes from corrupting the inference KV cache (which
// would cause duplicate entries and wrong RoPE positions), we strip the
// KV cache from the context used for compilation.
func (gen *Generator[T]) compileGraph(ctx context.Context, tokenTensor *tensor.TensorNumeric[T]) {
	gen.planOnce.Do(func() {
		// Use a cache-free context for compilation so tracing and
		// validation forward passes do not mutate the real KV cache.
		compileCtx := context.WithValue(ctx, kvCacheKey{}, nil)

		var compiled *graph.ExecutionPlan[T]
		var cErr error
		if proxy := gen.graph.EngineProxy(); proxy != nil {
			compiled, cErr = gen.graph.CompileTraced(compileCtx, tokenTensor)
			if cErr == nil {
				// Validate traced plan with a test run.
				if _, vErr := compiled.Run(compileCtx, tokenTensor); vErr != nil {
					log.Printf("generate: CompileTraced plan validation failed, falling back to Compile: %v", vErr)
					compiled, cErr = gen.graph.Compile(compileCtx, tokenTensor)
				}
			} else {
				log.Printf("generate: CompileTraced failed, falling back to Compile: %v", cErr)
				compiled, cErr = gen.graph.Compile(compileCtx, tokenTensor)
			}
		} else {
			compiled, cErr = gen.graph.Compile(compileCtx, tokenTensor)
		}
		if cErr == nil {
			// CUDA graph capture for the decode loop: position-independent
			// GPU ops run inside the captured graph for near-zero launch
			// overhead. Disable with ZERFOO_DISABLE_CUDA_GRAPH=1.
			if sp, ok := any(gen.engine).(compute.StreamProvider); ok && os.Getenv("ZERFOO_DISABLE_CUDA_GRAPH") == "" {
				if streamPtr := sp.Stream(); streamPtr != nil {
					if cuda.Available() && cuda.Lib().GraphAvailable() {
						// After capture, protect arena allocations so Reset
						// doesn't reclaim the GPU buffers referenced by the graph.
						var onCaptured func()
						if ap, ok := any(gen.engine).(interface{ ArenaUsedBytes() int }); ok {
							if asf, ok2 := any(gen.engine).(interface{ SetArenaResetFloor(int) }); ok2 {
								onCaptured = func() {
									floor := ap.ArenaUsedBytes()
									asf.SetArenaResetFloor(floor)
								}
							}
						}
						// Snapshot/restore KV cache on capture failure to prevent
					// double-updating the cache when falling back to RunInstructions.
					snapshotCache := func(ctx context.Context) func() {
						cache, ok := GetCache[T](ctx)
						if !ok {
							return func() {}
						}
						preCapSeqLen := cache.SeqLen()
						return func() {
							cache.Truncate(preCapSeqLen)
						}
					}
					ge := graph.NewCUDAGraphExecutor[T](compiled, streamPtr, 2, onCaptured, snapshotCache)
						compiled.SetMegakernelFn(func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
							return ge.Run(ctx, inputs...)
						})
					}
				}
			}
			gen.plan.Store(compiled)
			go tryCompileMegakernel(compiled, nil)
		}
	})
}

// Generate produces text from a prompt using the given sampling configuration.
// It tokenizes the prompt, runs the autoregressive loop with KV caching, and
// returns the generated text (excluding the prompt).
//
// When WithSpeculativeDraft is configured, Generate uses speculative decoding:
// the draft model proposes K tokens, the target model verifies them in one
// forward pass. If the rolling acceptance rate (alpha) drops below 0.4,
// generation falls back to standard autoregressive decoding.
func (gen *Generator[T]) Generate(ctx context.Context, prompt string, sc SamplingConfig) (string, error) {
	gen.mu.Lock()
	defer gen.mu.Unlock()

	if sc.MaxNewTokens <= 0 {
		sc.MaxNewTokens = 256
	}

	// Speculative decoding path: delegate to SpeculativeGenerator with
	// alpha-based fallback to standard decode.
	if gen.specDraft != nil {
		return gen.generateSpeculative(ctx, prompt, sc)
	}

	// Build grammar vocab cache once if grammar-constrained decoding is active.
	if sc.GrammarState != nil {
		vocabSize := gen.tokenizer.VocabSize()
		sc.grammarVocab = make([]string, vocabSize)
		for i := range vocabSize {
			if tok, ok := gen.tokenizer.GetToken(i); ok {
				sc.grammarVocab[i] = tok
			}
		}
	}

	promptIDs, err := gen.tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}

	pf, err := gen.prefillSetup(ctx, promptIDs, sc)
	if err != nil {
		return "", err
	}
	if pf.tieredStore != nil {
		defer pf.tieredStore.Close()
	}

	if debugOnnx {
		log.Printf("[DEBUG_ONNX] Generate: cacheType=%T numLayers=%d maxSeqLen=%d vocabSize=%d",
			pf.cacheProvider, gen.config.NumLayers, gen.config.MaxSeqLen, gen.config.VocabSize)
	}

	nextToken := pf.nextToken
	generatedIDs := pf.generatedIDs

	if debugOnnx {
		log.Printf("[DEBUG_ONNX] Generate: prefill produced token %d, promptIDs=%v", nextToken, promptIDs)
	}

	// Advance grammar state after sampling.
	if sc.GrammarState != nil {
		sc.GrammarState = advanceGrammar(sc.GrammarState, nextToken, sc.grammarVocab)
	}

	if pf.stopSet[nextToken] {
		return "", nil
	}
	generatedIDs = append(generatedIDs, nextToken)

	// Running state for incremental stop-string checking.
	var runningDecoded string
	var decodedCount int

	if stopped, text := gen.checkStop(generatedIDs, sc.StopStrings, &runningDecoded, &decodedCount); stopped {
		return text, nil
	}

	// Autoregressive decode loop.
	for range sc.MaxNewTokens - 1 {
		if err := ctx.Err(); err != nil {
			break
		}

		step, err := gen.runDecodeStep(ctx, pf.genCtx, pf.tokenTensor, pf.decodeBuf, nextToken, sc, generatedIDs, pf.stopSet)
		if err != nil {
			return "", err
		}
		nextToken = step.Token

		if debugOnnx {
			cacheSeq := -1
			if cache, ok := GetCache[T](pf.genCtx); ok {
				cacheSeq = cache.SeqLen()
			}
			log.Printf("[DEBUG_ONNX] decode step: token=%d cacheSeqLen=%d generatedSoFar=%d",
				nextToken, cacheSeq, len(generatedIDs)+1)
		}

		// Advance grammar state after sampling.
		if sc.GrammarState != nil {
			sc.GrammarState = advanceGrammar(sc.GrammarState, nextToken, sc.grammarVocab)
		}

		if step.Stop {
			break
		}
		generatedIDs = append(generatedIDs, nextToken)

		if stopped, text := gen.checkStop(generatedIDs, sc.StopStrings, &runningDecoded, &decodedCount); stopped {
			return text, nil
		}

		// Stop early if grammar is complete (valid JSON fully generated).
		if sc.GrammarState != nil && sc.GrammarState.IsComplete() {
			break
		}
	}

	syncGPUCounter[T](pf.cacheProvider)

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

	// GPU argmax fast path: copies back 4 bytes instead of ~1MB.
	if idx, ok := tryGPUArgmax(logits, gen.engine, sc); ok {
		return idx, nil
	}

	// Allocate a local buffer for logits. A struct-level buffer would be
	// racy when BatchGenerate spawns concurrent goroutines that each call
	// Generate on the same Generator instance.
	data, err := copyLogitsToCPU(logits, seqLen, vocabSize)
	if err != nil {
		return 0, err
	}

	lastStart := (seqLen - 1) * vocabSize
	if lastStart+vocabSize > len(data) {
		return 0, fmt.Errorf("logits data too short: %d < %d", len(data), lastStart+vocabSize)
	}

	// Debug: log top-5 logits for ONNX diagnosis.
	if debugOnnx {
		type tokenLogit struct {
			id    int
			logit float64
		}
		topN := make([]tokenLogit, vocabSize)
		allZero := true
		hasNaN := false
		for i := 0; i < vocabSize; i++ {
			v := float64(data[lastStart+i])
			topN[i] = tokenLogit{id: i, logit: v}
			if v != 0 {
				allZero = false
			}
			if math.IsNaN(v) || math.IsInf(v, 0) {
				hasNaN = true
			}
		}
		sort.Slice(topN, func(a, b int) bool { return topN[a].logit > topN[b].logit })
		top5 := topN
		if len(top5) > 5 {
			top5 = top5[:5]
		}
		log.Printf("[DEBUG_ONNX] sampleFromLogits: vocabSize=%d seqLen=%d allZero=%v hasNaN=%v top5=%v generated_so_far=%d",
			vocabSize, seqLen, allZero, hasNaN, top5, len(generatedTokens))
	}

	// Greedy fast path: find argmax directly in the T buffer without
	// allocating a float64 slice. This eliminates 2MB of allocation per token.
	// Disabled when grammar-constrained decoding is active (must mask first).
	if sc.GrammarState == nil && sc.Temperature <= 0 && (sc.RepetitionPenalty <= 0 || sc.RepetitionPenalty == 1.0) {
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

	// Apply grammar token mask before any other logit modification.
	if sc.GrammarState != nil && len(sc.grammarVocab) > 0 {
		mask := grammar.TokenMask(sc.GrammarState, sc.grammarVocab)
		applyTokenMask(logitsF64, mask)
	}

	if sc.RepetitionPenalty > 0 && sc.RepetitionPenalty != 1.0 {
		applyRepetitionPenalty(logitsF64, generatedTokens, sc.RepetitionPenalty)
	}

	return applyTemperatureAndTopP(logitsF64, sc, vocabSize), nil
}

// idsToTensor converts token IDs to a [1, seqLen] input tensor.
func (gen *Generator[T]) idsToTensor(ids []int) (*tensor.TensorNumeric[T], error) {
	data := make([]T, len(ids))
	for i, id := range ids {
		data[i] = T(id)
	}
	return tensor.New([]int{1, len(ids)}, data)
}

// selectCacheProvider creates the appropriate KV cache provider based on the
// generator's configuration. It returns the cache provider and, when tiered
// storage is used, the TieredKVStore that the caller must Close after use.
func (gen *Generator[T]) selectCacheProvider() (CacheProvider[T], *TieredKVStore[T], error) {
	if gen.tieredKVCfg != nil {
		cfg := *gen.tieredKVCfg
		if cfg.NumLayers == 0 {
			cfg.NumLayers = gen.config.NumLayers
		}
		if cfg.MaxSeqLen == 0 {
			cfg.MaxSeqLen = gen.config.MaxSeqLen
		}
		store, err := NewTieredKVStore[T](gen.engine, cfg)
		if err != nil {
			return nil, nil, fmt.Errorf("create tiered KV store: %w", err)
		}
		return &tieredKVAdapter[T]{store: store}, store, nil
	}
	if gen.compressedKVChunkSize > 0 {
		return NewCompressedKVCache[T](gen.engine, gen.config.NumLayers, 0, 0, gen.compressedKVChunkSize), nil, nil
	}
	if gen.blockPool != nil {
		return NewPagedKVCache[T](gen.blockPool, gen.config.NumLayers), nil, nil
	}
	if qc, ok := gen.newQuantizedCache(); ok {
		return qc, nil, nil
	}
	if _, ok := any(gen.engine).(compute.WeightUploader); ok {
		return gen.newTensorCache(), nil, nil
	}
	return NewKVCache[T](gen.config.NumLayers, gen.config.MaxSeqLen), nil, nil
}

// newQuantizedCache returns a Q4 or Q3 quantized KV cache when kvDtype
// requests one and T is float32. Returns nil, false otherwise.
func (gen *Generator[T]) newQuantizedCache() (CacheProvider[T], bool) {
	switch gen.kvDtype {
	case "q4":
		if cp, ok := any(NewKVCacheQ4(gen.config.NumLayers, gen.config.MaxSeqLen)).(CacheProvider[T]); ok {
			return cp, true
		}
	case "q3":
		if cp, ok := any(NewKVCacheQ3(gen.config.NumLayers, gen.config.MaxSeqLen)).(CacheProvider[T]); ok {
			return cp, true
		}
	}
	return nil, false
}

// newTensorCache creates a TensorCache with the generator's KV dtype setting.
func (gen *Generator[T]) newTensorCache() *TensorCache[T] {
	var opts []TensorCacheOption
	if gen.kvDtype == "fp16" {
		opts = append(opts, WithKVDtype("fp16"))
	}
	return NewTensorCache[T](gen.engine, gen.config.NumLayers, gen.config.MaxSeqLen, opts...)
}

// checkStop checks if the decoded generated tokens contain any stop string.
// It delegates to incrementalCheckStop which maintains a running decoded string
// across calls to avoid re-decoding all tokens on every step.
func (gen *Generator[T]) checkStop(generatedIDs []int, stopStrings []string, prevDecoded *string, prevCount *int) (bool, string) {
	if len(stopStrings) == 0 {
		return false, ""
	}
	return incrementalCheckStop(gen.tokenizer, generatedIDs, stopStrings, prevDecoded, prevCount)
}

// generateSpeculative runs speculative decoding using the configured draft
// model. It starts with speculative steps, tracking the rolling acceptance
// rate. If alpha drops below the fallback threshold (0.4), it switches to
// standard autoregressive decoding for the remaining tokens.
func (gen *Generator[T]) generateSpeculative(ctx context.Context, prompt string, sc SamplingConfig) (string, error) {
	draftGraph, ok := gen.specDraft.draftGraph.(*graph.Graph[T])
	if !ok {
		return "", fmt.Errorf("speculative draft graph type mismatch")
	}

	promptIDs, err := gen.tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encode prompt: %w", err)
	}
	if len(promptIDs) == 0 {
		return "", fmt.Errorf("prompt produced no tokens")
	}

	// Prepend BOS token if configured.
	if gen.config.BOSTokenID > 0 {
		promptIDs = append([]int{gen.config.BOSTokenID}, promptIDs...)
	}

	stopSet := make(map[int]bool, len(sc.StopTokenIDs)+1)
	for _, id := range sc.StopTokenIDs {
		stopSet[id] = true
	}
	stopSet[gen.config.EOSTokenID] = true

	// Create KV caches for both models.
	draftCache := NewKVCache[T](gen.specDraft.draftCfg.NumLayers, gen.specDraft.draftCfg.MaxSeqLen)
	targetCache := NewKVCache[T](gen.config.NumLayers, gen.config.MaxSeqLen)

	draftCtx := WithCache(ctx, CacheProvider[T](draftCache))
	targetCtx := WithCache(ctx, CacheProvider[T](targetCache))

	// Reset stateful nodes before generation.
	gen.graph.ResetStatefulNodes()
	draftGraph.ResetStatefulNodes()

	// Prefill both models with the prompt.
	prefillTensor, err := gen.idsToTensor(promptIDs)
	if err != nil {
		return "", fmt.Errorf("create prefill tensor: %w", err)
	}

	_, err = draftGraph.Forward(draftCtx, prefillTensor)
	if err != nil {
		return "", fmt.Errorf("draft prefill: %w", err)
	}

	targetLogits, err := gen.graph.Forward(targetCtx, prefillTensor)
	if err != nil {
		return "", fmt.Errorf("target prefill: %w", err)
	}

	// Sample first token from target.
	firstToken, err := gen.sampleFromLogits(targetLogits, sc, nil)
	if err != nil {
		return "", fmt.Errorf("sample after prefill: %w", err)
	}
	if stopSet[firstToken] {
		return "", nil
	}

	generatedIDs := []int{firstToken}
	nextDraftInput := firstToken

	// Running state for incremental stop-string checking.
	var runningDecoded string
	var decodedCount int

	tracker := newAdaptiveDraftLen(gen.specDraft.draftLen, 1, 8, 32)
	fellBack := false

	for len(generatedIDs) < sc.MaxNewTokens {
		if err := ctx.Err(); err != nil {
			break
		}

		// Check if acceptance rate has dropped below threshold.
		if tracker.Rate() < gen.specDraft.fallbackAlpha && tracker.count >= 8 {
			fellBack = true
			break
		}

		currentDraftLen := tracker.Current()
		draftN := min(currentDraftLen, sc.MaxNewTokens-len(generatedIDs))

		// Draft phase: generate draftN tokens greedily.
		draftTokens := make([]int, 0, draftN)
		draftInput := nextDraftInput

		for range draftN {
			tokenTensor, tErr := gen.idsToTensor([]int{draftInput})
			if tErr != nil {
				return "", fmt.Errorf("draft token tensor: %w", tErr)
			}

			draftLogits, fErr := draftGraph.Forward(draftCtx, tokenTensor)
			if fErr != nil {
				return "", fmt.Errorf("draft forward: %w", fErr)
			}

			draftToken, sErr := gen.sampleFromLogits(draftLogits, SamplingConfig{Temperature: 0}, nil)
			if sErr != nil {
				return "", fmt.Errorf("sample draft: %w", sErr)
			}
			draftTokens = append(draftTokens, draftToken)

			if stopSet[draftToken] {
				break
			}
			draftInput = draftToken
		}

		if len(draftTokens) == 0 {
			break
		}

		// Verify phase: target processes all draft tokens in one forward pass.
		verifyTensor, tErr := gen.idsToTensor(draftTokens)
		if tErr != nil {
			return "", fmt.Errorf("verify tensor: %w", tErr)
		}

		verifyLogits, fErr := gen.graph.Forward(targetCtx, verifyTensor)
		if fErr != nil {
			return "", fmt.Errorf("target verify forward: %w", fErr)
		}

		// Accept/reject: compare target's greedy output with draft tokens.
		accepted, bonusToken := verifyDraftTokens(verifyLogits, draftTokens, stopSet)

		// Emit accepted tokens and bonus token.
		var stopped bool
		generatedIDs, stopped = emitVerified(accepted, bonusToken, generatedIDs, sc.MaxNewTokens, stopSet)
		if stopped {
			break
		}

		// Record acceptance rate and update the Prometheus gauge.
		tracker.Record(len(accepted), len(draftTokens))
		gen.specAcceptRate.Set(tracker.Rate())

		// Roll back caches if tokens were rejected.
		correctSeqLen := len(promptIDs) + len(generatedIDs)
		if draftCache.SeqLen() > correctSeqLen {
			draftCache.Truncate(correctSeqLen)
		}
		if targetCache.SeqLen() > correctSeqLen {
			targetCache.Truncate(correctSeqLen)
		}

		if len(generatedIDs) > 0 {
			nextDraftInput = generatedIDs[len(generatedIDs)-1]
		}

		// Check stop strings.
		if len(sc.StopStrings) > 0 {
			if stopped, text := gen.checkStop(generatedIDs, sc.StopStrings, &runningDecoded, &decodedCount); stopped {
				return text, nil
			}
		}
	}

	// Fallback: continue with standard autoregressive decoding using the
	// target model's KV cache (already populated with accepted tokens).
	if fellBack && len(generatedIDs) < sc.MaxNewTokens {
		lastToken := generatedIDs[len(generatedIDs)-1]
		remaining := sc.MaxNewTokens - len(generatedIDs)

		decodeBuf := []T{T(lastToken)}
		tokenTensor, tErr := tensor.New([]int{1, 1}, decodeBuf)
		if tErr != nil {
			return "", fmt.Errorf("create fallback decode tensor: %w", tErr)
		}

		for range remaining {
			if err := ctx.Err(); err != nil {
				break
			}

			decodeBuf[0] = T(lastToken)
			logits, fErr := gen.graph.Forward(targetCtx, tokenTensor)
			if fErr != nil {
				return "", fmt.Errorf("fallback decode forward: %w", fErr)
			}

			nextToken, sErr := gen.sampleFromLogits(logits, sc, generatedIDs)
			if sErr != nil {
				return "", fmt.Errorf("fallback sample: %w", sErr)
			}

			if stopSet[nextToken] {
				break
			}
			generatedIDs = append(generatedIDs, nextToken)
			lastToken = nextToken

			if stopped, text := gen.checkStop(generatedIDs, sc.StopStrings, &runningDecoded, &decodedCount); stopped {
				return text, nil
			}
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

