package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/generate/grammar"
	"github.com/zerfoo/zerfoo/model/registry"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	tokenizer "github.com/zerfoo/ztoken"
)

// defaultMaxBatchConcurrency is the default limit on simultaneous goroutines
// spawned by GenerateBatch. This prevents resource exhaustion when batching
// many prompts on GPU-backed models.
const defaultMaxBatchConcurrency = 8

// Model is a loaded model ready for generation.
type Model struct {
	generator  *generate.Generator[float32]
	tokenizer  tokenizer.Tokenizer
	engine     compute.Engine[float32]
	config     ModelMetadata
	info       *registry.ModelInfo
	closer     io.Closer // mmap reader, if loaded with mmap
	embWeights []float32 // flattened embedding table [vocabSize * hiddenSize]
	hiddenSize int       // embedding dimension

	// maxBatchConcurrency limits the number of simultaneous goroutines in
	// GenerateBatch. Zero means use defaultMaxBatchConcurrency.
	maxBatchConcurrency int

	// sessionPool reuses sessions to preserve GPU memory addresses across calls.
	// CUDA graph capture records GPU pointers; reusing sessions keeps those
	// pointers valid for graph replay. The pool grows on demand for concurrency.
	sessionPool chan *generate.InferenceSession[float32]

	// pjrtPlan holds the PJRT compiled plan when WithPJRT is used.
	// Nil when using the standard Engine path. Closed by Model.Close().
	pjrtPlan *graph.PJRTPlan[float32]
}

// ModelMetadata holds model configuration loaded from config.json.
type ModelMetadata struct {
	Architecture          string `json:"architecture"`
	VocabSize             int    `json:"vocab_size"`
	HiddenSize            int    `json:"hidden_size"`
	NumLayers             int    `json:"num_layers"`
	MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	EOSTokenID            int    `json:"eos_token_id"`
	BOSTokenID            int    `json:"bos_token_id"`
	ChatTemplate          string `json:"chat_template"`

	// Extended fields for multi-architecture support.
	IntermediateSize    int                `json:"intermediate_size"`
	NumQueryHeads       int                `json:"num_attention_heads"`
	NumKeyValueHeads    int                `json:"num_key_value_heads"`
	RopeTheta           float64            `json:"rope_theta"`
	RopeScaling         *RopeScalingConfig `json:"rope_scaling,omitempty"`
	TieWordEmbeddings   bool               `json:"tie_word_embeddings"`
	SlidingWindow       int                `json:"sliding_window"`
	AttentionBias       bool               `json:"attention_bias"`
	PartialRotaryFactor float64            `json:"partial_rotary_factor"`
	LayerNormEps        float64            `json:"layer_norm_eps,omitempty"`

	// Granite-specific fields.
	EmbeddingMultiplier float64 `json:"embedding_multiplier,omitempty"`
	ResidualMultiplier  float64 `json:"residual_multiplier,omitempty"`
	LogitScale          float64 `json:"logit_scale,omitempty"`

	// Audio model fields.
	AudioNumMels int `json:"audio_num_mels,omitempty"` // Number of mel bins; 0 means use architecture default

	// DeepSeek MLA and MoE fields.
	KVLoRADim          int `json:"kv_lora_rank"`
	QLoRADim           int `json:"q_lora_rank"`
	QKRopeHeadDim      int `json:"qk_rope_head_dim"`
	NumExperts         int `json:"num_experts"`
	NumExpertsPerToken int `json:"num_experts_per_tok"`
	NumSharedExperts   int `json:"n_shared_experts"`
}

// modelAliases maps short model names to HuggingFace repo IDs.
var (
	modelAliasesMu sync.RWMutex
	modelAliases   = map[string]string{
		"gemma-3-1b-q4":  "google/gemma-3-1b-it-qat-q4_0-gguf",
		"gemma-3-2b-q4":  "google/gemma-3-2b-it-qat-q4_0-gguf",
		"llama-3-1b-q4":  "meta-llama/Llama-3.2-1B-Instruct-GGUF",
		"llama-3-8b-q4":  "meta-llama/Llama-3.1-8B-Instruct-GGUF",
		"mistral-7b-q4":  "mistralai/Mistral-7B-Instruct-v0.3-GGUF",
		"qwen-2.5-7b-q4": "Qwen/Qwen2.5-7B-Instruct-GGUF",
	}
)

// ResolveAlias returns the HuggingFace repo ID for a short alias.
// If the name is not an alias, it is returned unchanged.
func ResolveAlias(name string) string {
	modelAliasesMu.RLock()
	id, ok := modelAliases[name]
	modelAliasesMu.RUnlock()
	if ok {
		return id
	}
	return name
}

// RegisterAlias adds a custom short name -> HuggingFace repo ID mapping.
func RegisterAlias(shortName, repoID string) {
	modelAliasesMu.Lock()
	modelAliases[shortName] = repoID
	modelAliasesMu.Unlock()
}

// Option configures model loading.
type Option func(*loadOptions)

type loadOptions struct {
	cacheDir            string
	device              string
	maxSeqLen           int
	registry            registry.ModelRegistry
	backend             string // "" or "default" for standard engine, "tensorrt" for TRT
	precision           string // "" or "fp32" for float32, "fp16" for half precision (TRT only)
	dtype               string // "" or "fp32" for float32, "fp16" for FP16 compute
	kvDtype             string // "" or "fp32" for float32, "fp16" for FP16 KV cache
	mmap                bool   // use mmap for model loading (unix only); default true
	quarot              bool   // fuse QuaRot Hadamard rotation into weights
	maxBatchConcurrency int    // max goroutines in GenerateBatch (0 = default)
	sessionPoolSize     int    // session pool capacity (0 = default 16)
	pjrtPlugin          string // path to PJRT plugin .so (empty = disabled)
}

// WithCacheDir sets the model cache directory.
func WithCacheDir(dir string) Option {
	return func(o *loadOptions) {
		o.cacheDir = dir
	}
}

// WithDevice sets the compute device ("cpu" or "cuda").
func WithDevice(device string) Option {
	return func(o *loadOptions) {
		o.device = device
	}
}

// WithMaxSeqLen overrides the model's default max sequence length.
func WithMaxSeqLen(n int) Option {
	return func(o *loadOptions) {
		o.maxSeqLen = n
	}
}

// WithRegistry provides a custom model registry.
func WithRegistry(r registry.ModelRegistry) Option {
	return func(o *loadOptions) {
		o.registry = r
	}
}

// WithBackend selects the inference backend. Supported values: "" or "default"
// for the standard Engine path, "tensorrt" for TensorRT-optimized inference.
// TensorRT requires the cuda build tag and a CUDA device.
func WithBackend(backend string) Option {
	return func(o *loadOptions) {
		o.backend = backend
	}
}

// WithPrecision sets the compute precision for the TensorRT backend.
// Supported values: "" or "fp32" for full precision, "fp16" for half precision.
// Has no effect when the backend is not "tensorrt".
func WithPrecision(precision string) Option {
	return func(o *loadOptions) {
		o.precision = precision
	}
}

// WithDType sets the compute precision for the GPU engine.
// Supported values: "" or "fp32" for full precision, "fp16" for FP16 compute.
// FP16 mode converts activations F32->FP16 before GPU kernels and back after.
// Has no effect on CPU engines.
func WithDType(dtype string) Option {
	return func(o *loadOptions) {
		o.dtype = dtype
	}
}

// WithMmap controls memory-mapped model loading. When true (the default),
// the GGUF file is mapped into memory using mmap instead of reading into
// heap-allocated slices. This gives near-instant startup, keeps tensor data
// off the Go heap, and allows loading models larger than physical RAM —
// the OS pages data in from disk on demand.
func WithMmap(enabled bool) Option {
	return func(o *loadOptions) {
		o.mmap = enabled
	}
}

// WithQuaRot enables QuaRot (Quantization with Rotation) weight fusion.
// When enabled, a normalized Walsh-Hadamard rotation is fused into Q/K/V/O
// and FFN gate/up/down weight matrices after loading. This improves
// quantization quality at zero runtime cost (arXiv:2404.00456).
func WithQuaRot(enabled bool) Option {
	return func(o *loadOptions) {
		o.quarot = enabled
	}
}

// WithPJRT sets the path to a PJRT plugin shared library (.so). When set,
// the inference pipeline loads the plugin and uses the PJRT backend for
// compilation and execution instead of the standard Engine path.
// An empty string (the default) disables PJRT.
func WithPJRT(pluginPath string) Option {
	return func(o *loadOptions) {
		o.pjrtPlugin = pluginPath
	}
}

// WithKVDtype sets the KV cache storage dtype. Supported: "fp32" (default), "fp16".
// FP16 halves KV cache bandwidth by storing keys/values in half precision.
func WithKVDtype(dtype string) Option {
	return func(o *loadOptions) {
		o.kvDtype = dtype
	}
}

// WithMaxBatchConcurrency sets the maximum number of concurrent goroutines
// that GenerateBatch will use. Values <= 0 are ignored (the default of 8 is used).
func WithMaxBatchConcurrency(n int) Option {
	return func(o *loadOptions) {
		if n > 0 {
			o.maxBatchConcurrency = n
		}
	}
}

// defaultSessionPoolSize is the default capacity of the session pool.
const defaultSessionPoolSize = 16

// WithSessionPoolSize sets the session pool capacity. The pool buffers
// inference sessions for reuse so that CUDA graph-captured GPU pointers
// remain valid across calls. Minimum value is 1; values below 1 are
// clamped to 1.
func WithSessionPoolSize(n int) Option {
	return func(o *loadOptions) {
		if n < 1 {
			n = 1
		}
		o.sessionPoolSize = n
	}
}

// Load loads a model by ID, pulling it if not cached.
func Load(modelID string, opts ...Option) (*Model, error) {
	o := &loadOptions{
		device: "cpu",
		mmap:   true,
	}
	for _, opt := range opts {
		opt(o)
	}

	// Resolve short aliases to full repo IDs.
	modelID = ResolveAlias(modelID)

	// Get or create registry.
	reg := o.registry
	if reg == nil {
		var err error
		var lr *registry.LocalRegistry
		if o.cacheDir != "" {
			lr, err = registry.NewLocalRegistry(o.cacheDir)
		} else {
			lr, err = registry.NewLocalRegistry("")
		}
		if err != nil {
			return nil, fmt.Errorf("create registry: %w", err)
		}
		// Wire the HuggingFace pull function by default.
		lr.SetPullFunc(registry.NewHFPullFunc(registry.HFPullOptions{}))
		reg = lr
	}

	// Check cache first, pull if needed.
	info, ok := reg.Get(modelID)
	if !ok {
		var err error
		info, err = reg.Pull(context.Background(), modelID)
		if err != nil {
			return nil, fmt.Errorf("pull model %q: %w", modelID, err)
		}
	}

	// If a GGUF file exists in the model directory, use the GGUF loader.
	if ggufPath := findGGUF(info.Path); ggufPath != "" {
		return LoadFile(ggufPath, opts...)
	}

	// ZMF loading was removed; only GGUF is supported via LoadFile.
	return nil, fmt.Errorf("model %q has no GGUF file; ZMF loading is no longer supported", modelID)
}

// findGGUF looks for a .gguf file in the given directory.
// Returns the full path if found, empty string otherwise.
func findGGUF(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			return filepath.Join(dir, e.Name())
		}
	}
	return ""
}

// assembleModel wires together loaded components into a ready-to-use Model.
func assembleModel(
	g *graph.Graph[float32],
	tok tokenizer.Tokenizer,
	eng compute.Engine[float32],
	meta *ModelMetadata,
	info *registry.ModelInfo,
	maxSeqLenOverride int,
	kvDtype string,
) *Model {
	maxSeqLen := meta.MaxPositionEmbeddings
	if maxSeqLenOverride > 0 {
		maxSeqLen = maxSeqLenOverride
	}

	var genOpts []generate.GeneratorOption
	if kvDtype == "fp16" {
		genOpts = append(genOpts, generate.WithGeneratorKVDtype("fp16"))
	}

	gen := generate.NewGenerator(g, tok, eng, generate.ModelConfig{
		VocabSize:  meta.VocabSize,
		MaxSeqLen:  maxSeqLen,
		EOSTokenID: meta.EOSTokenID,
		BOSTokenID: meta.BOSTokenID,
		NumLayers:  meta.NumLayers,
	}, genOpts...)

	pool := make(chan *generate.InferenceSession[float32], defaultSessionPoolSize)
	pool <- gen.NewSession() // Pre-warm with one session for CUDA graph address reuse.

	return &Model{
		generator:   gen,
		tokenizer:   tok,
		engine:      eng,
		config:      *meta,
		info:        info,
		sessionPool: pool,
	}
}

// loadMetadata reads config.json and dispatches to the appropriate
// architecture-specific parser via the default config registry.
func loadMetadata(path string) (*ModelMetadata, error) {
	data, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	reg := DefaultArchConfigRegistry()
	meta, err := reg.Parse(raw)
	if err != nil {
		return nil, err
	}

	// Overlay fields that use our internal JSON tags (not HuggingFace names).
	if ct, ok := raw["chat_template"].(string); ok {
		meta.ChatTemplate = ct
	}

	return meta, nil
}

// GenerateOption configures a generation call.
type GenerateOption func(*generate.SamplingConfig)

// WithTemperature sets the sampling temperature.
func WithTemperature(t float64) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.Temperature = t
	}
}

// WithTopK sets the top-K sampling parameter.
func WithTopK(k int) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.TopK = k
	}
}

// WithTopP sets the top-P (nucleus) sampling parameter.
func WithTopP(p float64) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.TopP = p
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.MaxNewTokens = n
	}
}

// WithRepetitionPenalty sets the repetition penalty factor.
func WithRepetitionPenalty(p float64) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.RepetitionPenalty = p
	}
}

// WithStopStrings sets strings that stop generation.
func WithStopStrings(ss ...string) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.StopStrings = ss
	}
}

// WithGrammar sets a grammar state machine for constrained decoding.
// When set, a token mask is applied at each sampling step to restrict output
// to tokens that are valid according to the grammar.
func WithGrammar(g *grammar.Grammar) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.GrammarState = g
	}
}

// WithAdapter sets the LoRA adapter name for per-request adapter selection.
func WithAdapter(name string) GenerateOption {
	return func(sc *generate.SamplingConfig) {
		sc.AdapterName = name
	}
}

func buildSamplingConfig(opts []GenerateOption) generate.SamplingConfig {
	sc := generate.DefaultSamplingConfig()
	for _, opt := range opts {
		opt(&sc)
	}
	return sc
}

// acquireSession gets a session from the pool or creates a new one.
// Reusing sessions preserves GPU memory addresses for CUDA graph replay.
func (m *Model) acquireSession() *generate.InferenceSession[float32] {
	if m.sessionPool == nil {
		return m.generator.NewSession()
	}
	select {
	case sess := <-m.sessionPool:
		return sess
	default:
		return m.generator.NewSession()
	}
}

// releaseSession returns a session to the pool for reuse.
func (m *Model) releaseSession(sess *generate.InferenceSession[float32]) {
	if m.sessionPool == nil {
		return
	}
	select {
	case m.sessionPool <- sess:
	default:
		// Pool full, discard.
	}
}

// Generate produces text from a prompt. Sessions are pooled to reuse GPU
// memory addresses, enabling CUDA graph replay across calls. Concurrent
// Generate calls get separate sessions from the pool.
func (m *Model) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (string, error) {
	sc := buildSamplingConfig(opts)
	sess := m.acquireSession()
	defer m.releaseSession(sess)
	result, err := sess.Generate(ctx, prompt, sc)
	return result, err
}

// GenerateBatch processes multiple prompts concurrently and returns the
// generated text for each prompt. Results are returned in the same order as
// the input prompts. If a prompt fails, its corresponding error is non-nil.
//
// Concurrency is capped at maxBatchConcurrency (default 8) to prevent
// resource exhaustion on GPU-backed models.
//
// [Deviation: Architectural] Used parallel goroutines instead of shared
// PagedKV decode — full multi-seq requires deeper Generator refactor.
func (m *Model) GenerateBatch(ctx context.Context, prompts []string, opts ...GenerateOption) ([]string, error) {
	if len(prompts) == 0 {
		return nil, nil
	}

	sc := buildSamplingConfig(opts)
	results := make([]string, len(prompts))
	errs := make([]error, len(prompts))

	maxConc := m.maxBatchConcurrency
	if maxConc <= 0 {
		maxConc = defaultMaxBatchConcurrency
	}

	sem := make(chan struct{}, maxConc)
	var wg sync.WaitGroup
	wg.Add(len(prompts))
	for i, prompt := range prompts {
		go func(idx int, p string) {
			defer wg.Done()
			sem <- struct{}{}        // acquire
			defer func() { <-sem }() // release
			sess := m.acquireSession()
			text, err := sess.Generate(ctx, p, sc)
			m.releaseSession(sess)
			results[idx] = text
			errs[idx] = err
		}(i, prompt)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return results, fmt.Errorf("batch generation: %w", err)
		}
	}
	return results, nil
}

// SetMaxBatchConcurrency sets the maximum number of concurrent goroutines
// that GenerateBatch will use. Values <= 0 are ignored.
func (m *Model) SetMaxBatchConcurrency(n int) {
	if n > 0 {
		m.maxBatchConcurrency = n
	}
}

// GenerateStream delivers tokens one at a time via a callback. Sessions
// are pooled to preserve GPU memory addresses for CUDA graph replay.
func (m *Model) GenerateStream(ctx context.Context, prompt string, handler generate.TokenStream, opts ...GenerateOption) error {
	sc := buildSamplingConfig(opts)
	sess := m.acquireSession()
	defer m.releaseSession(sess)
	err := sess.GenerateStream(ctx, prompt, sc, handler)
	return err
}

// Message represents a chat message.
type Message struct {
	Role    string // "system", "user", or "assistant"
	Content string
	Images  [][]byte // optional raw image data for vision models
}

// Response holds the result of a chat completion.
type Response struct {
	Content          string
	TokensUsed       int
	PromptTokens     int
	CompletionTokens int
}

// Chat formats messages using the model's chat template and generates a response.
// Sessions are pooled to preserve CUDA graph replay.
func (m *Model) Chat(ctx context.Context, messages []Message, opts ...GenerateOption) (Response, error) {
	prompt := m.formatMessages(messages)
	sc := buildSamplingConfig(opts)
	sess := m.acquireSession()
	defer m.releaseSession(sess)
	result, err := sess.Generate(ctx, prompt, sc)
	if err != nil {
		return Response{}, err
	}

	// Count prompt and completion tokens separately.
	promptIDs, _ := m.tokenizer.Encode(prompt)
	resultIDs, _ := m.tokenizer.Encode(result)
	promptCount := len(promptIDs)
	completionCount := len(resultIDs)
	return Response{
		Content:          result,
		PromptTokens:     promptCount,
		CompletionTokens: completionCount,
		TokensUsed:       promptCount + completionCount,
	}, nil
}

// FormatMessages converts messages to the model's chat template format.
// This is useful when callers need the formatted prompt without running inference,
// e.g. for streaming paths that call GenerateStream separately.
func (m *Model) FormatMessages(messages []Message) string {
	return m.formatMessages(messages)
}

// ChatStream formats messages using the model's chat template and streams
// the response token-by-token via the provided handler. This is the streaming
// counterpart of Chat and ensures the same prompt formatting is applied.
func (m *Model) ChatStream(ctx context.Context, messages []Message, handler generate.TokenStream, opts ...GenerateOption) error {
	prompt := m.formatMessages(messages)
	return m.GenerateStream(ctx, prompt, handler, opts...)
}

// formatMessages converts messages to the model's chat template format.
func (m *Model) formatMessages(messages []Message) string {
	template := strings.ToLower(m.config.ChatTemplate)
	if template == "" {
		template = "gemma"
	}

	switch template {
	case "gemma":
		return formatGemma(messages)
	case "llama":
		return formatLlama(messages)
	case "mistral":
		return formatMistral(messages)
	case "qwen2":
		return formatQwen(messages)
	case "deepseek":
		return formatDeepSeek(messages)
	case "phi":
		return formatPhi(messages)
	default:
		return formatGeneric(messages)
	}
}

func formatGemma(messages []Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString("<start_of_turn>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<end_of_turn>\n")
	}
	sb.WriteString("<start_of_turn>model\n")
	return sb.String()
}

func formatLlama(messages []Message) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		sb.WriteString("<|start_header_id|>")
		sb.WriteString(msg.Role)
		sb.WriteString("<|end_header_id|>\n\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|eot_id|>")
	}
	sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return sb.String()
}

func formatMistral(messages []Message) string {
	var sb strings.Builder
	// Mistral merges system into first user message.
	pending := ""
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			pending = msg.Content + "\n\n"
		case "user":
			sb.WriteString("[INST] ")
			sb.WriteString(pending)
			sb.WriteString(msg.Content)
			sb.WriteString(" [/INST]")
			pending = ""
		case "assistant":
			sb.WriteString(msg.Content)
		}
	}
	return sb.String()
}

func formatQwen(messages []Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString("<|im_start|>")
		sb.WriteString(msg.Role)
		sb.WriteString("\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|im_end|>\n")
	}
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

func formatDeepSeek(messages []Message) string {
	var sb strings.Builder
	sb.WriteString("<|begin_of_sentence|>")
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			sb.WriteString(msg.Content)
			sb.WriteString("\n\n")
		case "user":
			sb.WriteString("User: ")
			sb.WriteString(msg.Content)
			sb.WriteString("\n\n")
		case "assistant":
			sb.WriteString("Assistant: ")
			sb.WriteString(msg.Content)
			sb.WriteString("\n\n")
		}
	}
	sb.WriteString("Assistant:")
	return sb.String()
}

func formatPhi(messages []Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString("<|")
		sb.WriteString(msg.Role)
		sb.WriteString("|>\n")
		sb.WriteString(msg.Content)
		sb.WriteString("<|end|>\n")
	}
	sb.WriteString("<|assistant|>\n")
	return sb.String()
}

func formatGeneric(messages []Message) string {
	var sb strings.Builder
	for _, msg := range messages {
		sb.WriteString(msg.Role)
		sb.WriteString(": ")
		sb.WriteString(msg.Content)
		sb.WriteString("\n")
	}
	sb.WriteString("assistant: ")
	return sb.String()
}

// EmbeddingWeights returns the flattened token embedding table and the
// hidden dimension. Returns nil, 0 if embeddings are not available.
func (m *Model) EmbeddingWeights() ([]float32, int) {
	return m.embWeights, m.hiddenSize
}

// Embed returns an L2-normalized embedding vector for the given text by
// looking up token embeddings from the model's embedding table and
// mean-pooling them.
func (m *Model) Embed(text string) ([]float32, error) {
	if len(m.embWeights) == 0 || m.hiddenSize == 0 {
		return nil, fmt.Errorf("model has no embedding weights")
	}
	ids, err := m.tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("text produced no tokens")
	}

	vocabSize := len(m.embWeights) / m.hiddenSize
	dim := m.hiddenSize

	// Mean-pool token embedding vectors.
	vec := make([]float32, dim)
	for _, id := range ids {
		if id < 0 || id >= vocabSize {
			continue
		}
		off := id * dim
		for j := 0; j < dim; j++ {
			vec[j] += m.embWeights[off+j]
		}
	}
	scale := float32(1.0 / float64(len(ids)))
	for j := range vec {
		vec[j] *= scale
	}

	// L2 normalize.
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		invNorm := float32(1.0 / math.Sqrt(norm))
		for j := range vec {
			vec[j] *= invNorm
		}
	}

	return vec, nil
}

// Close releases resources held by the model. If the model was loaded on a
// GPU, this frees the CUDA engine's handles, pool, and stream. If loaded
// with mmap, this releases the memory mapping. If a PJRT plan is held,
// its executables, weight buffers, and KV cache buffers are released.
func (m *Model) Close() error {
	var firstErr error
	if m.pjrtPlan != nil {
		if err := m.pjrtPlan.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		m.pjrtPlan = nil
	}
	if c, ok := m.engine.(io.Closer); ok {
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if m.closer != nil {
		if err := m.closer.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		m.closer = nil
	}
	return firstErr
}

// Generator returns the underlying generator.
func (m *Model) Generator() *generate.Generator[float32] {
	return m.generator
}

// SpeculativeGenerate runs speculative decoding using this model as the target
// and the draft model for token proposal. draftLen controls how many tokens
// are proposed per verification step.
func (m *Model) SpeculativeGenerate(
	ctx context.Context,
	draft *Model,
	prompt string,
	draftLen int,
	opts ...GenerateOption,
) (string, error) {
	sc := buildSamplingConfig(opts)
	sc.Temperature = 0 // speculative decode is greedy-only for now

	draftCfg := generate.ModelConfig{
		VocabSize:  draft.config.VocabSize,
		MaxSeqLen:  draft.config.MaxPositionEmbeddings,
		EOSTokenID: draft.config.EOSTokenID,
		BOSTokenID: draft.config.BOSTokenID,
		NumLayers:  draft.config.NumLayers,
	}
	targetCfg := generate.ModelConfig{
		VocabSize:  m.config.VocabSize,
		MaxSeqLen:  m.config.MaxPositionEmbeddings,
		EOSTokenID: m.config.EOSTokenID,
		BOSTokenID: m.config.BOSTokenID,
		NumLayers:  m.config.NumLayers,
	}

	// The target graph is not concurrency-safe and is shared with every
	// normal (non-speculative) generation path, which serializes on this
	// same mutex (see generate.Generator.Generate / InferenceSession.Generate).
	// Hold it for the whole speculative run so a concurrent normal request
	// can't call Forward on the same graph at the same time (CONC-H1).
	m.generator.LockGraph()
	defer m.generator.UnlockGraph()

	sg := generate.NewSpeculativeGenerator(
		draft.generator.Graph(),
		m.generator.Graph(),
		m.tokenizer,
		m.engine,
		draftCfg,
		targetCfg,
		draftLen,
	)

	return sg.Generate(ctx, prompt, sc)
}

// Config returns the model metadata.
func (m *Model) Config() ModelMetadata {
	return m.config
}

// Info returns the registry info for this model.
func (m *Model) Info() *registry.ModelInfo {
	return m.info
}

// Tokenizer returns the model's tokenizer for token counting.
func (m *Model) Tokenizer() tokenizer.Tokenizer {
	return m.tokenizer
}

// parseDevice parses a device string like "cpu", "cuda", "cuda:0", or "cuda:1"
// into its type and device ID. For "cuda" without a suffix, deviceID defaults to 0.
func parseDevice(device string) (devType string, deviceID int, err error) {
	device = strings.TrimSpace(strings.ToLower(device))
	if device == "" || device == "cpu" {
		return "cpu", 0, nil
	}
	if device == "cuda" {
		return "cuda", 0, nil
	}
	if strings.HasPrefix(device, "cuda:") {
		idStr := device[len("cuda:"):]
		id, err := strconv.Atoi(idStr)
		if err != nil {
			return "", 0, fmt.Errorf("invalid device ID in %q: %w", device, err)
		}
		if id < 0 {
			return "", 0, fmt.Errorf("negative device ID in %q", device)
		}
		return "cuda", id, nil
	}
	if device == "rocm" {
		return "rocm", 0, nil
	}
	if strings.HasPrefix(device, "rocm:") {
		idStr := device[len("rocm:"):]
		id, err := strconv.Atoi(idStr)
		if err != nil {
			return "", 0, fmt.Errorf("invalid device ID in %q: %w", device, err)
		}
		if id < 0 {
			return "", 0, fmt.Errorf("negative device ID in %q", device)
		}
		return "rocm", id, nil
	}
	if device == "opencl" {
		return "opencl", 0, nil
	}
	if strings.HasPrefix(device, "opencl:") {
		idStr := device[len("opencl:"):]
		id, err := strconv.Atoi(idStr)
		if err != nil {
			return "", 0, fmt.Errorf("invalid device ID in %q: %w", device, err)
		}
		if id < 0 {
			return "", 0, fmt.Errorf("negative device ID in %q", device)
		}
		return "opencl", id, nil
	}
	return "", 0, fmt.Errorf("unsupported device %q: expected \"cpu\", \"cuda\", \"cuda:N\", \"rocm\", \"rocm:N\", \"opencl\", or \"opencl:N\"", device)
}

// DTypeSetter is implemented by engines that support setting compute precision.
type DTypeSetter interface {
	SetDType(compute.DType)
}

// applyDType sets the compute precision on the engine if supported.
func applyDType(eng compute.Engine[float32], dtype string) {
	if dtype == "" || dtype == "fp32" {
		return
	}
	if dtype == "fp16" {
		if ds, ok := eng.(DTypeSetter); ok {
			ds.SetDType(compute.DTypeFP16)
		}
	}
	if dtype == "fp8" {
		if ds, ok := eng.(DTypeSetter); ok {
			ds.SetDType(compute.DTypeFP8)
		}
	}
}

// NewTestModel constructs a Model from pre-built components.
// Intended for use in external test packages that need a Model
// without going through the full Load pipeline.
func NewTestModel(
	gen *generate.Generator[float32],
	tok tokenizer.Tokenizer,
	eng compute.Engine[float32],
	meta ModelMetadata,
	info *registry.ModelInfo,
) *Model {
	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config:    meta,
		info:      info,
	}
}

// SetEmbeddingWeights sets the token embedding table for Embed().
// weights is a flattened [vocabSize, hiddenSize] matrix.
func (m *Model) SetEmbeddingWeights(weights []float32, hiddenSize int) {
	m.embWeights = weights
	m.hiddenSize = hiddenSize
}
