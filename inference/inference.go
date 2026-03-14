// Package inference provides a high-level API for loading models and
// generating text with minimal boilerplate.
package inference

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/generate"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/pkg/tokenizer"
	"github.com/zerfoo/zerfoo/registry"
)

// Model is a loaded model ready for generation.
type Model struct {
	generator *generate.Generator[float32]
	tokenizer tokenizer.Tokenizer
	engine    compute.Engine[float32]
	config    ModelMetadata
	info      *registry.ModelInfo
	closer    io.Closer // mmap reader, if loaded with mmap
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
	IntermediateSize  int                `json:"intermediate_size"`
	NumQueryHeads     int                `json:"num_attention_heads"`
	NumKeyValueHeads  int                `json:"num_key_value_heads"`
	RopeTheta         float64            `json:"rope_theta"`
	RopeScaling       *RopeScalingConfig `json:"rope_scaling,omitempty"`
	TieWordEmbeddings  bool               `json:"tie_word_embeddings"`
	SlidingWindow      int                `json:"sliding_window"`
	AttentionBias      bool               `json:"attention_bias"`
	PartialRotaryFactor float64           `json:"partial_rotary_factor"`

	// DeepSeek MLA and MoE fields.
	KVLoRADim        int `json:"kv_lora_rank"`
	QLoRADim         int `json:"q_lora_rank"`
	QKRopeHeadDim    int `json:"qk_rope_head_dim"`
	NumExperts       int `json:"num_experts"`
	NumExpertsPerToken int `json:"num_experts_per_tok"`
	NumSharedExperts int `json:"n_shared_experts"`
}

// modelAliases maps short model names to HuggingFace repo IDs.
var modelAliases = map[string]string{
	"gemma-3-1b-q4":  "google/gemma-3-1b-it-qat-q4_0-gguf",
	"gemma-3-2b-q4":  "google/gemma-3-2b-it-qat-q4_0-gguf",
	"llama-3-1b-q4":  "meta-llama/Llama-3.2-1B-Instruct-GGUF",
	"llama-3-8b-q4":  "meta-llama/Llama-3.1-8B-Instruct-GGUF",
	"mistral-7b-q4":  "mistralai/Mistral-7B-Instruct-v0.3-GGUF",
	"qwen-2.5-7b-q4": "Qwen/Qwen2.5-7B-Instruct-GGUF",
}

// ResolveAlias returns the HuggingFace repo ID for a short alias.
// If the name is not an alias, it is returned unchanged.
func ResolveAlias(name string) string {
	if id, ok := modelAliases[name]; ok {
		return id
	}
	return name
}

// RegisterAlias adds a custom short name -> HuggingFace repo ID mapping.
func RegisterAlias(shortName, repoID string) {
	modelAliases[shortName] = repoID
}

// Option configures model loading.
type Option func(*loadOptions)

type loadOptions struct {
	cacheDir  string
	device    string
	maxSeqLen int
	registry  registry.ModelRegistry
	backend   string // "" or "default" for standard engine, "tensorrt" for TRT
	precision string // "" or "fp32" for float32, "fp16" for half precision (TRT only)
	dtype     string // "" or "fp32" for float32, "fp16" for FP16 compute
	kvDtype   string // "" or "fp32" for float32, "fp16" for FP16 KV cache
	mmap      bool   // use mmap for model loading (unix only)
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

// WithMmap enables memory-mapped model loading. When true, the ZMF file
// is mapped into memory using syscall.Mmap instead of os.ReadFile, avoiding
// heap allocation for model weights. Only supported on unix platforms.
func WithMmap(enabled bool) Option {
	return func(o *loadOptions) {
		o.mmap = enabled
	}
}

// WithKVDtype sets the KV cache storage dtype. Supported: "fp32" (default), "fp16".
// FP16 halves KV cache bandwidth by storing keys/values in half precision.
func WithKVDtype(dtype string) Option {
	return func(o *loadOptions) {
		o.kvDtype = dtype
	}
}

// Load loads a model by ID, pulling it if not cached.
func Load(modelID string, opts ...Option) (*Model, error) {
	o := &loadOptions{
		device: "cpu",
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

	// Load config.json.
	configPath := filepath.Join(info.Path, "config.json")
	meta, err := loadMetadata(configPath)
	if err != nil {
		return nil, fmt.Errorf("load config: %w", err)
	}

	// Load tokenizer.
	tokPath := filepath.Join(info.Path, "tokenizer.json")
	tok, err := tokenizer.LoadFromJSON(tokPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	// Load ZMF model and build graph.
	zmfPath := filepath.Join(info.Path, "model.zmf")
	eng, err := createEngine(o.device)
	if err != nil {
		return nil, fmt.Errorf("create engine (%s): %w", o.device, err)
	}

	// Apply FP16 compute precision if requested.
	applyDType(eng, o.dtype)

	globalAttrs := map[string]interface{}{}
	if meta.RopeScaling != nil && meta.RopeScaling.Type == "yarn" {
		globalAttrs["rope_scaling_type"] = meta.RopeScaling.Type
		globalAttrs["rope_scaling_factor"] = meta.RopeScaling.Factor
		globalAttrs["rope_scaling_orig_max_len"] = meta.RopeScaling.OriginalMaxPositionEmbeddings
	}
	if meta.PartialRotaryFactor > 0 && meta.PartialRotaryFactor < 1.0 {
		globalAttrs["partial_rotary_factor"] = meta.PartialRotaryFactor
	}

	var buildOpts []model.BuildOption
	if len(globalAttrs) > 0 {
		buildOpts = append(buildOpts, model.WithGlobalAttributes(globalAttrs))
	}

	var mdl *model.Model[float32]
	var mmapCloser io.Closer
	if o.mmap {
		mdl, mmapCloser, err = loadZMFWithMmap(eng, zmfPath, buildOpts)
		if err != nil {
			return nil, fmt.Errorf("load model (mmap): %w", err)
		}
	} else {
		mdl, err = model.LoadModelFromZMF[float32](eng, numeric.Float32Ops{}, zmfPath, buildOpts...)
		if err != nil {
			return nil, fmt.Errorf("load model: %w", err)
		}
	}

	// Upload model weights to GPU if the engine supports it.
	if uploader, ok := eng.(compute.WeightUploader); ok {
		tensors := mdl.Graph.ConstantTensors()
		if mdl.Embedding != nil {
			for _, p := range mdl.Embedding.Parameters() {
				tensors = append(tensors, p.Value)
			}
		}
		// Also upload layer parameters (e.g. RMSNorm gain weights) that
		// are embedded inside compute nodes rather than registered as
		// top-level constant/parameter nodes in the graph.
		for _, p := range mdl.Graph.Parameters() {
			if p.Value != nil {
				tensors = append(tensors, p.Value)
			}
		}
		if err := uploader.UploadWeights(tensors); err != nil {
			return nil, fmt.Errorf("upload weights to GPU: %w", err)
		}
	}

	m := assembleModel(mdl.Graph, tok, eng, meta, info, o.maxSeqLen, o.kvDtype)
	m.closer = mmapCloser
	return m, nil
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

	return &Model{
		generator: gen,
		tokenizer: tok,
		engine:    eng,
		config:    *meta,
		info:      info,
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

func buildSamplingConfig(opts []GenerateOption) generate.SamplingConfig {
	sc := generate.DefaultSamplingConfig()
	for _, opt := range opts {
		opt(&sc)
	}
	return sc
}

// Generate produces text from a prompt.
func (m *Model) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (string, error) {
	sc := buildSamplingConfig(opts)
	return m.generator.Generate(ctx, prompt, sc)
}

// GenerateStream delivers tokens one at a time via a callback.
func (m *Model) GenerateStream(ctx context.Context, prompt string, handler generate.TokenStream, opts ...GenerateOption) error {
	sc := buildSamplingConfig(opts)
	return m.generator.GenerateStream(ctx, prompt, sc, handler)
}

// Message represents a chat message.
type Message struct {
	Role    string // "system", "user", or "assistant"
	Content string
}

// Response holds the result of a chat completion.
type Response struct {
	Content          string
	TokensUsed       int
	PromptTokens     int
	CompletionTokens int
}

// Chat formats messages using the model's chat template and generates a response.
func (m *Model) Chat(ctx context.Context, messages []Message, opts ...GenerateOption) (Response, error) {
	prompt := m.formatMessages(messages)
	sc := buildSamplingConfig(opts)
	result, err := m.generator.Generate(ctx, prompt, sc)
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

// Embed returns a float32 embedding vector for the given text.
// It runs the model forward and mean-pools the last layer's hidden states.
func (m *Model) Embed(ctx context.Context, text string) ([]float32, error) {
	ids, err := m.tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("text produced no tokens")
	}

	// Generate one token to get the last hidden state.
	// For embeddings, we use the output of a forward pass and mean-pool.
	result, err := m.generator.Generate(ctx, text, generate.SamplingConfig{
		Temperature:  0,
		MaxNewTokens: 1,
	})
	if err != nil {
		return nil, fmt.Errorf("forward: %w", err)
	}

	// Since we don't have direct access to hidden states, we return
	// a hash of the result as a placeholder. Real implementation would
	// need access to intermediate layer outputs.
	_ = result
	return nil, fmt.Errorf("embeddings not yet supported: model does not expose hidden states")
}

// Close releases resources held by the model. If the model was loaded on a
// GPU, this frees the CUDA engine's handles, pool, and stream. If loaded
// with mmap, this releases the memory mapping.
func (m *Model) Close() error {
	var firstErr error
	if c, ok := m.engine.(io.Closer); ok {
		firstErr = c.Close()
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
