# Package inference

```
import "github.com/zerfoo/zerfoo/inference"
```

Package inference provides a high-level API for loading GGUF models and running text generation, chat, embedding, and speculative decoding with minimal boilerplate.

For lower-level control over text generation, KV caching, and sampling, see the [generate](generate.md) package. For an OpenAI-compatible HTTP server built on top of this package, see the [serve](serve.md) package.

## Quick Start

```go
m, err := inference.Load("gemma-3-1b-q4",
    inference.WithDevice("cuda"),
    inference.WithMaxSeqLen(4096),
)
if err != nil {
    log.Fatal(err)
}
defer m.Close()

text, err := m.Generate(ctx, "Explain gradient descent briefly.",
    inference.WithMaxTokens(256),
    inference.WithTemperature(0.7),
)
```

---

## Model Loading

### func Load

```go
func Load(modelID string, opts ...Option) (*Model, error)
```

Load resolves a model by name or HuggingFace repo ID, pulling it from the registry if not already cached, and returns a ready-to-use Model.

Short aliases such as `"gemma-3-1b-q4"` and `"llama-3-8b-q4"` map to full HuggingFace repository IDs. Use `ResolveAlias` to look up the mapping.

```go
m, err := inference.Load("llama-3-8b-q4",
    inference.WithDevice("cuda:0"),
    inference.WithCacheDir("/models"),
)
if err != nil {
    log.Fatal(err)
}
defer m.Close()
```

### func LoadFile

```go
func LoadFile(path string, opts ...Option) (*Model, error)
```

LoadFile loads a model directly from a local GGUF file path and returns a ready-to-use Model.

```go
m, err := inference.LoadFile("/data/models/gemma-3-1b-q4_k_m.gguf",
    inference.WithDevice("cuda"),
)
if err != nil {
    log.Fatal(err)
}
defer m.Close()
```

### func LoadGGUF

```go
func LoadGGUF(path string) (*GGUFModel, error)
```

LoadGGUF loads a GGUF model file and returns its configuration and tensors as an intermediate representation. This is useful for inspecting model metadata or building custom computation graphs. Tensor names are mapped from GGUF convention (`blk.N.attn_q.weight`) to Zerfoo canonical names (`model.layers.N.self_attn.q_proj.weight`).

```go
gguf, err := inference.LoadGGUF("/data/models/llama-3-8b.gguf")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Architecture: %s\n", gguf.Config.Architecture)
fmt.Printf("Tensors: %d\n", len(gguf.Tensors))
```

---

## Load Options

### type Option

```go
type Option func(*loadOptions)
```

Option configures model loading. Pass these to `Load` or `LoadFile`.

### func WithDevice

```go
func WithDevice(device string) Option
```

Sets the compute device. Supported values: `"cpu"`, `"cuda"`, `"cuda:N"` (specific GPU), `"rocm"`, `"opencl"`.

```go
m, _ := inference.Load("gemma-3-1b-q4", inference.WithDevice("cuda:0"))
```

### func WithCacheDir

```go
func WithCacheDir(dir string) Option
```

Sets the local directory for cached model files.

### func WithMaxSeqLen

```go
func WithMaxSeqLen(n int) Option
```

Overrides the model's default maximum sequence length.

### func WithRegistry

```go
func WithRegistry(r registry.ModelRegistry) Option
```

Supplies a custom model registry for model resolution.

### func WithBackend

```go
func WithBackend(backend string) Option
```

Selects the inference backend. Supported values: `""` or `"default"` for the standard Engine path, `"tensorrt"` for TensorRT-optimized inference. TensorRT requires the cuda build tag and a CUDA device.

### func WithPrecision

```go
func WithPrecision(precision string) Option
```

Sets the compute precision for the TensorRT backend. Supported values: `""` or `"fp32"` for full precision, `"fp16"` for half precision. Has no effect when the backend is not `"tensorrt"`.

### func WithDType

```go
func WithDType(dtype string) Option
```

Sets the compute precision for the GPU engine. Supported values: `""` or `"fp32"` for full precision, `"fp16"` for FP16 compute. FP16 mode converts activations F32->FP16 before GPU kernels and back after. Has no effect on CPU engines.

### func WithKVDtype

```go
func WithKVDtype(dtype string) Option
```

Sets the KV cache storage dtype. Supported: `"fp32"` (default), `"fp16"`. FP16 halves KV cache bandwidth by storing keys/values in half precision.

### func WithMmap

```go
func WithMmap(enabled bool) Option
```

Enables memory-mapped model loading. When true, the file is mapped into memory using `syscall.Mmap` instead of `os.ReadFile`, avoiding heap allocation for model weights. Only supported on unix platforms.

---

## Model

### type Model

```go
type Model struct {
    // unexported fields
}
```

Model is a loaded model ready for generation. Created by `Load` or `LoadFile`.

### func (*Model) Generate

```go
func (m *Model) Generate(ctx context.Context, prompt string, opts ...GenerateOption) (string, error)
```

Produces text from a prompt. Sessions are pooled to reuse GPU memory addresses, enabling CUDA graph replay across calls. Concurrent `Generate` calls get separate sessions from the pool.

```go
text, err := m.Generate(ctx, "What is backpropagation?",
    inference.WithMaxTokens(512),
    inference.WithTemperature(0.7),
    inference.WithTopP(0.9),
)
```

### func (*Model) GenerateStream

```go
func (m *Model) GenerateStream(ctx context.Context, prompt string, handler generate.TokenStream, opts ...GenerateOption) error
```

Delivers tokens one at a time via a `TokenStream` callback. Sessions are pooled to preserve GPU memory addresses for CUDA graph replay.

```go
err := m.GenerateStream(ctx, "Tell me a story.",
    generate.TokenStreamFunc(func(token string, done bool) error {
        if !done {
            fmt.Print(token)
        }
        return nil
    }),
    inference.WithMaxTokens(256),
)
```

### func (*Model) GenerateBatch

```go
func (m *Model) GenerateBatch(ctx context.Context, prompts []string, opts ...GenerateOption) ([]string, error)
```

Processes multiple prompts concurrently and returns the generated text for each prompt. Results are returned in the same order as the input prompts.

```go
prompts := []string{
    "Explain neural networks.",
    "What is gradient descent?",
    "Define overfitting.",
}
results, err := m.GenerateBatch(ctx, prompts,
    inference.WithMaxTokens(128),
)
```

### func (*Model) Chat

```go
func (m *Model) Chat(ctx context.Context, messages []Message, opts ...GenerateOption) (Response, error)
```

Formats a slice of Message values using the model's chat template and generates a Response with token usage statistics.

```go
resp, err := m.Chat(ctx, []inference.Message{
    {Role: "system", Content: "You are a helpful assistant."},
    {Role: "user", Content: "What is machine learning?"},
},
    inference.WithMaxTokens(256),
)
fmt.Printf("Response: %s\n", resp.Content)
fmt.Printf("Tokens used: %d\n", resp.TokensUsed)
```

### func (*Model) Embed

```go
func (m *Model) Embed(text string) ([]float32, error)
```

Returns an L2-normalized embedding vector for the given text by looking up token embeddings from the model's embedding table and mean-pooling them.

```go
vec, err := m.Embed("machine learning")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Embedding dimension: %d\n", len(vec))
```

### func (*Model) SpeculativeGenerate

```go
func (m *Model) SpeculativeGenerate(
    ctx context.Context,
    draft *Model,
    prompt string,
    draftLen int,
    opts ...GenerateOption,
) (string, error)
```

Runs speculative decoding using this model as the target and the draft model for token proposal. `draftLen` controls how many tokens are proposed per verification step (typically 2-8).

```go
target, _ := inference.Load("llama-3-70b-q4", inference.WithDevice("cuda"))
draft, _ := inference.Load("llama-3-8b-q4", inference.WithDevice("cuda"))
defer target.Close()
defer draft.Close()

text, err := target.SpeculativeGenerate(ctx, draft,
    "Explain quantum computing.",
    4, // propose 4 tokens per step
    inference.WithMaxTokens(256),
)
```

### func (*Model) Close

```go
func (m *Model) Close() error
```

Releases resources held by the model. If the model was loaded on a GPU, this frees the CUDA engine's handles, pool, and stream. If loaded with mmap, this releases the memory mapping.

### func (*Model) Config

```go
func (m *Model) Config() ModelMetadata
```

Returns the model metadata.

### func (*Model) Generator

```go
func (m *Model) Generator() *generate.Generator[float32]
```

Returns the underlying generator for lower-level access.

### func (*Model) Tokenizer

```go
func (m *Model) Tokenizer() tokenizer.Tokenizer
```

Returns the model's tokenizer for token counting.

### func (*Model) Info

```go
func (m *Model) Info() *registry.ModelInfo
```

Returns the registry info for this model.

### func (*Model) EmbeddingWeights

```go
func (m *Model) EmbeddingWeights() ([]float32, int)
```

Returns the flattened token embedding table and the hidden dimension. Returns `nil, 0` if embeddings are not available.

### func (*Model) SetEmbeddingWeights

```go
func (m *Model) SetEmbeddingWeights(weights []float32, hiddenSize int)
```

Sets the token embedding table for `Embed()`. `weights` is a flattened `[vocabSize, hiddenSize]` matrix.

---

## Generate Options

### type GenerateOption

```go
type GenerateOption func(*generate.SamplingConfig)
```

GenerateOption configures a generation call. Pass these to `Generate`, `GenerateStream`, `GenerateBatch`, `Chat`, or `SpeculativeGenerate`.

### func WithTemperature

```go
func WithTemperature(t float64) GenerateOption
```

Sets the sampling temperature. Higher values produce more random output; 0 uses greedy (argmax) decoding.

### func WithTopK

```go
func WithTopK(k int) GenerateOption
```

Sets the top-K sampling cutoff. Only the top K most probable tokens are considered. 0 disables top-K filtering.

### func WithTopP

```go
func WithTopP(p float64) GenerateOption
```

Sets the nucleus (top-P) sampling threshold. Tokens are selected from the smallest set whose cumulative probability exceeds P. 1.0 disables top-P filtering.

### func WithMaxTokens

```go
func WithMaxTokens(n int) GenerateOption
```

Sets the maximum number of tokens to generate.

### func WithRepetitionPenalty

```go
func WithRepetitionPenalty(p float64) GenerateOption
```

Sets the repetition penalty factor. Values > 1.0 penalize repeated tokens.

### func WithStopStrings

```go
func WithStopStrings(ss ...string) GenerateOption
```

Sets strings that terminate generation when encountered in the output.

### func WithGrammar

```go
func WithGrammar(g *grammar.Grammar) GenerateOption
```

Sets a grammar state machine for constrained decoding. When set, a token mask is applied at each sampling step to restrict output to tokens valid according to the grammar.

---

## Response Types

### type Message

```go
type Message struct {
    Role    string   // "system", "user", or "assistant"
    Content string
    Images  [][]byte // optional raw image data for vision models
}
```

Message represents a chat message for the `Chat` method.

### type Response

```go
type Response struct {
    Content          string
    TokensUsed       int
    PromptTokens     int
    CompletionTokens int
}
```

Response holds the result of a chat completion.

---

## Model Metadata

### type ModelMetadata

```go
type ModelMetadata struct {
    Architecture          string
    VocabSize             int
    HiddenSize            int
    NumLayers             int
    MaxPositionEmbeddings int
    EOSTokenID            int
    BOSTokenID            int
    ChatTemplate          string

    // Extended fields for multi-architecture support.
    IntermediateSize    int
    NumQueryHeads       int
    NumKeyValueHeads    int
    RopeTheta           float64
    RopeScaling         *RopeScalingConfig
    TieWordEmbeddings   bool
    SlidingWindow       int
    AttentionBias       bool
    PartialRotaryFactor float64

    // DeepSeek MLA and MoE fields.
    KVLoRADim          int
    QLoRADim           int
    QKRopeHeadDim      int
    NumExperts         int
    NumExpertsPerToken int
    NumSharedExperts   int
}
```

ModelMetadata holds model configuration loaded from config.json or GGUF metadata.

### type RopeScalingConfig

```go
type RopeScalingConfig struct {
    Type                          string
    Factor                        float64
    OriginalMaxPositionEmbeddings int
}
```

RopeScalingConfig holds configuration for RoPE scaling methods (e.g., YaRN).

---

## GGUF Model Loading

### type GGUFModel

```go
type GGUFModel struct {
    Config  *gguf.ModelConfig
    Tensors map[string]*tensor.TensorNumeric[float32]
    File    *gguf.File
}
```

GGUFModel holds a loaded GGUF model's configuration and tensors. This is an intermediate representation; full inference requires an architecture-specific graph builder.

### func (*GGUFModel) ToModelMetadata

```go
func (m *GGUFModel) ToModelMetadata() *ModelMetadata
```

Converts a GGUF model config to `ModelMetadata`.

---

## Architecture Registry

The architecture registry maps GGUF `general.architecture` values to graph builder functions.

### type ArchBuilder

```go
type ArchBuilder func(
    tensors map[string]*tensor.TensorNumeric[float32],
    cfg *gguf.ModelConfig,
    engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)
```

ArchBuilder builds a computation graph for a model architecture from pre-loaded GGUF tensors. Returns the graph and the embedding table tensor.

### func RegisterArchitecture

```go
func RegisterArchitecture(name string, builder ArchBuilder)
```

Registers an architecture builder under the given name. Names correspond to GGUF `general.architecture` values (e.g. `"llama"`, `"gemma"`). Panics if name is empty or a builder is already registered.

```go
inference.RegisterArchitecture("custom", func(
    tensors map[string]*tensor.TensorNumeric[float32],
    cfg *gguf.ModelConfig,
    engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error) {
    // Build custom architecture graph...
    return g, embedTensor, nil
})
```

### func GetArchitecture

```go
func GetArchitecture(name string) (ArchBuilder, bool)
```

Returns the builder registered for the given architecture name. Returns `nil, false` if no builder is registered.

### func ListArchitectures

```go
func ListArchitectures() []string
```

Returns a sorted list of all registered architecture names.

### func BuildArchGraph

```go
func BuildArchGraph(
    arch string,
    tensors map[string]*tensor.TensorNumeric[float32],
    cfg *gguf.ModelConfig,
    engine compute.Engine[float32],
) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)
```

Dispatches to the appropriate architecture-specific graph builder. Exported for benchmark and integration tests that construct synthetic weight maps without loading from GGUF files.

---

## Model Aliases

### func RegisterAlias

```go
func RegisterAlias(shortName, repoID string)
```

Adds a custom short name to HuggingFace repo ID mapping.

```go
inference.RegisterAlias("my-model", "myorg/my-model-7b-q4")
m, _ := inference.Load("my-model")
```

### func ResolveAlias

```go
func ResolveAlias(name string) string
```

Returns the HuggingFace repo ID for a short alias. If the name is not an alias, it is returned unchanged.

---

## Architecture-Specific Builders

### func BuildJamba

```go
func BuildJamba(jc JambaConfig, tensors map[string]*tensor.TensorNumeric[float32], engine compute.Engine[float32]) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)
```

Constructs a computation graph for the Jamba hybrid architecture (mixed attention + SSM layers).

### type JambaConfig

```go
type JambaConfig struct {
    NumLayers            int
    HiddenSize           int
    IntermediateSize     int
    AttnHeads            int
    KVHeads              int
    SSMHeads             int
    AttentionLayerOffset int
    RMSEps               float32
    VocabSize            int
    MaxSeqLen            int
    RopeTheta            float64
    DConv                int
}
```

### func JambaConfigFromGGUF

```go
func JambaConfigFromGGUF(cfg *gguf.ModelConfig) JambaConfig
```

Extracts Jamba configuration from GGUF ModelConfig.

### func BuildMamba3

```go
func BuildMamba3(mc MambaConfig, tensors map[string]*tensor.TensorNumeric[float32], engine compute.Engine[float32]) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)
```

Constructs a computation graph for Mamba-3 from a weight map.

### type MambaConfig

```go
type MambaConfig struct {
    NumLayers  int
    DModel     int
    DState     int
    DConv      int
    DInner     int
    VocabSize  int
    EOSTokenID int
    RMSNormEps float32
}
```

### func MambaConfigFromGGUF

```go
func MambaConfigFromGGUF(cfg *gguf.ModelConfig) MambaConfig
```

Extracts Mamba configuration from GGUF ModelConfig.

### func MambaConfigFromMetadata

```go
func MambaConfigFromMetadata(meta map[string]interface{}) MambaConfig
```

Extracts Mamba configuration from a raw metadata map.

### func BuildRWKV

```go
func BuildRWKV(rc RWKVConfig, tensors map[string]*tensor.TensorNumeric[float32], engine compute.Engine[float32]) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)
```

Constructs a computation graph for the RWKV-6/7 architecture.

### type RWKVConfig

```go
type RWKVConfig struct {
    NumLayers    int
    HiddenSize   int
    VocabSize    int
    HeadSize     int // default 64
    NumHeads     int // HiddenSize / HeadSize
    LayerNormEps float32
}
```

### func RWKVConfigFromGGUF

```go
func RWKVConfigFromGGUF(cfg *gguf.ModelConfig) RWKVConfig
```

Extracts RWKV configuration from GGUF ModelConfig.

### func BuildWhisperEncoder

```go
func BuildWhisperEncoder(wc WhisperConfig, tensors map[string]*tensor.TensorNumeric[float32], engine compute.Engine[float32]) (*graph.Graph[float32], *tensor.TensorNumeric[float32], error)
```

Constructs a computation graph for the Whisper encoder.

### type WhisperConfig

```go
type WhisperConfig struct {
    NumMels    int
    HiddenDim  int
    NumHeads   int
    NumLayers  int
    KernelSize int
}
```

### func WhisperConfigFromGGUF

```go
func WhisperConfigFromGGUF(cfg *gguf.ModelConfig) WhisperConfig
```

Extracts Whisper configuration from GGUF ModelConfig. NumMels defaults to 80, KernelSize defaults to 3.

---

## Config Registry

### type ArchConfigRegistry

```go
type ArchConfigRegistry struct {
    // unexported fields
}
```

Maps model_type strings to config parsers.

### func DefaultArchConfigRegistry

```go
func DefaultArchConfigRegistry() *ArchConfigRegistry
```

Returns a registry with all built-in parsers registered.

### func (*ArchConfigRegistry) Register

```go
func (r *ArchConfigRegistry) Register(modelType string, parser ConfigParser)
```

Adds a parser for the given model type.

### func (*ArchConfigRegistry) Parse

```go
func (r *ArchConfigRegistry) Parse(raw map[string]interface{}) (*ModelMetadata, error)
```

Dispatches to the registered parser for the `model_type` in raw, or falls back to generic field extraction for unknown types.

### type ConfigParser

```go
type ConfigParser func(raw map[string]interface{}) (*ModelMetadata, error)
```

Parses a raw JSON map (from config.json) into ModelMetadata.

---

## TensorRT Integration

### func ConvertGraphToTRT

```go
func ConvertGraphToTRT(
    g *graph.Graph[float32],
    workspaceBytes int,
    fp16 bool,
    dynamicShapes *DynamicShapeConfig,
) (*trtConversionResult, error)
```

Walks a graph in topological order and maps each node to a TensorRT layer. Returns serialized engine bytes or an `UnsupportedOpError` if the graph contains operations that cannot be converted.

### type DynamicShapeConfig

```go
type DynamicShapeConfig struct {
    InputShapes []ShapeRange
}
```

Specifies per-input shape ranges for TensorRT optimization profiles.

### type ShapeRange

```go
type ShapeRange struct {
    Min []int32
    Opt []int32
    Max []int32
}
```

Defines min/opt/max dimensions for a single input tensor.

### func TRTCacheKey

```go
func TRTCacheKey(modelID, precision string) (string, error)
```

Builds a deterministic cache key from model ID, precision, and GPU architecture.

### func SaveTRTEngine / LoadTRTEngine

```go
func SaveTRTEngine(key string, data []byte) error
func LoadTRTEngine(key string) ([]byte, error)
```

Write/read serialized TensorRT engines to/from the cache directory. `LoadTRTEngine` returns `nil, nil` on cache miss.

### type TRTInferenceEngine

```go
type TRTInferenceEngine struct {
    // unexported fields
}
```

Holds a TensorRT engine and execution context for inference.

### func (*TRTInferenceEngine) Forward

```go
func (e *TRTInferenceEngine) Forward(inputs []*tensor.TensorNumeric[float32], outputSize int) (*tensor.TensorNumeric[float32], error)
```

Runs inference through TensorRT with the given input tensors. Input tensors must already be on GPU.

### func (*TRTInferenceEngine) Close

```go
func (e *TRTInferenceEngine) Close() error
```

Releases all TensorRT resources.

### type UnsupportedOpError

```go
type UnsupportedOpError struct {
    Ops []string
}
```

Lists the operations that cannot be converted to TensorRT.

---

## Testing Utilities

### func NewTestModel

```go
func NewTestModel(
    gen *generate.Generator[float32],
    tok tokenizer.Tokenizer,
    eng compute.Engine[float32],
    meta ModelMetadata,
    info *registry.ModelInfo,
) *Model
```

Constructs a Model from pre-built components. Intended for use in external test packages that need a Model without going through the full Load pipeline.

---

## Interfaces

### type ConstantValueGetter

```go
type ConstantValueGetter interface {
    GetValue() *tensor.TensorNumeric[float32]
}
```

Interface for nodes that hold constant tensor data.

### type DTypeSetter

```go
type DTypeSetter interface {
    SetDType(compute.DType)
}
```

Implemented by engines that support setting compute precision.
