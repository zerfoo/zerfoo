# Package serve

```
import "github.com/zerfoo/zerfoo/serve"
```

Package serve provides an OpenAI-compatible HTTP API server for model inference. The server exposes REST endpoints that follow the OpenAI API specification, enabling drop-in compatibility with existing OpenAI client libraries and tools.

For model loading, see the [inference](inference.md) package. For lower-level generation control, see the [generate](generate.md) package.

## Quick Start

```go
m, _ := inference.Load("gemma-3-1b-q4", inference.WithDevice("cuda"))
defer m.Close()

srv := serve.NewServer(m,
    serve.WithLogger(logger),
    serve.WithMetrics(collector),
)
log.Fatal(http.ListenAndServe(":8080", srv.Handler()))
```

---

## Server

### type Server

```go
type Server struct {
    // unexported fields
}
```

Wraps a loaded model and serves OpenAI-compatible HTTP endpoints.

### func NewServer

```go
func NewServer(m *inference.Model, opts ...ServerOption) *Server
```

Creates a Server for the given model.

```go
srv := serve.NewServer(m,
    serve.WithLogger(slog.Default()),
    serve.WithMetrics(collector),
    serve.WithDraftModel(draftModel),
)
```

### func (*Server) Handler

```go
func (s *Server) Handler() http.Handler
```

Returns the HTTP handler for this server. Use with `http.ListenAndServe` or any `net/http`-compatible router.

```go
mux := http.NewServeMux()
mux.Handle("/", srv.Handler())
http.ListenAndServe(":8080", mux)
```

### func (*Server) Close

```go
func (s *Server) Close(_ context.Context) error
```

Gracefully shuts down the server, draining the batch scheduler if one is attached. Implements `shutdown.Closer`.

---

## Server Options

### type ServerOption

```go
type ServerOption func(*Server)
```

Configures the server.

### func WithLogger

```go
func WithLogger(l log.Logger) ServerOption
```

Sets the logger for request logging.

### func WithMetrics

```go
func WithMetrics(c runtime.Collector) ServerOption
```

Sets the metrics collector for token rate and request tracking.

### func WithDraftModel

```go
func WithDraftModel(draft *inference.Model) ServerOption
```

Enables speculative decoding. When set, completion requests use the draft model to propose tokens and the target model to verify them.

```go
target, _ := inference.Load("llama-3-70b-q4", inference.WithDevice("cuda"))
draft, _ := inference.Load("llama-3-8b-q4", inference.WithDevice("cuda"))

srv := serve.NewServer(target, serve.WithDraftModel(draft))
```

### func WithBatchScheduler

```go
func WithBatchScheduler(bs *BatchScheduler) ServerOption
```

Attaches a batch scheduler for non-streaming requests. Incoming requests are grouped into batches for higher throughput.

### func WithTranscriber

```go
func WithTranscriber(t Transcriber) ServerOption
```

Sets the audio transcription backend for the `/v1/audio/transcriptions` endpoint.

---

## Endpoints

The server registers the following HTTP routes:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| POST | `/v1/completions` | Text completion (streaming and non-streaming) |
| POST | `/v1/embeddings` | Text embeddings |
| GET | `/v1/models` | List loaded models |
| GET | `/v1/models/{id}` | Get model info |
| DELETE | `/v1/models/{id}` | Unload a model |
| GET | `/openapi.yaml` | OpenAPI specification |
| GET | `/metrics` | Prometheus metrics |

---

## Chat Completions

### type ChatCompletionRequest

```go
type ChatCompletionRequest struct {
    Model          string          `json:"model"`
    Messages       []ChatMessage   `json:"messages"`
    Temperature    *float64        `json:"temperature,omitempty"`
    TopP           *float64        `json:"top_p,omitempty"`
    MaxTokens      *int            `json:"max_tokens,omitempty"`
    Stream         bool            `json:"stream"`
    Tools          []Tool          `json:"tools,omitempty"`
    ToolChoice     *ToolChoice     `json:"tool_choice,omitempty"`
    ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}
```

Represents the OpenAI chat completion request. Set `Stream: true` for SSE streaming.

### type ChatCompletionResponse

```go
type ChatCompletionResponse struct {
    ID      string                 `json:"id"`
    Object  string                 `json:"object"`
    Created int64                  `json:"created"`
    Model   string                 `json:"model"`
    Choices []ChatCompletionChoice `json:"choices"`
    Usage   UsageInfo              `json:"usage"`
}
```

### type ChatCompletionChoice

```go
type ChatCompletionChoice struct {
    Index        int         `json:"index"`
    Message      ChatMessage `json:"message"`
    FinishReason string      `json:"finish_reason"`
    ToolCalls    []ToolCall  `json:"tool_calls,omitempty"`
}
```

### type ChatMessage

```go
type ChatMessage struct {
    Role      string     `json:"role"`
    Content   string     `json:"content"`
    ImageURLs []ImageURL `json:"-"`
}
```

Content can be either a plain string or an array of content parts (for vision requests with `type:"text"` and `type:"image_url"`). Custom JSON unmarshaling handles both formats.

---

## Text Completions

### type CompletionRequest

```go
type CompletionRequest struct {
    Model       string   `json:"model"`
    Prompt      string   `json:"prompt"`
    Temperature *float64 `json:"temperature,omitempty"`
    MaxTokens   *int     `json:"max_tokens,omitempty"`
    Stream      bool     `json:"stream"`
}
```

### type CompletionResponse

```go
type CompletionResponse struct {
    ID      string             `json:"id"`
    Object  string             `json:"object"`
    Created int64              `json:"created"`
    Model   string             `json:"model"`
    Choices []CompletionChoice `json:"choices"`
    Usage   UsageInfo          `json:"usage"`
}
```

### type CompletionChoice

```go
type CompletionChoice struct {
    Index        int    `json:"index"`
    Text         string `json:"text"`
    FinishReason string `json:"finish_reason"`
}
```

---

## Embeddings

### type EmbeddingRequest

```go
type EmbeddingRequest struct {
    Model string      `json:"model"`
    Input interface{} `json:"input"` // string or []string
}
```

### type EmbeddingResponse

```go
type EmbeddingResponse struct {
    Object string            `json:"object"`
    Data   []EmbeddingObject `json:"data"`
    Model  string            `json:"model"`
    Usage  UsageInfo         `json:"usage"`
}
```

### type EmbeddingObject

```go
type EmbeddingObject struct {
    Object    string    `json:"object"`
    Embedding []float32 `json:"embedding"`
    Index     int       `json:"index"`
}
```

---

## SSE Streaming

When a chat or text completion request sets `"stream": true`, the server responds with Server-Sent Events (SSE). Each event contains a JSON chunk with incremental tokens. The stream terminates with a `"data: [DONE]"` sentinel.

Example client usage with `curl`:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-1b-q4",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

---

## Tool Calling

Chat completion requests may include OpenAI-compatible tool definitions. The server validates tool schemas, detects tool calls in model output, and returns structured `tool_calls` in the response.

### type Tool

```go
type Tool struct {
    Type     string       `json:"type"`
    Function ToolFunction `json:"function"`
}
```

### type ToolFunction

```go
type ToolFunction struct {
    Name        string          `json:"name"`
    Description string          `json:"description"`
    Parameters  json.RawMessage `json:"parameters,omitempty"`
}
```

### type ToolChoice

```go
type ToolChoice struct {
    Mode     string // "auto", "none", or "function"
    Function *ToolChoiceFunction
}
```

Represents the `tool_choice` field. Can be the string `"auto"`, `"none"`, or an object forcing a specific function. Custom JSON marshaling/unmarshaling handles both forms.

### type ToolChoiceFunction

```go
type ToolChoiceFunction struct {
    Name string `json:"name"`
}
```

### type ToolCall

```go
type ToolCall struct {
    ID       string           `json:"id"`
    Type     string           `json:"type"`
    Function ToolCallFunction `json:"function"`
}
```

### type ToolCallFunction

```go
type ToolCallFunction struct {
    Name      string `json:"name"`
    Arguments string `json:"arguments"`
}
```

### func DetectToolCall

```go
func DetectToolCall(text string, tools []Tool, choice ToolChoice) (*ToolCallResult, bool)
```

Examines generated text to determine if it is a tool call. Returns `(result, true)` if detected, `(nil, false)` otherwise. Detection heuristic:

1. If `tool_choice` is `"none"`: never detect.
2. Trim whitespace; if text starts with `{` and parses as valid JSON: it is a tool call.
3. If `tool_choice` forces a specific function: use that name.
4. Otherwise: look for `"name"` field in the JSON to match a tool.

### type ToolCallResult

```go
type ToolCallResult struct {
    ID           string
    FunctionName string
    Arguments    json.RawMessage
}
```

---

## Structured Output

### type ResponseFormat

```go
type ResponseFormat struct {
    Type       string            `json:"type"` // "text" | "json_object" | "json_schema"
    JSONSchema *JSONSchemaFormat `json:"json_schema,omitempty"`
}
```

Controls the output structure of a chat completion. When `Type` is `"json_schema"`, grammar-constrained decoding ensures model output conforms to the provided JSON Schema.

### type JSONSchemaFormat

```go
type JSONSchemaFormat struct {
    Name   string          `json:"name"`
    Strict bool            `json:"strict,omitempty"`
    Schema json.RawMessage `json:"schema"`
}
```

---

## Batch Scheduling

### type BatchScheduler

```go
type BatchScheduler struct {
    // unexported fields
}
```

Groups incoming non-streaming requests into batches for efficient GPU utilization.

### func NewBatchScheduler

```go
func NewBatchScheduler(config BatchConfig) *BatchScheduler
```

Creates a new batch scheduler.

```go
bs := serve.NewBatchScheduler(serve.BatchConfig{
    MaxBatchSize: 8,
    BatchTimeout: 10 * time.Millisecond,
})
bs.Start()
defer bs.Stop()

srv := serve.NewServer(m, serve.WithBatchScheduler(bs))
```

### func (*BatchScheduler) Start / Stop

```go
func (s *BatchScheduler) Start()
func (s *BatchScheduler) Stop()
```

Start begins the batch collection loop. Stop gracefully shuts down the scheduler.

### func (*BatchScheduler) Submit

```go
func (s *BatchScheduler) Submit(ctx context.Context, req BatchRequest) (BatchResult, error)
```

Adds a request to the next batch and waits for the result.

### type BatchConfig

```go
type BatchConfig struct {
    MaxBatchSize int
    BatchTimeout time.Duration
    Handler      BatchHandler
}
```

### type BatchHandler

```go
type BatchHandler func(ctx context.Context, reqs []BatchRequest) []BatchResult
```

Processes a batch of requests. The results slice must have the same length as requests.

### type BatchRequest

```go
type BatchRequest struct {
    Prompt string
    Phase  string // "prefill" or "decode"
}
```

### type BatchResult

```go
type BatchResult struct {
    Value string
    Err   error
}
```

---

## Metrics

### type ServerMetrics

```go
type ServerMetrics struct {
    // unexported fields
}
```

Records serving metrics using a `runtime.Collector`.

### func NewServerMetrics

```go
func NewServerMetrics(c runtime.Collector) *ServerMetrics
```

Creates a ServerMetrics backed by the given collector.

### func (*ServerMetrics) RecordRequest

```go
func (m *ServerMetrics) RecordRequest(tokens int, latency time.Duration)
```

Records a completed request's metrics.

The `GET /metrics` endpoint exposes Prometheus text exposition format metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `requests_total` | Counter | Total number of completed requests |
| `tokens_generated_total` | Counter | Total tokens generated |
| `tokens_per_second` | Gauge | Rolling token generation rate |
| `request_latency_ms` | Histogram | Request latency with configurable buckets |

---

## Vision Support

### type ContentPart

```go
type ContentPart struct {
    Type     string    `json:"type"`
    Text     string    `json:"text,omitempty"`
    ImageURL *ImageURL `json:"image_url,omitempty"`
}
```

A single element in a multi-part content array (for vision requests).

### type ImageURL

```go
type ImageURL struct {
    URL    string `json:"url"`
    Detail string `json:"detail,omitempty"`
}
```

---

## Audio Transcription

### type Transcriber

```go
type Transcriber interface {
    Transcribe(ctx context.Context, audio []byte, language string) (string, error)
}
```

Converts raw audio bytes into a text transcript. Set via `WithTranscriber`.

### type TranscriptionResponse

```go
type TranscriptionResponse struct {
    Text string `json:"text"`
}
```

---

## Models API

### type ModelObject

```go
type ModelObject struct {
    ID           string `json:"id"`
    Object       string `json:"object"`
    Created      int64  `json:"created"`
    OwnedBy      string `json:"owned_by"`
    Architecture string `json:"architecture,omitempty"`
}
```

### type ModelListResponse

```go
type ModelListResponse struct {
    Object string        `json:"object"`
    Data   []ModelObject `json:"data"`
}
```

### type ModelDeleteResponse

```go
type ModelDeleteResponse struct {
    ID      string `json:"id"`
    Object  string `json:"object"`
    Deleted bool   `json:"deleted"`
}
```

---

## Usage Info

### type UsageInfo

```go
type UsageInfo struct {
    PromptTokens     int `json:"prompt_tokens"`
    CompletionTokens int `json:"completion_tokens"`
    TotalTokens      int `json:"total_tokens"`
}
```

Reports token counts in completion responses.
