# LangChain-Go Chatbot with Zerfoo

This example demonstrates using Zerfoo as an LLM provider through the
`integrations/langchain` adapter. The adapter exposes the same method
signatures as LangChain-Go's `schema.LLM` interface, so it works as a
drop-in replacement pointing at Zerfoo's OpenAI-compatible API.

## Prerequisites

- A compiled `zerfoo` binary (or `go run ./cmd/zerfoo`)
- A GGUF model file (e.g. Llama 3 1B)

## Setup

1. Start the Zerfoo server:

```bash
zerfoo serve --model path/to/model.gguf --port 8080
```

2. Run the chatbot:

```bash
go run ./examples/langchain-chatbot/ \
  --server http://localhost:8080 \
  --model llama3 \
  --temperature 0.7 \
  --max-tokens 512
```

3. Type a message and press Enter. Type `quit` to exit.

## How It Works

The adapter (`integrations/langchain.Adapter`) wraps Zerfoo's
`/v1/chat/completions` endpoint. It implements `Call`, `Generate`, and `Type`
methods matching LangChain-Go's LLM interface without importing the
`langchaingo` module, keeping the dependency footprint minimal.

```go
llm := langchain.NewAdapter(
    "http://localhost:8080",
    "llama3",
    langchain.WithTemperature(0.7),
    langchain.WithMaxTokens(512),
)

reply, err := llm.Call(ctx, "What is the capital of France?")
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://localhost:8080` | Zerfoo server URL |
| `--model` | `llama3` | Model name |
| `--temperature` | `0.7` | Sampling temperature (0-1) |
| `--max-tokens` | `512` | Maximum tokens to generate |
