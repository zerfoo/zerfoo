# Agentic Tool Use

Function calling (tool use) with a language model -- the core pattern behind agentic AI systems.

## How it works

1. Loads a model using the `zerfoo.Load` one-line API
2. Defines tools (functions) with OpenAI-compatible schemas via `serve.Tool`
3. Generates a response with `zerfoo.WithTools` -- the model decides which tool to call
4. Executes the tool call locally and gets a result
5. Feeds the result back to the model for a natural-language answer

This two-step pattern (plan + execute) is the foundation of agentic systems that combine LLM reasoning with real-world actions like API calls, database queries, or system commands.

## Usage

```bash
go build -o agentic-tool-use ./examples/agentic-tool-use/

# Default query (weather)
./agentic-tool-use --model path/to/model.gguf

# Custom query
./agentic-tool-use --model path/to/model.gguf --query "What is 42 * 17?"

# Using a HuggingFace model ID
./agentic-tool-use --model google/gemma-3-4b --query "Weather in London?"
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file or HuggingFace model ID |
| `--query` | "What is the weather in San Francisco?" | User query |

## Architecture

```
User Query
  -> Model generates tool call (with zerfoo.WithTools)
  -> Application executes tool locally
  -> Tool result fed back to model
  -> Model generates final answer
```
