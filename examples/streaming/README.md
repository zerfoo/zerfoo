# Streaming Chat Example

Demonstrates real-time token streaming using `Model.ChatStream`. Each token is printed to the terminal as it arrives, rather than waiting for the full response.

## Build

```bash
go build -o streaming ./examples/streaming/
```

## Usage

```bash
# With a local GGUF model file
./streaming --model ./models/gemma-3-1b-Q4_K_M.gguf

# With a HuggingFace model ID (downloads on first use)
./streaming --model google/gemma-3-1b
```

Type a message and press Enter. Tokens stream to the terminal in real time. Type `quit` to exit.
