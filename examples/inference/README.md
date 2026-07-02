# Inference Example

Load a GGUF model and generate text from a prompt.

## Prerequisites

- Go 1.25+
- A GGUF model file (e.g., Gemma 3 1B or Llama 3.2 1B)

### Downloading a test model

Download a quantized Gemma 3 1B model from HuggingFace:

```bash
# Install huggingface-cli if you don't have it
pip install huggingface-hub

# Download the Q4_0 quantized model (~700 MB)
huggingface-cli download google/gemma-3-1b-it-qat-q4_0-gguf \
  --local-dir ./models
```

## Build

```bash
go build -o inference-example ./examples/inference/
```

## Run

```bash
./inference-example path/to/model.gguf "Explain what a tensor is in one sentence."
```

### Streaming output

```bash
./inference-example -stream path/to/model.gguf "Write a haiku about Go."
```

### GPU inference

```bash
./inference-example -device cuda path/to/model.gguf "Hello, world!"
```

### All flags

```
Usage: inference-example [flags] <model.gguf> <prompt>

Flags:
  -device string       compute device: "cpu", "cuda", "cuda:0", "rocm" (default "cpu")
  -max-tokens int      maximum tokens to generate (default 256)
  -stream              stream tokens as they are generated
  -temperature float   sampling temperature (0 = greedy) (default 0.7)
  -top-k int           top-K sampling (0 = disabled)
  -top-p float         top-P nucleus sampling (1.0 = disabled) (default 1)
```

## Expected output

```
$ ./inference-example models/gemma-3-1b-it-qat-q4_0.gguf "What is 2+2?"
Loaded gemma model (26 layers, vocab 262144)
2+2 equals 4.
```

The model metadata (architecture, layers, vocab size) is printed to stderr.
The generated text is printed to stdout.
