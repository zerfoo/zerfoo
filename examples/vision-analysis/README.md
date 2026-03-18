# Vision Analysis

Analyze images using a vision-capable GGUF language model.

## How it works

1. Reads an image file (JPEG or PNG) from disk
2. Loads a vision-capable GGUF model via `inference.LoadFile`
3. Sends the image as part of an `inference.Message` with the `Images` field
4. Generates a text description or analysis using `model.Chat`

This uses the same multimodal API that powers the OpenAI-compatible `/v1/chat/completions` endpoint for vision requests.

## Prerequisites

Requires a vision-capable model (e.g., LLaVA, Gemma 3 with vision encoder). Text-only models will ignore the image data.

## Usage

```bash
go build -o vision-analysis ./examples/vision-analysis/

# Describe an image
./vision-analysis --model path/to/vision-model.gguf --image photo.jpg

# Ask a specific question about an image
./vision-analysis --model path/to/vision-model.gguf --image chart.png \
    --prompt "What trend does this chart show?"

# With GPU
./vision-analysis --model path/to/vision-model.gguf --device cuda --image photo.jpg
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to a vision-capable GGUF model file |
| `--image` | (required) | Path to an image file (JPEG or PNG) |
| `--device` | cpu | Compute device: "cpu", "cuda" |
| `--prompt` | "Describe this image in detail." | Question or instruction about the image |
| `--max-tokens` | 512 | Maximum tokens to generate |
