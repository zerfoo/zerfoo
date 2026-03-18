# Translation

Translate text between languages using a GGUF language model.

## How it works

1. Loads a GGUF model via `inference.LoadFile`
2. Constructs a translation prompt with source and target languages
3. Generates the translation
4. Prints the original text and its translation

Works best with multilingual models such as Qwen 2.5, Gemma 3, or Llama 3.

## Usage

```bash
go build -o translation ./examples/translation/

# English to French (default)
./translation --model path/to/model.gguf --text "Hello, world!"

# French to English
./translation --model path/to/model.gguf --text "Bonjour le monde" --source French --target English

# English to Spanish with GPU
./translation --model path/to/model.gguf --device cuda --text "Good morning" --target Spanish
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file |
| `--device` | cpu | Compute device: "cpu", "cuda" |
| `--text` | "The quick brown fox..." | Text to translate |
| `--source` | English | Source language |
| `--target` | French | Target language |
| `--max-tokens` | 256 | Maximum tokens to generate |
