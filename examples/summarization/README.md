# Summarization

Summarize text using a GGUF language model.

## How it works

1. Reads input text from `--text`, `--file`, or uses a built-in example
2. Loads a GGUF model via `inference.LoadFile`
3. Constructs a summarization prompt and generates a concise summary
4. Prints the summary to stdout

## Usage

```bash
go build -o summarization ./examples/summarization/

# Summarize inline text
./summarization --model path/to/model.gguf --text "Your long text here..."

# Summarize a file
./summarization --model path/to/model.gguf --file article.txt

# Use the built-in example
./summarization --model path/to/model.gguf

# With GPU acceleration
./summarization --model path/to/model.gguf --device cuda
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file |
| `--device` | cpu | Compute device: "cpu", "cuda" |
| `--text` | (example) | Inline text to summarize |
| `--file` | | Path to a text file to summarize |
| `--max-tokens` | 256 | Maximum tokens in the summary |
