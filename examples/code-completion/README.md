# Code Completion

Generate code completions from a partial code snippet using a GGUF language model.

## How it works

1. Reads a code prefix from the `--code` flag or piped stdin
2. Loads a GGUF model via `inference.LoadFile`
3. Constructs a completion prompt and generates the continuation
4. Prints the original code followed by the generated completion

Works best with code-capable models (CodeLlama, DeepSeek Coder, Qwen 2.5 Coder) but any instruction-tuned model can produce reasonable completions.

## Usage

```bash
go build -o code-completion ./examples/code-completion/

# Pass code via flag
./code-completion --model path/to/model.gguf --code "func fibonacci(n int) int {"

# Pass code via stdin
echo "func add(a, b int) int {" | ./code-completion --model path/to/model.gguf

# Use GPU and lower temperature for more deterministic output
./code-completion --model path/to/model.gguf --device cuda --temperature 0.1 \
    --code "// quicksort sorts a slice of integers in place.
func quicksort(arr []int) {"
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file |
| `--device` | cpu | Compute device: "cpu", "cuda" |
| `--code` | (stdin) | Code prefix to complete |
| `--max-tokens` | 256 | Maximum tokens to generate |
| `--temperature` | 0.2 | Sampling temperature (lower = more deterministic) |
