# Classification

Text classification with grammar-constrained JSON output using a GGUF language model.

## How it works

1. Defines a JSON schema with allowed labels (`positive`, `negative`, `neutral`) and a confidence score
2. Converts the schema to a grammar state machine via `grammar.Convert`
3. Loads a GGUF model via `inference.LoadFile`
4. Prompts the model to classify the input text
5. Uses `inference.WithGrammar` to constrain output to valid JSON matching the schema

The grammar constraint guarantees the model output is always valid JSON with the correct structure, eliminating the need for fragile string parsing or retry loops.

## Usage

```bash
go build -o classification ./examples/classification/

# Classify sentiment
./classification --model path/to/model.gguf --text "I love this product!"
# Output: {"label":"positive","confidence":0.95}

./classification --model path/to/model.gguf --text "The package arrived damaged."
# Output: {"label":"negative","confidence":0.87}

# With GPU
./classification --model path/to/model.gguf --device cuda --text "It was okay I guess."
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to GGUF model file |
| `--device` | cpu | Compute device: "cpu", "cuda" |
| `--text` | "I absolutely love..." | Text to classify |
