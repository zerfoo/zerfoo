# Structured JSON Output

This example demonstrates grammar-guided decoding, which constrains the model's
output to valid JSON matching a predefined schema.

## How It Works

Grammar-guided decoding works at the token level during generation:

1. A JSON Schema is converted into a context-free grammar state machine
   (`grammar.Grammar`).
2. At each decoding step, the grammar reports which bytes are valid next
   characters (`ValidBytes`).
3. The decoder masks logits for tokens that would produce invalid bytes,
   ensuring every generated token advances the grammar legally.
4. The result is guaranteed to be well-formed JSON conforming to the schema —
   no post-processing or retry logic needed.

This is particularly useful for extracting structured data from LLMs, building
tool-calling pipelines, or any scenario where the output must conform to a
strict format.

## Usage

```bash
# Build
go build -o json-output ./examples/json-output/

# Run with a local GGUF model
./json-output --model path/to/model.gguf

# Run with a HuggingFace model (downloads automatically)
./json-output --model google/gemma-3-1b

# Custom prompt
./json-output --model path/to/model.gguf --prompt "Generate a person named Bob who is 25"
```

## Schema

The example uses a simple object schema with two required fields:

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "number" }
  },
  "required": ["name", "age"]
}
```

The output will always be valid JSON matching this schema, for example:

```json
{"name": "Alice", "age": 30}
```

## Supported Schema Features

The grammar converter supports these JSON Schema types:

- `object` with `properties` and `required`
- `array` with `items`
- `string` (with optional `minLength` / `maxLength`)
- `number`, `integer`, `boolean`, `null`
- `enum` and `const` constraints
