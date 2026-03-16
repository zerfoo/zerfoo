# Structured JSON Output

This example demonstrates grammar-guided decoding, which constrains the model's
output to valid JSON matching a predefined schema.

## How It Works

Grammar-guided decoding works at the token level during generation:

1. A JSON Schema is converted into a context-free grammar state machine
   (`grammar.Grammar`) via `grammar.Convert`.
2. At each decoding step, the grammar reports which bytes are valid next
   characters (`ValidBytes`).
3. The decoder masks logits for tokens that would produce invalid bytes,
   ensuring every generated token advances the grammar legally.
4. The result is guaranteed to be well-formed JSON conforming to the schema --
   no post-processing or retry logic needed.

This is particularly useful for extracting structured data from LLMs, building
tool-calling pipelines, or any scenario where the output must conform to a
strict format.

## Usage

### Library -- high-level API (recommended)

The simplest approach uses `zerfoo.WithSchema`, which handles grammar conversion
internally:

```go
schema := grammar.JSONSchema{
    Type: "object",
    Properties: map[string]*grammar.JSONSchema{
        "name": {Type: "string"},
        "age":  {Type: "number"},
    },
    Required: []string{"name", "age"},
}

result, err := model.Generate(ctx, prompt, zerfoo.WithSchema(schema))
```

### Library -- low-level API

For custom inference pipelines, convert the schema to a grammar explicitly with
`grammar.Convert` and pass it via `inference.WithGrammar`:

```go
g, err := grammar.Convert(&schema)
// ...
text, err := model.Generate(ctx, prompt, inference.WithGrammar(g))
```

This gives you direct control over the grammar lifecycle and the inference
package's `Model` type.

### Running the example

```bash
# Build
go build -o json-output ./examples/json-output/

# High-level API (default)
./json-output --model path/to/model.gguf

# Low-level API
./json-output --model path/to/model.gguf --low-level

# Custom prompt
./json-output --model path/to/model.gguf --prompt "Generate a person named Bob who is 25"
```

### CLI -- `--json-schema` flag

The `zerfoo run` command supports structured output directly from the command
line via the `--json-schema` flag:

```bash
zerfoo run path/to/model.gguf \
  --json-schema '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"number"}},"required":["name","age"]}' \
  --prompt "Generate a person named Alice who is 30"
```

This is equivalent to using `zerfoo.WithSchema` in the library API.

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
