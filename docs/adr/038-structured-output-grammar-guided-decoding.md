# ADR 038: Structured Output via Grammar-Guided Decoding

## Status
Accepted

## Date
2026-03-15

## Context
Go developers embedding Zerfoo need to extract structured data from LLM responses
-- JSON objects, typed records, enum values -- without post-processing regex hacks
or retry loops. OpenAI, vLLM, and llama.cpp all ship constrained decoding. Zerfoo
must offer parity to be a credible alternative.

The generation pipeline currently samples the full vocabulary at each step.
Grammar-guided decoding constrains the token selection to a subset that keeps the
output valid against a schema at every step, producing guaranteed-valid JSON
without post-hoc parsing.

## Decision
Implement grammar-guided decoding in the generation pipeline using a token mask
approach:

1. Accept a JSON Schema as input alongside the prompt (via `response_format` in
   the OpenAI-compatible API, and as a `GenerateOptions` field in the library API).

2. Convert JSON Schema to a context-free grammar (CFG) at request time. The CFG
   encodes the structural constraints (object braces, required keys, value types,
   enum literals, array brackets).

3. At each decode step, compute a token mask from the current CFG state -- a
   bitset over the vocabulary indicating which tokens are valid continuations.
   Apply the mask before sampling (set logits of invalid tokens to -inf).

4. Advance the CFG state after each sampled token.

5. The grammar engine runs on CPU (it is string/state manipulation, not
   tensor math). It produces a `[]bool` mask that the GPU sampling kernel
   consumes.

### API Surface

Library:
```go
schema := zerfoo.JSONSchema(`{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}`)
result, _ := model.Generate(ctx, prompt, zerfoo.WithSchema(schema))
// result.Text is guaranteed-valid JSON matching the schema
```

OpenAI API:
```json
{
  "model": "gemma-3-4b",
  "messages": [{"role": "user", "content": "Extract name and age from: John is 30"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "person",
      "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}
    }
  }
}
```

### Implementation Approach

- Grammar representation: recursive descent parser state machine over the JSON
  Schema CFG. Each state maps to a set of valid next-byte classes.
- Token masking: precompute a byte-trie from the tokenizer vocabulary. At each
  step, intersect the CFG's valid-byte-set with the trie to produce the token
  mask. This is O(vocab_size) per step but runs on CPU in parallel with GPU
  forward pass.
- Supported JSON Schema subset: object, array, string, number, integer, boolean,
  null, enum, const, required, minLength/maxLength, minimum/maximum. Nested
  objects and arrays. Not supporting: $ref, oneOf, anyOf, allOf, pattern
  (deferred to future work).

## Consequences

**Positive:**
- Guaranteed-valid JSON output from any model, eliminating retry loops.
- OpenAI API compatibility for `response_format.type = "json_schema"`.
- Library API gets first-class structured output support.
- No model fine-tuning or special tokens required -- works with any GGUF model.

**Negative:**
- Adds ~500-1000 lines to the generation pipeline (grammar engine + token masking).
- Token mask computation adds per-step CPU overhead (expected <1ms per step for
  typical schemas, negligible vs GPU forward pass).
- Complex schemas with deeply nested structures may produce slower generation due
  to constrained vocabulary reducing effective sampling diversity.
- JSON Schema subset limitations mean some schemas require simplification.
