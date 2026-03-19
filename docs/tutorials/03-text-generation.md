# Part 3: Text Generation Deep Dive

This tutorial explores how Zerfoo generates text: sampling strategies, streaming responses token by token, KV cache behavior, and batch generation for throughput.

## How Autoregressive Generation Works

Transformer models generate text one token at a time. At each step, the model computes a probability distribution over the vocabulary (logits), a token is selected, and it becomes part of the input for the next step. The `generate` package implements this loop with configurable sampling, stopping conditions, and KV caching.

When you call `model.Generate`, this is what happens internally:

1. The prompt is tokenized using the BPE tokenizer embedded in the GGUF file.
2. A `SamplingConfig` is built from the options you pass.
3. The prompt tokens run through the computation graph in a single forward pass (prefill).
4. The KV cache stores key/value activations so they are not recomputed on subsequent steps.
5. One token is generated per step (decode) until a stop condition is met.

## Sampling Strategies

Sampling controls how the next token is chosen from the logit distribution. Zerfoo supports several strategies that can be combined.

### Temperature

Temperature scales the logits before converting them to probabilities. Lower values make the distribution sharper (more deterministic), higher values make it flatter (more creative).

```go
// Deterministic output (greedy decoding).
result, _ := model.Generate(ctx, prompt,
	inference.WithTemperature(0),
)

// Creative output.
result, _ := model.Generate(ctx, prompt,
	inference.WithTemperature(1.2),
)
```

A temperature of 0 selects the highest-probability token every time (greedy). A temperature of 1.0 samples proportionally to the probabilities. Values above 1.0 increase randomness.

### Top-K Sampling

Top-K restricts the candidate set to the K most probable tokens before sampling. This prevents the model from selecting very unlikely tokens.

```go
result, _ := model.Generate(ctx, prompt,
	inference.WithTemperature(0.8),
	inference.WithTopK(40),
)
```

When `TopK` is 0 (the default), all tokens are candidates.

### Top-P (Nucleus) Sampling

Top-P keeps the smallest set of tokens whose cumulative probability exceeds P. This adapts the candidate set size dynamically -- confident predictions use fewer candidates, uncertain predictions use more.

```go
result, _ := model.Generate(ctx, prompt,
	inference.WithTemperature(0.8),
	inference.WithTopP(0.9),
)
```

When `TopP` is 1.0 (the default), no filtering is applied. Top-K and Top-P can be combined: Top-K filters first, then Top-P filters the remainder.

### Repetition Penalty

Repetition penalty reduces the probability of tokens that have already appeared in the output. A value of 1.0 disables the penalty; values above 1.0 penalize repetition.

```go
result, _ := model.Generate(ctx, prompt,
	inference.WithRepetitionPenalty(1.1),
)
```

### Recommended Defaults

For most use cases, a good starting point is:

```go
result, _ := model.Generate(ctx, prompt,
	inference.WithTemperature(0.7),
	inference.WithTopP(0.9),
	inference.WithMaxTokens(256),
)
```

## Streaming Responses

For interactive applications, you often want to display tokens as they are generated rather than waiting for the full response. The `GenerateStream` method accepts a callback that receives each token:

```go
err := model.GenerateStream(ctx, "Tell me a story.",
	func(token string) bool {
		fmt.Print(token)
		// Return true to continue, false to stop early.
		return true
	},
	inference.WithTemperature(0.8),
	inference.WithMaxTokens(512),
)
```

The callback function implements the `generate.TokenStream` type. It receives each decoded token string and returns a boolean: `true` to continue generation, `false` to stop immediately.

## Stop Conditions

Generation stops when any of these conditions is met:

1. The end-of-sequence (EOS) token is generated.
2. `MaxNewTokens` is reached.
3. A stop string is found in the output.
4. The streaming callback returns `false`.
5. The context is cancelled.

You can set custom stop strings:

```go
result, _ := model.Generate(ctx, prompt,
	inference.WithMaxTokens(512),
	inference.WithStopStrings("\n\n", "END"),
)
```

## Constrained Decoding with Grammars

Zerfoo supports grammar-constrained generation using the `grammar` package. At each sampling step, a token mask restricts output to tokens valid according to the grammar:

```go
import "github.com/zerfoo/zerfoo/generate/grammar"

g, err := grammar.Parse(`root ::= "{" ws "\"name\"" ws ":" ws string "}" ...`)
result, _ := model.Generate(ctx, "Generate a JSON object with a name field.",
	inference.WithGrammar(g),
	inference.WithMaxTokens(128),
)
```

This is useful for generating structured output like JSON, SQL, or code that must conform to a specific syntax.

## KV Cache and Performance

The KV (Key-Value) cache is the single most important optimization for autoregressive generation. Without it, every decode step would reprocess the entire sequence from scratch.

### How It Works

During the prefill phase, the model computes attention keys and values for all prompt tokens and stores them in the KV cache. During decode, only the new token is processed -- its keys and values are appended to the cache, and attention is computed against all cached entries.

### Memory Considerations

KV cache memory grows linearly with sequence length and model size. For a 7B model with 32 layers and 4096 context length, the KV cache can use 1-2 GB of memory in FP32. You can halve this with FP16 KV storage:

```go
model, err := inference.LoadFile("model.gguf",
	inference.WithDevice("cuda"),
	inference.WithKVDtype("fp16"),
)
```

### Paged KV Cache

For serving multiple concurrent requests, Zerfoo supports paged KV caching at the generator level. Paged KV allocates memory in blocks from a shared pool rather than pre-allocating the full sequence length per request. This significantly improves memory utilization when serving requests of varying lengths.

### CUDA Graph Capture

On CUDA devices, Zerfoo captures the decode step as a CUDA graph after the first execution. Subsequent decode steps replay the captured graph, eliminating CPU-side kernel launch overhead. This is why sessions are pooled in `inference.Model` -- reusing sessions preserves GPU memory addresses required for graph replay.

## Batch Generation

When you have multiple prompts to process, batch generation is more efficient than sequential calls:

```go
prompts := []string{
	"Summarize quantum computing in one sentence.",
	"What is the capital of Japan?",
	"Explain REST APIs briefly.",
}

results, err := model.GenerateBatch(ctx, prompts,
	inference.WithTemperature(0.5),
	inference.WithMaxTokens(64),
)
for i, r := range results {
	fmt.Printf("Prompt %d: %s\n", i+1, r)
}
```

`GenerateBatch` processes prompts concurrently using the session pool, taking advantage of GPU parallelism when available.

## What is Next

- [Part 4: Running the OpenAI-Compatible API Server](04-api-server.md) -- serve your model over HTTP.
- [Part 5: Tabular and Time-Series ML](05-tabular-timeseries.md) -- structured data prediction and forecasting.
