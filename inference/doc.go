// Package inference provides a high-level API for loading GGUF models and
// running text generation, chat, embedding, and speculative decoding with
// minimal boilerplate. (Stability: stable)
//
// # Loading Models
//
// There are two entry points for loading a model:
//
//   - [Load] resolves a model by name or HuggingFace repo ID, pulling it from
//     the registry if not already cached, and returns a ready-to-use [Model].
//   - [LoadFile] loads a model directly from a local GGUF file path.
//
// Both accept functional [Option] values to configure the compute device,
// cache directory, sequence length, and other parameters:
//
//	m, err := inference.Load("gemma-3-1b-q4",
//		inference.WithDevice("cuda"),
//		inference.WithMaxSeqLen(4096),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer m.Close()
//
//	text, err := m.Generate(ctx, "Explain gradient descent briefly.",
//		inference.WithMaxTokens(256),
//		inference.WithTemperature(0.7),
//	)
//
// # Model Methods
//
// A loaded [Model] exposes several generation methods:
//
//   - [Model.Generate] produces text from a prompt and returns the full result.
//   - [Model.GenerateStream] delivers tokens incrementally via a callback.
//   - [Model.GenerateBatch] processes multiple prompts concurrently.
//   - [Model.Chat] formats a slice of [Message] values using the model's chat
//     template and generates a [Response] with token usage statistics.
//   - [Model.Embed] returns an L2-normalized embedding vector for a text input
//     by mean-pooling the model's token embedding table.
//   - [Model.SpeculativeGenerate] runs speculative decoding with a smaller draft
//     model to accelerate generation from a larger target model.
//
// # Load Options
//
// The following [Option] functions configure model loading:
//
//   - [WithDevice] — compute device: "cpu", "cuda", "cuda:N", "rocm", "opencl"
//   - [WithCacheDir] — local directory for cached model files
//   - [WithMaxSeqLen] — override the model's default maximum sequence length
//   - [WithRegistry] — supply a custom model registry
//   - [WithBackend] — select "tensorrt" for TensorRT-optimized inference
//   - [WithPrecision] — set TensorRT compute precision ("fp16")
//   - [WithDType] — set GPU compute precision ("fp16", "fp8")
//   - [WithKVDtype] — set KV cache storage precision ("fp16")
//   - [WithMmap] — control memory-mapped model loading (default: enabled)
//
// # Generate Options
//
// The following [GenerateOption] functions configure sampling for generation methods:
//
//   - [WithTemperature] — sampling temperature (higher = more random)
//   - [WithTopK] — top-K sampling cutoff
//   - [WithTopP] — nucleus (top-P) sampling threshold
//   - [WithMaxTokens] — maximum number of tokens to generate
//   - [WithRepetitionPenalty] — penalize repeated tokens
//   - [WithStopStrings] — strings that terminate generation
//   - [WithGrammar] — constrained decoding via a grammar state machine
//
// # Model Aliases
//
// Short aliases such as "gemma-3-1b-q4" and "llama-3-8b-q4" map to full
// HuggingFace repository IDs. Use [ResolveAlias] to look up the mapping and
// [RegisterAlias] to add custom aliases.
//
// # Related Packages
//
// For lower-level control over text generation, KV caching, and sampling,
// see the [github.com/zerfoo/zerfoo/generate] package. For an
// OpenAI-compatible HTTP server built on top of this package, see
// [github.com/zerfoo/zerfoo/serve].
// Stability: stable
package inference
