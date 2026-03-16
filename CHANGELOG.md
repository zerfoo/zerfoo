# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-03-16

### Added

- `zerfoo.Load(pathOrID)` one-line model loader supporting local GGUF paths and HuggingFace model IDs (e.g. `"google/gemma-3-4b"`)
- `Model.Chat(prompt)` single-turn text generation
- `Model.Generate(ctx, prompt, opts...)` generation with options (max tokens, temperature, top-p)
- `Model.Embed(texts)` text embedding extraction with L2 normalization
- `Model.ChatStream(ctx, prompt)` streaming token generation via channel
- `Model.Close()` resource cleanup
- `WithSchema(schema)` grammar-guided JSON output constrained to a JSON Schema
- `WithTools(tools...)` / `WithToolChoice(choice)` tool/function calling via chat completions API
- `zerfoo pull` CLI command to download GGUF models from HuggingFace with resume, SHA256 verification, and progress display
- `zerfoo list` CLI command to list cached models
- `zerfoo rm` CLI command to remove a cached model
- `response_format: json_schema` support in OpenAI-compatible chat completions API
- GGUF graph builders for Mistral (sliding window attention), Qwen 2 (attention bias, RoPE theta=1M), Phi 3/4 (partial rotary), and DeepSeek V3/V2-Lite (MLA + MoE) bringing total supported architectures to six
- Example applications: `examples/chat/` (chatbot), `examples/rag/` (retrieval-augmented generation), `examples/json-output/` (structured output)

### Fixed

- FP16 inference: GQA tensor storage mismatch causing incorrect outputs
- FP8 inference: same GQA root cause in E4M3FN path
- CUDA graph capture: eliminated device-to-host transfer in GQA decode that prevented graph closure
- Concurrent generation race condition in `Generator` serialized with mutex
- Q5_K_M and Q6_K weights: removed lossy re-quantization to Q4_0; now dequantize accurately to float32

### Changed

- `zerfoo.Load` now accepts HuggingFace model IDs in addition to local file paths
- All exported symbols in the top-level `zerfoo` package now have stability markers (`Stable.` / `Experimental.`)
