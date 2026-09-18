# Hacker News: Show HN Post

## Title

Show HN: Zerfoo -- Run LLMs in Pure Go, 1.25x Ollama on Gemma 3 1B

## Suggested Top-Level Comment

Zerfoo is an ML inference framework written entirely in Go. It loads GGUF
models and runs transformer inference -- Llama 3, Gemma 3, Mistral, Qwen 2,
Phi 3/4, and DeepSeek V3 -- zero CGo in the default build, no Python, no
external runtime.
GPU acceleration (CUDA, ROCm, OpenCL) is loaded dynamically at runtime via
purego/dlopen, so `go build` works everywhere with no C compiler required.

On Gemma 3 1B Q4_K_M, Zerfoo decodes at 235 tok/s on a DGX Spark --
1.25x Ollama (188 tok/s) on the same hardware with the same model file,
at parity by 3B.
CUDA graph capture covers 99.5% of the decode path. Full benchmarking
methodology with reproduction steps is linked below.

You can use it as a CLI or embed it as a library:

```bash
go install github.com/zerfoo/zerfoo/cmd/zerfoo@latest
zerfoo pull gemma-3-1b-q4
zerfoo run gemma-3-1b-q4 "The quick brown fox"
```

Or import it directly in your Go code -- three lines to load a model and
generate text.

Repo: https://github.com/zerfoo/zerfoo

Benchmarks and methodology: https://github.com/zerfoo/zerfoo/blob/main/docs/benchmarks.md

Happy to answer questions about the architecture, GPU binding approach, or
performance numbers.

## Posting Guidelines

- **Timing**: US morning (9-11am ET), Tuesday through Thursday
- **Format**: Post as a link to the GitHub repo; add the comment above immediately after posting
- **Do not**: Editorialize in the title; let the numbers speak
- **Do not**: Ask for upvotes or coordinate voting
- **Be prepared to answer**: How does purego GPU binding work? What are the limitations vs llama.cpp? Why not just wrap llama.cpp?
