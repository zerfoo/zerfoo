# Zerfoo vs Ollama Benchmark Comparison

**Date:** 2026-03-25
**Commit:** 294aa43
**Hardware:** DGX Spark GB10 (sm_121, 128GB LPDDR5x)
**Methodology:** 3 runs per model, median reported, 128 decode tokens, greedy sampling
**Ollama version:** 0.17.7

## Results

| Model | Architecture | Size | Zerfoo (tok/s) | Ollama (tok/s) | Ratio | Winner |
|-------|-------------|------|----------------|----------------|-------|--------|
| gemma3-1b | gemma3 | 1B | 236.38 | 204.37 | 1.16x | Zerfoo |
| deepseek-r1-1.5b | deepseek2 | 1.5B | 192.83 | 184.75 | 1.04x | Zerfoo |
| llama3.2-3b | llama | 3B | 96.06 | 97.66 | 0.98x | Ollama |
| phi3-mini | phi3 | 3.8B | FAIL | 90.80 | N/A | N/A |
| mistral-7b | mistral | 7B | 11.61 | 46.77 | 0.25x | Ollama |
| llama3.1-8b | llama | 8B | FAIL | 42.85 | N/A | N/A |
| gemma3-4b | gemma3 | 4B | SKIP | SKIP | N/A | SKIP |
| qwen2.5-7b | qwen2 | 7B | SKIP | SKIP | N/A | SKIP |
| mixtral-8x7b | mixtral | 47B | SKIP | SKIP | N/A | SKIP |
| command-r-35b | command-r | 35B | SKIP | SKIP | N/A | SKIP |
| falcon-7b | falcon | 7B | SKIP | N/A | N/A | SKIP |
| mamba-2.8b | mamba | 2.8B | SKIP | N/A | N/A | SKIP |
| rwkv-7b | rwkv | 7B | SKIP | N/A | N/A | SKIP |

## Key Findings

1. **Gemma 3 1B advantage confirmed**: 1.16x faster than Ollama (target was >= 1.15x).
2. **DeepSeek R1 1.5B**: Zerfoo slightly faster (1.04x).
3. **Llama 3.2 3B**: Roughly even (0.98x, within margin of error).
4. **Mistral 7B: major regression**: Zerfoo 4x slower than Ollama (0.25x). Needs investigation.
5. **GGUF load failures**: Phi 3 mini and Llama 3.1 8B failed to load — likely HuggingFace GGUF format incompatibility.
6. **7 models skipped**: GGUF files not available or not downloaded in time.

## Regressions Requiring Issues

- **Mistral 7B**: 11.61 tok/s vs Ollama 46.77 tok/s (0.25x). Critical regression.
- **Phi 3 mini**: GGUF load failure. Needs GGUF parser fix.
- **Llama 3.1 8B**: GGUF load failure. Needs GGUF parser fix.

## Notes

- Ratio > 1.0 means Zerfoo is faster.
- N/A in Ollama column means the model is not supported by Ollama.
- SKIP means the model file was not available or not pulled.
- FAIL means the benchmark run encountered an error.
