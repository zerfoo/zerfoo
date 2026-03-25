# Multi-Model Benchmark: Zerfoo vs Ollama

## Context

### Problem Statement

Zerfoo claims 245 tok/s on Gemma 3 1B Q4_K_M, 20% faster than Ollama (204 tok/s).
This claim was established on a single model and a single quantization. To make
credible performance claims, the benchmark must cover all 13 text-generation
architectures that Zerfoo supports for inference, across the same hardware, using
the same methodology, and comparing head-to-head against Ollama on every model
Ollama also supports.

### Objectives

- Benchmark all 13 text-generation architectures Zerfoo supports on DGX Spark (GB10).
- Run the same benchmarks with Ollama using identical GGUF files where possible.
- Confirm the Gemma 3 1B 20% advantage still holds on current HEAD.
- Produce a public comparison table for the website and README.
- Identify architectures where Zerfoo underperforms Ollama and file issues.

### Non-Goals

- Benchmarking non-generative models (BERT, Whisper are excluded -- they do not
  produce tok/s metrics comparable to LLM decode).
- Benchmarking on hardware other than DGX Spark (future work).
- Optimizing underperforming architectures (separate plan).
- Benchmarking training throughput.

### Constraints and Assumptions

- Hardware: DGX Spark at ssh ndungu@192.168.86.250 (GB10, sm_121, 128GB LPDDR5x).
- All benchmarks use CUDA (device=cuda). CPU-only benchmarks are out of scope.
- GGUF is the model format for both Zerfoo and Ollama.
- Ollama must be installed on the DGX Spark and models pulled before benchmarking.
- Models must fit in 128GB memory. Large models (70B+) may require quantization.
- Existing bench_tps tool at cmd/bench_tps/main.go is the primary Zerfoo benchmark.
- Existing parity test at tests/parity/throughput_parity_test.go covers Gemma only.
- 3 runs per model, 32-token warmup, median reported (existing methodology).
- Greedy sampling (temperature=0) for reproducibility.

### Success Metrics

| Metric | Target |
|--------|--------|
| Models benchmarked | 13 text-generation architectures |
| Ollama comparison | Head-to-head on all Ollama-supported models |
| Gemma 3 1B advantage | >= 1.15x (confirm 20% claim still holds) |
| Results published | Updated benchmarks page on website |
| Regressions filed | GitHub issues for any model where Zerfoo < Ollama |

---

## Discovery Summary

**Work type:** Engineering (benchmarking, scripting) + Content (results publication)

**Models to benchmark (13 text-generation architectures):**

| # | Architecture | Model Variant | Size | Ollama Name | Ollama Support |
|---|-------------|---------------|------|-------------|----------------|
| 1 | Gemma 3 | gemma-3-1b Q4_K_M | 1B | gemma3:1b | Yes |
| 2 | Gemma 3 | gemma-3-4b Q4_K_M | 4B | gemma3:4b | Yes |
| 3 | Llama 3 | llama-3.2-3b Q4_K_M | 3B | llama3.2:3b | Yes |
| 4 | Llama 3 | llama-3.1-8b Q4_K_M | 8B | llama3.1:8b | Yes |
| 5 | Mistral | mistral-7b Q4_K_M | 7B | mistral:7b | Yes |
| 6 | Mixtral | mixtral-8x7b Q4_K_M | 47B | mixtral:8x7b | Yes |
| 7 | Qwen 2 | qwen2.5-7b Q4_K_M | 7B | qwen2.5:7b | Yes |
| 8 | Phi 3 | phi-3-mini Q4_K_M | 3.8B | phi3:mini | Yes |
| 9 | DeepSeek V3 | deepseek-r1:1.5b Q4_K_M | 1.5B | deepseek-r1:1.5b | Yes |
| 10 | Command R | command-r Q4_K_M | 35B | command-r:35b | Yes |
| 11 | Falcon | falcon-7b Q4_K_M | 7B | N/A | No |
| 12 | Mamba | mamba-2.8b Q4_K_M | 2.8B | N/A | No |
| 13 | RWKV | rwkv-7b Q4_K_M | 7B | N/A | No |

**Notes:**
- LLaVA and Qwen-VL are vision-language models benchmarked separately (image+text).
- Whisper is audio transcription, not text generation.
- BERT is encoder-only, no decode tok/s metric.
- Falcon, Mamba, RWKV have no Ollama equivalents -- Zerfoo-only benchmarks.
- Mixtral and Command R are large; may need to verify they fit on GB10 128GB.

**Existing infrastructure:**
- bench_tps CLI: ready, supports -model, -tokens, -device, -dtype flags.
- Parity test: covers Gemma 3 only. Needs expansion.
- CI workflow: benchmark.yml runs weekly on DGX Spark self-hosted runner.
- Devlog: detailed regression history for Gemma 3 baseline.

---

## Scope and Deliverables

### In Scope

- Download/acquire GGUF files for all 13 models on DGX Spark.
- Install and configure Ollama on DGX Spark with all 10 supported models pulled.
- Run bench_tps for each model (Zerfoo side).
- Run equivalent Ollama benchmark for each model (ollama run --verbose).
- Write a benchmark automation script that runs both runtimes sequentially.
- Record results in a structured format (JSON + Markdown table).
- Update the website benchmarks page with the full comparison.
- Update README.md benchmark claims if any numbers change.
- File GitHub issues for any architecture where Zerfoo < Ollama.

### Out of Scope

- Fixing performance regressions (separate plan per issue).
- Benchmarking on non-DGX hardware.
- Benchmarking prefill (prompt processing) separately from decode.
- Benchmarking batch inference or concurrent requests.
- Benchmarking with different quantization formats (Q8_0, FP16, etc.).

### Deliverables

| ID | Description | Acceptance Criterion |
|----|-------------|---------------------|
| D1 | Benchmark automation script | Single command runs all 13 models on both runtimes |
| D2 | Results JSON file | Machine-readable results with model, runtime, tok/s, metadata |
| D3 | Comparison table | Markdown table: model, Zerfoo tok/s, Ollama tok/s, ratio |
| D4 | Website update | benchmarks page updated with full comparison table |
| D5 | README update | Performance claims updated if numbers changed |
| D6 | Regression issues | GitHub issues filed for any Zerfoo < Ollama results |

---

## Checkable Work Breakdown

### E1: Environment Setup [DGX Spark]

- [ ] T1.1 Verify Ollama is installed on DGX Spark  Owner: TBD  Est: 15m  verifies: [infrastructure]
  - ssh ndungu@192.168.86.250
  - Check: `ollama --version`
  - If not installed: `curl -fsSL https://ollama.com/install.sh | sh`
  - Acceptance: ollama binary available and ollama serve running

- [ ] T1.2 Pull all Ollama models  Owner: TBD  Est: 1h  verifies: [infrastructure]
  - Pull each: gemma3:1b, gemma3:4b, llama3.2:3b, llama3.1:8b, mistral:7b,
    mixtral:8x7b, qwen2.5:7b, phi3:mini, deepseek-r1:1.5b, command-r:35b
  - Acceptance: `ollama list` shows all 10 models

- [ ] T1.3 Acquire GGUF files for all 13 Zerfoo models  Owner: TBD  Est: 1h  verifies: [infrastructure]
  - Download Q4_K_M variants from HuggingFace for each model
  - Store in /data/models/ on DGX Spark
  - For Falcon, Mamba, RWKV: convert from HuggingFace using zonnx if GGUF not available
  - Acceptance: all 13 GGUF files present and loadable by Zerfoo

- [ ] T1.4 Sync Zerfoo main branch on DGX Spark  Owner: TBD  Est: 15m  verifies: [infrastructure]
  - `git pull origin main` on DGX Spark
  - `go build ./...` passes
  - `go build ./cmd/bench_tps/` produces working binary
  - Acceptance: bench_tps binary runs on DGX Spark with CUDA

### E2: Benchmark Automation Script

- [x] T2.1 Write benchmark runner script  Owner: Claude  Est: 2h  verifies: [infrastructure]  Done: 2026-03-25 (89c9b9e)
  - Create scripts/bench-compare-ollama.sh
  - For each model:
    a) Run bench_tps with: -model <path> -tokens 128 -device cuda -warmup 32
    b) Run ollama: `ollama run <model> --verbose "Explain quantum computing" 2>&1`
       Parse "eval rate: XXX.XX tokens/s" from stderr
    c) Record: model name, architecture, size, zerfoo_tps, ollama_tps, ratio, timestamp
  - Output: results/benchmark-YYYY-MM-DD.json
  - Output: results/benchmark-YYYY-MM-DD.md (formatted table)
  - Run 3 iterations per model, report median
  - Skip Ollama for models it does not support (Falcon, Mamba, RWKV)
  - Acceptance: script runs end-to-end, produces JSON and Markdown outputs

- [ ] T2.2 Test benchmark script with Gemma 3 1B  Owner: TBD  Est: 30m  verifies: [infrastructure]
  - Depends on: T1.1, T1.2, T1.3, T1.4, T2.1
  - Run the script for Gemma 3 1B only as a validation
  - Confirm Zerfoo tok/s is in the 240+ range
  - Confirm Ollama tok/s is captured correctly
  - Confirm JSON and Markdown output are well-formed
  - Acceptance: Gemma 3 1B results match expected range (240+ Zerfoo, ~200 Ollama)

### E3: Run Full Benchmark Suite [DGX Spark]

- [ ] T3.1 Benchmark small models (1B-4B)  Owner: TBD  Est: 1h  verifies: [infrastructure]
  - Depends on: T2.2
  - Models: Gemma 3 1B, Gemma 3 4B, Llama 3.2 3B, Phi 3 mini, DeepSeek R1 1.5B
  - Run on DGX Spark with benchmark script
  - Acceptance: all 5 models produce valid tok/s results for both runtimes

- [ ] T3.2 Benchmark medium models (7B-8B)  Owner: TBD  Est: 1h  verifies: [infrastructure]
  - Depends on: T2.2
  - Models: Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, Falcon 7B, RWKV 7B
  - Run on DGX Spark with benchmark script
  - Acceptance: all 5 models produce valid tok/s results

- [ ] T3.3 Benchmark large models (35B+)  Owner: TBD  Est: 1h  verifies: [infrastructure]
  - Depends on: T2.2
  - Models: Mixtral 8x7B, Command R 35B
  - These may be slow; allow longer timeout
  - If a model does not fit in 128GB, skip and document
  - Acceptance: results or documented skip reason for each model

- [ ] T3.4 Benchmark alternative architectures (Mamba, RWKV)  Owner: TBD  Est: 30m  verifies: [infrastructure]
  - Depends on: T2.2
  - Models: Mamba 2.8B (Zerfoo only, no Ollama comparison)
  - Acceptance: Mamba results recorded, Ollama column marked N/A

### E4: Analyze Results and Publish

- [ ] T4.1 Compile results into comparison table  Owner: TBD  Est: 30m  delivers: [benchmark comparison table]
  - Depends on: T3.1, T3.2, T3.3, T3.4
  - Merge all JSON results into a single table
  - Calculate: Zerfoo/Ollama ratio for each model
  - Flag: any model where ratio < 1.0 (Zerfoo slower)
  - Highlight: Gemma 3 1B ratio (confirm >= 1.15x)
  - Format: Markdown table sorted by model size
  - Acceptance: complete table with all 13 models

- [ ] T4.2 Update website benchmarks page  Owner: TBD  Est: 30m  delivers: [updated benchmarks page]
  - Depends on: T4.1
  - Update /Users/dndungu/Code/zerfoo/zerfoo.github.io/content/docs/reference/benchmarks.md
  - Add full Zerfoo vs Ollama comparison table
  - Add methodology section (hardware, settings, date)
  - Commit and push to zerfoo.github.io
  - Acceptance: benchmarks page shows full comparison at zerfoo.feza.ai/docs/reference/benchmarks/

- [ ] T4.3 Update README.md benchmark claims  Owner: TBD  Est: 15m  delivers: [updated README]
  - Depends on: T4.1
  - If Gemma 3 1B numbers changed, update the "244 tok/s" claim
  - If the 20% advantage changed, update that claim
  - Acceptance: README numbers match actual benchmark results

- [ ] T4.4 File GitHub issues for regressions  Owner: TBD  Est: 30m  delivers: [regression issues]
  - Depends on: T4.1
  - For each model where Zerfoo < Ollama, create a GitHub issue with:
    - Title: "perf: [arch] Zerfoo slower than Ollama ([ratio]x)"
    - Body: benchmark numbers, hardware, reproduction steps
  - If no regressions: skip this task
  - Acceptance: issues filed or documented as unnecessary

- [ ] T4.5 Record results in devlog  Owner: TBD  Est: 15m  delivers: [devlog entry]
  - Depends on: T4.1
  - Append benchmark results to docs/devlog.md
  - Format: date, model, tok/s for each runtime, commit hash
  - Acceptance: devlog entry with all results and commit hash

---

## Parallel Work

| Track | Tasks | Description |
|-------|-------|-------------|
| A: Environment | T1.1, T1.2, T1.3, T1.4 | DGX Spark setup, model acquisition |
| B: Script | T2.1 | Write benchmark automation |
| C: Execution | T3.1, T3.2, T3.3, T3.4 | Run benchmarks (sequential on DGX) |
| D: Publication | T4.1-T4.5 | Analyze and publish results |

**Note:** Tracks C tasks (T3.1-T3.4) MUST run sequentially on DGX Spark to avoid
GPU contention. They cannot be parallelized. Tracks A and B can run in parallel.

### Waves

#### Wave 1: Setup (3 agents)

- [ ] T1.1 Verify Ollama on DGX
- [ ] T1.2 Pull Ollama models (depends on T1.1, but can start immediately after)
- [x] T2.1 Write benchmark runner script (independent of DGX setup)

Note: T1.3 and T1.4 also run in this wave but depend on DGX SSH access.
In practice, one agent handles T1.1 + T1.2 + T1.3 + T1.4 on DGX sequentially,
while another writes the script locally.

#### Wave 2: Validate (1 agent)

- [ ] T2.2 Test script with Gemma 3 1B on DGX

#### Wave 3: Execute (1 agent -- sequential GPU access)

- [ ] T3.1 Small models (1B-4B)
- [ ] T3.2 Medium models (7B-8B)
- [ ] T3.3 Large models (35B+)
- [ ] T3.4 Alternative architectures

Note: All T3.x tasks run on the same DGX GPU. One agent runs them sequentially.

#### Wave 4: Publish (3 agents)

- [ ] T4.1 Compile comparison table
- [ ] T4.2 Update website benchmarks page (after T4.1)
- [ ] T4.3 Update README (after T4.1)
- [ ] T4.4 File regression issues (after T4.1)
- [ ] T4.5 Record in devlog (after T4.1)

Note: T4.2-T4.5 can run in parallel after T4.1 produces the table.

---

## Timeline and Milestones

| ID | Milestone | Exit Criteria | Depends On |
|----|-----------|---------------|------------|
| M1 | Environment ready | Ollama + 13 GGUF files on DGX, script written | Wave 1 |
| M2 | Gemma validation | Gemma 3 1B results match expected range | Wave 2 |
| M3 | Full results | All 13 models benchmarked | Wave 3 |
| M4 | Published | Website, README, devlog updated | Wave 4 |

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|----|------|--------|------------|------------|
| R1 | Model does not fit in 128GB (Mixtral, Command R) | Medium | Medium | Skip model, document, benchmark smaller variant |
| R2 | GGUF file not available for some architecture | Medium | Low | Convert from HuggingFace using zonnx |
| R3 | Ollama version has different performance characteristics | Low | Low | Record Ollama version, pin it |
| R4 | Zerfoo slower than Ollama on some models | High | Medium | File issue, investigate root cause separately |
| R5 | DGX Spark unavailable during benchmark window | High | Low | Schedule around known maintenance |
| R6 | Gemma 20% claim no longer holds after recent changes | High | Low | If <15% advantage, update marketing claims |

---

## Operating Procedure

### Definition of Done

- All 13 models benchmarked with 3 runs each, median reported.
- Results recorded in JSON format with commit hash and timestamp.
- Website benchmarks page updated and deployed.
- README claims verified or updated.
- Regression issues filed if any.
- Devlog entry added with full results.

### Benchmark Methodology

- Prompt: "Explain the theory of relativity in simple terms."
- Tokens: 128 generated tokens (decode phase).
- Warmup: 32 tokens discarded before measurement.
- Sampling: greedy (temperature=0) for reproducibility.
- Runs: 3 per model, median reported.
- Metric: tokens per second during decode (eval rate).
- Hardware: DGX Spark GB10, CUDA, 128GB LPDDR5x.

---

## Progress Log

### 2026-03-25: T2.1 benchmark script written

- Created scripts/bench-compare-ollama.sh (89c9b9e)
- Covers all 13 text-generation architectures
- Runs bench_tps for Zerfoo, parses Ollama --verbose eval rate
- 3 runs per model, median reported, outputs JSON + Markdown
- Supports --list flag and single-model filtering
- T1.1-T1.4 blocked: DGX Spark is busy (user confirmed)
- DGX Spark state: Ollama v0.17.7 installed, Go 1.26.1 available,
  partial model set (gemma3-q4km, mistral-7b, phi-3.5-mini, deepseek-v2-lite, qwen2.5-0.5b)
- Missing GGUF files: gemma3-4b, llama3.2-3b, llama3.1-8b, mixtral-8x7b,
  qwen2.5-7b, deepseek-r1-1.5b, command-r-35b, falcon-7b, mamba-2.8b, rwkv-7b

### 2026-03-25: Plan created

- Created plan-benchmark.md with 4 epics, 14 tasks across 4 waves.
- Identified 13 text-generation architectures to benchmark.
- 10 of 13 have Ollama equivalents for head-to-head comparison.
- 3 architectures (Falcon, Mamba, RWKV) are Zerfoo-only benchmarks.
- Existing infrastructure: bench_tps CLI, parity test for Gemma, CI workflow.

---

## Hand-off Notes

### For the benchmark operator

- The DGX Spark is at ssh ndungu@192.168.86.250.
- bench_tps is built with: `go build ./cmd/bench_tps/`
- Ollama outputs eval rate to stderr with --verbose flag.
- The existing parity test at tests/parity/throughput_parity_test.go is a reference
  for how to parse Ollama output programmatically.
- GGUF files should be stored in /data/models/ on DGX Spark.
- Results go in the results/ directory (gitignored; copy to docs/ for publication).

### Key URLs

- DGX Spark: ssh ndungu@192.168.86.250
- Benchmarks page: https://zerfoo.feza.ai/docs/reference/benchmarks/
- Ollama: https://ollama.com
- HuggingFace GGUF models: https://huggingface.co/models?library=gguf
