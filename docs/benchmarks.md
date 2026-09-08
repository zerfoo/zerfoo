# Zerfoo Benchmarks

Reproducible throughput record for the Zerfoo inference engine.

This file is the benchmark half of the evidence rule in
[`docs/verified-models.md`](verified-models.md): *marketing may not exceed the
verified record*. Every number below is either (a) traceable to a committed
result file, a manifest, or a dated `docs/devlog.md` entry, or (b) explicitly
marked as unbacked. Numbers that cannot be traced are listed as defects in
[Open documentation defects](#open-documentation-defects) rather than repeated.

**Document status:** written against commit `8e9d02ae`, module version
`v1.56.0`, `github.com/zerfoo/ztensor v1.19.2`, Go 1.26. Unless a row says
otherwise, its figure was **recorded on a previous date and has not been re-run
for this document.**

---

## Evidence status at a glance

| Claim | Status | Where the evidence is |
|---|---|---|
| Gemma 3 1B Q4_K_M beats Ollama on GB10 | **Recorded** | `results/benchmark-2026-03-27.json` — 235 vs 188 tok/s (1.25x) |
| 241 tok/s / 1.28x specifically | **Not backed by a committed run** | See [The 241 tok/s figure](#the-241-toks-figure) |
| Faster than Ollama on *all four* benchmarked models | **Contradicted by the repo's own record** | Two of four are ties/losses — see table below |
| Runs models larger than RAM (229B on 128 GB) | **Recorded, single run** | `docs/devlog.md` 2026-03-29 |
| Zero CGo on the default build | **Verified** | `.goreleaser.yml:8,22,36` pins `CGO_ENABLED=0`; see [Architecture notes](#architecture-notes) |
| Zero CGo on the GPU-accelerated build | **False as stated** | `-tags cuda` is a CGo path (`.github/workflows/benchmark.yml:79-82`) |
| Fused Q4_K GEMV, 14x | **Unsupported, and the kernel is not on the default path** | See [Fused Q4_K GEMV](#fused-q4_k-gemv) |
| EAGLE speculative decoding | **Code + tests exist; no measured speedup; not reachable from the public API** | See [Speculative decoding](#speculative-decoding) |
| QuaRot quantization | **Implemented and wired; no accuracy or speed number recorded** | `inference/quarot.go:29` |
| 41 architectures / 25 families | **Neither figure matches the registry** | 45 registered strings, 38 distinct builders |

---

## Throughput vs Ollama

### The recorded run

This is the most recent Zerfoo-vs-Ollama comparison that exists as a committed,
machine-readable artifact in this repository:
[`results/benchmark-2026-03-27.json`](../results/benchmark-2026-03-27.json).

| Model | Quant (Zerfoo side) | Zerfoo tok/s | Ollama tok/s | Ratio | Winner |
|---|---|---|---|---|---|
| Gemma 3 1B | Q4_K_M | 235 | 188 | 1.25x | Zerfoo |
| DeepSeek-R1-Distill 1.5B | Q4_K_M | 186 | 167 | 1.11x | Zerfoo |
| Llama 3.2 3B | Q4_K_M | 92 | 93 | 0.99x | **Even / Ollama** |
| Mistral 7B | Q5_K_M | 44 | 44 | 1.00x | **Even** |

- **Hardware:** NVIDIA DGX Spark GB10 (Grace Blackwell, `sm_121`, 128 GB LPDDR5x unified memory)
- **Date:** 2026-03-27
- **Prompt:** `"Explain the theory of relativity in simple terms."`
- **Generated tokens:** 128
- **Sampling:** greedy (`temperature = 0`)
- **Runs:** 3 per model per runtime, median reported
- **Ollama version:** 0.17.7 (recorded in the sibling
  [`results/benchmark-2026-03-25.json`](../results/benchmark-2026-03-25.json);
  the 03-27 file omits the field)
- **Harness:** [`scripts/bench-compare-ollama.sh`](../scripts/bench-compare-ollama.sh)
  driving [`cmd/bench_tps`](../cmd/bench_tps/main.go)

The honest one-line summary of this run is: **Zerfoo is meaningfully faster than
Ollama at 1B–1.5B, and at parity at 3B–7B.** It is not "faster on all four".

### The 241 tok/s figure

Several documents in this repository publish **241 tok/s / 1.28x vs Ollama 188**,
dated 2026-03-31 — see defect 13 below for the current list. `README.md` did too
until it was corrected on 2026-07-28. That specific pair of numbers cannot be
traced to a run in this repository. What is verifiable:

1. **The cited results file does not exist.** The previous revision of this file
   pointed at `results/benchmark-2026-03-31.json`. `git log --all` shows only
   `benchmark-2026-03-25.{json,md}` and `benchmark-2026-03-27.json` have ever
   been committed. `docs/verified-models.md:136-142` already records this gap.
2. **No devlog entry records a benchmark run on 2026-03-31.** Every
   `## 2026-03-31` entry in `docs/devlog.md` is a GPU *regression* investigation
   — "GPU engine produces wrong inference output (ONGOING)" (`:3512`), "GPU
   kernel recompilation produces garbage output" (`:3539`), "GPU inference
   regression fully diagnosed" (`:3433`).
3. **The number was reached by two documentation edits 74 minutes apart, not by
   one measurement.** Commit `2dbc52d8` (2026-03-31 20:16) set the row to
   `236 | 197 | 1.20x`; commit `174c2ca8` (21:30) changed it to
   `241 | 188 | 1.28x`. The Zerfoo value rose by 5 and the Ollama baseline fell
   by 9 in the same edit. The replacement Ollama value, 188, is exactly the
   figure recorded four days earlier in `results/benchmark-2026-03-27.json`.
   Neither commit added a result file.
4. **`2dbc52d8` also deleted the README's raw-results link** (`Raw results:
   results/benchmark-2026-03-30.json`), and that file was never committed either.
5. **The published methodology describes code that had already been removed.**
   The methodology block added at 20:16 credits "fused kernels: softmax+V
   multiply" and "Q4_K/Q5_K/Q6_K/Q5_0 weights re-quantized to Q4_0". The fused
   softmax kernels were disabled at 18:18 that day as producing wrong decode
   output (`999f2fdf`), and commit `1d56d2e5` at 15:37 was titled
   "use native Q4_K storage instead of lossy re-quantization to Q4_0".

Separately, the devlog states outright that an earlier 241 measurement was not
on the format the README names: *"The 241 tok/s benchmark was on Q4_0 ZMF, not
GGUF Q4_K_M"* (`docs/devlog.md:5018`, 2026-03-16). ZMF has since been removed
from the codebase entirely (`CLAUDE.md`, "GGUF is the sole model format").

**The best-supported Gemma 3 1B figures in this repository are:**

| Figure | Conditions | Source |
|---|---|---|
| 235 tok/s | 128 tokens, Q4_K_M GGUF, CUDA graphs | `results/benchmark-2026-03-27.json` |
| 245.15 tok/s | **256** tokens, Q4_K_M GGUF, CUDA graphs, Q4_0 re-quant path | `docs/devlog.md:4200` (2026-03-18) |
| 174.44 tok/s | 256 tokens, **native Q4_K** + `sm_121` kernel, no re-quant | `docs/devlog.md:4207` (2026-03-18) |

Throughput is sensitive to token count (longer runs amortize prefill and graph
capture), so 235 @ 128t and 245 @ 256t are not in conflict — but they are also
not interchangeable, and a published figure must name its token count.

### Recertifying the Gemma 3 1B number

[`docs/bench/manifests/gemma3-tps.yaml`](bench/manifests/gemma3-tps.yaml) exists
specifically to settle this. Its submit wrapper states the problem plainly:

> The original 241 tok/s was measured with a different binary; after the ztensor
> module extraction the same kernel hit 186 tok/s. This script runs
> `cmd/bench_tps` against the current main branch to establish the real current
> number.
> — [`scripts/gemma3-spark.sh:8-10`](../scripts/gemma3-spark.sh)

**That recertification has not been run and recorded.** Until it is, treat the
Gemma 3 1B headline as *"235 tok/s at 128 tokens, 1.25x Ollama 0.17.7, measured
2026-03-27 on GB10"*.

---

## Methodology

### What `cmd/bench_tps` measures

[`cmd/bench_tps/main.go`](../cmd/bench_tps/main.go):

| Step | Line | Behaviour |
|---|---|---|
| Model load | `:108-116` | Excluded from the timed region |
| Warm-up | `:124` | One `Generate` call capped at **4 tokens** |
| Timer start | `:138` | Immediately before `GenerateStream` |
| Token counting | `:130-136` | Increments once per streamed non-terminal token |
| Timer stop | `:140` | After `GenerateStream` returns |
| Reported metric | `:146` | `generatedTokens / elapsedSeconds` |

Two consequences a reviewer should know:

- **The timed region includes prompt evaluation.** `GenerateStream` tokenizes and
  prefills the prompt inside the timer, so Zerfoo's reported tok/s is
  *generated tokens ÷ (prefill + decode)*, not a pure decode rate.
- **Warm-up is 4 tokens.** Enough to trigger CUDA graph capture, but a short
  warm-up by conventional standards.

### What the Ollama side measures

[`scripts/bench-compare-ollama.sh:96-117`](../scripts/bench-compare-ollama.sh)
runs `ollama run --verbose` and parses the line matching `^eval rate:`,
explicitly excluding `prompt eval rate:`.

**These two metrics are not the same measurement.** Ollama's `eval rate` is
decode-only; Zerfoo's includes prefill. For a ~10-token prompt and 128 generated
tokens the asymmetry is small, and it runs **against** Zerfoo — the comparison
understates Zerfoo rather than flattering it — but it is not apples-to-apples and
should be corrected before the number is used adversarially.

### Other asymmetries in the comparison

| Asymmetry | Detail |
|---|---|
| **Compute dtype** | `bench_tps` defaults to `-dtype fp32` and `-kv-dtype fp32` (`main.go:63-64`), and the compare script does not override them. Ollama uses its own defaults. |
| **Effective quantization** | The GGUF decode path re-quantizes Q4_K to Q4_0 at load (`model/gguf/loader.go:265`). Zerfoo therefore runs **Q4_0 arithmetic on a Q4_K_M file**, while Ollama runs true Q4_K_M. The quantization label describes the file, not the math. |
| **Ollama tags are unpinned** | Models are named by tag (`gemma3:1b`, `mistral:7b`) at `bench-compare-ollama.sh:49-63`. The tag's quantization is whatever Ollama's registry serves; it is not asserted to match the Zerfoo-side GGUF. The Mistral row is the clearest exposure — Zerfoo's side is recorded as Q5_K_M. |
| **Process model** | Each Zerfoo run is a fresh process (fresh load, fresh 4-token warm-up); Ollama keeps the model resident across the three runs. |
| **mmap off** | `bench_tps:74-76` force-disables mmap on any `cuda` device. The GPU throughput numbers and the over-RAM mmap story are different code paths and are never exercised together. |

### Per-model reproduction manifests

Only Gemma 3 1B has one (`gemma3-tps.yaml`). The DeepSeek, Llama, and Mistral
rows are marked "per-model reproduction manifest pending" in
`docs/verified-models.md`.

---

## Reproduction

### CPU-only microbenchmarks — runnable on any machine

No GPU, no model weights, no network beyond `go mod download`.

```bash
git clone https://github.com/zerfoo/zerfoo && cd zerfoo
go test -bench='BenchmarkGemmQ4F32_GEMV' -benchmem -count=1 -run='^$' \
    -timeout 300s ./internal/xblas/
```

Expected output shape — two sub-benchmarks, `dequant+sgemm` and `fused`, at
M=1, K=N=4096:

```
goos: darwin
goarch: arm64
pkg: github.com/zerfoo/zerfoo/internal/xblas
cpu: Apple M4
BenchmarkGemmQ4F32_GEMV/dequant+sgemm-10   1059   1115582 ns/op   0 B/op   0 allocs/op
BenchmarkGemmQ4F32_GEMV/fused-10           1160   1055201 ns/op   0 B/op   0 allocs/op
PASS
ok      github.com/zerfoo/zerfoo/internal/xblas 2.993s
```

**This output was produced live while writing this document** (2026-07-28,
Apple M4, Go 1.26, commit `8e9d02ae`, single run, `-count=1`). Runtime ~3 s after
compilation. It is the only benchmark in this file that was observed rather than
recorded.

Reading it honestly: the fused CPU Q4 GEMV is **1.06x** the dequant-then-SGEMM
path here, not 14x. This is the CPU `xblas` Q4_0 kernel, a *different* kernel
from the CUDA Q4_K GEMV that the 14x claim refers to — but it is also the only
fused-vs-unfused GEMV comparison in the repo that anyone can run without a GB10,
and it is the metric CI tracks (`scripts/bench.sh:23`, `:48-58`).

The full CI benchmark set:

```bash
./scripts/bench.sh          # emits one JSON object per line
```

### Full Zerfoo-vs-Ollama comparison — requires a GPU host and weights

Not re-run for this document. Prerequisites:

1. A CUDA host with the CUDA 13 runtime at `/usr/local/cuda`. The numbers above
   are GB10 / `sm_121`; any other GPU produces different numbers.
2. Ollama installed and serving, with the comparison models pulled:
   `ollama pull gemma3:1b llama3.2:3b mistral:7b deepseek-r1:1.5b`.
3. GGUF files under `$MODEL_BASE` (default `$HOME/models`) at the relative paths
   listed in `scripts/bench-compare-ollama.sh:49-63` — e.g.
   `gemma3-q4km/model.gguf`. Gemma 3 1B is `ggml-org/gemma-3-1b-it-GGUF`, file
   `gemma-3-1b-it-Q4_K_M.gguf` (~0.8 GB); the four-model set is roughly 8 GB.

```bash
go build -o bench_tps ./cmd/bench_tps/
MODEL_BASE=/path/to/models ./scripts/bench-compare-ollama.sh           # all models
MODEL_BASE=/path/to/models ./scripts/bench-compare-ollama.sh gemma3-1b # one model
```

Writes `results/benchmark-<YYYY-MM-DD>.{json,md}`. Expected runtime: 3 runs per
model per runtime — a few minutes for the 1B–3B models, longer at 7B.

Single-model sanity check without the compare harness:

```bash
./bench_tps -model /path/to/gemma-3-1b-it-Q4_K_M.gguf \
            -prompt "Explain the theory of relativity in simple terms." \
            -tokens 128 -device cuda -dtype fp32 -temp 0
```

Expected output shape (values are hardware-dependent):

```
Loading model from ... (device=cuda, dtype=fp32, kv-dtype=fp32)...
Loaded in N.Ns
Warm-up...
Generating (temp=0.0)...
--- Results ---
Output: ...
Generated tokens: 128
Time: N.NNNs
Throughput: NNN.NN tok/s
```

### Recertifying Gemma 3 1B on the DGX host

GPU benchmarks are submitted as pods to the Spark orchestrator, never over
interactive SSH — see [`docs/adr/083-spark-bench-runner.md`](adr/083-spark-bench-runner.md)
for why (an SSH-channel leak took the bench host down on 2026-04-07).

Stage the binary and weights on the bench host, then:

```bash
GOOS=linux GOARCH=arm64 go build -o bench_tps ./cmd/bench_tps
# stage bench_tps at /var/lib/zerfoo/bin/ and the GGUF at /var/lib/zerfoo/models/

SPARK_HOST=<host:port> scripts/gemma3-spark.sh \
  -gguf /var/lib/zerfoo/models/gemma-3-1b-it-Q4_K_M.gguf \
  -prompt "Explain the theory of relativity in simple terms." \
  -tokens 128 -device cuda -dtype fp32 -cleanup
```

The script renders [`docs/bench/manifests/gemma3-tps.yaml`](bench/manifests/gemma3-tps.yaml),
POSTs it, polls to a terminal phase, and prints the pod log. Note the manifest's
own instruction: `ZERFOO_DISABLE_CUDA_GRAPH` is deliberately left unset, because
the published baseline was measured with graph capture enabled (184/185
instructions, 99.5%).

### GPU kernel microbenchmark

```bash
go test -tags cuda -bench='BenchmarkGemmQ4F32_1024' -benchmem -count=1 \
    -run='^$' -timeout 300s ./internal/cuda/kernels/
```

Reported as GFLOPS by `.github/workflows/benchmark.yml:82-94`. This is the only
GPU benchmark wired into CI, and it runs on schedule/dispatch only, on a
self-hosted runner. **`-tags cuda` is a CGo build.**

---

## What is and is not claimed

**Claimed, with the stated evidence:**

- On one NVIDIA DGX Spark GB10, on 2026-03-27, generating 128 greedy tokens from
  a fixed 8-word prompt, Zerfoo decoded Gemma 3 1B Q4_K_M at 235 tok/s against
  Ollama 0.17.7 at 188 tok/s, and DeepSeek-R1-Distill 1.5B at 186 vs 167.
- On that same run, Llama 3.2 3B (92 vs 93) and Mistral 7B (44 vs 44) were at
  parity — Zerfoo did not win either.
- The default `go build` produces a binary with no CGo, and release artifacts pin
  `CGO_ENABLED=0`.
- A 229B MoE model in a 128.8 GB three-shard GGUF loaded in 6.3 s and generated
  tokens on a 128 GB machine, where Ollama returned a 500 error.

**Not claimed:**

- **Nothing here is independently reproduced.** Every GPU figure comes from a
  single operator on a single machine. No third party has re-run any of it.
- **These are single-configuration numbers, not a benchmark suite.** One prompt,
  one token count, greedy sampling, batch size 1, short context. There is no
  batched-throughput, long-context, concurrent-request, or prompt-eval figure in
  this repository. Nothing here says anything about serving under load.
- **GB10-specific.** Grace Blackwell unified memory, roughly 200 GB/s. Results
  will not transfer to a discrete-VRAM datacenter GPU, and the ranking against
  Ollama may invert.
- **Zerfoo is not claimed to be faster than Ollama in general.** It is faster at
  1B–1.5B on this hardware and at parity by 3B. The trend across the four models
  is that the advantage shrinks as the model grows — which is what a
  memory-bandwidth-bound decode with a well-optimized small-model path looks
  like. Extrapolating a win to larger models is not supported.
- **Not compared against llama.cpp, vLLM, TensorRT-LLM, or MLX.** Ollama is a
  llama.cpp wrapper and a convenience baseline, not the performance ceiling.
- **The over-RAM result is a capability demonstration, not a throughput result.**
  0.06 tok/s is NVMe-bound and unusable for interactive work. What it proves is
  that the model loads and decodes at all.
- **Quantization labels describe the file, not the arithmetic.** Zerfoo
  re-quantizes Q4_K to Q4_0 at load. Same file, cheaper math than Ollama does on
  it. No perplexity or accuracy comparison has been run to quantify what that
  costs, so the throughput comparison is **not quality-normalized**.
- **Zero-CGo does not extend to the CUDA build tag.** The default and release
  builds are CGo-free; `-tags cuda` — which is how the kernel benchmarks build —
  is not.
- **No speculative-decoding speedup is claimed.** See below.
- **The 2026-03-31 figures published in `README.md` are not claimed by this
  document.** They are listed as defects below.

---

## Architecture notes

Factual description of the mechanisms behind the numbers, each with its
implementing source.

### Zero-CGo GPU binding

CUDA is loaded at runtime by a hand-rolled `dlopen`/`dlsym` layer rather than
CGo, which is what lets `go build ./...` work with no C toolchain present.

- `internal/cuda/doc.go:2` — "low-level bindings for the CUDA runtime API using dlopen/dlsym (no CGo)"
- `internal/cuda/purego.go:54` — `Open`, resolving `libcudart.so.12` / `libcudart.so`
- `internal/cuda/purego.go:174` — `DlopenKernels`; `:198` — `Dlsym`
- `internal/cuda/purego_linux_arm64.go:29` — `//go:linkname runtime_dlopen runtime.dlopen`, with asm trampolines in `purego_linux_arm64.s`, `purego_darwin_arm64.s`, `purego_darwin_amd64.s`
- `distributed/nccl.go:66` — `openNCCL`, same mechanism for NCCL
- `.goreleaser.yml:8,22,36` — `CGO_ENABLED=0` on all three release builds

27 files do contain `import "C"`, every one behind an off-by-default tag
(`cuda`, `cuda && cutlass`, `cuda && tensorrt`, `opencl`, `linux && arm64 && cgo`).

### Fused Q4_K GEMV

The kernel is real and non-trivial — one warp per row, 6-bit sub-block scale/min
decode in registers, plus a Blackwell-specific variant:

- `internal/cuda/kernels/gemv_q4k.cu:65` — `gemv_q4k_kernel`
- `internal/cuda/kernels/gemv_q4k.cu:34` — `decode_scales_mins`, the 6-bit sub-block decode
- `internal/cuda/kernels/gemv_q4k.cu:146` — `gemv_q4k_f32` dispatcher
- `internal/cuda/kernels/gemv_q4k_sm121.cu:87` — `sm_121` variant
- `internal/cuda/kernels/gemv_q4k_purego.go:14` — purego binding
- `internal/gpuapi/cuda_kernels.go:112` — `(*CUDAKernels).GemvQ4KF32`

**It is not on the default inference path.** `model/gguf/loader.go:265`
re-quantizes Q4_K to Q4_0 at load time:

```go
// Re-quantize Q4_K → Q4_0 for uniform fast GEMV decode path.
```

so tensors carry `Q4Storage`, not `Q4KStorage`, and dispatch goes to the Q4_0
GEMV. `GemvQ4K` has zero call sites in `inference/`, `layers/`, or `model/`. The
only escape is `ZERFOO_GEMMA4_PLE_NATIVE_Q4K=1` (`model/gguf/loader.go:41`),
which applies to embedding-shaped tensors only.

This is a deliberate, measured choice rather than an oversight —
`docs/devlog.md:4207` (2026-03-18) records the native Q4_K + `sm_121` path at
**174.44 tok/s** against the Q4_0 re-quant path at **245.15 tok/s**, and states:
*"the Q4_K GEMV kernel (including the `sm_121`-optimized variant) is never
reached during inference."* Q4_0's 18-byte blocks cache better than Q4_K's
144-byte super-blocks.

The **"14x faster" figure has no runnable benchmark.** The only Go benchmark,
`internal/cuda/kernels/gemv_q4k_test.go:384`, skips without CUDA, runs the fused
kernel alone, and reports a single GFLOPS metric — no baseline, no ratio. The
figure appears only as prose in `README.md:10` and `docs/updates.md:162`
("DGX: 47.8ms → 3.36ms"); those two timings appear in no test, result file, or
commit message.

### Memory-mapped weights

- `inference/load_gguf.go:18` — `mmap: true` is the default
- `inference/load_gguf.go:27-29` — auto-disabled on any `cuda` device, so mmap is in practice a CPU-path feature
- `inference/gguf.go:153` — `LoadGGUFMmap`; `:181` `tensor.MmapFile`; `:202` `tensor.MadviseRandom`
- `model/gguf/loader_mmap.go:17` — `LoadTensorsMmap`
- `model/mmap_unix.go:36` — this repo's own `syscall.Mmap` call. Note it is currently exercised only by `model/mmap_test.go`; the production GGUF path maps through `tensor.MmapFile` in ztensor.

Split-GGUF sharding, which is what makes the 3-shard 128.8 GB load work:

- `model/gguf/split_file.go:40` — `ParseSplit`, the entry point (called at `inference/gguf.go:155`)
- `model/gguf/split_file.go:88` — `discoverShards`, matching `-NNNNN-of-NNNNN.gguf` and `os.Stat`-ing every sibling
- `model/gguf/split_file.go:141` — `LoadTensorsMmapSplit`
- `inference/gguf.go:207` — `loadGGUFMmapSplit`

### Speculative decoding

Implemented with real tests, but **not reachable from the public API**:

- `generate/eagle_speculative.go:57` — `NewEAGLEGenerator`; decode loop at `:91`, draft generation at `:220`
- `layers/core/eagle_head.go:20` — `EAGLEHead`; `:158` `NewEAGLEHeadFromWeights`
- `inference/eagle.go:22` — `BuildEAGLEHead`; `:118` `LoadEAGLEWeights`; `:251` `GenerateDraftTokens`
- `cmd/cli/eagle_train.go:24` — the `eagle-train` command

Gaps a reviewer will find: `NewEAGLEGenerator` has no non-test call site;
`generate/generator.go:157` `WithEAGLE` stores a path that no generation path
reads; head training uses synthetic pairs because graph-level intermediate
capture is unimplemented (`inference/arch_common.go:93`). **No speedup number
exists anywhere** — the repo's only speculative-decoding benchmark
(`docs/devlog.md:4398`) is draft-model speculation, not EAGLE, and records "DGX
results pending".

### QuaRot

- `inference/quarot.go:29` — `FuseQuaRotWeights`; `:98` `fuseHadamardIntoWeight`; `:164` `fwht` (in-place Fast Walsh–Hadamard, O(n log n))
- `inference/inference.go:215` — `WithQuaRot`, consumed at `inference/load_gguf.go:76` before graph build
- `cmd/cli/run.go:150` — the `--quarot` flag

Wired end-to-end with 13 tests. **No perplexity, accuracy, or speed number is
recorded.** Real limitation: the hidden dimension must be a power of two
(`inference/quarot.go:47`); non-power-of-two tensors are skipped with a warning
(`inference/quarot.go:127`).

### CUDA graph capture

The decode loop is captured as a CUDA graph at 184/185 instructions (99.5%) and
replayed, removing per-step launch overhead. Recorded at `docs/devlog.md:4909`
and in the `gemma3-tps.yaml` header. Its measured contribution has varied widely
across the codebase's history — `docs/devlog.md:4880` records a period where the
graph was worth only 1.4x (122 → 166 tok/s).

### Architecture registry

- `inference/registry.go:24` — `archRegistry`; `:35` `RegisterArchitecture`; `:60` `ListArchitectures`
- `inference/registry_init.go:5` — 29 names, plus per-file `init()` registrations in `arch_glm.go`, `arch_lfm2.go`, `arch_kimi.go`, `arch_commandr.go`, `arch_mixtral.go`, `arch_rwkv.go`, `arch_falcon.go`, `arch_gemma3n.go`, `arch_llama4.go`, `arch_llava.go`, `arch_qwenvl.go`
- `inference/load_gguf.go:196` — `buildArchGraph`, a hardcoded switch falling through to the registry at `:242`

**45 distinct architecture strings resolving to 38 distinct builder functions**
(aliases collapse: `gemma`/`gemma3`, `phi`/`phi3`,
`deepseek_v3`/`deepseek2`/`deepseek2-ocr`, `chatglm`/`glm4`/`glm-dsa`,
`exaone`/`exaone4`). Registration is not verification — see
`docs/verified-models.md`, where only six rows are `verified` and the rest are
`pending` or `arch-unconfirmed`.

---

## Over-RAM Inference

Single recorded run, 2026-03-29 (`docs/devlog.md:3606`).

| Model | Params | Quant | File size | Shards | RAM | Load time | Throughput | Ollama |
|---|---|---|---|---|---|---|---|---|
| MiniMax-M2 | 229B (MoE) | Q4_K_M | 128.8 GB | 3 | 128 GB | 6.3 s | 0.06 tok/s | fails to load (500) |

Conditions: DGX Spark GB10, CPU-only, 809 tensors mapped, prompt
`"The meaning of life is"`, 4 tokens generated, output `"a priori is something"`.

Caveats, all from the devlog entry itself:

- **CPU-only.** There is no GPU path for over-RAM inference.
- **NVMe-bound**, hence 0.06 tok/s. A capability proof, not a serving
  configuration.
- **4 tokens.** The output was checked for non-degeneracy, not for quality.
- Reaching this required fixing four separate OOM sources: streaming GEMM for
  `MmapStorage`, a 23.5 GB KV allocation at graph build, expert-slice
  materialization, and an 857 GB random-weight allocation in `NewFFN`. The
  `-max-seq-len` cap (`cmd/bench_tps/main.go:61`) was added for this run.
- The Ollama failure is reported as a 500 on the same hardware; no Ollama-side
  diagnosis was done.

---

## Training: PatchTST (GPU)

DGX GB10, commit `2ecf473a`, 2026-04-09.

| Workload | v1.37 | v1.38.4 | v1.42+ | Speedup (v1.37 → v1.42+) |
|---|---|---|---|---|
| 28K×20×10 | 596 s | 128.5 s | **40.3 s** | **14.8x** |
| 20K×20×5 | — | — | 15.0 s | — |
| 5K×10×3 | — | — | 3.0 s | — |

Pre-allocated batch workspace plus GPU dst-memory reuse (ztensor #84/#85). Loss
converges 99.9% (0.0178 → 0.000022) on 28K×20×10.

The v1.38.4 baseline (128.5 s) had regressed to OOM after E85 buffer
pre-allocation (`09a318c6`) introduced per-op GPU memory leaks; fixed in ztensor
#85 by reusing dst device pointers instead of allocating per call.

Note: `README.md` previously advertised "Time-series training 4.6x faster with
CUDA graphs". 596 / 128.5 = 4.64, so that figure was the **v1.37 → v1.38.4**
column and was two releases stale. It was removed from the README on 2026-07-28
rather than updated to 14.8x, because of the provenance caveat below.

Submit via [`scripts/bench-spark.sh`](../scripts/bench-spark.sh) with
[`docs/bench/manifests/patchtst-train.yaml`](bench/manifests/patchtst-train.yaml).

**Provenance caveat:** these timings trace to the previous revision of this file
and to commit `2ecf473a`. Unlike the inference rows, they have no committed
result file and no corresponding `docs/devlog.md` entry — a grep for `40.3` and
`14.8x` finds them only here. Treat the 14.8x as recorded-but-unsourced until a
re-run is committed.

---

## Granite Time Series vs Python granite-tsfm

Parity against the reference Python implementation. Results are recorded in
[`docs/devlog.md`](devlog.md) under 2026-03-27; no throughput comparison or
result file has been committed, and no speed claim is made here. Golden-file
parity fixtures live at `tests/parity/ttm_golden_test.go` (10 TTM golden files
from granite-tsfm 0.3.5, per `docs/updates.md`).

---

## Fused GQA repeat-interleave restored (2026-08-23, T148.2)

Target: ≥ 1.5x speedup at seqLen_KV > 1024 (E43). To be benchmarked on DGX Spark after GPU streaming GEMM lands.

The fused GQA KV-head-expansion kernel is back in use on GPU. It was disabled
outright in `layers/attention/grouped_query_attention.go` from 2026-08-09 to
2026-08-23 as the mitigation for ztensor#180, during which every GQA model
(Llama, Mistral, Qwen, Gemma) ran the slower `Reshape -> Repeat -> Reshape`
chain instead. Fixed upstream in ztensor#183 (an optional kernel symbol was
launched through a null function pointer, killing the process instead of
returning an error) plus a refresh of the DGX's deployed
`/opt/zerfoo/lib/libkernels.so`, which had not been rebuilt since 2026-07-03
and did not export `launch_repeat_interleave_f32` at all.

**Every throughput number above this section was measured on a build where
this path was NOT disabled** (they predate 2026-08-09) **and remains true for
that build.** No figure in this file has been restated. Nothing was
re-benchmarked for this change: it restores a previously-shipped code path
rather than introducing a new one, and a throughput comparison for it is not
yet measured. Treat the perf effect as unquantified until an A/B run lands.

Correctness on the restored path, GB10, ztensor
`v1.19.3-0.20260823235001-c29246e8ac17`:

```
--- PASS: TestGPUParity_GQA
    gqa_forward: maxDiff=7.629395e-06 at idx=5, mismatches=0/64
--- PASS: TestGPUParity_GQA_FusedRepeatInterleave
    cpu reference range=8.011882e+01 (min=-46.216557 max=33.902267) over 64 elements
    gqa_forward_fused_repeat_interleave: maxDiff=7.629395e-06 at idx=5, mismatches=0/64
```

`TestGPUParity_GQA` alone does not establish that the fused path ran --
GroupedQueryAttention falls back silently and correctly, so that test is green
either way (verified: it passes against the pre-refresh library too). The
companion test asserts `FusedRepeatInterleaveAvailable()` first and fails
rather than skips, which is what makes the claim in this section checkable.

## Gemma 4 E2B (edge) — do not cite as a performance claim

Recorded figures: 3.85 / 2.69 / 1.23 / 3.15 tok/s across four dates.

| Figure | Conditions | Status |
|---|---|---|
| 3.85 tok/s | commit `72828131` | **Unverified.** Could not be reproduced on 2026-04-16: the same binary at the same commit on the same host, `-mode generate -device cuda -steps 64 -prompt "The quick brown fox"` with `ZERFOO_DISABLE_CUDA_GRAPH=1`, measured 2.69. Likely measured with different parameters. |
| 2.69 tok/s | `72828131`, capture disabled, 2026-04-16 | Best reproducible figure at the time |
| 1.23 tok/s | main `6ad8bceb`, capture disabled | A ~2.2x regression, bisected to `96c7540a` — 70 synchronous H2D copies per decode step in `inference/gemma4_edge_ple_nodes.go` |
| 3.15 tok/s | PR #490 tip `8bb7e1a1`, 2026-04-20 | Fix: 2 H2D + 70 D2D per step |

**Correctness caveat (2026-04-16):** gemma4e generate produces degenerate tokens
(`"ly\ns\ns\ns..."` on CPU, multilingual gibberish on GPU) on both `72828131` and
main. The earlier "40 bytes non-degenerate output" note referred to
`-mode forward`, not decode. These figures are a **regression floor only** until
decode coherence is restored. No Ollama comparison exists — Ollama does not
support the architecture.

---

## Flash decode — pending

Target: ≥1.5x speedup at `seqLen_KV > 1024` (E43). Not benchmarked. Note that
`flash_decode.cu` was absent from this repo's kernel Makefile until 2026-07-03
and had to be ported from ztensor (`docs/devlog.md`, T135.3), so no historical
flash-decode figure from this repository should be trusted.

---

## Open documentation defects

Tracked here so they stay visible. Each is a claim published elsewhere in the
repo that this file does not support.

**`README.md` was corrected against this document on 2026-07-28.** Defects 1–10
were README defects and are now closed. They are kept in the table with their
resolution rather than deleted, so the record of what was claimed and what
replaced it stays auditable.

| # | Defect | Location | Status |
|---|---|---|---|
| 1 | "Faster than Ollama on all models" — the repo's own recorded run has Llama 3.2 3B at 0.99x and Mistral 7B at 1.00x | `README.md` | **Closed.** README now states parity by 3B and prints the 0.99x/1.00x rows unbolded. |
| 2 | README benchmark table (241/188, 190/174, 95/93, 46/45) matches no committed result file | `README.md` | **Closed.** Table rewritten from `results/benchmark-2026-03-27.json` (235/188, 186/167, 92/93, 44/44) and links to it. |
| 3 | Cited results file `results/benchmark-2026-03-31.json` has never existed | previously cited in this file and at `docs/verified-models.md` | **Closed.** Citation removed here; `verified-models.md` updated to cite the 03-27 file. |
| 4 | README's Mistral row said Q5_K_M 46 tok/s while its own supported-models table said 44 tok/s Q4_K_M; the result file says Q5_K_M 44 | `README.md` | **Closed.** Both README locations now say 44 tok/s Q5_K_M. |
| 5 | Published methodology credited fused softmax kernels and Q4_0 re-quantization; the former were disabled hours before it was written | `README.md` | **Closed.** Methodology block rewritten; the removed softmax-kernel credit is gone and the Q4_0 re-quantization is now disclosed as an asymmetry rather than credited as an optimization. |
| 6 | "Q4_K fused GEMV (14x faster)" — no benchmark produces the ratio, and the kernel is not reached on the default path | `README.md` | **Closed.** Claim removed. The README now discloses the kernel as measured-slower (174.44 vs 245.15) and off the default path. |
| 7 | "41 model architectures (25 families)" — the registry has 45 strings / 38 builders, and no "family" grouping exists in code | `README.md` | **Closed.** README now says 38 builders / 45 architecture strings and states that registration is not verification. |
| 8 | "Zero CGo" is unqualified; true for default and release builds, false for `-tags cuda` | `README.md` | **Closed.** README adds a Zero CGo section that names the `-tags cuda`/`cutlass`/`tensorrt`/`opencl` exception. |
| 9 | README's EAGLE example calls `zerfoo.WithEAGLE`, which the root package does not export; the snippet does not compile | `README.md` | **Closed.** Snippet removed; EAGLE is described as implemented but not wired to the Model API. |
| 10 | "Time-series training 4.6x faster" is the v1.37 → v1.38.4 figure; current is 14.8x | `README.md` | **Closed.** Claim dropped from the README rather than restated, because the 14.8x replacement has no committed result file either (see the PatchTST provenance caveat above). |
| 11 | Ollama baseline for Gemma 3 1B appears as 188, 197, and 204 in different documents | `docs/gophercon-2026-proposal.md:87`, `docs/roadmap-progress-2026-03-17.md:29`, `docs/adr/033-how-we-beat-ollama.md:11` | **Open.** See below. |
| 12 | `docs/QUALITY.md` test counts are stale (says 6 QuaRot tests; there are 13) | `docs/QUALITY.md:129` | **Open.** |
| 13 | Stale 241 tok/s and derived ratios still published outside the README | `docs/gophercon-2026-proposal.md:17,87,140,149`, `docs/product-strategy-2026-H2.md:15,161`, `docs/roadmap-progress-2026-03-17.md:29`, `docs/updates.md:162` | **Open.** These now contradict the corrected README. |

### On the three Ollama baselines (defect 11)

The divergence has a traceable cause, and two of the three numbers are real:

| Value | Origin | Status |
|---|---|---|
| **188** | `results/benchmark-2026-03-27.json` | Real, committed. The figure this document and the README use. |
| **204** | `results/benchmark-2026-03-25.json` (204.37) | Real, committed, but from the earlier 03-25 run. Cited in `docs/gophercon-2026-proposal.md:87` alongside a 241 Zerfoo number from neither run. |
| **197** | none | **Unsourced.** Predates both result files; appears at `docs/roadmap-progress-2026-03-17.md:29` and `docs/adr/033-how-we-beat-ollama.md:11`. No committed run records it. |

Pairing a Zerfoo number from one run with an Ollama number from another produces
a ratio that describes no experiment. The GopherCon proposal does this twice, and
also publishes a fourth Zerfoo-side figure — 233 tok/s (`:17`) — that matches no
result file. Its "+14%" (`:87`) and "28% faster" (`:17`) cannot both be true.

Note also `docs/product-strategy-2026-H2.md:161`, which sets a regression gate of
"≥241 (no regressions)". Since the best committed figure is 235, that gate cannot
pass and should be reset to 235 or to a freshly measured number.

### Closing the remaining defects

Run [`scripts/gemma3-spark.sh`](../scripts/gemma3-spark.sh) and
[`scripts/bench-compare-ollama.sh`](../scripts/bench-compare-ollama.sh), commit
the resulting `results/benchmark-<date>.json`, then update this file, the README
table, and the documents listed in defects 11 and 13 from that single file.
