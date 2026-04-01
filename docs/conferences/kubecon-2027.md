# KubeCon + CloudNativeCon 2027 Talk Proposal

## Talk Title

**Running ML Inference at Scale with Go and Kubernetes**

## Format

Breakout session (30 minutes)

## Abstract

Running large language models in production used to mean operating a separate
Python fleet alongside your Go services — a costly, operationally complex split
that undermines the reliability and simplicity Kubernetes was designed to
provide. Zerfoo eliminates that split.

Zerfoo is a production-grade ML inference framework written entirely in Go. It
loads GGUF models, serves transformer architectures (Llama 3, Gemma 3, DeepSeek
V3, and twelve others) through an OpenAI-compatible API, and delivers 233
tokens/second on Gemma 3 1B — 28% faster than Ollama on identical hardware.
Because it has zero CGo by default and ships as a single static binary, it
deploys identically to any other Go service on Kubernetes.

This talk covers the full cloud-native serving stack: the `ZerfooInferenceService`
custom resource definition (CRD) that lets platform teams declaratively manage
model deployments; the adaptive batch scheduler that dynamically adjusts batch
size from 1 to 32 based on queue depth and latency EMA to hit p99 SLOs; the
`ModelManager` with LRU GPU-memory eviction for multi-model serving from a
single pod; and prefill/decode disaggregation via gRPC for separating compute
profiles across node pools. We will also cover edge deployments on ARM64 nodes
using NEON SIMD acceleration and cross-compiled binaries with no runtime
dependencies.

Attendees will leave with a concrete architecture for running ML inference on
Kubernetes using Go — observable via Prometheus, configurable via Helm, and
deployable with `kubectl apply`.

## Target Audience

- Platform and infrastructure engineers managing Kubernetes clusters who want to
  add ML inference without introducing a Python operational surface
- Go developers building services that need LLM capabilities (chat, embeddings,
  code completion, document processing) and want to run inference in-process
- ML platform engineers evaluating alternatives to vLLM or TGI for
  latency-sensitive or edge-constrained workloads
- Site reliability engineers responsible for ML serving SLOs

**Prerequisites:** Familiarity with Kubernetes concepts (Deployments, CRDs,
ConfigMaps). Go experience is helpful but not required. No ML internals knowledge
needed.

## Detailed Outline (30 minutes)

### 1. The Cloud-Native ML Serving Problem (4 min)

- Why Python ML runtimes and Kubernetes are an awkward fit: image size, startup
  latency, cross-compilation blockers, and separate operational runbooks
- The cost of a Python sidecar: additional pod, separate scaling policy, network
  hop on every inference call
- What the Kubernetes community actually wants: a single binary, declarative
  configuration, standard observability hooks
- The thesis: ML inference is a library problem, not a runtime problem

### 2. Zerfoo Architecture in 90 Seconds (4 min)

- Three-layer stack: `ztensor` (tensors + compute + graph), `ztoken`
  (BPE tokenizer), `zerfoo` (inference, training, serving)
- GGUF as the sole model format: mmap-friendly, llama.cpp compatible,
  self-describing — models load from a `PersistentVolumeClaim`
- Zero CGo by default: `go build ./...` compiles everywhere; GPU acceleration
  loads dynamically via purego/dlopen at runtime
- `compute.Engine[T]` — the single interface all tensor arithmetic flows through,
  enabling transparent CPU/GPU switching without code changes
- Performance baseline: 241 tok/s on Gemma 3 1B Q4_K_M with CUDA graph capture,
  +14% vs Ollama on DGX Spark GB10

### 3. The Kubernetes Operator: ZerfooInferenceService CRD (6 min)

- CRD overview: declare model, quantization format, replica count, GPU memory
  budget, and SLO targets in a single manifest
- Operator responsibilities: downloading GGUF models to PVCs, managing
  Deployments and Services, health-check integration, rolling model updates
- Example manifest: deploying Gemma 3 1B Q4_K_M with a memory budget of 4 GiB
  and a 100 ms p99 latency target
- Status subresource: `loadedModels`, `activeRequests`, `tokensPerSecond` fields
  surfaced via `kubectl get zerfooinferenceservice`
- Integration with HPA: custom metrics from Prometheus (tokens/second,
  queue depth) drive horizontal scaling

### 4. Adaptive Batching: Hitting SLOs Under Variable Load (6 min)

- Why fixed batch sizes fail: over-batching blows the latency budget; under-
  batching wastes GPU cycles
- `serve/adaptive.Batcher`: dynamically adjusts batch size between `MinBatchSize`
  and `MaxBatchSize` (default 1–32) each dispatch cycle
- The control loop: exponential moving average of batch latency (alpha=0.3);
  scale up when queue depth >= current batch size and EMA is under target;
  scale down by 25% when EMA exceeds the latency SLO
- Queue timeout (`QueueTimeoutMS`, default 50 ms): dispatch whatever is queued
  rather than stall waiting for a full batch under low load
- Continuous batching via `serve/batcher.Scheduler`: ragged batches with zero
  padding; completed sequences evicted immediately, vacated slots filled from
  queue — 2x+ throughput over fixed batching at the same concurrency
- Live trace: show latency EMA and batch size oscillating in response to a load
  spike on Grafana

### 5. Multi-Model Serving with LRU GPU Eviction (4 min)

- `serve/multimodel.ModelManager`: single pod serves N models within a GPU
  memory budget (`MaxGPUMemoryBytes`)
- LRU eviction: when loading a new model would exceed the budget, the least-
  recently-used model is unloaded first — no manual eviction policy needed
- Preload list: warm the cache at startup for latency-critical models
- Kubernetes mapping: one `ZerfooInferenceService` with `multiModel: true` and
  a shared memory budget across the model list
- Practical sizing: a 24 GiB GPU can host Gemma 3 1B Q4_K_M (1 GiB), Llama 3
  8B Q4_K_M (5 GiB), and Mistral 7B Q4_K_M (4 GiB) simultaneously

### 6. Prefill/Decode Disaggregation for Heterogeneous Node Pools (3 min)

- Why disaggregate: prefill is compute-bound (large matrix multiply over the
  prompt); decode is memory-bandwidth-bound (one token at a time)
- `serve/disaggregated`: Gateway → PrefillWorker (gRPC KV block stream) →
  DecodeWorker (gRPC token stream)
- Kubernetes topology: prefill workers on compute-optimized nodes (A100/H100),
  decode workers on memory-optimized nodes (L40S), connected via cluster-local
  gRPC
- Cost savings: right-size each tier independently; scale prefill workers for
  prompt-heavy workloads without over-provisioning decode capacity

### 7. Edge Deployment on ARM64 (2 min)

- Cross-compiled static binaries: `GOOS=linux GOARCH=arm64 go build` — no
  runtime dependencies, no Python, no shared libraries required on node
- ARM NEON SIMD acceleration via Plan 9 assembly in `internal/xblas` — CPU
  inference at 8+ tok/s on edge hardware
- Kubernetes DaemonSet pattern for edge inference: one pod per node, model
  served from local NVMe
- Tested on: Raspberry Pi 5 (arm64), NVIDIA Jetson Orin Nano

### 8. Observability and Demo (1 min)

- Prometheus metrics: `requests_total`, `tokens_generated_total`,
  `tokens_per_second`, `request_latency_ms` histogram
- `kubectl port-forward` to Grafana: live tokens/second and queue depth dashboard
- Health endpoints: `/health/live`, `/health/ready` for Kubernetes probes
- Demo: `kubectl apply` a `ZerfooInferenceService`, watch the operator pull the
  GGUF model to a PVC, then `curl` the OpenAI-compatible completions endpoint

## Key Demos

1. **Operator deploy**: Apply a `ZerfooInferenceService` manifest in a KinD
   cluster; watch the operator create a Deployment, download the model, and
   surface status fields with `kubectl get` and `kubectl describe`.

2. **Adaptive batching under load**: Use `hey` or `vegeta` to ramp request
   concurrency; show on Grafana the latency EMA rising, batch size scaling up,
   then batch size scaling back down as load drops — all within the SLO window.

3. **Multi-model switching**: Issue requests for two different models from a
   single pod; show `UsedBytes` and `Loaded()` from the model manager changing
   in the metrics output; trigger an LRU eviction by requesting a third model
   that pushes the pod over its GPU budget.

4. **Edge binary**: Cross-compile to `linux/arm64` in one command, `scp` to a
   Raspberry Pi 5, `kubectl apply` a DaemonSet manifest, and show inference
   running at the edge with zero additional dependencies installed on the node.

## Speaker Bio

Daniel Ndungu is the founder of Feza, Inc and creator of Zerfoo, a
production-grade ML inference and training framework for Go. He has spent
the past several years building GPU-accelerated systems in pure Go — including
tensor libraries, CUDA kernel bindings via purego/dlopen, SIMD-optimized compute
paths, and a Kubernetes operator for declarative ML serving. Daniel is passionate
about making ML infrastructure a first-class citizen of the cloud-native
ecosystem, without sacrificing the operational simplicity Go developers expect.

- GitHub: github.com/zerfoo
- Company: feza.ai

## Submission Notes

This talk complements the GopherCon 2026 proposal ("Native ML Inference in Go:
Zero CGo, Maximum Performance") by shifting focus from the Go language internals
to the Kubernetes operational story. The audience overlap is small: GopherCon
attendees are primarily language-focused; KubeCon attendees are primarily
platform and infrastructure focused. The two proposals share performance numbers
and architecture context but cover distinct use cases and deployment patterns.

The demos are self-contained and can run in a KinD cluster on a laptop for the
operator and multi-model sections, with a pre-recorded fallback for the edge
ARM64 section.
