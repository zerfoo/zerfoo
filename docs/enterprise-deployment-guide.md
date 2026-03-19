# Enterprise Deployment Guide

This guide covers deploying Zerfoo in production Kubernetes environments with
autoscaling, monitoring, security hardening, and operational best practices.

For single-node deployments with systemd and nginx, see
[Production Deployment Guide](production-deployment.md).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Scaling](#scaling)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Security](#security)
7. [High Availability](#high-availability)
8. [Model Management](#model-management)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Software

| Component | Minimum Version | Notes |
|-----------|----------------|-------|
| Go | 1.25+ | Required only for building from source |
| Kubernetes | 1.28+ | `autoscaling/v2` API required for HPA |
| Helm | 3.12+ | For chart-based deployment |
| NVIDIA GPU Operator | 24.3+ | Required for GPU nodes (installs drivers, device plugin, container toolkit) |
| CUDA | 12.x | Loaded dynamically at runtime via purego -- no CGo needed |

### Cluster Requirements

- **GPU nodes**: NVIDIA GPU Operator installed, nodes labeled with
  `nvidia.com/gpu.present=true`. Zerfoo loads CUDA at runtime via `dlopen` --
  no special build flags are needed.
- **Storage**: A `PersistentVolume` provisioner (e.g., `local-path`, EBS CSI,
  GCE PD) for model weight storage.
- **Container registry access**: Images are published to `ghcr.io/zerfoo/zerfoo`.
  Configure `imagePullSecrets` if your cluster requires authentication.

---

## Installation

### Pre-built Container Images

```bash
# Pull the latest release image
docker pull ghcr.io/zerfoo/zerfoo:latest

# Or pin to a specific version
docker pull ghcr.io/zerfoo/zerfoo:0.1.0
```

The container image includes the `zerfoo` binary with all CLI commands
(`serve`, `run`, `pull`, `predict`, `tokenize`, `worker`).

### Building from Source

```bash
# CPU-only build (zero CGo, compiles everywhere)
go build -o zerfoo ./cmd/zerfoo

# Build container image
docker build -t ghcr.io/zerfoo/zerfoo:custom .
```

No build tags are required for CPU-only operation. GPU acceleration is loaded
dynamically at runtime when CUDA libraries are available on the host.

---

## Kubernetes Deployment

### Helm Chart

Zerfoo ships a Helm chart at `deploy/helm/zerfoo/`.

#### Install

```bash
helm install zerfoo deploy/helm/zerfoo/ \
  --namespace zerfoo \
  --create-namespace \
  --set model.name="google/gemma-3-1b" \
  --set model.quantization="Q4_K_M"
```

#### Key Values

| Value | Default | Description |
|-------|---------|-------------|
| `replicaCount` | `1` | Number of inference pods |
| `image.repository` | `ghcr.io/zerfoo/zerfoo` | Container image |
| `image.tag` | Chart `appVersion` | Image tag |
| `model.name` | `""` | Model ID (e.g., `google/gemma-3-1b`) |
| `model.quantization` | `Q4_K_M` | Quantization level |
| `model.cacheDir` | `/models` | Model cache directory inside the container |
| `server.port` | `8080` | Inference server listen port |
| `resources.requests.cpu` | `2` | CPU request |
| `resources.limits.memory` | `8Gi` | Memory limit |
| `autoscaling.enabled` | `false` | Enable HPA |
| `autoscaling.minReplicas` | `1` | Minimum replicas |
| `autoscaling.maxReplicas` | `10` | Maximum replicas |
| `ingress.enabled` | `false` | Enable Ingress resource |

#### GPU-Enabled Deployment

```yaml
# values-gpu.yaml
resources:
  requests:
    cpu: "4"
    memory: 16Gi
  limits:
    cpu: "8"
    memory: 32Gi
    nvidia.com/gpu: "1"

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule

nodeSelector:
  nvidia.com/gpu.present: "true"

volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: zerfoo-models

volumeMounts:
  - name: model-storage
    mountPath: /models
```

```bash
helm install zerfoo deploy/helm/zerfoo/ \
  --namespace zerfoo \
  --create-namespace \
  -f values-gpu.yaml \
  --set model.name="meta-llama/Llama-3-8B"
```

### Zerfoo Operator (Custom Resource)

The Zerfoo Kubernetes operator (`serve/operator/`) manages inference services
through the `ZerfooInferenceService` custom resource. It reconciles Deployments,
Services, and HorizontalPodAutoscalers automatically.

#### Custom Resource Definition

```yaml
apiVersion: zerfoo.feza.ai/v1
kind: ZerfooInferenceService
metadata:
  name: llama3-8b
  namespace: zerfoo
spec:
  modelRef: "meta-llama/Llama-3-8B-Q4_K_M"
  replicas: 3
  minReplicas: 2
  maxReplicas: 10
  resources:
    cpu: "4"
    memory: "16Gi"
    gpuMemory: "24Gi"
  healthCheck:
    path: "/health"
    interval: 10s
    timeout: 5s
```

The operator creates the following Kubernetes resources:

| Resource | Naming Convention | Purpose |
|----------|------------------|---------|
| Deployment | `<name>-primary` | Primary inference pods |
| Service | `<name>-svc` | ClusterIP service with selector `app: <name>` |
| HPA | `<name>-hpa` | Autoscaler (when `minReplicas` and `maxReplicas` are set) |

#### Canary Deployments

The operator supports canary deployments with weighted traffic splitting:

```yaml
apiVersion: zerfoo.feza.ai/v1
kind: ZerfooInferenceService
metadata:
  name: llama3-8b
  namespace: zerfoo
spec:
  modelRef: "meta-llama/Llama-3-8B-Q4_K_M"
  replicas: 3
  minReplicas: 2
  maxReplicas: 10
  resources:
    cpu: "4"
    memory: "16Gi"
    gpuMemory: "24Gi"
  healthCheck:
    path: "/health"
    interval: 10s
    timeout: 5s
  canary:
    modelRef: "meta-llama/Llama-3-8B-Q8_0"
    weight: 10  # 10% traffic to canary
```

This creates a `<name>-canary` Deployment alongside the primary, with the
Service distributing traffic according to the configured weights (90/10 in
this example).

---

## Scaling

### Horizontal Pod Autoscaling

Enable HPA via the Helm chart:

```yaml
# values-autoscaling.yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

This creates a `HorizontalPodAutoscaler` using the `autoscaling/v2` API:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: zerfoo
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: zerfoo
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 80
```

For GPU workloads, consider scaling on custom metrics instead of CPU. Use
the Prometheus adapter to expose `tokens_per_second` or `request_latency_ms`
as Kubernetes custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: zerfoo
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: zerfoo
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Pods
      pods:
        metric:
          name: request_latency_ms_p99
        target:
          type: AverageValue
          averageValue: "500"
```

### Multi-GPU Deployment

Zerfoo supports distributing a single model across multiple GPUs via the
`--gpus` flag:

```bash
zerfoo serve meta-llama/Llama-3-70B --gpus 0,1,2,3
```

In Kubernetes, request multiple GPUs per pod for large models:

```yaml
resources:
  limits:
    nvidia.com/gpu: "4"

env:
  - name: ZERFOO_GPUS
    value: "0,1,2,3"
```

### Disaggregated Prefill/Decode

For high-throughput deployments, Zerfoo supports disaggregated serving where
prefill and decode phases run on separate worker pools. The `serve/disaggregated/`
package provides:

- **Gateway**: Routes requests, distributes KV blocks between prefill and decode
  workers using least-loaded scheduling.
- **Prefill workers**: Handle prompt processing (compute-intensive, benefits from
  high-bandwidth GPUs).
- **Decode workers**: Handle autoregressive token generation (memory-bandwidth
  bound).

Configure the gateway with separate worker pools:

```yaml
# Prefill workers (compute-optimized GPU nodes)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zerfoo-prefill
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: zerfoo
          image: ghcr.io/zerfoo/zerfoo:latest
          args: ["worker", "--role", "prefill", "--port", "50051"]
          resources:
            limits:
              nvidia.com/gpu: "1"
---
# Decode workers (memory-optimized GPU nodes)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zerfoo-decode
spec:
  replicas: 4
  template:
    spec:
      containers:
        - name: zerfoo
          image: ghcr.io/zerfoo/zerfoo:latest
          args: ["worker", "--role", "decode", "--port", "50052"]
          resources:
            limits:
              nvidia.com/gpu: "1"
```

---

## Monitoring and Observability

### Prometheus Metrics

Every Zerfoo server exposes a `GET /metrics` endpoint in Prometheus text
exposition format. The following metrics are available:

| Metric | Type | Description |
|--------|------|-------------|
| `requests_total` | Counter | Total number of completed inference requests |
| `tokens_generated_total` | Counter | Total tokens generated across all requests |
| `tokens_per_second` | Gauge | Rolling token generation rate (tok/s) |
| `speculative_acceptance_rate` | Gauge | Speculative decoding acceptance rate (when enabled) |
| `request_latency_ms` | Histogram | Request latency distribution (buckets: 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000 ms) |

### Prometheus Scrape Configuration

```yaml
# prometheus.yaml
scrape_configs:
  - job_name: zerfoo
    scrape_interval: 15s
    metrics_path: /metrics
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [zerfoo]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app_kubernetes_io_name]
        regex: zerfoo
        action: keep
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        target_label: __address__
        regex: (.+)
        replacement: ${1}
```

Or use a `PodMonitor` if you have the Prometheus Operator:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: zerfoo
  namespace: zerfoo
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: zerfoo
  podMetricsEndpoints:
    - port: http
      path: /metrics
      interval: 15s
```

### Grafana Dashboard

Recommended panels for a Zerfoo operations dashboard:

| Panel | PromQL Query | Description |
|-------|-------------|-------------|
| Request Rate | `rate(requests_total[5m])` | Requests per second |
| Token Throughput | `rate(tokens_generated_total[5m])` | Tokens per second (cluster-wide) |
| Tokens/s per Pod | `tokens_per_second` | Per-pod generation rate |
| P50 Latency | `histogram_quantile(0.5, rate(request_latency_ms_bucket[5m]))` | Median request latency |
| P99 Latency | `histogram_quantile(0.99, rate(request_latency_ms_bucket[5m]))` | Tail latency |
| Speculative Acceptance | `speculative_acceptance_rate` | Draft model acceptance rate |
| GPU Memory | `nvidia_gpu_memory_used_bytes` (from DCGM exporter) | GPU memory utilization |

### Alerting Rules

```yaml
# alerts.yaml
groups:
  - name: zerfoo
    rules:
      - alert: ZerfooHighLatency
        expr: histogram_quantile(0.99, rate(request_latency_ms_bucket[5m])) > 5000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Zerfoo P99 latency exceeds 5s"

      - alert: ZerfooNoRequests
        expr: rate(requests_total[5m]) == 0
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Zerfoo is not processing any requests"

      - alert: ZerfooOOM
        expr: increase(requests_total{status="503"}[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Zerfoo returning 503 (possible OOM)"
```

### Health Checks

The Helm chart configures liveness and readiness probes by default:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: http
  initialDelaySeconds: 10
  periodSeconds: 5
```

Adjust `initialDelaySeconds` based on model size -- larger models take longer
to load into GPU memory. A 70B model on a single GPU may need 60-120 seconds.

### Structured Logging

Zerfoo logs every request with structured fields:

```
method=POST path=/v1/chat/completions model=gemma-3-1b prompt_tokens=0 completion_tokens=0 latency_ms=142 status_code=200
```

Collect logs via your cluster's logging stack (Fluentd, Loki, CloudWatch).
Filter on `status_code >= 500` for error alerting.

---

## Security

### TLS Termination

**Option 1: Ingress TLS (recommended for most deployments)**

Use the Helm chart's built-in Ingress with TLS:

```yaml
# values-tls.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: inference.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: zerfoo-tls
      hosts:
        - inference.example.com
```

**Option 2: Application-level TLS**

For end-to-end encryption without an Ingress controller, embed TLS directly
in the server. The `serve.Server` returns a standard `http.Handler`, so wrap
it with Go's TLS support:

```go
srv := serve.NewServer(model,
    serve.WithLogger(logger),
    serve.WithMetrics(collector),
)
httpServer := &http.Server{
    Addr:    ":8443",
    Handler: srv.Handler(),
    TLSConfig: &tls.Config{
        MinVersion: tls.VersionTLS13,
    },
}
httpServer.ListenAndServeTLS("server.crt", "server.key")
```

### Mutual TLS (mTLS)

For service-to-service authentication, Zerfoo's distributed training layer
(`distributed/tlsconfig.go`) provides mTLS support with:

- CA certificate verification for all peers
- Client certificate authentication (`tls.RequireAndVerifyClientCert`)
- Minimum TLS 1.2

For the HTTP serving layer, configure mTLS at the application level:

```go
caCert, _ := os.ReadFile("ca.crt")
caCertPool := x509.NewCertPool()
caCertPool.AppendCertsFromPEM(caCert)

tlsConfig := &tls.Config{
    MinVersion: tls.VersionTLS13,
    ClientAuth: tls.RequireAndVerifyClientCert,
    ClientCAs:  caCertPool,
}
```

Or use a service mesh (Istio, Linkerd) for transparent mTLS between all pods.

### Network Policies

Restrict traffic to only what is needed:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: zerfoo-inference
  namespace: zerfoo
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: zerfoo
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow traffic from the Ingress controller
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
    # Allow Prometheus scraping
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # Allow DNS resolution
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    # Allow model downloads (if pulling at startup)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

### RBAC

The Helm chart creates a dedicated ServiceAccount with
`automountServiceAccountToken: false` by default. This prevents pods from
accessing the Kubernetes API unless explicitly needed.

For the operator, create a scoped Role:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: zerfoo-operator
  namespace: zerfoo
rules:
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "create", "update", "delete"]
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["get", "list", "create", "update"]
  - apiGroups: ["autoscaling"]
    resources: ["horizontalpodautoscalers"]
    verbs: ["get", "list", "create", "update"]
  - apiGroups: ["zerfoo.feza.ai"]
    resources: ["zerfooinferenceservices"]
    verbs: ["get", "list", "watch", "update"]
```

### Secrets Management

Store model repository credentials and TLS certificates as Kubernetes Secrets:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: zerfoo-model-credentials
  namespace: zerfoo
type: Opaque
stringData:
  HF_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxxx"
```

Reference in the Deployment:

```yaml
env:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: zerfoo-model-credentials
        key: HF_TOKEN
```

For production, use an external secrets operator (e.g., External Secrets
Operator, HashiCorp Vault) to inject secrets from your secrets manager.

---

## High Availability

### Multi-Replica Deployment

Run at least 2 replicas for production workloads:

```yaml
replicaCount: 3

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
```

### Pod Disruption Budget

Prevent voluntary disruptions from taking down all replicas simultaneously:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: zerfoo
  namespace: zerfoo
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: zerfoo
```

### Pod Anti-Affinity

Spread inference pods across nodes to survive node failures:

```yaml
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app.kubernetes.io/name
                operator: In
                values: [zerfoo]
          topologyKey: kubernetes.io/hostname
```

### Load Balancing

The Helm chart creates a `ClusterIP` Service by default. For external access,
use an Ingress controller or change the service type:

```yaml
service:
  type: LoadBalancer
  port: 80
```

For streaming (SSE) support, ensure your load balancer or Ingress controller
disables response buffering. With nginx:

```yaml
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
```

### Graceful Shutdown

Zerfoo uses a `shutdown.Coordinator` that handles `SIGINT` and `SIGTERM`:

1. The HTTP server stops accepting new connections.
2. In-flight requests complete (respects Kubernetes `terminationGracePeriodSeconds`).
3. The batch scheduler (if attached) is drained.
4. The model is closed and GPU memory is released.

Set an appropriate termination grace period for your model size:

```yaml
podAnnotations:
  terminationGracePeriodSeconds: "60"
```

---

## Model Management

### Model Format

Zerfoo uses **GGUF** as its sole model format. GGUF files are memory-mapped
for efficient loading and support quantized formats (Q4_K_M, Q8_0, F16, F32).

### Model Loading

Models are loaded at startup via the `model.name` Helm value. The server runs
`zerfoo serve <model-id>` which:

1. Resolves the model ID to a GGUF file (local path or HuggingFace download).
2. Memory-maps the model weights.
3. Builds the computation graph for the model architecture.
4. Compiles the graph (with optional CUDA graph capture on GPU).

### Persistent Model Storage

Use a PersistentVolumeClaim to avoid re-downloading models on pod restart:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zerfoo-models
  namespace: zerfoo
spec:
  accessModes: [ReadWriteMany]  # ReadWriteOnce if single-node
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi
```

Reference in Helm values:

```yaml
volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: zerfoo-models

volumeMounts:
  - name: model-storage
    mountPath: /models
```

### Multi-Model Serving

The `serve/multimodel/` package provides a `ModelManager` that loads and
unloads models on demand with LRU eviction when GPU memory budget is exceeded.
Configure `MaxGPUMemoryBytes` to match your available VRAM and optionally
`PreloadModels` for models that should always be warm.

### Model Version Registry

The `serve/registry/` package provides a bbolt-backed model version registry
for tracking, activating, and managing model versions. It supports:

- Registering model versions with metadata and performance metrics
- A/B testing between model versions (`serve/registry/ab_router.go`)
- Canary deployments with traffic splitting (`serve/registry/canary.go`)
- Shadow mode for comparing model outputs without affecting production traffic
  (`serve/registry/shadow.go`)

### Supported Architectures

| Architecture | Status | Notes |
|-------------|--------|-------|
| Llama 3 | Production | RoPE theta=500K |
| Gemma 3 | Production | Tied embeddings, QK norms, logit softcap |
| Mistral | Production | Sliding window attention |
| Qwen 2 | Production | Attention bias, RoPE theta=1M |
| Phi 3/4 | Production | Partial rotary factor |
| DeepSeek V3 | Production | MLA, MoE |

---

## Performance Tuning

### Quantization

Choose quantization based on your latency/quality trade-off:

| Quantization | Memory | Quality | Speed |
|-------------|--------|---------|-------|
| F32 | 4x | Baseline | Slowest |
| F16 | 2x | Near-lossless | Moderate |
| Q8_0 | 1x | Minimal degradation | Fast |
| Q4_K_M | 0.5x | Good quality/size ratio | Fastest |

Set via Helm:

```yaml
model:
  quantization: "Q4_K_M"
```

### Batch Scheduling

For throughput-oriented workloads (non-streaming), attach a `BatchScheduler`
to group requests and improve GPU utilization. The adaptive batcher
(`serve/adaptive/`) dynamically adjusts batch size based on queue depth and
latency targets:

| Config | Default | Description |
|--------|---------|-------------|
| `MinBatchSize` | 1 | Smallest batch to form |
| `MaxBatchSize` | 32 | Largest batch to form |
| `TargetLatencyMS` | 100 | Latency SLO in milliseconds |
| `QueueTimeoutMS` | 50 | Max wait time to fill a batch |

### CUDA Graph Capture

On GPU, Zerfoo captures the inference computation graph as a CUDA graph on
first execution, then replays it for subsequent requests. This eliminates
kernel launch overhead and achieves up to 99.5% instruction coverage on the
GGUF inference path.

No configuration is needed -- CUDA graph capture is automatic when a GPU is
available.

### Speculative Decoding

Enable speculative decoding with a small draft model to increase throughput
for large models:

```go
srv := serve.NewServer(targetModel,
    serve.WithDraftModel(draftModel),
)
```

Monitor the `speculative_acceptance_rate` metric to verify the draft model
is effective. Acceptance rates above 70% typically yield significant speedups.

### Resource Sizing

#### CPU-Only

| Model Size | RAM | CPU Cores | Notes |
|-----------|-----|-----------|-------|
| 1B (Q4_K_M) | 2 GB | 4+ | Development and light traffic |
| 3B (Q4_K_M) | 4 GB | 8+ | Moderate throughput |
| 7B (Q4_K_M) | 8 GB | 8+ | Recommended minimum for production |

#### GPU (CUDA)

| Model Size | VRAM | System RAM | Notes |
|-----------|------|------------|-------|
| 1B (Q4_K_M) | 1 GB | 4 GB | Single consumer GPU |
| 7B (Q4_K_M) | 6 GB | 8 GB | RTX 3060 or better |
| 13B (Q4_K_M) | 10 GB | 16 GB | RTX 3080/4080 or better |
| 70B (Q4_K_M) | 40 GB | 64 GB | A100/H100 or multi-GPU with `--gpus` |

### Memory

Model weights are memory-mapped. Pod RSS will be close to the GGUF file size
plus KV cache overhead. Set resource limits accordingly and avoid memory
overcommit on GPU nodes.

---

## Troubleshooting

### Common Issues

**Pod stuck in `Pending`**

- Check GPU availability: `kubectl describe node <node> | grep nvidia.com/gpu`
- Verify NVIDIA GPU Operator is running: `kubectl get pods -n gpu-operator`
- Check PVC binding: `kubectl get pvc -n zerfoo`

**Pod in `CrashLoopBackOff`**

- Check logs: `kubectl logs -n zerfoo deploy/zerfoo --previous`
- Common causes:
  - Model not found (invalid `model.name`)
  - Insufficient memory (increase `resources.limits.memory`)
  - GPU out of memory (use a smaller quantization or more GPUs)

**503 Service Unavailable**

Zerfoo returns 503 for out-of-memory errors during inference. Solutions:
- Reduce concurrent requests (lower HPA `maxReplicas` target)
- Use a smaller quantization (Q4_K_M instead of Q8_0)
- Add more GPU memory (multi-GPU with `--gpus`)

**Slow model loading**

- Large models take time to download and memory-map. Use a PVC to persist
  models across pod restarts.
- Increase `livenessProbe.initialDelaySeconds` to prevent Kubernetes from
  killing pods during model loading.

**Streaming not working through Ingress**

Ensure response buffering is disabled on your Ingress controller:

```yaml
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-buffering: "off"
```

### Debug Logging

Set the log level via environment variable:

```yaml
env:
  - name: ZERFOO_LOG_LEVEL
    value: "debug"
```

### Useful kubectl Commands

```bash
# Check pod status and events
kubectl get pods -n zerfoo -o wide
kubectl describe pod -n zerfoo <pod-name>

# Stream logs
kubectl logs -n zerfoo deploy/zerfoo -f

# Check metrics endpoint
kubectl port-forward -n zerfoo svc/zerfoo 8080:80
curl localhost:8080/metrics

# Check model info
curl localhost:8080/v1/models

# Test inference
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-1b", "messages": [{"role": "user", "content": "Hello"}]}'

# Check HPA status
kubectl get hpa -n zerfoo
```
