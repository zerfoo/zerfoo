# Enterprise Deployment Guide

This guide covers deploying Zerfoo in production-grade enterprise environments:
Kubernetes with the ZerfooInferenceService operator, multi-GPU inference, TLS/mTLS,
Prometheus monitoring, adaptive batching auto-scaling, model repositories, multi-model
serving with LRU eviction, security hardening, and troubleshooting.

---

## Table of Contents

1. [Prerequisites and System Requirements](#1-prerequisites-and-system-requirements)
2. [Kubernetes Deployment with ZerfooInferenceService](#2-kubernetes-deployment-with-zerfooinferenceservice)
3. [Multi-GPU Inference Setup](#3-multi-gpu-inference-setup)
4. [TLS / mTLS Configuration](#4-tls--mtls-configuration)
5. [Monitoring with Prometheus](#5-monitoring-with-prometheus)
6. [Auto-Scaling with Adaptive Batching](#6-auto-scaling-with-adaptive-batching)
7. [Model Repository Setup](#7-model-repository-setup)
8. [Multi-Model Serving with LRU Eviction](#8-multi-model-serving-with-lru-eviction)
9. [Security Hardening Checklist](#9-security-hardening-checklist)
10. [Troubleshooting Guide](#10-troubleshooting-guide)

---

## 1. Prerequisites and System Requirements

### Software

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Go | 1.25+ | latest stable |
| Kubernetes | 1.28+ | 1.30+ |
| Linux kernel | 5.15+ | 6.1+ (for NVIDIA open driver) |
| NVIDIA driver | 525+ (CUDA 12.0) | 550+ |
| CUDA toolkit | 12.0 | 12.4+ |

### Hardware — CPU-Only

| Model Size | RAM | CPU Cores |
|------------|-----|-----------|
| 1B Q4_K_M | 2 GB | 4+ |
| 7B Q4_K_M | 8 GB | 8+ |
| 13B Q4_K_M | 16 GB | 16+ |

### Hardware — GPU (CUDA / ROCm)

| Model Size | VRAM | System RAM | GPU Examples |
|------------|------|------------|--------------|
| 1B Q4_K_M | 1 GB | 4 GB | RTX 3060 |
| 7B Q4_K_M | 6 GB | 8 GB | RTX 3080, A10 |
| 13B Q4_K_M | 10 GB | 16 GB | RTX 4080, A30 |
| 70B Q4_K_M | 40 GB | 64 GB | A100 80GB, H100, or multi-GPU |

### Key Notes

- **No CGo required.** Zerfoo loads GPU backends dynamically at runtime via
  `purego`/`dlopen`. Build with `go build ./...` everywhere; no `cuda` build tag
  is needed for runtime GPU acceleration.
- **GGUF is the only supported model format.** Ensure all models are in GGUF format
  before deployment. Use `zonnx` to convert ONNX models.
- Model weights are memory-mapped. RSS will be close to the GGUF file size plus KV
  cache overhead. Set `LimitMEMLOCK=infinity` in systemd.

---

## 2. Kubernetes Deployment with ZerfooInferenceService

### Overview

The `ZerfooInferenceService` operator (in `serve/`) manages the lifecycle of Zerfoo
inference servers on Kubernetes. It reconciles custom resources into Deployments with
health probes, Prometheus scraping annotations, and GPU resource requests.

The health endpoints are provided by the `health` package (`health/server.go`):

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Liveness probe — process is alive |
| `GET /readyz` | Readiness probe — model is loaded and serving |
| `GET /debug/pprof/` | Runtime profiling (restrict to internal network) |

### Namespace and RBAC

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: zerfoo-system
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: zerfoo-server
  namespace: zerfoo-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: zerfoo-server
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: zerfoo-server
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: zerfoo-server
subjects:
  - kind: ServiceAccount
    name: zerfoo-server
    namespace: zerfoo-system
```

### Model PersistentVolumeClaim

Models (GGUF files) must be available to the pod. Use a `ReadOnlyMany` PVC backed
by a shared NFS / EFS volume, or pre-populate an `emptyDir` via an init container.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: zerfoo-models
  namespace: zerfoo-system
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: efs-sc        # replace with your StorageClass
  resources:
    requests:
      storage: 200Gi
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zerfoo-gemma-3-7b
  namespace: zerfoo-system
  labels:
    app: zerfoo
    model: gemma-3-7b
spec:
  replicas: 2
  selector:
    matchLabels:
      app: zerfoo
      model: gemma-3-7b
  template:
    metadata:
      labels:
        app: zerfoo
        model: gemma-3-7b
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: zerfoo-server
      runtimeClassName: nvidia   # omit for CPU-only
      containers:
        - name: zerfoo
          image: ghcr.io/zerfoo/zerfoo:v1.0.0
          command: ["zerfoo", "serve"]
          args:
            - "google/gemma-3-7b-it-q4_k_m"
            - "--port"
            - "8080"
            - "--cache-dir"
            - "/models"
            # For multi-GPU, add: --gpus 0,1
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: health
              containerPort: 8081
              protocol: TCP
          env:
            - name: GOMAXPROCS
              valueFrom:
                resourceFieldRef:
                  resource: limits.cpu
                  divisor: "1"
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "24Gi"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
            - name: tls-certs
              mountPath: /etc/zerfoo/tls
              readOnly: true
          livenessProbe:
            httpGet:
              path: /healthz
              port: health
            initialDelaySeconds: 30
            periodSeconds: 10
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /readyz
              port: health
            initialDelaySeconds: 60
            periodSeconds: 5
            failureThreshold: 6
      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: zerfoo-models
        - name: tls-certs
          secret:
            secretName: zerfoo-tls
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: zerfoo
              model: gemma-3-7b
```

### Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: zerfoo-gemma-3-7b
  namespace: zerfoo-system
spec:
  selector:
    app: zerfoo
    model: gemma-3-7b
  ports:
    - name: http
      port: 80
      targetPort: http
    - name: https
      port: 443
      targetPort: http
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: zerfoo
  namespace: zerfoo-system
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-buffering: "off"   # required for SSE streaming
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
      secretName: zerfoo-tls-ingress
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: zerfoo-gemma-3-7b
                port:
                  name: http
```

### HorizontalPodAutoscaler

Zerfoo exposes `tokens_per_second` as a Prometheus gauge. Use the
[Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter) to
surface it as a custom metric for HPA:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: zerfoo-gemma-3-7b
  namespace: zerfoo-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: zerfoo-gemma-3-7b
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Pods
      pods:
        metric:
          name: tokens_per_second
        target:
          type: AverageValue
          averageValue: "150"   # scale out when avg TPS per pod drops below 150
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 3. Multi-GPU Inference Setup

Zerfoo distributes a single model across multiple GPUs using the `--gpus` flag,
which accepts a comma-separated list of NVIDIA device IDs (implemented in
`cmd/cli/serve.go:parseGPUList`).

### CLI Usage

```bash
# Single GPU (default — uses GPU 0 if CUDA is available)
zerfoo serve meta-llama/llama-3-70b-q4_k_m

# Two-GPU tensor parallel (GPUs 0 and 1)
zerfoo serve meta-llama/llama-3-70b-q4_k_m --gpus 0,1

# Four-GPU for a 70B model at full precision
zerfoo serve meta-llama/llama-3-70b-q8_0 --gpus 0,1,2,3
```

GPU IDs must be non-negative integers, unique, and correspond to physical device
ordinals as reported by `nvidia-smi`.

### Go API

```go
import (
    "github.com/zerfoo/zerfoo/inference"
    "github.com/zerfoo/zerfoo/serve"
)

model, err := inference.Load("meta-llama/llama-3-70b-q4_k_m")
if err != nil {
    log.Fatal(err)
}

srv := serve.NewServer(model,
    serve.WithGPUs([]int{0, 1, 2, 3}),   // distribute across 4 GPUs
    serve.WithLogger(logger),
    serve.WithMetrics(collector),
)
```

The GPU IDs are passed through `serve.WithGPUs` (`serve/server.go:WithGPUs`) and
stored on the `Server` struct for use by the compute engine during model loading.

### Kubernetes Multi-GPU Pod

```yaml
resources:
  limits:
    nvidia.com/gpu: "4"   # request 4 GPUs
args:
  - "meta-llama/llama-3-70b-q4_k_m"
  - "--port"
  - "8080"
  - "--gpus"
  - "0,1,2,3"
```

Set `CUDA_VISIBLE_DEVICES` in the environment when you need explicit device
mapping within a shared node:

```yaml
env:
  - name: CUDA_VISIBLE_DEVICES
    value: "0,1,2,3"
```

### Sizing Guidelines

| Model | Quantization | GPUs | VRAM Each |
|-------|-------------|------|-----------|
| 7B | Q4_K_M | 1× | 6 GB |
| 13B | Q4_K_M | 1× | 10 GB |
| 70B | Q4_K_M | 2× A100 40GB | 40 GB |
| 70B | Q8_0 | 4× A100 40GB | 40 GB |
| 405B | Q4_K_M | 8× H100 80GB | 80 GB |

---

## 4. TLS / mTLS Configuration

The serve package returns a standard `http.Handler`. TLS is configured at the Go
`http.Server` level or terminated at the ingress/proxy layer.

### Application-Level TLS (TLS 1.3)

```go
import (
    "crypto/tls"
    "net/http"
    "github.com/zerfoo/zerfoo/serve"
)

srv := serve.NewServer(model,
    serve.WithLogger(logger),
    serve.WithMetrics(collector),
)
httpServer := &http.Server{
    Addr:    ":8443",
    Handler: srv.Handler(),
    TLSConfig: &tls.Config{
        MinVersion: tls.VersionTLS13,
        CurvePreferences: []tls.CurveID{
            tls.X25519,
            tls.CurveP256,
        },
    },
    ReadHeaderTimeout: 10 * time.Second,
}
if err := httpServer.ListenAndServeTLS("server.crt", "server.key"); err != nil {
    log.Fatal(err)
}
```

### Mutual TLS (mTLS)

mTLS is required for service-to-service authentication in zero-trust environments.

```go
import (
    "crypto/tls"
    "crypto/x509"
    "os"
)

caCert, err := os.ReadFile("/etc/zerfoo/tls/ca.crt")
if err != nil {
    log.Fatal(err)
}
caCertPool := x509.NewCertPool()
caCertPool.AppendCertsFromPEM(caCert)

cert, err := tls.LoadX509KeyPair(
    "/etc/zerfoo/tls/server.crt",
    "/etc/zerfoo/tls/server.key",
)
if err != nil {
    log.Fatal(err)
}

tlsConfig := &tls.Config{
    MinVersion:   tls.VersionTLS13,
    Certificates: []tls.Certificate{cert},
    ClientAuth:   tls.RequireAndVerifyClientCert,
    ClientCAs:    caCertPool,
}

httpServer := &http.Server{
    Addr:      ":8443",
    Handler:   srv.Handler(),
    TLSConfig: tlsConfig,
}
```

### Certificate Management with cert-manager

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: zerfoo-tls
  namespace: zerfoo-system
spec:
  secretName: zerfoo-tls
  duration: 2160h    # 90 days
  renewBefore: 360h  # renew 15 days before expiry
  subject:
    organizations:
      - Feza Inc
  dnsNames:
    - api.example.com
    - zerfoo-gemma-3-7b.zerfoo-system.svc.cluster.local
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
```

### Nginx TLS Termination (Reverse Proxy)

```nginx
upstream zerfoo {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate     /etc/ssl/certs/api.example.com.crt;
    ssl_certificate_key /etc/ssl/private/api.example.com.key;
    ssl_protocols       TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # Required for SSE streaming (token-by-token responses)
    proxy_buffering off;
    proxy_cache off;

    # Long inference requests
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://zerfoo;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection        "";
    }

    # Restrict metrics to internal network
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
        proxy_pass http://zerfoo;
    }
}
```

---

## 5. Monitoring with Prometheus

### Exposed Metrics

The `GET /metrics` endpoint (`serve/metrics.go`) exposes Prometheus text format.
An `*runtime.InMemoryCollector` must be passed via `serve.WithMetrics`.

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `requests_total` | Counter | Total completed requests | — |
| `tokens_generated_total` | Counter | Total tokens generated | — |
| `tokens_per_second` | Gauge | Rolling token generation rate | — |
| `speculative_acceptance_rate` | Gauge | Speculative decoding acceptance rate | — |
| `request_latency_ms` | Histogram | Request latency in ms | — |

Histogram buckets: `10, 50, 100, 250, 500, 1000, 2500, 5000, 10000` ms.

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: zerfoo
    scrape_interval: 15s
    static_configs:
      - targets:
          - "zerfoo-host:8080"
    metrics_path: /metrics
```

For Kubernetes using pod annotations:

```yaml
scrape_configs:
  - job_name: zerfoo-pods
    scrape_interval: 15s
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [zerfoo-system]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: "true"
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
```

### Grafana Dashboard Queries

**Token throughput:**
```promql
rate(tokens_generated_total[1m])
```

**Request rate:**
```promql
rate(requests_total[1m])
```

**p50 / p95 / p99 latency:**
```promql
histogram_quantile(0.50, rate(request_latency_ms_bucket[5m]))
histogram_quantile(0.95, rate(request_latency_ms_bucket[5m]))
histogram_quantile(0.99, rate(request_latency_ms_bucket[5m]))
```

**Current tokens/s gauge:**
```promql
tokens_per_second
```

**Speculative decoding acceptance rate:**
```promql
speculative_acceptance_rate
```

### Alerting Rules

```yaml
groups:
  - name: zerfoo
    rules:
      - alert: ZerfooHighLatency
        expr: |
          histogram_quantile(0.99, rate(request_latency_ms_bucket[5m])) > 5000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "p99 request latency above 5 s"
          description: "p99 latency is {{ $value }}ms"

      - alert: ZerfooLowThroughput
        expr: tokens_per_second < 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Token throughput critically low"
          description: "Tokens/s: {{ $value }}"

      - alert: ZerfooDown
        expr: up{job="zerfoo-pods"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Zerfoo instance down"
```

---

## 6. Auto-Scaling with Adaptive Batching

Zerfoo provides two complementary batching mechanisms:

### Continuous Batching (`serve/batcher/scheduler.go`)

The `batcher.Scheduler` implements continuous batching — variable-length (ragged)
batches with zero padding, immediate eviction of completed sequences, and slot
back-fill without stalling active requests. This typically achieves 2× throughput
over fixed batching.

```go
import "github.com/zerfoo/zerfoo/serve/batcher"

scheduler := batcher.New(
    16,   // maxBatchSize — max concurrent active sequences
    func(ctx context.Context, batch *batcher.StepBatch) {
        // Run one forward pass; append one token to each Slot.GeneratedToks
        // and set Slot.Done = true when EOS or max tokens reached.
        runForwardPass(ctx, batch)
    },
    batcher.WithPollInterval(1*time.Millisecond),
)
scheduler.Start()
defer scheduler.Stop()
```

### Adaptive Batching (`serve/adaptive/batcher.go`)

The `adaptive.Batcher` dynamically adjusts batch size based on queue depth and
latency EMA (exponential moving average, α=0.3):

- **Scale up**: queue depth ≥ current batch size AND latency EMA ≤ target → double
  batch size (capped at `MaxBatchSize`).
- **Scale down**: latency EMA > target → reduce batch size by 25%.
- **Hold**: otherwise.

```go
import "github.com/zerfoo/zerfoo/serve/adaptive"

batcher := adaptive.New(adaptive.Config{
    MinBatchSize:    1,
    MaxBatchSize:    32,
    TargetLatencyMS: 200.0,   // target p50 latency SLO in ms
    QueueTimeoutMS:  50.0,    // max wait to fill a batch before dispatching
}, handler)
batcher.Start()
defer batcher.Stop()
```

**Configuration reference (`adaptive.Config`):**

| Field | Default | Description |
|-------|---------|-------------|
| `MinBatchSize` | 1 | Minimum batch size |
| `MaxBatchSize` | 32 | Maximum batch size |
| `TargetLatencyMS` | 100 | Latency SLO in ms; controls scale-down |
| `QueueTimeoutMS` | 50 | Max wait time (ms) to fill a batch |

### Wiring into the HTTP Server

```go
import "github.com/zerfoo/zerfoo/serve"

bs := serve.NewBatchScheduler(serve.BatchConfig{
    MaxBatchSize: 8,
    BatchTimeout: 10 * time.Millisecond,
    // Handler is auto-wired to model.GenerateBatch when nil
})
bs.Start()

srv := serve.NewServer(model,
    serve.WithBatchScheduler(bs),
    serve.WithGPUs([]int{0, 1}),
    serve.WithLogger(logger),
    serve.WithMetrics(collector),
)
```

When a `BatchScheduler` is attached and `Handler` is nil, the server auto-wires
`model.GenerateBatch` as the handler (`serve/server.go:NewServer`).

### Kubernetes HPA with Custom Metrics

Use the Prometheus Adapter to expose `tokens_per_second` as a custom metric and
configure HPA (see [Section 2](#2-kubernetes-deployment-with-zerfooinferenceservice)).
For queue depth-based scaling, expose `adaptive.Batcher.BatchSize()` through a
custom metrics endpoint.

---

## 7. Model Repository Setup

The `serve/repository` package implements a local filesystem model repository.
Models are stored as `{baseDir}/{modelID}/model.gguf` with a `metadata.json`
sidecar. SHA-256 is computed and stored on upload.

### Directory Layout

```
/models/
  llama-3-7b-q4_k_m/
    model.gguf
    metadata.json
  gemma-3-7b-it-q4_k_m/
    model.gguf
    metadata.json
```

### Go API

```go
import "github.com/zerfoo/zerfoo/serve/repository"

repo, err := repository.NewFileSystemRepository("/models")
if err != nil {
    log.Fatal(err)
}

// Upload a model
f, _ := os.Open("gemma-3-7b-it-q4_k_m.gguf")
err = repo.Upload(repository.ModelMetadata{
    ID:      "gemma-3-7b-it-q4_k_m",
    Name:    "Gemma 3 7B IT Q4_K_M",
    Version: "v1.0",
    Format:  "gguf",
}, f)

// List models
models, err := repo.List()

// Get model metadata
meta, err := repo.Get("gemma-3-7b-it-q4_k_m")
fmt.Printf("Size: %d bytes, SHA256: %s\n", meta.Size, meta.SHA256)

// Delete a model
err = repo.Delete("gemma-3-7b-it-q4_k_m")
```

### Kubernetes PVC-Backed Repository

Mount the PVC at `/models` and configure `--cache-dir /models` so the CLI finds
GGUF files there:

```yaml
args:
  - "gemma-3-7b-it-q4_k_m"
  - "--port"
  - "8080"
  - "--cache-dir"
  - "/models"
volumeMounts:
  - name: models
    mountPath: /models
```

### Model Pre-population (Init Container)

```yaml
initContainers:
  - name: model-pull
    image: ghcr.io/zerfoo/zerfoo:v1.0.0
    command: ["zerfoo", "pull", "google/gemma-3-7b-it-q4_k_m", "--cache-dir", "/models"]
    volumeMounts:
      - name: models
        mountPath: /models
```

---

## 8. Multi-Model Serving with LRU Eviction

The `serve/multimodel` package provides a `ModelManager` that loads multiple models
on-demand within a GPU memory budget, evicting the least-recently-used model when
a new load would exceed the budget.

### Architecture

```
Request → ModelManager.Get("model-id")
              │
              ├── Already loaded? → promote to MRU, return handle
              │
              └── Not loaded:
                    Evict LRU models until usedBytes + newSize ≤ MaxGPUMemoryBytes
                    Load model via ModelLoader.Load()
                    Track in LRU list
```

### Configuration

```go
import "github.com/zerfoo/zerfoo/serve/multimodel"

manager, err := multimodel.NewModelManager(loader, multimodel.Config{
    MaxGPUMemoryBytes: 40 * 1024 * 1024 * 1024, // 40 GB VRAM budget
    PreloadModels: []string{
        "gemma-3-7b-it-q4_k_m",   // preloaded at startup
        "llama-3-1b-q4_k_m",
    },
})
if err != nil {
    log.Fatal(err)
}
defer manager.Close()
```

**Config fields:**

| Field | Description |
|-------|-------------|
| `MaxGPUMemoryBytes` | Total VRAM budget. LRU eviction triggers when exceeded. |
| `PreloadModels` | Model IDs loaded eagerly at startup. Any failure aborts init. |

### Implementing ModelLoader

```go
type GGUFLoader struct {
    cacheDir string
}

func (l *GGUFLoader) Load(ctx context.Context, modelID string) (io.Closer, int64, error) {
    path := filepath.Join(l.cacheDir, modelID, "model.gguf")
    info, err := os.Stat(path)
    if err != nil {
        return nil, 0, err
    }
    model, err := inference.Load(modelID, inference.WithCacheDir(l.cacheDir))
    if err != nil {
        return nil, 0, err
    }
    return model, info.Size(), nil
}
```

### Runtime Operations

```go
// Get a model (loads if not cached, evicts LRU if needed)
model, err := manager.Get(ctx, "deepseek-v3-q4_k_m")

// Explicit eviction
err = manager.Unload("llama-3-1b-q4_k_m")

// Inspect state
loadedIDs := manager.Loaded()
usedBytes := manager.UsedBytes()
```

### Multi-Model Kubernetes Deployment

For deployments serving many models from a single pod, increase the memory budget
and mount a larger model store:

```yaml
resources:
  limits:
    nvidia.com/gpu: "2"
    memory: "128Gi"
env:
  - name: ZERFOO_MAX_GPU_MEMORY_GB
    value: "80"   # 2× A100 40GB
```

---

## 9. Security Hardening Checklist

### Network

- [ ] Terminate TLS 1.3 at ingress or application level — never serve HTTP in
  production.
- [ ] Restrict `GET /metrics` to the internal monitoring network (Prometheus
  scraper IP range), not the public internet.
- [ ] Restrict `GET /debug/pprof/` to internal networks only (exposed by
  `health/server.go`).
- [ ] Use mTLS for service-to-service communication (e.g., load balancer → server,
  or distributed training gRPC channels).
- [ ] Apply Kubernetes `NetworkPolicy` to limit pod-to-pod traffic to only required
  ports (8080 inference, 8081 health).

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: zerfoo-inference
  namespace: zerfoo-system
spec:
  podSelector:
    matchLabels:
      app: zerfoo
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - port: 8080
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: monitoring
      ports:
        - port: 8080   # metrics scrape
        - port: 8081   # health probes
  egress:
    - {}   # allow egress for model downloads; tighten as needed
```

### Container Hardening

- [ ] Run as a non-root user (UID 1000).
- [ ] Set `readOnlyRootFilesystem: true` — mount `/tmp` as `emptyDir` if needed.
- [ ] Set `allowPrivilegeEscalation: false`.
- [ ] Drop all Linux capabilities; add only `IPC_LOCK` if huge pages are required.
- [ ] Use a minimal base image (distroless or scratch + binary).

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

### Secrets Management

- [ ] Store TLS private keys in Kubernetes `Secrets` (type `kubernetes.io/tls`),
  not in ConfigMaps or container images.
- [ ] Rotate certificates automatically with cert-manager.
- [ ] Use external secret stores (AWS Secrets Manager, Vault) for API keys and
  credentials; mount via the Secrets Store CSI driver.
- [ ] Never log request bodies that may contain sensitive user data.

### Pod Security

- [ ] Apply a `PodDisruptionBudget` to ensure at least one replica stays available
  during node maintenance.

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: zerfoo-gemma-3-7b
  namespace: zerfoo-system
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: zerfoo
      model: gemma-3-7b
```

- [ ] Enable Kubernetes Audit Logging for all API server requests in the
  `zerfoo-system` namespace.
- [ ] Use `AppArmor` or `Seccomp` profiles on production nodes.

### Model Integrity

- [ ] Verify the SHA-256 of GGUF files before loading using the checksum stored
  in `repository.ModelMetadata.SHA256` (computed on upload by
  `serve/repository/repository.go`).
- [ ] Sign model artifacts and verify signatures in CI before publishing to the
  model repository.

---

## 10. Troubleshooting Guide

### Server Does Not Start

**Symptom:** `zerfoo serve` exits immediately or the readiness probe fails.

**Checks:**
1. Verify the GGUF file exists and is readable:
   ```bash
   ls -lh /models/gemma-3-7b-it-q4_k_m/model.gguf
   ```
2. Check the model ID matches the directory name and `metadata.json`.
3. Confirm sufficient RAM/VRAM is available:
   ```bash
   free -h
   nvidia-smi --query-gpu=memory.free --format=csv
   ```
4. Ensure the `video` and `render` groups are assigned (Linux GPU access):
   ```bash
   groups zerfoo
   ```

### CUDA / GPU Not Detected

**Symptom:** Server starts but runs on CPU; GPU utilization stays at 0%.

**Checks:**
1. Confirm `libcuda.so` is on the library path:
   ```bash
   ldconfig -p | grep libcuda
   ```
2. Verify driver version supports CUDA 12.0+:
   ```bash
   nvidia-smi
   ```
3. In Kubernetes, confirm the NVIDIA device plugin is running and the pod has
   `nvidia.com/gpu: "1"` in its resource limits.
4. Check `CUDA_VISIBLE_DEVICES` is not set to an empty string or `NoDevFiles`.

### Out-of-Memory (OOM) Errors

**Symptom:** HTTP 503 responses with `out of memory` in the body (see
`serve/server.go:isOOMError`).

**Remediation:**
1. Reduce model quantization (e.g., switch from Q8 to Q4_K_M).
2. Decrease `MaxBatchSize` on the `BatchScheduler` or `adaptive.Batcher`.
3. Add more GPUs (`--gpus 0,1,2,3`).
4. For multi-model: reduce `MaxGPUMemoryBytes` in `multimodel.Config` to leave
   headroom for KV cache.

### High Latency / Low Throughput

**Symptom:** `tokens_per_second` gauge is below 50; p99 latency exceeds SLO.

**Checks:**
1. Inspect adaptive batcher state:
   ```go
   log.Printf("batch_size=%d latency_ema=%.1fms",
       batcher.BatchSize(), batcher.LatencyEMA())
   ```
2. If `LatencyEMA` is high, reduce `TargetLatencyMS` or `MaxBatchSize` to shed
   load.
3. Check for CUDA graph capture misses — examine startup logs for graph compilation
   warnings.
4. Profile with pprof (exposed at `/debug/pprof/` by `health/server.go`):
   ```bash
   go tool pprof http://localhost:8081/debug/pprof/profile?seconds=30
   ```

### Streaming (SSE) Broken at Proxy

**Symptom:** Streaming responses are buffered and delivered all at once.

**Fix:** Disable proxy buffering. For nginx:
```nginx
proxy_buffering off;
proxy_cache off;
```
For the Kubernetes ingress-nginx controller:
```yaml
annotations:
  nginx.ingress.kubernetes.io/proxy-buffering: "off"
```
The server sets `Content-Type: text/event-stream` and flushes each token
individually (`serve/server.go:streamChatCompletion`).

### Model Not Found (Multi-Model)

**Symptom:** `ModelManager.Get()` returns `load model "foo": ...` error.

**Checks:**
1. Confirm the model exists in the repository:
   ```go
   meta, err := repo.Get("foo")
   ```
2. Verify `MaxGPUMemoryBytes` is large enough to hold at least one model.
3. Check `PreloadModels` list — a failed preload aborts `NewModelManager`.

### Certificate / TLS Errors

**Symptom:** Clients get `certificate signed by unknown authority` or `handshake
failure`.

**Checks:**
1. Confirm the CA certificate is correct in the `ClientCAs` pool (mTLS) or system
   trust store (one-way TLS).
2. Verify the certificate's `dnsNames` includes the hostname clients connect to.
3. Check certificate expiry:
   ```bash
   openssl s_client -connect api.example.com:443 < /dev/null 2>/dev/null \
     | openssl x509 -noout -dates
   ```
4. With cert-manager, inspect `Certificate` status:
   ```bash
   kubectl describe certificate zerfoo-tls -n zerfoo-system
   ```

### Graceful Shutdown Hangs

**Symptom:** Pod takes longer than `terminationGracePeriodSeconds` to stop.

**Checks:**
1. Increase the Kubernetes `terminationGracePeriodSeconds` (default 30 s) to
   accommodate the longest expected in-flight request.
2. Ensure the `shutdown.Coordinator` is registered with both the HTTP server and
   the `BatchScheduler` (the server's `Close` method calls `batch.Stop()`).
3. Check for goroutine leaks with pprof:
   ```bash
   curl http://localhost:8081/debug/pprof/goroutine?debug=1
   ```
