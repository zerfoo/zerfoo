# Enterprise Deployment Guide

This guide covers deploying Zerfoo in production-grade enterprise environments:
Kubernetes with Helm and the ZerfooInferenceService operator, multi-GPU inference,
TLS/mTLS, Prometheus monitoring, adaptive batching, auto-scaling, model repositories,
multi-model serving with LRU eviction, security hardening, and troubleshooting.

For single-node deployments with systemd and nginx, see
[Production Deployment Guide](production-deployment.md).

---

## Table of Contents

1. [Prerequisites and System Requirements](#1-prerequisites-and-system-requirements)
2. [Installation](#2-installation)
3. [Kubernetes Deployment](#3-kubernetes-deployment)
4. [Scaling](#4-scaling)
5. [Multi-GPU Inference](#5-multi-gpu-inference)
6. [TLS / mTLS Configuration](#6-tls--mtls-configuration)
7. [Monitoring and Observability](#7-monitoring-and-observability)
8. [Model Management](#8-model-management)
9. [Multi-Model Serving with LRU Eviction](#9-multi-model-serving-with-lru-eviction)
10. [Performance Tuning](#10-performance-tuning)
11. [Security Hardening](#11-security-hardening)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites and System Requirements

### Software

| Requirement | Minimum | Recommended | Notes |
|-------------|---------|-------------|-------|
| Go | 1.25+ | latest stable | Required only for building from source |
| Kubernetes | 1.28+ | 1.30+ | `autoscaling/v2` API required for HPA |
| Helm | 3.12+ | latest stable | For chart-based deployment |
| Linux kernel | 5.15+ | 6.1+ | 6.1+ recommended for NVIDIA open driver |
| NVIDIA GPU Operator | 24.3+ | latest | Installs drivers, device plugin, container toolkit |
| NVIDIA driver | 525+ | 550+ | CUDA 12.0+ support |
| CUDA | 12.0 | 12.4+ | Loaded dynamically at runtime via purego -- no CGo needed |

### Hardware -- CPU-Only

| Model Size | RAM | CPU Cores | Notes |
|------------|-----|-----------|-------|
| 1B Q4_K_M | 2 GB | 4+ | Development and light traffic |
| 3B Q4_K_M | 4 GB | 8+ | Moderate throughput |
| 7B Q4_K_M | 8 GB | 8+ | Recommended minimum for production |
| 13B Q4_K_M | 16 GB | 16+ | |

### Hardware -- GPU (CUDA / ROCm)

| Model Size | VRAM | System RAM | GPU Examples |
|------------|------|------------|--------------|
| 1B Q4_K_M | 1 GB | 4 GB | RTX 3060 |
| 7B Q4_K_M | 6 GB | 8 GB | RTX 3080, A10 |
| 13B Q4_K_M | 10 GB | 16 GB | RTX 4080, A30 |
| 70B Q4_K_M | 40 GB | 64 GB | A100 80GB, H100, or multi-GPU |

### Cluster Requirements

- **GPU nodes**: NVIDIA GPU Operator installed, nodes labeled with
  `nvidia.com/gpu.present=true`. Zerfoo loads CUDA at runtime via `dlopen` --
  no special build flags are needed.
- **Storage**: A `PersistentVolume` provisioner (e.g., `local-path`, EBS CSI,
  GCE PD) for model weight storage.
- **Container registry access**: Images are published to `ghcr.io/zerfoo/zerfoo`.
  Configure `imagePullSecrets` if your cluster requires authentication.

### Key Notes

- **No CGo required.** Zerfoo loads GPU backends dynamically at runtime via
  `purego`/`dlopen`. Build with `go build ./...` everywhere; no `cuda` build tag
  is needed for runtime GPU acceleration.
- **GGUF is the only supported model format.** Ensure all models are in GGUF format
  before deployment. Use `zonnx` to convert ONNX models.
- Model weights are memory-mapped. RSS will be close to the GGUF file size plus KV
  cache overhead. Set `LimitMEMLOCK=infinity` in systemd for non-Kubernetes deployments.

---

## 2. Installation

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

## 3. Kubernetes Deployment

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

### ZerfooInferenceService Operator

The `ZerfooInferenceService` operator (in `serve/operator/`) manages the lifecycle
of Zerfoo inference servers on Kubernetes. It reconciles custom resources into
Deployments with health probes, Prometheus scraping annotations, GPU resource
requests, Services, and HorizontalPodAutoscalers.

The health endpoints are provided by the `health` package (`health/server.go`):

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Liveness probe -- process is alive |
| `GET /readyz` | Readiness probe -- model is loaded and serving |
| `GET /health` | Combined health check |
| `GET /debug/pprof/` | Runtime profiling (restrict to internal network) |

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

### Deployment Manifest

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

Adjust `livenessProbe.initialDelaySeconds` based on model size -- larger models
take longer to load into GPU memory. A 70B model on a single GPU may need
60-120 seconds.

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

For streaming (SSE) support, ensure your load balancer or Ingress controller
disables response buffering. The `proxy-buffering: "off"` annotation above
handles this for nginx.

---

## 4. Scaling

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

### Custom Metrics HPA

For GPU workloads, scale on custom metrics instead of CPU. Use the
[Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter) to
expose `tokens_per_second` or `request_latency_ms` as Kubernetes custom metrics:

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

You can also scale on request latency:

```yaml
metrics:
  - type: Pods
    pods:
      metric:
        name: request_latency_ms_p99
      target:
        type: AverageValue
        averageValue: "500"
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

## 5. Multi-GPU Inference

Zerfoo distributes a single model across multiple GPUs using the `--gpus` flag,
which accepts a comma-separated list of NVIDIA device IDs (implemented in
`cmd/cli/serve.go:parseGPUList`).

### CLI Usage

```bash
# Single GPU (default -- uses GPU 0 if CUDA is available)
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
| 7B | Q4_K_M | 1x | 6 GB |
| 13B | Q4_K_M | 1x | 10 GB |
| 70B | Q4_K_M | 2x A100 40GB | 40 GB |
| 70B | Q8_0 | 4x A100 40GB | 40 GB |
| 405B | Q4_K_M | 8x H100 80GB | 80 GB |

---

## 6. TLS / mTLS Configuration

The serve package returns a standard `http.Handler`. TLS is configured at the Go
`http.Server` level or terminated at the ingress/proxy layer.

### Ingress TLS (Recommended)

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

### Application-Level TLS (TLS 1.3)

For end-to-end encryption without an Ingress controller, embed TLS directly
in the server:

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
Zerfoo's distributed training layer (`distributed/tlsconfig.go`) provides mTLS
support with CA certificate verification, client certificate authentication
(`tls.RequireAndVerifyClientCert`), and minimum TLS 1.2.

For the HTTP serving layer, configure mTLS at the application level:

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

Or use a service mesh (Istio, Linkerd) for transparent mTLS between all pods.

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

## 7. Monitoring and Observability

### Prometheus Metrics

Every Zerfoo server exposes a `GET /metrics` endpoint in Prometheus text
exposition format. An `*runtime.InMemoryCollector` must be passed via
`serve.WithMetrics`.

| Metric | Type | Description |
|--------|------|-------------|
| `requests_total` | Counter | Total completed inference requests |
| `tokens_generated_total` | Counter | Total tokens generated across all requests |
| `tokens_per_second` | Gauge | Rolling token generation rate (tok/s) |
| `speculative_acceptance_rate` | Gauge | Speculative decoding acceptance rate (when enabled) |
| `request_latency_ms` | Histogram | Request latency distribution |

Histogram buckets: `10, 50, 100, 250, 500, 1000, 2500, 5000, 10000` ms.

### Prometheus Scrape Configuration

Static target:

```yaml
scrape_configs:
  - job_name: zerfoo
    scrape_interval: 15s
    static_configs:
      - targets:
          - "zerfoo-host:8080"
    metrics_path: /metrics
```

Kubernetes pod discovery:

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
| P95 Latency | `histogram_quantile(0.95, rate(request_latency_ms_bucket[5m]))` | 95th percentile latency |
| P99 Latency | `histogram_quantile(0.99, rate(request_latency_ms_bucket[5m]))` | Tail latency |
| Speculative Acceptance | `speculative_acceptance_rate` | Draft model acceptance rate |
| GPU Memory | `nvidia_gpu_memory_used_bytes` (from DCGM exporter) | GPU memory utilization |

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

### Structured Logging

Zerfoo logs every request with structured fields:

```
method=POST path=/v1/chat/completions model=gemma-3-1b prompt_tokens=0 completion_tokens=0 latency_ms=142 status_code=200
```

Collect logs via your cluster's logging stack (Fluentd, Loki, CloudWatch).
Filter on `status_code >= 500` for error alerting.

---

## 8. Model Management

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
  namespace: zerfoo-system
spec:
  accessModes:
    - ReadOnlyMany           # ReadWriteOnce if single-node
  storageClassName: efs-sc   # replace with your StorageClass
  resources:
    requests:
      storage: 200Gi
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

### Model Repository

The `serve/repository` package implements a local filesystem model repository.
Models are stored as `{baseDir}/{modelID}/model.gguf` with a `metadata.json`
sidecar. SHA-256 is computed and stored on upload.

#### Directory Layout

```
/models/
  llama-3-7b-q4_k_m/
    model.gguf
    metadata.json
  gemma-3-7b-it-q4_k_m/
    model.gguf
    metadata.json
```

#### Go API

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

## 9. Multi-Model Serving with LRU Eviction

The `serve/multimodel` package provides a `ModelManager` that loads multiple models
on-demand within a GPU memory budget, evicting the least-recently-used model when
a new load would exceed the budget.

### Architecture

```
Request -> ModelManager.Get("model-id")
              |
              +-- Already loaded? -> promote to MRU, return handle
              |
              +-- Not loaded:
                    Evict LRU models until usedBytes + newSize <= MaxGPUMemoryBytes
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

### Kubernetes Multi-Model Deployment

For deployments serving many models from a single pod, increase the memory budget
and mount a larger model store:

```yaml
resources:
  limits:
    nvidia.com/gpu: "2"
    memory: "128Gi"
env:
  - name: ZERFOO_MAX_GPU_MEMORY_GB
    value: "80"   # 2x A100 40GB
```

---

## 10. Performance Tuning

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

#### Continuous Batching (`serve/batcher/scheduler.go`)

The `batcher.Scheduler` implements continuous batching -- variable-length (ragged)
batches with zero padding, immediate eviction of completed sequences, and slot
back-fill without stalling active requests. This typically achieves 2x throughput
over fixed batching.

```go
import "github.com/zerfoo/zerfoo/serve/batcher"

scheduler := batcher.New(
    16,   // maxBatchSize -- max concurrent active sequences
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

#### Adaptive Batching (`serve/adaptive/batcher.go`)

The `adaptive.Batcher` dynamically adjusts batch size based on queue depth and
latency EMA (exponential moving average, alpha=0.3):

- **Scale up**: queue depth >= current batch size AND latency EMA <= target -> double
  batch size (capped at `MaxBatchSize`).
- **Scale down**: latency EMA > target -> reduce batch size by 25%.
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

| Field | Default | Description |
|-------|---------|-------------|
| `MinBatchSize` | 1 | Minimum batch size |
| `MaxBatchSize` | 32 | Maximum batch size |
| `TargetLatencyMS` | 100 | Latency SLO in ms; controls scale-down |
| `QueueTimeoutMS` | 50 | Max wait time (ms) to fill a batch |

#### Wiring into the HTTP Server

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

### Memory

Model weights are memory-mapped. Pod RSS will be close to the GGUF file size
plus KV cache overhead. Set resource limits accordingly and avoid memory
overcommit on GPU nodes.

---

## 11. Security Hardening

### Network

- [ ] Terminate TLS 1.3 at ingress or application level -- never serve HTTP in
  production.
- [ ] Restrict `GET /metrics` to the internal monitoring network (Prometheus
  scraper IP range), not the public internet.
- [ ] Restrict `GET /debug/pprof/` to internal networks only (exposed by
  `health/server.go`).
- [ ] Use mTLS for service-to-service communication (e.g., load balancer -> server,
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

### Container Hardening

- [ ] Run as a non-root user (UID 1000).
- [ ] Set `readOnlyRootFilesystem: true` -- mount `/tmp` as `emptyDir` if needed.
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

Store model repository credentials as Kubernetes Secrets:

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

### RBAC

The Helm chart creates a dedicated ServiceAccount with
`automountServiceAccountToken: false` by default.

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

## 12. Troubleshooting

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
3. Check for CUDA graph capture misses -- examine startup logs for graph compilation
   warnings.
4. Profile with pprof (exposed at `/debug/pprof/` by `health/server.go`):
   ```bash
   go tool pprof http://localhost:8081/debug/pprof/profile?seconds=30
   ```

### Pod Stuck in `Pending`

- Check GPU availability: `kubectl describe node <node> | grep nvidia.com/gpu`
- Verify NVIDIA GPU Operator is running: `kubectl get pods -n gpu-operator`
- Check PVC binding: `kubectl get pvc -n zerfoo`

### Pod in `CrashLoopBackOff`

- Check logs: `kubectl logs -n zerfoo deploy/zerfoo --previous`
- Common causes:
  - Model not found (invalid `model.name`)
  - Insufficient memory (increase `resources.limits.memory`)
  - GPU out of memory (use a smaller quantization or more GPUs)

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
3. Check `PreloadModels` list -- a failed preload aborts `NewModelManager`.

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
