# Production Deployment Guide

This guide covers deploying `zerfoo serve` in production with TLS, monitoring,
and reverse proxy configuration.

## Quick Start

```bash
zerfoo serve google/gemma-3-1b --port 8080
```

This starts an OpenAI-compatible API server on port 8080 with the following
endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming/non-streaming) |
| `/v1/completions` | POST | Text completion (streaming/non-streaming) |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/models` | GET | List loaded models |
| `/v1/models/{id}` | GET/DELETE | Get or unload a model |
| `/metrics` | GET | Prometheus metrics |
| `/openapi.yaml` | GET | OpenAPI specification |

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8080` | Listen port |
| `--cache-dir` | System default | Override model cache directory |

## TLS / mTLS

The serve package returns a standard `http.Handler`. To enable TLS, wrap it
with Go's `tls.Config` or terminate TLS at your reverse proxy (see nginx
section below).

For direct TLS termination at the application level, embed the server in a
custom `main.go`:

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

For mTLS (mutual TLS), add client certificate verification:

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

## Prometheus Metrics

The `GET /metrics` endpoint exposes metrics in Prometheus text exposition
format. Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `requests_total` | Counter | Total completed requests |
| `tokens_generated_total` | Counter | Total tokens generated |
| `tokens_per_second` | Gauge | Rolling token generation rate |
| `request_latency_ms` | Histogram | Request latency (buckets: 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000 ms) |

Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: zerfoo
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics
```

## Graceful Shutdown

The server uses a `shutdown.Coordinator` that closes components in reverse
registration order when the process receives SIGINT or SIGTERM:

1. The HTTP server stops accepting new connections.
2. In-flight requests are allowed to complete.
3. The batch scheduler (if attached) is drained and stopped.
4. The model is closed and GPU memory is released.

No special configuration is needed — the CLI wires this automatically.

## systemd Unit File

```ini
[Unit]
Description=Zerfoo Inference Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=zerfoo
Group=zerfoo
ExecStart=/usr/local/bin/zerfoo serve google/gemma-3-1b --port 8080
Restart=on-failure
RestartSec=5
LimitNOFILE=65536
LimitMEMLOCK=infinity
Environment=HOME=/var/lib/zerfoo
WorkingDirectory=/var/lib/zerfoo

# GPU access
SupplementaryGroups=video render

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/zerfoo

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now zerfoo
sudo journalctl -u zerfoo -f
```

## Reverse Proxy (nginx)

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

    # Streaming support — disable buffering for SSE
    proxy_buffering off;
    proxy_cache off;

    # Timeouts for long-running inference requests
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://zerfoo;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
    }

    # Metrics — restrict to internal network
    location /metrics {
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
        proxy_pass http://zerfoo;
    }
}
```

## Resource Sizing

### CPU-Only

| Model Size | RAM | CPU Cores | Notes |
|-----------|-----|-----------|-------|
| 1B (Q4_K_M) | 2 GB | 4+ | Suitable for development and light traffic |
| 3B (Q4_K_M) | 4 GB | 8+ | Good for moderate throughput |
| 7B (Q4_K_M) | 8 GB | 8+ | Recommended minimum for production |

### GPU (CUDA)

| Model Size | VRAM | System RAM | Notes |
|-----------|------|------------|-------|
| 1B (Q4_K_M) | 1 GB | 4 GB | Single consumer GPU |
| 7B (Q4_K_M) | 6 GB | 8 GB | RTX 3060 or better |
| 13B (Q4_K_M) | 10 GB | 16 GB | RTX 3080/4080 or better |
| 70B (Q4_K_M) | 40 GB | 64 GB | A100/H100 or multi-GPU |

### General Guidelines

- **Memory**: Model weights are memory-mapped. RSS will be close to the GGUF
  file size plus KV cache overhead. Set `LimitMEMLOCK=infinity` in systemd to
  prevent swapping.
- **File descriptors**: Set `LimitNOFILE=65536` for high-concurrency workloads.
- **GPU**: Ensure the `video` and `render` groups are assigned for GPU access.
  Zerfoo loads GPU backends dynamically via purego — no CGo or special build
  flags are needed.
- **Batch scheduling**: For throughput-oriented workloads (non-streaming),
  attach a `BatchScheduler` to group requests and improve GPU utilization.
