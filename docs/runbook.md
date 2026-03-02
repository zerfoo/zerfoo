# Zerfoo Deployment Runbook

## System Requirements

- Go 1.25 or later.
- Linux amd64 or Darwin arm64/amd64.
- CUDA Toolkit 12.x (only for GPU mode).
- At least 4 GB RAM for CPU-only inference. 8 GB recommended for training.
- Network access between nodes for distributed training (default port 50051/TCP).

## Installation

```
git clone <repo-url> && cd zerfoo
go build ./cmd/zerfoo
go build ./cmd/zerfoo-predict
```

For GPU support:

```
go build -tags cuda ./cmd/zerfoo
```

Verify the build:

```
./zerfoo --help
```

## Configuration

Zerfoo loads configuration from a JSON file with optional environment variable
overrides. Use `config.LoadWithEnv[T](path, prefix)` in code, or pass a JSON
file path to the CLI.

### Engine Configuration

| Field            | JSON key          | Env var            | Required | Default | Description                    |
|------------------|-------------------|--------------------|----------|---------|--------------------------------|
| Device           | device            | DEVICE             | Yes      | --      | "cpu" or "cuda"                |
| MemoryLimitMB    | memory_limit_mb   | MEMORY_LIMIT_MB    | No       | 0       | Max memory in MB (0=unlimited) |
| LogLevel         | log_level         | LOG_LEVEL          | No       | "info"  | debug, info, warn, error       |

### Training Configuration

| Field              | JSON key             | Env var              | Required | Default | Description                    |
|--------------------|----------------------|----------------------|----------|---------|--------------------------------|
| BatchSize          | batch_size           | BATCH_SIZE           | Yes      | --      | Training batch size            |
| LearningRate       | learning_rate        | LEARNING_RATE        | Yes      | --      | Learning rate (e.g. "0.001")   |
| Optimizer          | optimizer            | OPTIMIZER            | Yes      | --      | "sgd" or "adam"                |
| Epochs             | epochs               | EPOCHS               | No       | 0       | Number of epochs               |
| CheckpointInterval | checkpoint_interval  | CHECKPOINT_INTERVAL  | No       | 0       | Steps between checkpoints      |

### Distributed Configuration

| Field              | JSON key              | Env var                | Required | Default | Description                    |
|--------------------|-----------------------|------------------------|----------|---------|--------------------------------|
| CoordinatorAddress | coordinator_address   | COORDINATOR_ADDRESS    | Yes      | --      | Host:port of coordinator       |
| TimeoutSeconds     | timeout_seconds       | TIMEOUT_SECONDS        | No       | 0       | RPC timeout in seconds         |
| TLSEnabled         | tls_enabled           | TLS_ENABLED            | No       | false   | Enable TLS for gRPC            |

Example config file (`config.json`):

```json
{
  "device": "cpu",
  "memory_limit_mb": 4096,
  "log_level": "info"
}
```

Override with environment variables:

```
DEVICE=cuda MEMORY_LIMIT_MB=8192 ./zerfoo predict --config config.json ...
```

## Startup Sequence

1. Load configuration from file (and apply env overrides).
2. Create the compute engine (CPU or GPU).
3. Set memory limit if configured.
4. Register the engine with the shutdown coordinator.
5. Start the health check server (default :8081).
6. If distributed: start the gRPC server, connect to peers.
7. Load the model and begin serving.

## Health Check Verification

The health server exposes:

- `GET /healthz` -- Liveness probe. Returns 200 if the process is alive.
- `GET /readyz` -- Readiness probe. Returns 200 if all checks pass, 503 otherwise.
- `GET /debug/pprof/` -- pprof index for profiling.

Verify health after startup:

```
curl -s http://localhost:8081/healthz
# {"status":"healthy"}

curl -s http://localhost:8081/readyz
# {"status":"ready","checks":{"engine":"ok"}}
```

## Shutdown Procedure

Zerfoo handles SIGINT and SIGTERM for graceful shutdown:

1. Signal is received.
2. The shutdown coordinator is triggered.
3. Closers are called in reverse registration order:
   - Distributed strategy disconnects from peers.
   - Engine releases resources.
4. The root context is canceled.
5. Process exits.

To trigger graceful shutdown manually:

```
kill -TERM <pid>
```

The default shutdown timeout is governed by the context deadline. If a closer
does not complete within the deadline, it is skipped and the error is logged.

## Log Interpretation

Zerfoo uses structured leveled logging:

- **DEBUG**: Detailed operation-level information (tensor shapes, timing).
- **INFO**: Startup, configuration, connection events.
- **WARN**: Recoverable errors (e.g. OOM fallback from GPU to CPU).
- **ERROR**: Unrecoverable errors (connection failures, invalid configuration).

JSON log output example:

```json
{"level":"info","msg":"engine created","device":"cpu","memory_limit_mb":4096}
{"level":"warn","msg":"GPU OOM fallback to CPU","op":"MatMul","shape":[1024,1024]}
```

## Common Operational Tasks

### Scale Workers (Distributed Training)

1. Start additional worker nodes with the same `coordinator_address`.
2. Workers auto-register with the coordinator on connection.
3. The coordinator redistributes work on the next barrier synchronization.

### Update Model

1. Save the new model file to the model path.
2. Restart the serving process to pick up the new model.
3. Verify via health check that the engine is ready.

### Restart a Node

1. Send SIGTERM to the process.
2. Wait for graceful shutdown to complete.
3. Start the process again with the same configuration.

### Monitor Performance

Use pprof endpoints for runtime profiling:

```
# CPU profile (30 seconds)
go tool pprof http://localhost:8081/debug/pprof/profile?seconds=30

# Heap profile
go tool pprof http://localhost:8081/debug/pprof/heap

# Goroutine dump
curl http://localhost:8081/debug/pprof/goroutine?debug=2
```

## Metrics

The metrics interface provides counters, gauges, and histograms:

- `op_count_<OpName>` -- Counter incremented per operation.
- `op_duration_seconds` -- Histogram of operation durations.
- `allreduce_count` -- Counter for all-reduce operations.
- `allreduce_duration_seconds` -- Histogram of all-reduce durations.

Metrics are accessed programmatically via `metrics.Collector.Snapshot()`.
Integration with Prometheus or StatsD requires implementing the Collector
interface for the target backend.

## TLS Configuration

For production gRPC:

1. Generate CA cert, server cert/key, and optionally client cert/key.
2. Configure the TLS paths in `distributed/TLSConfig`.
3. Set `tls_enabled: true` in the distributed config.
4. For mutual TLS (mTLS), provide the CA cert path on both server and client.

Server:

```go
tlsCfg := &distributed.TLSConfig{
    CACertPath: "/path/to/ca.pem",
    CertPath:   "/path/to/server.pem",
    KeyPath:    "/path/to/server-key.pem",
}
```

Client:

```go
tlsCfg := &distributed.TLSConfig{
    CACertPath: "/path/to/ca.pem",
    CertPath:   "/path/to/client.pem",   // for mTLS
    KeyPath:    "/path/to/client-key.pem", // for mTLS
}
```
