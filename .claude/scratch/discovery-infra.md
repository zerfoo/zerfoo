# Infrastructure, Deployment, CI/CD & Dependencies Discovery

**Date:** 2026-03-21
**Repo:** `/Users/dndungu/Code/zerfoo/zerfoo` (module `github.com/zerfoo/zerfoo`)

---

## 1. Dependencies (go.mod)

**Go version:** 1.26

### Direct Dependencies

| Module | Version | Purpose |
|--------|---------|---------|
| `github.com/zerfoo/float16` | v0.2.0 | IEEE 754 half-precision arithmetic |
| `github.com/zerfoo/float8` | v0.2.0 | FP8 E4M3FN arithmetic |
| `github.com/zerfoo/ztensor` | v0.3.0 | Tensor/compute/graph/GPU kernels |
| `github.com/zerfoo/ztoken` | v0.2.0 | BPE tokenizer |
| `go.etcd.io/bbolt` | v1.4.3 | Embedded key-value store (BoltDB) |
| `golang.org/x/image` | v0.37.0 | Image processing |
| `gonum.org/v1/gonum` | v0.17.0 | Numerical computing |
| `google.golang.org/grpc` | v1.65.0 | gRPC for distributed training |
| `google.golang.org/protobuf` | v1.36.8 | Protocol Buffers |

### Indirect Dependencies

| Module | Version |
|--------|---------|
| `github.com/google/go-cmp` | v0.7.0 |
| `golang.org/x/net` | v0.25.0 |
| `golang.org/x/sys` | v0.29.0 |
| `golang.org/x/text` | v0.35.0 |
| `golang.org/x/sync` | v0.20.0 |
| `google.golang.org/genproto/googleapis/rpc` | 2024-05-28 |

### Test-only (in go.sum but not go.mod)

| Module | Version |
|--------|---------|
| `github.com/stretchr/testify` | v1.10.0 |
| `github.com/davecgh/go-spew` | v1.1.1 |
| `github.com/pmezard/go-difflib` | v1.0.0 |
| `gopkg.in/yaml.v3` | v3.0.1 |

### Notable: `google.golang.org/genproto` Exclusion & Replace

Two specific genproto versions are excluded, and genproto is replaced with a pinned 2024-05-28 snapshot. This works around version conflicts between grpc and genproto.

---

## 2. CI Pipeline Structure

Four GitHub Actions workflows in `.github/workflows/`:

### 2.1 `ci.yml` -- Main CI

- **Triggers:** push/PR to `main`
- **Concurrency:** cancel-in-progress per ref
- **Matrix:** Go 1.26
- **Steps:** checkout, setup-go, cache go modules, `go vet ./...`, `go test ./... -race -timeout 120s`
- **SBOM Generation:** `anchore/sbom-action@v0` producing CycloneDX JSON
- **Env:** `GONOSUMDB` and `GONOPROXY` set for `github.com/zerfoo/*` (private modules)

### 2.2 `release-please.yml` -- Automated Releases

- **Triggers:** push to `main`
- **Permissions:** contents:write, pull-requests:write
- **Steps:** release-please-action (Go release type), SBOM generation on release, upload SBOM to GitHub release via `softprops/action-gh-release`

### 2.3 `arm64-build.yml` -- ARM64 Cross-Compilation

- **Triggers:** push/PR to `main`
- **Matrix:** linux/arm64, darwin/arm64
- **Builds:** `cmd/zerfoo-edge` binary with `go vet` and cross-compile
- **Verification:** checks binary exists

### 2.4 `benchmark.yml` -- Performance Regression Detection

- **Triggers:** push to main, PRs, weekly cron (Monday 6am UTC), workflow_dispatch
- **CPU job** (`ubuntu-latest`): runs `scripts/bench.sh`, uploads JSON artifacts, downloads previous baseline, builds `cmd/bench-compare/` tool, posts regression report as PR comment
- **GPU job** (`self-hosted, dgx-spark`): runs only on schedule/dispatch, executes CUDA Q4 GEMM benchmark with `-tags cuda`, computes GFLOPS, uploads artifact
- **Baseline management:** saves current results as "previous" on main branch pushes

---

## 3. Docker

### `deploy/aws/Dockerfile` -- Multi-Stage Build

**Build stage:**
- Base: `golang:1.25-bookworm`
- `CGO_ENABLED=0 GOOS=linux GOARCH=amd64` -- fully static binary
- Flags: `-trimpath -ldflags="-s -w"` (stripped, no debug info)

**Runtime stage:**
- Base: `gcr.io/distroless/static-debian12:nonroot` -- **NOT running as root**
- No shell, no package manager -- minimal attack surface
- Binary copied as `/zerfoo`
- Port: 8080
- Healthcheck: `/zerfoo health` every 30s
- Env: `ZERFOO_PORT=8080`, `ZERFOO_LOG_FORMAT=json`
- Entrypoint: `["/zerfoo"]`, CMD: `["serve", "--port", "8080"]`

**Security assessment:** Excellent. Distroless nonroot image, static binary, no shell, no root user.

---

## 4. AWS Deployment

### 4.1 CloudFormation (`deploy/aws/cloudformation.yaml`)

Full ECS Fargate deployment with:

- **ECS Cluster** with Container Insights enabled
- **IAM Roles:** TaskExecutionRole (standard ECS), TaskRole (Marketplace MeterUsage + CloudWatch PutMetricData), AutoScalingRole
- **Networking:** ALB security group (ports 80/443 from 0.0.0.0/0), Task security group (port 8080 from ALB only)
- **Load Balancer:** Internet-facing ALB with HTTP and optional HTTPS (ACM certificate)
- **HTTP-to-HTTPS redirect** when certificate is provided
- **ECS Task:** Fargate, awsvpc, configurable CPU/memory (default 4 vCPU / 16 GiB)
- **Service:** private subnets (AssignPublicIp: DISABLED), min 50% healthy during deploy
- **Autoscaling:** Target tracking on ALB requests/task (default 50), min 1 / max 10, cooldowns 60s out / 300s in
- **CloudWatch Alarms:** High P99 latency (>5s), unhealthy tasks, high 5xx rate (>10/min)
- **Logs:** CloudWatch log group with 30-day retention

### 4.2 AWS Marketplace Metering (`deploy/aws/metering.go`)

- Interface-based `MeteringClient` (no AWS SDK dependency at compile time)
- `HTTPMeteringClient` calls Marketplace Metering API directly over HTTP
- Three metering dimensions: `inference-requests`, `tokens-processed`, `gpu-hours`
- `Meter` struct aggregates and reports usage

### 4.3 AWS Marketplace Listing (`deploy/aws/listing.json`)

- Product code: `PLACEHOLDER_PRODUCT_CODE` (not yet published)
- Pricing: SaaS model with 3 dimensions (inference requests, tokens, GPU hours)
- Regions: us-east-1, us-west-2, eu-west-1, ap-southeast-1, ap-northeast-1
- Container image: `709825985650.dkr.ecr.us-east-1.amazonaws.com/zerfoo/zerfoo:latest`

---

## 5. Helm Chart (`deploy/helm/zerfoo/`)

- Chart version: 0.1.0, appVersion: 0.1.0
- Image: `ghcr.io/zerfoo/zerfoo` (GHCR, not ECR)
- Default resources: 2 CPU / 4 GiB request, 4 CPU / 8 GiB limit
- GPU support commented out (`nvidia.com/gpu` toleration and resource limit)
- Service: ClusterIP on port 80
- Ingress: disabled by default, supports cert-manager annotations
- HPA: disabled by default, targets 80% CPU
- Probes: liveness on `/health` (30s initial, 10s period), readiness on `/health` (10s initial, 5s period)
- ConfigMap checksum annotation for rolling restarts on config change
- ServiceAccount: created by default

**Templates:** deployment, service, serviceaccount, configmap, hpa, ingress

---

## 6. Kubernetes DaemonSet (`deploy/modelcache-daemonset.yaml`)

- Model cache DaemonSet in `zerfoo` namespace
- Tolerates GPU nodes (`nvidia.com/gpu`)
- hostPath volume at `/var/lib/zerfoo/models`
- 10 GiB cache max, 20 GiB ephemeral storage limit
- Rolling update strategy

---

## 7. GCP Terraform (`infra/terraform/zerfoo-cloud/`)

### Provider & Backend

- Terraform >= 1.5
- Google provider ~> 5.0 (both `google` and `google-beta`)
- State backend: GCS bucket `zerfoo-terraform-state`

### Resources

- **VPC:** Custom network with subnet (10.0.0.0/20), pod CIDR (10.16.0.0/14), service CIDR (10.20.0.0/20)
- **NAT Gateway:** Auto-allocated IPs
- **GKE Cluster:** Regular release channel, Workload Identity enabled, separate node pools
- **CPU Node Pool:** e2-standard-4, 2 nodes default, auto-repair/upgrade
- **GPU Node Pool:** n1-standard-8, nvidia-tesla-t4 default, **spot instances**, GPU driver auto-install, taint for NoSchedule, auto-repair/upgrade
- **Cloud Run API Gateway:** Min 1 / max 10 instances, public access (`allUsers` invoker)
- **GCS Model Bucket:** Versioned (keep 3 versions), uniform bucket-level access
- **Workload Identity:** Service account `zerfoo-model-reader` with `storage.objectViewer` on model bucket

### Sensitive Outputs

- `cluster_endpoint` and `cluster_ca_certificate` marked as `sensitive = true`

### Security Note

- Cloud Run service grants `roles/run.invoker` to `allUsers` -- this is intentional for a public API gateway but should be reviewed if authentication is required at the gateway level.

---

## 8. Edge Deployments

### 8.1 Jetson Orin Nano (`deploy/jetson/`)

- Cross-compile: `GOOS=linux GOARCH=arm64 CGO_ENABLED=0`
- Build tags: `edge`
- Binary: `zerfoo-edge-jetson`
- Also has a `build-cuda.sh` for CUDA-enabled Jetson builds

### 8.2 Raspberry Pi 5 (`deploy/rpi5/`)

- Cross-compile: `GOOS=linux GOARCH=arm64 CGO_ENABLED=0`
- Build tags: `edge,!cuda,!rocm,!opencl` (explicitly excludes GPU backends)
- Binary: `zerfoo-edge-rpi5`

Both produce static binaries with version info via `-X main.version=...` and stripped debug info (`-s -w`).

---

## 9. Cloud Marketplace Integrations

### 9.1 Unified Abstraction (`marketplace/marketplace.go`)

- `Provider` interface combining `MeteringService`, `SubscriptionManager`, `EntitlementChecker`
- Common types: `UsageRecord`, `Subscription`, `CloudProvider` (aws/gcp/azure)

### 9.2 AWS (`marketplace/aws/`)

- `EntitlementChecker` with in-memory cache and configurable TTL
- `EntitlementStore` interface with `MemoryEntitlementStore` for testing
- Metering, billing, subscription management
- CloudFormation quickstart template (`cfn/quickstart.yaml`)

### 9.3 Azure (`marketplace/azure/`)

- SaaS Fulfillment API v2 client (Resolve, Activate, GetSubscription, Update, Suspend, Delete, List)
- `TokenProvider` interface for Azure AD authentication
- Metering, subscription lifecycle, webhook handler
- ARM template (`arm/template.json`)

### 9.4 GCP (`marketplace/gcp/`)

- Cloud Commerce Partner Procurement API client
- Account management, entitlement lifecycle (approve/reject/suspend/reinstate)
- `TokenSource` interface for OAuth2 authentication
- Metering, billing, entitlement management
- Deployment Manager template (`dm/quickstart.yaml`)

---

## 10. Cloud Module (`cloud/`)

Multi-tenant managed inference service (alpha stability):

- **CloudServer:** HTTP handler with auth, rate limiting, billing middleware chain
- **TenantManager:** CRUD on tenants, O(1) lookup by ID or API key
- **API Key Authentication:** Bearer token extraction, constant-time comparison (`crypto/subtle`)
- **Rate Limiting:** Per-tenant per-minute request and token budget limits (lock-free atomics)
- **Token Billing:** `TokenMeter` with `BillingStore` interface, records input/output tokens per request
- **Audit Logging:** SOC 2 compliant audit entries (no sensitive data stored), AuditAction/AuditResult enums
- **SSO:** SAML 2.0 provider with metadata XML parsing, assertion validation (signature verification noted as TODO for production)

---

## 11. Security Module (`security/`)

### API Key Management (`apikey.go`)

- 32-byte random keys with `zf_` prefix
- SHA-256 hash storage (raw key never persisted)
- Scopes: `inference`, `training`, `admin`, `read_only`
- Key lifecycle: Create, Lookup, Revoke, Rotate, List
- Expiry support

### Encryption (`encryption.go`)

- AES-256-GCM encryption/decryption
- TLS configuration with mTLS support
- Minimum TLS 1.2 default
- Certificate validation and loading

### Network Security (`network.go`)

- Token-bucket rate limiter per client IP
- IP allowlist/denylist filtering
- CORS policy middleware
- Client IP extraction (X-Forwarded-For, X-Real-IP, RemoteAddr)
- Listen address validation

### Secrets Management (`secrets.go`)

- `KMS` interface for external key management (AWS KMS, GCP Cloud KMS, Vault)
- `SecretConfig` reads secrets from environment variables with configurable prefix
- No secrets are hardcoded anywhere in the codebase

### Incident Response (`incident.go`)

- Incident severity levels: critical, high, medium, low
- Alert hooks for security events
- Automatic IP lockout after repeated suspicious activity
- Configurable lockout threshold and duration
- Integration with IPFilter for automatic deny-listing

### Vulnerability Scanning (`vuln.go`)

- `DependencyAuditor` and `CVEChecker` interfaces
- `AuditReport` with severity-based filtering
- Severity levels: critical, high, medium, low, info

---

## 12. Configuration Module (`config/`)

- Generic JSON config loader: `Load[T](path)`
- Environment variable overrides via `LoadWithEnv[T](path, prefix)` using `env` struct tags
- Field validation via `validate:"required"` struct tags
- Config structs: `EngineConfig`, `TrainingConfig`, `DistributedConfig`

---

## 13. Compliance Module (`compliance/`)

SOC 2 Type II compliance automation:

- **Controls:** Full SOC 2 Trust Services Criteria mapping (CC1-CC9, A1, C1, PI1, P1) -- 35 controls
- **Categories:** Security, Availability, Confidentiality, Processing Integrity, Privacy
- **Assessment tracking:** per-control status (compliant, partial, non-compliant, not assessed, N/A)
- **Policy templates:** 6 policy types (access control, change management, incident response, data classification, risk assessment, vendor management)
- **Evidence collection:** `compliance/audit/` and `compliance/observation/` sub-packages
- **Audit readiness, gap analysis, deviation tracking, evidence management**

---

## 14. Provenance Module (`provenance/`)

Model lifecycle provenance tracking:

- Cryptographic hash chain (SHA-256) forming a DAG
- Event types: training, dataset, evaluation
- Records: `TrainingRecord`, `DatasetRecord`, `EvaluationRecord`
- Trace/verify operations with cycle detection
- Deterministic canonical serialization for hash computation

---

## 15. Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `bench.sh` | Run GEMM/KV cache/memory/Q4 benchmarks, emit JSON metrics |
| `run_distributed.sh` | Launch 3-worker distributed training on localhost |
| `dgx-spark-multigpu.sh` | Multi-GPU benchmarks on DGX Spark |
| `dgx-spark-parity.sh` | Model parity tests on DGX Spark |
| `generate_docs.sh` | Documentation generation |
| `make_prompts.sh` | Prompt generation |
| `make_seed.sh` | Seed data generation |
| `nsight-decode-profile.sh` | NVIDIA Nsight decode profiling |

---

## 16. Security Assessment Summary

### Secrets & Credentials

- **No hardcoded secrets, API keys, or credentials found** in any source file
- AWS product code uses `PLACEHOLDER_PRODUCT_CODE`
- ECR image URI `709825985650.dkr.ecr.us-east-1.amazonaws.com` is a public marketplace identifier (not a secret)
- Secrets loaded from environment variables via `security.SecretConfig`
- No `.env` files present in the repository
- `HF_TOKEN` read from environment in `registry/pull.go` (HuggingFace token for model pulls)

### Docker Security

- Distroless nonroot base image (no shell, no root)
- Multi-stage build (build artifacts not in final image)
- Static binary with CGO disabled
- Stripped debug symbols

### TLS & Authentication

- TLS 1.2 minimum enforced
- mTLS support via CA file configuration
- Bearer token authentication for cloud API
- Constant-time API key comparison (timing attack mitigation)
- SAML 2.0 SSO support (XML signature verification noted as production TODO)

### Network Security

- Per-IP rate limiting
- IP allowlist/denylist
- CORS middleware
- Automatic IP lockout after repeated incidents
- ECS tasks on private subnets (no public IP)

### Compliance

- SOC 2 Type II control framework built into the codebase
- Audit logging with no sensitive data stored
- SBOM generation (CycloneDX) in CI and on releases
- Dependency vulnerability scanning interfaces

### Areas for Review

1. **Cloud Run `allUsers` invoker** (`cloud_run.tf` line 69): API gateway is publicly accessible. Ensure authentication is handled at the application layer.
2. **SAML signature verification** (`cloud/sso.go` line 179): Comment says "In production, this would also verify the XML signature." This is not yet implemented.
3. **ALB Security Group** (`cloudformation.yaml`): Allows 0.0.0.0/0 on ports 80/443. Expected for public-facing service but noted.
4. **GPU node pool uses spot instances** (`main.tf` line 128): Cost-effective but can cause availability interruptions for inference workloads.
