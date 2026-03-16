# Contributing to Zerfoo

Thank you for your interest in contributing to Zerfoo, the Go-native ML inference and training framework. This guide will help you get started.

## Development Setup

### Prerequisites

- **Go 1.25+** (generics with `tensor.Numeric` constraint)
- **Git**
- **CUDA Toolkit** (optional, for GPU-accelerated tests and development)

### Clone and Verify

```bash
git clone https://github.com/zerfoo/zerfoo.git
cd zerfoo
go mod tidy
go test ./...
```

Zerfoo depends on several sibling packages fetched automatically by `go mod tidy`:

- [`github.com/zerfoo/ztensor`](https://github.com/zerfoo/ztensor) — tensor, compute engine, computation graph
- [`github.com/zerfoo/ztoken`](https://github.com/zerfoo/ztoken) — BPE tokenizer
- [`github.com/zerfoo/float16`](https://github.com/zerfoo/float16) — IEEE 754 half-precision arithmetic
- [`github.com/zerfoo/float8`](https://github.com/zerfoo/float8) — FP8 E4M3FN arithmetic

No CGo is required for CPU-only builds. GPU support is loaded dynamically at runtime via purego/dlopen, so `go build ./...` works on any platform without a C compiler.

## Running Tests

```bash
go test ./...            # All CPU tests (no GPU required)
go test -race ./...      # Tests with race detector (required before submitting)
go test -tags cuda ./... # GPU tests (requires CUDA toolkit and a GPU)
go test -coverprofile=coverage.out ./...  # Coverage report
go tool cover -html=coverage.out -o coverage.html
```

### Testing Requirements

- All new code must have tests
- Use **table-driven tests** with `t.Run` subtests
- Always run with the **`-race` flag** before submitting
- CI enforces a **75% coverage gate** on new packages

## Code Style

### Formatting and Linting

- **`gofmt`** — all code must be formatted with `gofmt`
- **`goimports`** — imports must be organized (stdlib, external, internal)
- **`golangci-lint`** — run `golangci-lint run` before submitting

### Go Conventions

- Prefer the **Go standard library** over third-party dependencies
- Follow standard Go naming: PascalCase for exported, camelCase for unexported
- Write documentation comments for all exported functions, types, and methods
- Use generics with `[T tensor.Numeric]` constraints — avoid type-specific code where generics work
- All tensor arithmetic must flow through `compute.Engine[T]` (see [Key Conventions](#key-conventions))

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning with release-please.

```
<type>(<scope>): <description>
```

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `perf` | A performance improvement |
| `docs` | Documentation only changes |
| `test` | Adding or correcting tests |
| `chore` | Maintenance tasks, CI, dependencies |
| `refactor` | Code change that neither fixes a bug nor adds a feature |

Examples:

```
feat(inference): add Qwen 2.5 architecture support
fix(generate): correct KV cache eviction for sliding window attention
perf(layers): fuse SiLU and gate projection into single kernel
```

## Pull Request Process

1. **Branch from `main`** and keep your branch up to date with rebase
2. **One logical change per PR** — keep PRs focused and reviewable
3. **All CI checks must pass** — tests, linting, formatting
4. **Rebase and merge** — we do not use squash merges or merge commits
5. **Reference related issues** — use `Fixes #123` or `Closes #123` in the PR description

### Before Submitting

```bash
go test -race ./...
go vet ./...
golangci-lint run
```

## Issue Reporting

### Bug Reports

Include: clear description, steps to reproduce, expected vs actual behavior, environment (Go version, OS, architecture, GPU), and model details if applicable.

### Feature Requests

Include: problem statement, proposed solution, alternatives considered, and use case.

## Good First Issues

See [`docs/good-first-issues.md`](docs/good-first-issues.md) for a curated list of starter tasks, or browse issues labeled [`good first issue`](https://github.com/zerfoo/zerfoo/labels/good%20first%20issue) on GitHub.

**How to claim an issue:**

1. Comment on the issue to let maintainers know you're working on it
2. Fork the repo and create a feature branch
3. Submit a PR referencing the issue

## Key Conventions

### Engine[T] is law

All tensor arithmetic must flow through `compute.Engine[T]`. Never operate on raw slices outside the engine — this enables transparent CPU/GPU switching and CUDA graph capture.

### No CGo by default

GPU bindings use purego/dlopen. A plain `go build ./...` must compile on any platform without a C compiler.

### GGUF is the sole model format

Do not add support for other formats (ONNX, SafeTensors, etc.) in this repo. Use [`zonnx`](https://github.com/zerfoo/zonnx) to convert ONNX models to GGUF.

### Fuse, don't fragment

Prefer fused operations (`FusedAddRMSNorm`, `FusedSiluGate`, etc.) over sequences of primitive ops. Every eliminated kernel launch matters for tok/s.
