# Contributing to Zerfoo

Thank you for your interest in contributing to Zerfoo, the Go-native ML inference and training framework. This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Good First Issues](#good-first-issues)
- [Key Conventions](#key-conventions)

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

Zerfoo depends on several sibling packages in the Zerfoo ecosystem:

- [`github.com/zerfoo/ztensor`](https://github.com/zerfoo/ztensor) — tensor, compute engine, computation graph
- [`github.com/zerfoo/ztoken`](https://github.com/zerfoo/ztoken) — BPE tokenizer
- [`github.com/zerfoo/float16`](https://github.com/zerfoo/float16) — IEEE 754 half-precision arithmetic
- [`github.com/zerfoo/float8`](https://github.com/zerfoo/float8) — FP8 E4M3FN arithmetic

These are fetched automatically by `go mod tidy`.

## Building from Source

```bash
go build ./...
```

No CGo is required for CPU-only builds. GPU support is loaded dynamically at runtime via purego/dlopen, so `go build` works on any platform without a CUDA toolkit installed.

## Running Tests

```bash
# Run all CPU tests (no GPU required)
go test ./...

# Run tests with race detector
go test -race ./...

# Run GPU tests (requires CUDA toolkit and a GPU)
go test -tags cuda ./...

# Run model parity tests (requires model files on disk)
go test -run TestParity -count=1 ./tests/parity/...

# Run tests with coverage
go test -cover ./...
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

All new code must have tests. Aim for at least 80% coverage on new packages.

## Code Style

### Formatting and Linting

- **`gofmt`** — all code must be formatted with `gofmt`
- **`goimports`** — imports must be organized (stdlib, external, internal)
- **`golangci-lint`** — run `golangci-lint run` before submitting

### Go Conventions

- Follow standard Go naming: PascalCase for exported symbols, camelCase for unexported
- Use table-driven tests with `t.Run` subtests
- Write documentation comments for all exported functions, types, and methods
- Use the `Engine[T]` pattern for all tensor operations (see [Key Conventions](#key-conventions))
- Use generics with `[T tensor.Numeric]` constraints — avoid type-specific code where generics work

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/) for automated versioning with release-please.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `perf` | A performance improvement |
| `docs` | Documentation only changes |
| `test` | Adding or correcting tests |
| `chore` | Maintenance tasks, CI, dependencies |
| `refactor` | Code change that neither fixes a bug nor adds a feature |

### Examples

```
feat(inference): add Qwen 2.5 architecture support
fix(generate): correct KV cache eviction for sliding window attention
perf(layers): fuse SiLU and gate projection into single kernel
docs(serve): document rate limiting configuration
test(training): add gradient accumulation integration tests
```

## Pull Request Process

1. **One logical change per PR** — keep PRs focused and reviewable
2. **Branch from `main`** and keep your branch up to date with rebase
3. **All CI checks must pass** — tests, linting, formatting
4. **Rebase and merge** — we do not use squash merges or merge commits
5. **Reference related issues** — use `Fixes #123` or `Closes #123` in the PR description
6. **Respond to review feedback** promptly

### Before Submitting

```bash
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
```

## Issue Reporting

### Bug Reports

Please include:

- **Description**: Clear summary of the bug
- **Steps to reproduce**: Minimal code or commands to trigger the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What happens instead
- **Environment**: Go version, OS, architecture, GPU (if relevant)
- **Model details**: Architecture, quantization type, file size (if model-related)

### Feature Requests

Please include:

- **Problem statement**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you thought about
- **Use case**: How would you use this feature in practice?

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/zerfoo/zerfoo/labels/good%20first%20issue) on GitHub. These are scoped, well-defined tasks suitable for new contributors.

**How to claim an issue:**

1. Comment on the issue to let maintainers know you're working on it
2. Fork the repo and create a feature branch
3. Submit a PR referencing the issue

Good areas for first contributions:

- Adding test coverage for existing packages
- Documentation improvements
- Small bug fixes in CLI commands (`cmd/`)
- Adding new sampling strategies in `generate/`

## Key Conventions

These conventions are critical to maintaining consistency across the codebase:

### Engine[T] is law

All tensor arithmetic must flow through `compute.Engine[T]`. Never operate on raw slices outside the engine — this enables transparent CPU/GPU switching and CUDA graph capture.

```go
// Good
engine.MatMul(ctx, out, a, b)

// Bad — bypasses the engine, breaks GPU support
for i := range out.Data() {
    out.Data()[i] = a.Data()[i] * b.Data()[i]
}
```

### No CGo by default

GPU bindings use purego/dlopen. A plain `go build ./...` must compile on any platform without a C compiler. Build tags (`cuda`, `rocm`, `opencl`) are optional and only used for CGo-based alternative paths.

### GGUF is the sole model format

GGUF is the only supported model format. Do not add support for other formats (ONNX, SafeTensors, etc.) in this repo. Use `zonnx` to convert ONNX models to GGUF.

### Generics throughout

Use `[T tensor.Numeric]` constraints. Do not write float32-specific code where generics work.

### Fuse, don't fragment

Prefer fused operations (`FusedAddRMSNorm`, `FusedSiluGate`, etc.) over sequences of primitive ops. Every eliminated kernel launch matters for tok/s.
