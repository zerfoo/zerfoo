# Contributing to Zerfoo

Thank you for your interest in contributing to Zerfoo, the Go-native ML inference and training framework. This guide covers the full Zerfoo ecosystem and applies to all six repositories.

## Code of Conduct

All participants in the Zerfoo community are expected to treat each other with respect and professionalism. We are committed to providing a welcoming and inclusive environment for everyone.

## Repository Structure

Zerfoo is an ecosystem of six independent repositories (each with its own `go.mod`, CI, and releases):

| Repository | Module | Purpose |
|-----------|--------|---------|
| [zerfoo](https://github.com/zerfoo/zerfoo) | `github.com/zerfoo/zerfoo` | Core ML framework: inference, training, serving |
| [ztensor](https://github.com/zerfoo/ztensor) | `github.com/zerfoo/ztensor` | GPU-accelerated tensor, compute engine, computation graph |
| [ztoken](https://github.com/zerfoo/ztoken) | `github.com/zerfoo/ztoken` | BPE tokenizer with HuggingFace compatibility |
| [zonnx](https://github.com/zerfoo/zonnx) | `github.com/zerfoo/zonnx` | ONNX-to-GGUF converter CLI |
| [float16](https://github.com/zerfoo/float16) | `github.com/zerfoo/float16` | IEEE 754 half-precision (Float16/BFloat16) arithmetic |
| [float8](https://github.com/zerfoo/float8) | `github.com/zerfoo/float8` | FP8 E4M3FN arithmetic for quantized inference |

**Dependency graph:**

```
float16 ─┐
float8  ─┼──► ztensor ──► zerfoo
ztoken ──┘

zonnx (standalone)
```

Each repo is versioned and released independently. Do not treat this as a monorepo — submit PRs to the repository where the change belongs.

## Development Setup

### Prerequisites

- **Go 1.25+** (generics with `tensor.Numeric` constraint)
- **Git**
- **CUDA Toolkit** (optional, for GPU-accelerated tests and development)

### Clone and Build

Each repository builds independently:

```bash
# Clone whichever repo you want to work on
git clone https://github.com/zerfoo/<repo>.git
cd <repo>
go mod tidy
go test ./...
```

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

### Review Process

- All PRs require at least one maintainer approval
- Maintainers may request changes — address feedback and force-push to update your branch
- Once approved and CI is green, a maintainer will rebase-merge your PR

## GPU Development

### Build Tags

GPU backends are opt-in via build tags. CPU-only builds require no tags:

```bash
go build ./...                # CPU only (default, works everywhere)
go build -tags cuda ./...     # CUDA backend (requires CUDA toolkit)
go build -tags rocm ./...     # ROCm backend (requires ROCm SDK)
go build -tags opencl ./...   # OpenCL backend (requires OpenCL SDK)
```

### purego Bindings

GPU libraries are loaded at runtime via purego/dlopen — not linked at compile time. This means:

- `go build` never requires a C compiler or GPU SDK
- GPU availability is detected at runtime
- The same binary runs on CPU-only machines (gracefully falls back)

When writing GPU code, use the `compute.Engine[T]` interface. Do not call CUDA/ROCm/OpenCL APIs directly outside of `internal/gpuapi/`.

## Release Process

All six repositories use [release-please](https://github.com/googleapis/release-please) for automated releases:

1. Conventional Commit messages drive version bumps (`feat` = minor, `fix` = patch)
2. release-please opens a release PR automatically when changes land on `main`
3. Merging the release PR creates a GitHub release and Git tag
4. Semantic versioning (`vMAJOR.MINOR.PATCH`) is enforced across all repos

Breaking changes require a `BREAKING CHANGE:` footer in the commit message, which triggers a major version bump.

## Issue Reporting

### Bug Reports

Include: clear description, steps to reproduce, expected vs actual behavior, environment (Go version, OS, architecture, GPU), and model details if applicable.

### Feature Requests

Include: problem statement, proposed solution, alternatives considered, and use case.

## Good First Issues

See the [Contributing guide](https://zerfoo.feza.ai/docs/contributing/) on the documentation site for a curated list of starter tasks, or browse issues labeled [`good first issue`](https://github.com/zerfoo/zerfoo/labels/good%20first%20issue) on GitHub.

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

## Getting Help

- **GitHub Discussions** — ask questions and share ideas on each repo's Discussions tab
- **GitHub Issues** — report bugs or request features
