# Contributing to Zerfoo

We welcome contributions to the Zerfoo project! By contributing, you help us build a robust, high-performance, and idiomatic Go framework for machine learning.

## Getting Started

1.  **Fork the repository:** Start by forking the `zerfoo` repository on GitHub.
2.  **Clone your fork:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/zerfoo.git
    cd zerfoo
    ```
3.  **Set up Go workspace:** Ensure you have Go 1.25 or later installed. This project uses Go modules and a `go.work` file to manage multiple related modules (`zerfoo`, `zonnx`, `zmf`, `float16`, `float8`, `gemma3`, `audacity`, `finbert`, `numerapi`).

    To ensure all modules are correctly recognized, run:
    ```bash
    go work tidy
    ```

## Development Environment

### Prerequisites

*   Go 1.25+
*   `protoc` (Protocol Buffers compiler) for generating Go code from `.proto` files.
*   `golangci-lint` for linting.
*   `gofumpt` and `goimports` for formatting.

### Building

To build the entire project, including all sub-modules:

```bash
go build ./...
```

To build the `zonnx` CLI tool:

```bash
go build -o zonnx ./zonnx/cmd/zonnx
```

## Coding Guidelines

We adhere to strict coding standards to maintain code quality and consistency.

### Formatting

Always format your code before committing. You can use the `make format` target:

```bash
make format
```

This runs `gofmt`, `goimports`, and `gofumpt`.

### Linting

We use `golangci-lint` to enforce code quality and catch potential issues. You can run the linter locally:

```bash
make lint
```

To automatically fix some linting issues:

```bash
make lint-fix
```

### Architectural Principles

Zerfoo, ZONNX, and ZMF follow strict architectural boundaries. Please familiarize yourself with these principles outlined in `zerfoo/README.md`, `zonnx/README.md`, and `zerfoo/docs/design.md`.

Key points:

*   **`zerfoo` (runtime):** Must NOT import `onnx` or `zonnx` packages. It consumes only ZMF models.
*   **`zonnx` (converter/tooling):** Must NOT import `github.com/zerfoo/zerfoo`. It converts ONNX to ZMF (and vice-versa).
*   **`zmf` (model format):** Must ONLY depend on `protobuf` (and test-only utilities).

These boundaries are enforced by CI checks. You can run these checks locally using:

```bash
make verify-architecture
```

## Testing

All new code must be accompanied by tests. We aim for high test coverage.

### Running Tests

To run all tests:

```bash
make test
```

To run tests with coverage analysis:

```bash
make test-coverage
```

To view a detailed coverage report:

```bash
make coverage-report
```

## Submitting Changes

We use GitHub Pull Requests for all contributions.

1.  **Create a new branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```
2.  **Make your changes:** Implement your feature or bug fix.
3.  **Write tests:** Ensure your changes are covered by tests.
4.  **Format and lint:** Run `make format` and `make lint` (or `make lint-fix`).
5.  **Verify architecture:** Run `make verify-architecture`.
6.  **Commit your changes:** Write clear, concise commit messages.
7.  **Push your branch:**
    ```bash
    git push origin feature/your-feature-name
    ```
8.  **Open a Pull Request:** Go to the `zerfoo` GitHub repository and open a new pull request from your branch.

### Pull Request Checklist

Our PR template includes a checklist to ensure compliance with our standards, including the architectural boundaries. Please ensure all items are checked before requesting a review.

## Code of Conduct

We adhere to the [Contributor Covenant Code of Conduct](LINK_TO_CODE_OF_CONDUCT). Please read it before contributing.

## License

By contributing to Zerfoo, you agree that your contributions will be licensed under the Apache 2.0 License.
