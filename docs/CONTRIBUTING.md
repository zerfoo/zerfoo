# Contributing to Zerfoo

Thank you for your interest in contributing to Zerfoo! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Code Style and Conventions](#code-style-and-conventions)
4. [Testing Guidelines](#testing-guidelines)
5. [Submitting Changes](#submitting-changes)
6. [Issue Guidelines](#issue-guidelines)
7. [Architecture Guidelines](#architecture-guidelines)
8. [Performance Considerations](#performance-considerations)

## Getting Started

### Prerequisites

- **Go 1.24+**: Zerfoo requires Go 1.24 or later for generics support
- **Git**: For version control
- **Make**: For build automation (optional but recommended)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/zerfoo.git
   cd zerfoo
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/zerfoo/zerfoo.git
   ```

## Development Environment

### Initial Setup

```bash
# Install dependencies
go mod tidy

# Run tests to ensure everything works
go test ./...

# Build all packages
go build ./...

# Install development tools
go install golang.org/x/tools/cmd/godoc@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
```

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests and linting:
   ```bash
   # Run all tests
   go test ./...
   
   # Run tests with coverage
   go test -cover ./...
   
   # Run linting
   golangci-lint run
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. Push and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style and Conventions

### General Go Guidelines

- Follow standard Go conventions and idioms
- Use `gofmt` to format your code
- Use `golint` and `go vet` to check for issues
- Follow the [Effective Go](https://golang.org/doc/effective_go.html) guidelines

### Zerfoo-Specific Conventions

#### Package Organization

```go
// Package documentation should clearly explain the package purpose
// and provide usage examples.
package tensor

import (
    // Standard library imports first
    "fmt"
    "math"
    
    // Third-party imports second
    "github.com/some/external/package"
    
    // Internal imports last
    "github.com/zerfoo/zerfoo/compute"
)
```

#### Naming Conventions

- **Types**: Use PascalCase (e.g., `Tensor`, `Engine`, `NodeHandle`)
- **Functions**: Use PascalCase for exported, camelCase for unexported
- **Variables**: Use camelCase
- **Constants**: Use PascalCase or UPPER_CASE for package-level constants
- **Interfaces**: Should describe behavior, often ending in -er (e.g., `Engine`, `Optimizer`)

#### Generic Types

```go
// Use clear, descriptive type parameters
type Tensor[T Numeric] struct {
    // ...
}

// Use consistent parameter naming across the codebase
type Node[T Numeric] interface {
    Forward(inputs ...*Tensor[T]) (*Tensor[T], error)
    Backward(gradient *Tensor[T]) ([]*Tensor[T], error)
}
```

#### Error Handling

```go
// Use descriptive error messages
func NewTensor[T Numeric](shape []int, data []T) (*Tensor[T], error) {
    if len(shape) == 0 {
        return nil, fmt.Errorf("tensor shape cannot be empty")
    }
    // ...
}

// Define package-level error variables for common errors
var (
    ErrInvalidShape = fmt.Errorf("invalid tensor shape")
    ErrShapeMismatch = fmt.Errorf("tensor shape mismatch")
)
```

#### Documentation

```go
// Package documentation should include usage examples
/*
Package tensor provides n-dimensional array operations for machine learning.

Basic usage:

    tensor, err := tensor.New[float32]([]int{2, 3}, nil)
    if err != nil {
        return err
    }
    
    // Set values
    tensor.Set([]int{0, 1}, 3.14)
    
    // Get values
    value := tensor.At([]int{0, 1})
*/
package tensor

// Document all exported types
// Tensor represents an n-dimensional array of numeric values.
// It supports various numeric types through Go generics.
type Tensor[T Numeric] struct {
    // ...
}

// Document all exported functions with examples when helpful
// New creates a new tensor with the specified shape.
// If data is nil, the tensor is initialized with zero values.
//
// Example:
//   tensor, err := New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
func New[T Numeric](shape []int, data []T) (*Tensor[T], error) {
    // ...
}
```

### Interface Design

Follow the Interface Segregation Principle:

```go
// Good: Small, focused interfaces
type Forward[T Numeric] interface {
    Forward(inputs ...*Tensor[T]) (*Tensor[T], error)
}

type Backward[T Numeric] interface {
    Backward(gradient *Tensor[T]) ([]*Tensor[T], error)
}

type Parameters[T Numeric] interface {
    Parameters() []*Parameter[T]
}

// Combine when needed
type Node[T Numeric] interface {
    Forward[T]
    Backward[T]
    Parameters[T]
    OutputShape() []int
}
```

## Testing Guidelines

### Test Organization

```go
// Test file naming: *_test.go
package tensor

import (
    "testing"
    
    "github.com/zerfoo/zerfoo/testing/testutils"
)

// Test function naming: TestFunctionName or TestTypeName_Method
func TestNew(t *testing.T) {
    // ...
}

func TestTensor_At(t *testing.T) {
    // ...
}
```

### Test Structure

```go
func TestTensor_At(t *testing.T) {
    tests := []struct {
        name     string
        shape    []int
        data     []float32
        indices  []int
        expected float32
        wantErr  bool
    }{
        {
            name:     "valid 2D access",
            shape:    []int{2, 3},
            data:     []float32{1, 2, 3, 4, 5, 6},
            indices:  []int{1, 2},
            expected: 6,
            wantErr:  false,
        },
        {
            name:     "out of bounds",
            shape:    []int{2, 3},
            data:     []float32{1, 2, 3, 4, 5, 6},
            indices:  []int{2, 0},
            expected: 0,
            wantErr:  true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            tensor, err := New(tt.shape, tt.data)
            if err != nil {
                t.Fatalf("failed to create tensor: %v", err)
            }
            
            result, err := tensor.At(tt.indices)
            if (err != nil) != tt.wantErr {
                t.Errorf("At() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            
            if !tt.wantErr && result != tt.expected {
                t.Errorf("At() = %v, want %v", result, tt.expected)
            }
        })
    }
}
```

### Test Coverage

- Aim for >80% test coverage for new code
- Test both happy paths and error cases
- Include edge cases and boundary conditions
- Test generic functions with different numeric types

```bash
# Run tests with coverage
go test -cover ./...

# Generate detailed coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Benchmarks

```go
func BenchmarkTensor_MatMul(b *testing.B) {
    engine := compute.NewCPUEngine()
    a, _ := tensor.New[float32]([]int{100, 100}, nil)
    c, _ := tensor.New[float32]([]int{100, 100}, nil)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        engine.MatMul(context.Background(), a, c, c)
    }
}
```

### Mock Objects

Use interfaces for testability:

```go
// MockEngine for testing
type MockEngine[T Numeric] struct {
    MatMulFunc func(ctx context.Context, a, b, dst *Tensor[T]) error
}

func (m *MockEngine[T]) MatMul(ctx context.Context, a, b, dst *Tensor[T]) error {
    if m.MatMulFunc != nil {
        return m.MatMulFunc(ctx, a, b, dst)
    }
    return nil
}
```

## Submitting Changes

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

Examples:
```
feat(tensor): add broadcasting support for element-wise operations

Add automatic broadcasting for tensors with compatible shapes.
This enables operations like adding a scalar to a matrix.

Closes #123

fix(graph): resolve memory leak in parameter gradient accumulation

The gradient tensors were not being properly freed after each
training step, causing memory usage to grow over time.

test(layers): add comprehensive tests for dense layer backward pass

Includes tests for gradient computation with various input shapes
and parameter configurations.
```

### Pull Request Process

1. **Update documentation** if you're changing APIs
2. **Add tests** for new functionality
3. **Update changelog** if applicable
4. **Ensure CI passes** before requesting review
5. **Write a clear PR description** explaining:
   - What changes you made
   - Why you made them
   - How to test them

### PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added new tests for this change

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings or errors
```

## Issue Guidelines

### Bug Reports

Include:
- **Go version**: `go version`
- **OS and architecture**: `uname -a`
- **Zerfoo version**: git commit hash
- **Minimal reproduction case**
- **Expected vs actual behavior**
- **Stack trace if applicable**

### Feature Requests

Include:
- **Use case**: Why do you need this feature?
- **Proposed API**: How should it work?
- **Alternatives considered**: What other approaches did you consider?
- **Breaking changes**: Will this break existing code?

## Architecture Guidelines

### Package Dependencies

Follow the dependency hierarchy:

```
examples/
├── depends on all packages
│
pkg/
├── depends on core packages
│
training/
├── depends on: graph, tensor, compute
│
layers/
├── depends on: graph, tensor, compute
│
graph/
├── depends on: tensor, compute
│
compute/
├── depends on: tensor, device
│
tensor/
├── depends on: device (optional)
│
device/
├── no internal dependencies
```

### Interface Design Principles

1. **Keep interfaces small**: Follow the Interface Segregation Principle
2. **Define interfaces where they're used**: Don't create interfaces just because
3. **Use composition over inheritance**: Embed structs instead of deep hierarchies
4. **Make zero values useful**: Structs should be usable without explicit initialization

### Generic Design

```go
// Good: Clear constraints
type Numeric interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint32 | ~uint64 |
    ~float32 | ~float64 |
    float8.Float8 | float16.Float16
}

// Good: Consistent parameter naming
type Engine[T Numeric] interface {
    MatMul(ctx context.Context, a, b, dst *Tensor[T]) error
    Add(ctx context.Context, a, b, dst *Tensor[T]) error
}

// Good: Type parameter constraints where needed
func NewTensor[T Numeric](shape []int, data []T) (*Tensor[T], error) {
    // ...
}
```

## Performance Considerations

### Memory Management

- **Avoid unnecessary allocations** in hot paths
- **Reuse tensors** when possible
- **Use memory pools** for frequently allocated objects
- **Profile memory usage** with `go tool pprof`

### Optimization Guidelines

```go
// Good: Reuse destination tensor
func (e *CPUEngine[T]) Add(ctx context.Context, a, b, dst *Tensor[T]) error {
    // Operate in-place on dst
    for i := range dst.data {
        dst.data[i] = a.data[i] + b.data[i]
    }
    return nil
}

// Good: Batch operations
func (e *CPUEngine[T]) MatMul(ctx context.Context, a, b, dst *Tensor[T]) error {
    // Use efficient BLAS implementation
    return e.blas.Gemm(/* ... */)
}
```

### Profiling

```bash
# CPU profiling
go test -cpuprofile cpu.prof -bench .
go tool pprof cpu.prof

# Memory profiling
go test -memprofile mem.prof -bench .
go tool pprof mem.prof

# Trace analysis
go test -trace trace.out -bench .
go tool trace trace.out
```

## Questions or Help?

- **Documentation**: Check the [Developer Guide](DEVELOPER_GUIDE.md)
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Architecture**: Review the [Design Document](design.md)

Thank you for contributing to Zerfoo! 🚀