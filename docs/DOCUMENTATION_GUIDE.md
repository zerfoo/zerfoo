# Package Documentation Generation

This document describes how to generate and maintain comprehensive API documentation for Zerfoo.

## Generating Documentation

### Using Go's Built-in Tools

#### 1. Local Documentation Server

Start a local documentation server:

```bash
# Install godoc if not already installed
go install golang.org/x/tools/cmd/godoc@latest

# Start the documentation server
godoc -http=:6060

# Open browser and navigate to:
# http://localhost:6060/pkg/github.com/zerfoo/zerfoo/
```

#### 2. Command Line Documentation

View documentation from the command line:

```bash
# Package overview
go doc github.com/zerfoo/zerfoo/tensor

# Specific type documentation  
go doc github.com/zerfoo/zerfoo/tensor.Tensor

# Function documentation
go doc github.com/zerfoo/zerfoo/tensor.New

# All package documentation
go doc -all github.com/zerfoo/zerfoo/tensor
```

#### 3. Generate Documentation Files

Generate documentation for all packages:

```bash
#!/bin/bash
# generate_docs.sh

PACKAGES=(
    "tensor"
    "graph" 
    "compute"
    "layers/core"
    "layers/activations"
    "training/optimizer"
    "training/loss"
    "distributed"
    "model"
    "pkg/onnx"
    "pkg/tokenizer"
)

mkdir -p docs/api

for pkg in "${PACKAGES[@]}"; do
    echo "Generating documentation for $pkg..."
    go doc -all "github.com/zerfoo/zerfoo/$pkg" > "docs/api/${pkg//\//_}.md"
done

echo "Documentation generation complete!"
```

### Using pkg.go.dev

Zerfoo's documentation is automatically available at:
https://pkg.go.dev/github.com/zerfoo/zerfoo

This includes:
- Package documentation
- Type definitions
- Function signatures
- Examples
- Cross-references

## Documentation Standards

### Package Documentation

Each package should have comprehensive documentation:

```go
/*
Package tensor provides n-dimensional array operations for machine learning.

The tensor package is the foundation of Zerfoo, providing the core data structure
for representing and manipulating multi-dimensional numeric data. It supports
various numeric types through Go generics and provides essential operations
for machine learning workloads.

Basic Usage

Create a new tensor:

    t, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
    if err != nil {
        return err
    }

Access and modify elements:

    value, err := t.At([]int{1, 2})  // Get element at position [1, 2]
    err = t.Set([]int{1, 2}, 42.0)   // Set element at position [1, 2]

Reshape tensors:

    reshaped, err := t.Reshape([]int{3, 2})

Element-wise operations:

    a, _ := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
    b, _ := tensor.New[float32]([]int{2, 2}, []float32{5, 6, 7, 8})
    
    sum, _ := tensor.Add(a, b)        // Element-wise addition
    product, _ := tensor.Mul(a, b)    // Element-wise multiplication

Matrix operations:

    matrix, _ := tensor.New[float32]([]int{2, 3}, nil)
    transposed, _ := tensor.Transpose(matrix)
    
Performance Considerations

The tensor package is designed for performance:

- Memory layout is optimized for cache efficiency
- Operations reuse memory when possible  
- Broadcasting is supported for compatible shapes
- Generic types allow compile-time optimization

Type Safety

Zerfoo uses Go's type system to prevent common errors:

- Shape mismatches are caught at compile time when possible
- Numeric types are constrained to prevent invalid operations
- The generic type system ensures type safety across operations

*/
package tensor
```

### Function Documentation

Each exported function should have clear documentation:

```go
// New creates a new tensor with the specified shape and optional initial data.
// 
// The shape parameter defines the dimensions of the tensor. For example,
// []int{2, 3} creates a 2x3 matrix, while []int{10} creates a vector of length 10.
//
// If data is provided, its length must match the total number of elements
// calculated from the shape (product of all dimensions). If data is nil,
// the tensor is initialized with zero values.
//
// Parameters:
//   - shape: Slice defining the tensor dimensions. Must not be empty.
//   - data: Optional initial data. Can be nil for zero initialization.
//
// Returns:
//   - *Tensor[T]: New tensor with the specified shape and data
//   - error: Error if shape is invalid or data length doesn't match
//
// Example:
//   // Create a 2x3 matrix with specific values
//   tensor, err := New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
//   
//   // Create a zero-filled 3x3 matrix
//   zeros, err := New[float32]([]int{3, 3}, nil)
//
//   // Create a scalar (0-dimensional tensor)
//   scalar, err := New[float32]([]int{}, []float32{42})
func New[T Numeric](shape []int, data []T) (*Tensor[T], error) {
    // Implementation...
}
```

### Type Documentation

Document all exported types thoroughly:

```go
// Tensor represents an n-dimensional array of numeric values.
//
// Tensors are the fundamental data structure in Zerfoo, providing efficient
// storage and manipulation of multi-dimensional numeric data. They support
// various operations including element access, reshaping, and mathematical
// operations.
//
// Generic Type Parameter:
//   T must satisfy the Numeric constraint, allowing for type-safe operations
//   with various numeric types including integers, floats, and custom types
//   like float16 and float8.
//
// Memory Layout:
//   Tensors use row-major order (C-style) for memory layout, which provides
//   good cache locality for most operations. The underlying data is stored
//   in a contiguous slice.
//
// Thread Safety:
//   Tensors are not thread-safe. Concurrent access must be synchronized
//   by the caller using appropriate synchronization primitives.
//
// Example:
//   var t Tensor[float32]
//   t, err := New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
//   if err != nil {
//       return err
//   }
//   
//   fmt.Printf("Shape: %v\n", t.Shape())     // [2 3]
//   fmt.Printf("Size: %d\n", t.Size())      // 6
//   
//   value, _ := t.At([]int{1, 2})           // Get element at [1,2]
//   t.Set([]int{1, 2}, 42.0)               // Set element at [1,2]
type Tensor[T Numeric] struct {
    shape   []int
    strides []int
    data    []T
    isView  bool
}
```

### Interface Documentation

Provide comprehensive interface documentation:

```go
// Node represents a computational node in a neural network graph.
//
// The Node interface defines the contract for all operations that can be
// part of a computational graph. This includes layers, activation functions,
// loss functions, and other transformations.
//
// Implementation Requirements:
//   - OutputShape() must return the shape of the tensor this node will produce
//   - Forward() must compute the node's output given its inputs
//   - Backward() must compute gradients with respect to inputs and parameters  
//   - Parameters() must return all trainable parameters owned by this node
//
// Thread Safety:
//   Node implementations should be thread-safe for concurrent forward passes,
//   but backward passes should be synchronized by the caller.
//
// Error Handling:
//   Nodes should return descriptive errors for invalid inputs or internal
//   failures. Shape mismatches should be detected early when possible.
//
// Generic Type Parameter:
//   T must satisfy the Numeric constraint to ensure type safety across
//   all tensor operations.
//
// Example Implementation:
//   type ReLU[T Numeric] struct {
//       engine compute.Engine[T]
//   }
//   
//   func (r *ReLU[T]) Forward(inputs ...*Tensor[T]) (*Tensor[T], error) {
//       if len(inputs) != 1 {
//           return nil, ErrInvalidInputCount
//       }
//       // Apply ReLU function...
//   }
type Node[T Numeric] interface {
    // OutputShape returns the shape of the tensor this node will produce.
    // This must be deterministic given the input shapes and should not
    // depend on the actual input data values.
    OutputShape() []int
    
    // Forward computes the output of this node given the input tensors.
    // The number and shapes of inputs must match what this node expects.
    // Returns the computed output tensor or an error if computation fails.
    Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)
    
    // Backward computes gradients with respect to inputs and parameters.
    // The outputGradient is the gradient of the loss with respect to this
    // node's output. Returns gradients with respect to each input tensor.
    // Parameter gradients should be accumulated in the Parameter objects.
    Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error)
    
    // Parameters returns all trainable parameters owned by this node.
    // Returns an empty slice if the node has no trainable parameters.
    // The returned parameters should include both the values and gradients.
    Parameters() []*Parameter[T]
}
```

## Example Documentation

### Comprehensive Package Example

Here's what good package documentation looks like:

```go
/*
Package compute provides hardware abstraction for tensor operations.

The compute package defines the Engine interface and implementations for different
hardware backends. This abstraction allows the same neural network code to run
on different devices (CPU, GPU, etc.) by simply changing the engine.

Architecture

The package follows a simple layered architecture:

    Application Code
         |
    Engine Interface  
         |
    Hardware Implementation (CPU, GPU, etc.)

All tensor operations are routed through the Engine interface, which provides
hardware-specific implementations of mathematical operations.

Engines

CPU Engine:
    The default engine that runs on CPU using optimized Go code and optionally
    BLAS libraries for linear algebra operations.

    engine := compute.NewCPUEngine[float32]()

Debug Engine:
    A wrapper around another engine that logs all operations for debugging.
    Should only be used during development due to performance overhead.

    debug := compute.NewDebugEngine[float32](cpuEngine)

Custom Engines:
    New engines can be implemented by satisfying the Engine interface.
    This allows supporting new hardware or optimization techniques.

Context and Cancellation

All engine operations accept a context.Context parameter for cancellation
and timeout support:

    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()
    
    err := engine.MatMul(ctx, a, b, result)
    if err != nil {
        // Handle timeout or cancellation
    }

Memory Management

Engines can optionally accept a destination tensor to avoid allocations:

    // Allocates new tensor for result
    result, err := tensor.MatMul(a, b)
    
    // Reuses existing tensor (more efficient)
    err := engine.MatMul(ctx, a, b, existingTensor)

Performance Considerations

- CPU engine uses multiple goroutines for large operations
- Debug engine has significant overhead and should not be used in production
- Consider batch operations for better throughput
- Memory reuse reduces garbage collection pressure

Thread Safety

Engine implementations must be thread-safe for concurrent operations.
The same engine instance can be safely used from multiple goroutines.

Error Handling

Engines return descriptive errors for:
- Invalid tensor shapes or dimensions
- Hardware-specific failures
- Context cancellation or timeout
- Memory allocation failures

Example Usage

    // Create engine
    engine := compute.NewCPUEngine[float32]()
    
    // Create tensors
    a, _ := tensor.New[float32]([]int{2, 3}, nil)
    b, _ := tensor.New[float32]([]int{3, 2}, nil)  
    result, _ := tensor.New[float32]([]int{2, 2}, nil)
    
    // Perform matrix multiplication
    ctx := context.Background()
    err := engine.MatMul(ctx, a, b, result)
    if err != nil {
        log.Fatal(err)
    }

*/
package compute
```

## Documentation Checklist

When writing documentation:

### ✅ **Required Elements**
- [ ] Package overview with purpose and usage
- [ ] All exported types documented
- [ ] All exported functions documented  
- [ ] Interface contracts clearly defined
- [ ] Parameter and return value descriptions
- [ ] Error conditions explained
- [ ] Examples for complex operations
- [ ] Performance considerations noted
- [ ] Thread safety guarantees specified

### ✅ **Best Practices**
- [ ] Use clear, concise language
- [ ] Provide runnable examples
- [ ] Explain the "why" not just the "what"
- [ ] Include common pitfalls and solutions
- [ ] Reference related functions/types
- [ ] Update documentation with code changes
- [ ] Test examples to ensure they work

### ✅ **Style Guidelines**
- [ ] Start with a brief summary sentence
- [ ] Use present tense ("returns", not "will return")
- [ ] Be consistent with terminology
- [ ] Follow Go documentation conventions
- [ ] Use proper capitalization and punctuation
- [ ] Include code examples in comments
- [ ] Reference other packages when appropriate

## Maintenance

### Automated Checks

Set up CI to verify documentation:

```bash
# Check for missing documentation
go doc -all ./... | grep -E "^(func|type|var|const)" | grep -v "^func.*{"

# Verify examples compile
go test -run Example

# Check documentation coverage
go test -cover ./...
```

### Manual Reviews

Regularly review documentation for:
- Accuracy with current implementation
- Clarity for new users
- Completeness of examples
- Links and references still work
- Formatting and style consistency

### Documentation Updates

When making code changes:
1. Update function/type documentation
2. Update package overview if needed
3. Add/update examples
4. Test all examples
5. Update changelog if breaking changes
6. Review related documentation

## Tools and Resources

### Useful Tools
- `godoc` - Local documentation server
- `go doc` - Command line documentation
- `gofmt` - Code formatting (affects doc comments)
- `golint` - Checks for missing documentation
- `govet` - Checks for common mistakes

### External Resources
- [Go Doc Comments](https://golang.org/doc/comment)
- [Effective Go Documentation](https://golang.org/doc/effective_go#commentary)
- [Go Package Documentation](https://blog.golang.org/package-names)

This comprehensive documentation strategy ensures that Zerfoo remains accessible to developers at all levels while maintaining professional standards.