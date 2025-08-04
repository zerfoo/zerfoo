# Zerfoo API Reference

This document provides comprehensive API reference for all Zerfoo packages.

## Table of Contents

1. [tensor](#tensor-package)
2. [graph](#graph-package)
3. [compute](#compute-package)
4. [layers](#layers-package)
5. [training](#training-package)
6. [device](#device-package)
7. [distributed](#distributed-package)
8. [model](#model-package)
9. [pkg](#pkg-package)

## tensor Package

The `tensor` package provides the core tensor data structure and operations for Zerfoo.

### Types

#### `type Tensor[T Numeric]`

Tensor represents an n-dimensional array of numeric values with generic type T.

```go
type Tensor[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Methods:**

```go
// Shape returns the dimensions of the tensor
func (t *Tensor[T]) Shape() []int

// Data returns the underlying data slice
func (t *Tensor[T]) Data() []T

// At returns the value at the specified indices
func (t *Tensor[T]) At(indices []int) (T, error)

// Set sets the value at the specified indices
func (t *Tensor[T]) Set(indices []int, value T) error

// Reshape returns a new tensor with the same data but different shape
func (t *Tensor[T]) Reshape(newShape []int) (*Tensor[T], error)

// Copy creates a deep copy of the tensor
func (t *Tensor[T]) Copy() *Tensor[T]

// String returns a string representation of the tensor
func (t *Tensor[T]) String() string
```

#### `type Numeric`

Numeric defines the constraint for numeric types that can be used in tensors.

```go
type Numeric interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
    ~uint | ~uint32 | ~uint64 |
    ~float32 | ~float64 |
    float8.Float8 |
    float16.Float16
}
```

### Functions

#### `func New[T Numeric](shape []int, data []T) (*Tensor[T], error)`

Creates a new tensor with the given shape and data. If data is nil, allocates a zero-filled slice.

**Parameters:**
- `shape`: Dimensions of the tensor
- `data`: Optional initial data (can be nil)

**Returns:**
- `*Tensor[T]`: New tensor
- `error`: Error if shape is invalid or data length doesn't match

**Example:**
```go
// Create a 2x3 matrix
tensor, err := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
if err != nil {
    panic(err)
}

// Create a zero-filled 3x3 matrix
zeros, err := tensor.New[float32]([]int{3, 3}, nil)
```

#### `func NewScalar[T Numeric](value T) (*Tensor[T], error)`

Creates a scalar (0-dimensional) tensor with the given value.

#### `func Zeros[T Numeric](shape []int) (*Tensor[T], error)`

Creates a tensor filled with zeros.

#### `func Ones[T Numeric](shape []int) (*Tensor[T], error)`

Creates a tensor filled with ones.

#### `func Eye[T Numeric](n int) (*Tensor[T], error)`

Creates an identity matrix of size n×n.

### Operations

#### Element-wise Operations

```go
// Add performs element-wise addition
func Add[T Numeric](a, b *Tensor[T]) (*Tensor[T], error)

// Sub performs element-wise subtraction
func Sub[T Numeric](a, b *Tensor[T]) (*Tensor[T], error)

// Mul performs element-wise multiplication
func Mul[T Numeric](a, b *Tensor[T]) (*Tensor[T], error)

// Div performs element-wise division
func Div[T Numeric](a, b *Tensor[T]) (*Tensor[T], error)
```

#### Matrix Operations

```go
// MatMul performs matrix multiplication
func MatMul[T Numeric](a, b *Tensor[T]) (*Tensor[T], error)

// Transpose returns the transpose of a 2D tensor
func Transpose[T Numeric](t *Tensor[T]) (*Tensor[T], error)
```

#### Reduction Operations

```go
// Sum computes the sum along specified axes
func Sum[T Numeric](t *Tensor[T], axes []int) (*Tensor[T], error)

// Mean computes the mean along specified axes
func Mean[T Numeric](t *Tensor[T], axes []int) (*Tensor[T], error)

// Max finds the maximum value along specified axes
func Max[T Numeric](t *Tensor[T], axes []int) (*Tensor[T], error)

// Min finds the minimum value along specified axes
func Min[T Numeric](t *Tensor[T], axes []int) (*Tensor[T], error)
```

---

## graph Package

The `graph` package provides computational graph construction and automatic differentiation.

### Types

#### `type Node[T Numeric]`

Node represents a node in the computation graph.

```go
type Node[T Numeric] interface {
    // OutputShape returns the shape of the output tensor
    OutputShape() []int
    
    // Forward computes the output given inputs
    Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)
    
    // Backward computes gradients with respect to inputs and parameters
    Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error)
    
    // Parameters returns the trainable parameters
    Parameters() []*Parameter[T]
}
```

#### `type Parameter[T Numeric]`

Parameter is a container for trainable tensors with their gradients.

```go
type Parameter[T Numeric] struct {
    Name     string
    Value    *tensor.Tensor[T]
    Gradient *tensor.Tensor[T]
}
```

#### `type Builder[T Numeric]`

Builder helps construct computational graphs.

```go
type Builder[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Methods:**

```go
// Input creates an input node with the specified shape
func (b *Builder[T]) Input(shape []int) NodeHandle

// AddNode adds a node to the graph with given inputs
func (b *Builder[T]) AddNode(node Node[T], inputs ...NodeHandle) (NodeHandle, error)

// Build finalizes the graph and returns forward/backward functions
func (b *Builder[T]) Build(output NodeHandle) (ForwardFunc[T], BackwardFunc[T], error)

// Parameters returns all parameters in the graph
func (b *Builder[T]) Parameters() []*Parameter[T]
```

#### `type NodeHandle`

NodeHandle is an opaque identifier for nodes in the graph.

```go
type NodeHandle struct {
    // contains filtered or unexported fields
}
```

### Functions

#### `func NewBuilder[T Numeric](engine compute.Engine[T]) *Builder[T]`

Creates a new graph builder with the specified compute engine.

#### `func NewParameter[T Numeric](name string, value *tensor.Tensor[T], newTensorFn func([]int, []T) (*tensor.Tensor[T], error)) (*Parameter[T], error)`

Creates a new parameter with the given name and value tensor.

### Function Types

#### `type ForwardFunc[T Numeric]`

```go
type ForwardFunc[T Numeric] func(inputs map[NodeHandle]*tensor.Tensor[T]) *tensor.Tensor[T]
```

Function for executing forward pass through the graph.

#### `type BackwardFunc[T Numeric]`

```go
type BackwardFunc[T Numeric] func(gradient *tensor.Tensor[T], inputs map[NodeHandle]*tensor.Tensor[T]) map[NodeHandle]*tensor.Tensor[T]
```

Function for executing backward pass through the graph.

---

## compute Package

The `compute` package provides hardware abstraction for tensor operations.

### Types

#### `type Engine[T Numeric]`

Engine interface defines the contract for compute backends.

```go
type Engine[T Numeric] interface {
    // Basic operations
    MatMul(ctx context.Context, a, b, dst *tensor.Tensor[T]) error
    Add(ctx context.Context, a, b, dst *tensor.Tensor[T]) error
    Sub(ctx context.Context, a, b, dst *tensor.Tensor[T]) error
    Mul(ctx context.Context, a, b, dst *tensor.Tensor[T]) error
    Div(ctx context.Context, a, b, dst *tensor.Tensor[T]) error
    
    // Memory operations
    Zero(ctx context.Context, t *tensor.Tensor[T]) error
    Copy(ctx context.Context, dst, src *tensor.Tensor[T]) error
    
    // Activation functions
    ReLU(ctx context.Context, input, output *tensor.Tensor[T]) error
    Sigmoid(ctx context.Context, input, output *tensor.Tensor[T]) error
    Tanh(ctx context.Context, input, output *tensor.Tensor[T]) error
    
    // Reduction operations
    Sum(ctx context.Context, input *tensor.Tensor[T], axes []int, output *tensor.Tensor[T]) error
    Mean(ctx context.Context, input *tensor.Tensor[T], axes []int, output *tensor.Tensor[T]) error
}
```

#### `type CPUEngine[T Numeric]`

CPU implementation of the Engine interface.

```go
type CPUEngine[T Numeric] struct {
    // contains filtered or unexported fields
}
```

#### `type DebugEngine[T Numeric]`

Debug wrapper that logs all operations.

```go
type DebugEngine[T Numeric] struct {
    // contains filtered or unexported fields
}
```

### Functions

#### `func NewCPUEngine[T Numeric]() *CPUEngine[T]`

Creates a new CPU compute engine.

#### `func NewDebugEngine[T Numeric](underlying Engine[T]) *DebugEngine[T]`

Creates a debug wrapper around an existing engine.

---

## layers Package

The `layers` package provides pre-built neural network layers.

### core Subpackage

#### `type Dense[T Numeric]`

Fully connected (dense) layer.

```go
type Dense[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Methods:**

```go
func (d *Dense[T]) OutputShape() []int
func (d *Dense[T]) Forward(inputs ...*tensor.Tensor[T]) (*tensor.Tensor[T], error)
func (d *Dense[T]) Backward(outputGradient *tensor.Tensor[T]) ([]*tensor.Tensor[T], error)
func (d *Dense[T]) Parameters() []*graph.Parameter[T]
```

**Constructor:**

```go
func NewDense[T Numeric](inputSize, outputSize int, engine compute.Engine[T]) *Dense[T]
```

#### `type Dropout[T Numeric]`

Dropout regularization layer.

```go
type Dropout[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewDropout[T Numeric](rate float64) *Dropout[T]
```

#### `type LayerNorm[T Numeric]`

Layer normalization.

```go
type LayerNorm[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewLayerNorm[T Numeric](normalizedShape []int, engine compute.Engine[T]) *LayerNorm[T]
```

### activations Subpackage

#### `type ReLU[T Numeric]`

Rectified Linear Unit activation.

```go
type ReLU[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewReLU[T Numeric](engine compute.Engine[T]) *ReLU[T]
```

#### `type Sigmoid[T Numeric]`

Sigmoid activation function.

```go
type Sigmoid[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewSigmoid[T Numeric](engine compute.Engine[T]) *Sigmoid[T]
```

#### `type Tanh[T Numeric]`

Hyperbolic tangent activation.

```go
type Tanh[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewTanh[T Numeric](engine compute.Engine[T]) *Tanh[T]
```

### attention Subpackage

#### `type MultiHeadAttention[T Numeric]`

Multi-head attention mechanism.

```go
type MultiHeadAttention[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewMultiHeadAttention[T Numeric](dModel, numHeads int, engine compute.Engine[T]) *MultiHeadAttention[T]
```

---

## training Package

The `training` package provides training utilities and optimizers.

### optimizer Subpackage

#### `type Optimizer[T Numeric]`

Interface for parameter optimizers.

```go
type Optimizer[T Numeric] interface {
    Step(parameters []*graph.Parameter[T]) error
    SetLearningRate(lr float64)
    GetLearningRate() float64
}
```

#### `type SGD[T Numeric]`

Stochastic Gradient Descent optimizer.

```go
type SGD[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewSGD[T Numeric](learningRate float64) *SGD[T]
func NewSGDWithMomentum[T Numeric](learningRate, momentum float64) *SGD[T]
```

#### `type Adam[T Numeric]`

Adam optimizer.

```go
type Adam[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewAdam[T Numeric](learningRate float64) *Adam[T]
func NewAdamWithParams[T Numeric](learningRate, beta1, beta2, epsilon float64) *Adam[T]
```

### loss Subpackage

#### `type Loss[T Numeric]`

Interface for loss functions.

```go
type Loss[T Numeric] interface {
    Forward(predictions, targets *tensor.Tensor[T]) *tensor.Tensor[T]
    Backward(predictions, targets *tensor.Tensor[T]) *tensor.Tensor[T]
}
```

#### `type CrossEntropy[T Numeric]`

Cross-entropy loss function.

```go
type CrossEntropy[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewCrossEntropy[T Numeric]() *CrossEntropy[T]
```

#### `type MSE[T Numeric]`

Mean Squared Error loss function.

```go
type MSE[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewMSE[T Numeric]() *MSE[T]
```

### scheduler Subpackage

#### `type LRScheduler[T Numeric]`

Interface for learning rate schedulers.

```go
type LRScheduler[T Numeric] interface {
    Step(optimizer Optimizer[T])
    GetLR() float64
}
```

#### `type StepLR[T Numeric]`

Step-based learning rate scheduler.

```go
type StepLR[T Numeric] struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewStepLR[T Numeric](stepSize int, gamma float64) *StepLR[T]
```

---

## device Package

The `device` package provides hardware device abstractions.

### Types

#### `type Device`

Interface representing a compute device.

```go
type Device interface {
    Name() string
    Type() DeviceType
    ID() int
    Available() bool
}
```

#### `type DeviceType`

Enumeration of device types.

```go
type DeviceType int

const (
    CPU DeviceType = iota
    CUDA
    OpenCL
)
```

#### `type Allocator`

Interface for memory allocation.

```go
type Allocator interface {
    Allocate(size int) ([]byte, error)
    Free(ptr []byte) error
}
```

### Functions

#### `func ListDevices() []Device`

Returns a list of available devices.

#### `func GetDevice(deviceType DeviceType, id int) (Device, error)`

Gets a specific device by type and ID.

---

## distributed Package

The `distributed` package provides distributed training capabilities.

### Types

#### `type DistributedStrategy`

Interface for distributed training strategies.

```go
type DistributedStrategy interface {
    Init() error
    AllReduceGradients(parameters []*graph.Parameter[T]) error
    Barrier() error
    Cleanup() error
}
```

#### `type AllReduceStrategy`

All-reduce distributed training strategy.

```go
type AllReduceStrategy struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewAllReduceStrategy(config Config) *AllReduceStrategy
```

#### `type ParameterServerStrategy`

Parameter server distributed training strategy.

```go
type ParameterServerStrategy struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewParameterServerStrategy(config Config) *ParameterServerStrategy
```

---

## model Package

The `model` package provides model abstraction and serialization.

### Types

#### `type Model[T Numeric]`

Interface representing a trainable model.

```go
type Model[T Numeric] interface {
    Forward(inputs map[string]*tensor.Tensor[T]) map[string]*tensor.Tensor[T]
    Parameters() []*graph.Parameter[T]
    Train(mode bool)
    IsTraining() bool
}
```

### Functions

#### `func SaveModel[T Numeric](model Model[T], path string) error`

Saves a model to disk.

#### `func LoadModel[T Numeric](path string, engine compute.Engine[T]) (Model[T], error)`

Loads a model from disk.

---

## pkg Package

The `pkg` package contains external integrations and utilities.

### onnx Subpackage

#### `type Exporter`

ONNX model exporter.

```go
type Exporter struct {
    // contains filtered or unexported fields
}
```

**Methods:**

```go
func (e *Exporter) Export(model model.Model[T], path string) error
```

**Constructor:**

```go
func NewExporter() *Exporter
```

#### `type Importer`

ONNX model importer.

```go
type Importer struct {
    // contains filtered or unexported fields
}
```

**Methods:**

```go
func (i *Importer) Import(path string, engine compute.Engine[T]) (model.Model[T], error)
```

**Constructor:**

```go
func NewImporter() *Importer
```

### tokenizer Subpackage

#### `type Tokenizer`

Interface for text tokenization.

```go
type Tokenizer interface {
    Encode(text string) ([]int, error)
    Decode(tokens []int) (string, error)
    Vocabulary() map[string]int
    VocabularySize() int
}
```

#### `type SentencePieceTokenizer`

SentencePiece-based tokenizer.

```go
type SentencePieceTokenizer struct {
    // contains filtered or unexported fields
}
```

**Constructor:**

```go
func NewSentencePieceTokenizer(modelPath string) (*SentencePieceTokenizer, error)
```

---

## Error Handling

### Common Errors

All packages define common error variables:

```go
var (
    ErrInvalidShape     = errors.New("invalid tensor shape")
    ErrShapeMismatch    = errors.New("tensor shape mismatch")
    ErrOutOfBounds      = errors.New("index out of bounds")
    ErrInvalidOperation = errors.New("invalid operation")
    ErrNotImplemented   = errors.New("operation not implemented")
)
```

### Error Types

#### `type ShapeError`

Error type for shape-related issues.

```go
type ShapeError struct {
    Expected []int
    Actual   []int
    Op       string
}

func (e *ShapeError) Error() string
```

#### `type DeviceError`

Error type for device-related issues.

```go
type DeviceError struct {
    Device string
    Op     string
    Err    error
}

func (e *DeviceError) Error() string
```

---

## Examples

### Basic Tensor Operations

```go
package main

import (
    "fmt"
    "github.com/zerfoo/zerfoo/tensor"
)

func main() {
    // Create tensors
    a, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
    b, _ := tensor.New[float32]([]int{2, 3}, []float32{6, 5, 4, 3, 2, 1})
    
    // Element-wise addition
    c, _ := tensor.Add(a, b)
    fmt.Printf("a + b = %v\n", c.Data()) // [7 7 7 7 7 7]
    
    // Matrix multiplication (reshape first)
    a2d, _ := a.Reshape([]int{2, 3})
    b2d, _ := b.Reshape([]int{3, 2})
    result, _ := tensor.MatMul(a2d, b2d)
    fmt.Printf("a @ b = %v\n", result.Data())
}
```

### Building a Simple Model

```go
package main

import (
    "github.com/zerfoo/zerfoo/compute"
    "github.com/zerfoo/zerfoo/graph"
    "github.com/zerfoo/zerfoo/layers/activations"
    "github.com/zerfoo/zerfoo/layers/core"
)

func main() {
    engine := compute.NewCPUEngine[float32]()
    builder := graph.NewBuilder[float32](engine)
    
    // Input layer
    input := builder.Input([]int{1, 784}) // MNIST-like input
    
    // Hidden layers
    hidden1 := builder.AddNode(core.NewDense[float32](784, 128, engine), input)
    relu1 := builder.AddNode(activations.NewReLU[float32](engine), hidden1)
    
    hidden2 := builder.AddNode(core.NewDense[float32](128, 64, engine), relu1)
    relu2 := builder.AddNode(activations.NewReLU[float32](engine), hidden2)
    
    // Output layer
    output := builder.AddNode(core.NewDense[float32](64, 10, engine), relu2)
    
    // Build the graph
    forward, backward, err := builder.Build(output)
    if err != nil {
        panic(err)
    }
    
    // Now you can use forward and backward for training
    _ = forward
    _ = backward
}
```

For more examples, see the [examples directory](../examples/).

---

## Performance Notes

### Memory Management

- Tensors reuse underlying data when possible
- Use in-place operations to reduce allocations
- Consider using memory pools for frequently allocated objects

### Compute Optimization

- CPU engine uses optimized BLAS operations when available
- Debug engine should only be used during development
- Consider batch operations for better throughput

### Generic Type Performance

- Go's generics have minimal runtime overhead
- Type specialization occurs at compile time
- Use specific numeric types when possible (e.g., `float32` vs `Numeric`)

---

This API reference is generated from the source code. For the most up-to-date information, refer to the Go documentation:

```bash
go doc github.com/zerfoo/zerfoo/tensor
go doc github.com/zerfoo/zerfoo/graph
# etc.
```