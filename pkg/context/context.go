package context

// Context defines an interface for fundamental tensor operations on a specific hardware backend:contentReference[oaicite:25]{index=25}.
// This allows the same model code to run on different hardware by injecting the appropriate Context.
type Context interface {
    MatMul(a interface{}, b interface{}) interface{}
    Add(a interface{}, b interface{}) interface{}
    // TODO: define other operations (activation functions, etc.)
}

// CPUContext is a stub implementation of Context for CPU computations.
type CPUContext struct {
    // Fields for CPU-specific optimizations or state
}

// MatMul multiplies two matrices on CPU (placeholder implementation).
func (ctx *CPUContext) MatMul(a interface{}, b interface{}) interface{} {
    // TODO: perform matrix multiplication (e.g., using Gonum BLAS)
    return nil
}

// Add adds two tensors on CPU (placeholder implementation).
func (ctx *CPUContext) Add(a interface{}, b interface{}) interface{} {
    // TODO: perform element-wise addition
    return nil
}
