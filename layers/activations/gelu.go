package activations

import (
	"math"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// Gelu represents a standard Gelu activation layer.
// Implements: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
type Gelu[T tensor.Float] struct {
	*BaseActivation[T]
}

// NewGelu creates a new standard Gelu activation layer.
func NewGelu[T tensor.Float](engine compute.Engine[T], ops numeric.Arithmetic[T]) *Gelu[T] {
	// Define forward operation for standard Gelu
	forwardOp := func(x T) T {
		// Constants for Gelu computation
		// sqrt(2/π) ≈ 0.7978845608
		// 0.044715 is the coefficient for x³ term
		sqrtTwoPi := T(0.7978845608)
		coeff := T(0.044715)
		half := T(0.5)
		one := T(1.0)
		
		// Compute x³ term
		xCubed := x * x * x
		
		// Compute the argument to tanh: sqrt(2/π) * (x + 0.044715 * x³)
		tanhArg := sqrtTwoPi * (x + coeff*xCubed)
		
		// Compute tanh using built-in math function
		tanhVal := T(math.Tanh(float64(tanhArg)))
		
		// Final computation: 0.5 * x * (1 + tanh(...))
		return half * x * (one + tanhVal)
	}
	
	// Define backward operation for standard Gelu
	backwardOp := func(x T) T {
		// Constants for Gelu gradient computation
		sqrtTwoPi := T(0.7978845608)
		coeff := T(0.044715)
		half := T(0.5)
		one := T(1.0)
		
		// Compute x² and x³
		xSquared := x * x
		xCubed := xSquared * x
		
		// Compute the argument to tanh
		tanhArg := sqrtTwoPi * (x + coeff*xCubed)
		tanhVal := T(math.Tanh(float64(tanhArg)))
		
		// Compute sech²(tanhArg) = 1 - tanh²(tanhArg)
		sechSquared := one - tanhVal*tanhVal
		
		// Compute derivative of the argument to tanh: sqrt(2/π) * (1 + 3 * 0.044715 * x²)
		tanhArgDerivative := sqrtTwoPi * (one + T(3.0)*coeff*xSquared)
		
		// Compute the full gradient:
		// d/dx[0.5 * x * (1 + tanh(...))] = 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d/dx[tanhArg]
		return half*(one+tanhVal) + half*x*sechSquared*tanhArgDerivative
	}
	
	base := NewBaseActivation(engine, ops, "Gelu", 
		WithForwardOp(forwardOp),
		WithBackwardOp(backwardOp))
	return &Gelu[T]{BaseActivation: base}
}

// Forward and Backward are handled by BaseActivation using the function operations

// BuildGelu constructs a standard Gelu layer for the registry.
func BuildGelu[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewGelu(engine, ops), nil
}