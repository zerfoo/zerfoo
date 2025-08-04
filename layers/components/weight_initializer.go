package components

import (
	"math"
	"math/rand"

	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// WeightInitializer defines the interface for weight initialization strategies.
type WeightInitializer[T tensor.Numeric] interface {
	Initialize(inputSize, outputSize int) ([]T, error)
}

// XavierInitializer implements Xavier/Glorot initialization.
// Weights are sampled from a uniform distribution with variance 2/(fan_in + fan_out).
type XavierInitializer[T tensor.Numeric] struct {
	ops numeric.Arithmetic[T]
}

// NewXavierInitializer creates a new Xavier initializer.
func NewXavierInitializer[T tensor.Numeric](ops numeric.Arithmetic[T]) *XavierInitializer[T] {
	return &XavierInitializer[T]{ops: ops}
}

// Initialize generates weights using Xavier initialization.
func (x *XavierInitializer[T]) Initialize(inputSize, outputSize int) ([]T, error) {
	fanIn := float64(inputSize)
	fanOut := float64(outputSize)
	limit := math.Sqrt(6.0 / (fanIn + fanOut))

	weights := make([]T, inputSize*outputSize)
	for i := range weights {
		// Generate random value in [-limit, limit]
		// #nosec G404 - math/rand is acceptable for ML weight initialization
		val := (rand.Float64()*2 - 1) * limit
		weights[i] = x.ops.FromFloat32(float32(val))
	}

	return weights, nil
}

// HeInitializer implements He initialization.
// Weights are sampled from a normal distribution with variance 2/fan_in.
type HeInitializer[T tensor.Numeric] struct {
	ops numeric.Arithmetic[T]
}

// NewHeInitializer creates a new He initializer.
func NewHeInitializer[T tensor.Numeric](ops numeric.Arithmetic[T]) *HeInitializer[T] {
	return &HeInitializer[T]{ops: ops}
}

// Initialize generates weights using He initialization.
func (h *HeInitializer[T]) Initialize(inputSize, outputSize int) ([]T, error) {
	fanIn := float64(inputSize)
	stddev := math.Sqrt(2.0 / fanIn)

	weights := make([]T, inputSize*outputSize)
	for i := range weights {
		// Generate random value from normal distribution
		// #nosec G404 - math/rand is acceptable for ML weight initialization
		val := rand.NormFloat64() * stddev
		weights[i] = h.ops.FromFloat32(float32(val))
	}

	return weights, nil
}

// UniformInitializer implements simple uniform initialization.
type UniformInitializer[T tensor.Numeric] struct {
	ops   numeric.Arithmetic[T]
	scale float64
}

// NewUniformInitializer creates a new uniform initializer with the given scale.
func NewUniformInitializer[T tensor.Numeric](ops numeric.Arithmetic[T], scale float64) *UniformInitializer[T] {
	return &UniformInitializer[T]{ops: ops, scale: scale}
}

// Initialize generates weights using uniform initialization.
func (u *UniformInitializer[T]) Initialize(inputSize, outputSize int) ([]T, error) {
	weights := make([]T, inputSize*outputSize)
	for i := range weights {
		// #nosec G404 - math/rand is acceptable for ML weight initialization
		val := (rand.Float64()*2 - 1) * u.scale
		weights[i] = u.ops.FromFloat32(float32(val))
	}

	return weights, nil
}
