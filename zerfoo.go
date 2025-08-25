// Package zerfoo provides the core building blocks for creating and training neural networks.
// It offers a prelude of commonly used types to simplify development and enhance
// readability of model construction code.
package zerfoo

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/zmf"

	"github.com/zerfoo/zerfoo/layers/normalization"
)

// Prelude of commonly used types for building models.
// This allows developers to use `zerfoo.Graph` instead of `graph.Graph`,
// improving discoverability and developer experience.
type (
	// Graph represents a computation graph.
	Graph[T tensor.Numeric] struct {
		*graph.Graph[T]
	}

	// Node represents a node in the computation graph.
	Node[T tensor.Numeric] interface {
		graph.Node[T]
	}

	// Parameter represents a trainable parameter in the model.
	Parameter[T tensor.Numeric] struct {
		*graph.Parameter[T]
	}

	// Engine represents a computation engine (e.g., CPU).
	Engine[T tensor.Numeric] interface {
		compute.Engine[T]
	}

	// Tensor represents a multi-dimensional array.
	Tensor[T tensor.Numeric] struct {
		*tensor.TensorNumeric[T]
	}

	// Numeric represents a numeric type constraint.
	Numeric tensor.Numeric

	// Model is a ZMF model.
	ZMFModel = zmf.Model

	// LayerBuilder is a function that builds a layer.
	LayerBuilder[T tensor.Numeric] func(
		engine compute.Engine[T],
		ops numeric.Arithmetic[T],
		name string,
		params map[string]*graph.Parameter[T],
		attributes map[string]interface{},
	) (graph.Node[T], error)
)

// NewGraph creates a new computation graph.
func NewGraph[T tensor.Numeric](engine compute.Engine[T]) *graph.Builder[T] {
	return graph.NewBuilder[T](engine)
}

// BuildFromZMF builds a graph from a ZMF model.
func BuildFromZMF[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	m *zmf.Model,
) (*graph.Graph[T], error) {
	return model.BuildFromZMF[T](engine, ops, m)
}

// RegisterLayer registers a new layer builder.
func RegisterLayer[T tensor.Numeric](opType string, builder model.LayerBuilder[T]) {
	model.RegisterLayer[T](opType, builder)
}

// UnregisterLayer unregisters a layer builder.
func UnregisterLayer(opType string) {
	model.UnregisterLayer(opType)
}

// NewCPUEngine creates a new CPU engine for the given numeric type.
func NewCPUEngine[T tensor.Numeric]() compute.Engine[T] {
	var ops numeric.Arithmetic[T]
	switch any(ops).(type) {
	case numeric.Arithmetic[float32]:
		ops = any(numeric.Float32Ops{}).(numeric.Arithmetic[T])
	case numeric.Arithmetic[float64]:
		ops = any(numeric.Float64Ops{}).(numeric.Arithmetic[T])
	default:
		ops = any(numeric.Float32Ops{}).(numeric.Arithmetic[T]) // Default to float32
	}
	return compute.NewCPUEngine[T](ops)
}

// NewFloat32Ops returns the float32 arithmetic operations.
func NewFloat32Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

// NewTensor creates a new tensor with the given shape and data.
func NewTensor[T tensor.Numeric](shape []int, data []T) (*tensor.TensorNumeric[T], error) {
	return tensor.New[T](shape, data)
}

// NewMSE creates a new Mean Squared Error loss function.
func NewMSE[T tensor.Numeric](engine compute.Engine[T]) *loss.MSE[T] {
	var ops numeric.Arithmetic[T]
	switch any(ops).(type) {
	case numeric.Arithmetic[float32]:
		ops = any(numeric.Float32Ops{}).(numeric.Arithmetic[T])
	case numeric.Arithmetic[float64]:
		ops = any(numeric.Float64Ops{}).(numeric.Arithmetic[T])
	default:
		ops = any(numeric.Float32Ops{}).(numeric.Arithmetic[T]) // Default to float32
	}
	return loss.NewMSE[T](engine, ops)
}

// NewAdamW creates a new AdamW optimizer.
func NewAdamW[T tensor.Numeric](learningRate, beta1, beta2, epsilon, weightDecay T) *optimizer.AdamW[T] {
	var ops numeric.Arithmetic[T]
	switch any(ops).(type) {
	case numeric.Arithmetic[float32]:
		ops = any(numeric.Float32Ops{}).(numeric.Arithmetic[T])
	case numeric.Arithmetic[float64]:
		ops = any(numeric.Float64Ops{}).(numeric.Arithmetic[T])
	default:
		ops = any(numeric.Float32Ops{}).(numeric.Arithmetic[T]) // Default to float32
	}
	engine := compute.NewCPUEngine[T](ops)
	return optimizer.NewAdamW[T](engine, learningRate, beta1, beta2, epsilon, weightDecay)
}

// NewDefaultTrainer creates a new default trainer.
func NewDefaultTrainer[T tensor.Numeric](
	graph *graph.Graph[T],
	loss graph.Node[T],
	opt optimizer.Optimizer[T],
	strategy training.GradientStrategy[T],
) *training.DefaultTrainer[T] {
	return training.NewDefaultTrainer[T](graph, loss, opt, strategy)
}

// Batch represents a training batch.
type Batch[T tensor.Numeric] struct {
	Inputs  map[graph.Node[T]]*tensor.TensorNumeric[T]
	Targets *tensor.TensorNumeric[T]
}

// NewRMSNorm is a factory function for creating RMSNorm layers.
func NewRMSNorm[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim int, options ...normalization.RMSNormOption[T]) (*normalization.RMSNorm[T], error) {
	return normalization.NewRMSNorm[T](name, engine, ops, modelDim, options...)
}
