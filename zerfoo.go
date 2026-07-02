// Package zerfoo provides the core building blocks for creating and training neural networks.
// It offers a prelude of commonly used types to simplify development and enhance
// readability of model construction code.
package zerfoo

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"

	"github.com/zerfoo/zerfoo/layers/normalization"
)

// Prelude of commonly used types for building models.
// This allows developers to use zerfoo.Graph instead of graph.Graph,
// improving discoverability and developer experience.
type (
	// Graph represents a computation graph.
	//
	// Stable.
	Graph[T tensor.Numeric] struct {
		*graph.Graph[T]
	}

	// Node represents a node in the computation graph.
	//
	// Stable.
	Node[T tensor.Numeric] interface {
		graph.Node[T]
	}

	// Parameter represents a trainable parameter in the model.
	//
	// Stable.
	Parameter[T tensor.Numeric] struct {
		*graph.Parameter[T]
	}

	// Engine represents a computation engine (e.g., CPU or GPU).
	//
	// Stable.
	Engine[T tensor.Numeric] interface {
		compute.Engine[T]
	}

	// Tensor represents a multi-dimensional array.
	//
	// Stable.
	Tensor[T tensor.Numeric] struct {
		*tensor.TensorNumeric[T]
	}

	// Numeric represents a numeric type constraint for tensor elements.
	//
	// Stable.
	Numeric tensor.Numeric

	// LayerBuilder is a function that builds a computation graph layer.
	//
	// Stable.
	LayerBuilder[T tensor.Numeric] func(
		engine compute.Engine[T],
		ops numeric.Arithmetic[T],
		name string,
		params map[string]*graph.Parameter[T],
		attributes map[string]interface{},
	) (graph.Node[T], error)
)

// NewGraph creates a new computation graph builder for the given engine.
//
// Stable.
func NewGraph[T tensor.Numeric](engine compute.Engine[T]) *graph.Builder[T] {
	return graph.NewBuilder[T](engine)
}

// RegisterLayer registers a new layer builder for the given operation type.
//
// Stable.
func RegisterLayer[T tensor.Numeric](opType string, builder model.LayerBuilder[T]) {
	model.RegisterLayer[T](opType, builder)
}

// UnregisterLayer unregisters the layer builder for the given operation type.
//
// Stable.
func UnregisterLayer(opType string) {
	model.UnregisterLayer(opType)
}

// NewCPUEngine creates a new CPU computation engine for the given numeric type.
//
// Stable.
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
//
// Stable.
func NewFloat32Ops() numeric.Arithmetic[float32] {
	return numeric.Float32Ops{}
}

// NewTensor creates a new tensor with the given shape and data.
//
// Stable.
func NewTensor[T tensor.Numeric](shape []int, data []T) (*tensor.TensorNumeric[T], error) {
	return tensor.New[T](shape, data)
}

// NewMSE creates a new Mean Squared Error loss function.
//
// Stable.
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

// NewAdamW creates a new AdamW optimizer with the given hyperparameters.
//
// Stable.
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

// NewDefaultTrainer creates a new default trainer for the given graph, loss, optimizer, and gradient strategy.
//
// Stable.
func NewDefaultTrainer[T tensor.Numeric](
	g *graph.Graph[T],
	lossNode graph.Node[T],
	opt optimizer.Optimizer[T],
	strategy training.GradientStrategy[T],
) *training.DefaultTrainer[T] {
	return training.NewDefaultTrainer[T](g, lossNode, opt, strategy)
}

// Batch represents a training batch of inputs and targets.
//
// Stable.
type Batch[T tensor.Numeric] struct {
	Inputs  map[graph.Node[T]]*tensor.TensorNumeric[T]
	Targets *tensor.TensorNumeric[T]
}

// NewRMSNorm creates a new RMSNorm normalization layer with the given configuration.
//
// Stable.
func NewRMSNorm[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim int, options ...normalization.RMSNormOption[T]) (*normalization.RMSNorm[T], error) {
	return normalization.NewRMSNorm[T](name, engine, ops, modelDim, options...)
}
