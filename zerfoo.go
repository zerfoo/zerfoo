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
	"github.com/zerfoo/zmf"

	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/training/optimizer"
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

// NewRMSNorm is a factory function for creating RMSNorm layers.
func NewRMSNorm[T tensor.Numeric](name string, engine compute.Engine[T], ops numeric.Arithmetic[T], modelDim int, options ...normalization.RMSNormOption[T]) (*normalization.RMSNorm[T], error) {
	return normalization.NewRMSNorm[T](name, engine, ops, modelDim, options...)
}

// NewAdamW creates a new AdamW optimizer.
// Wrapper matching training/optimizer.NewAdamW signature.
func NewAdamW[T tensor.Numeric](engine compute.Engine[T], learningRate, beta1, beta2, epsilon, weightDecay T) *optimizer.AdamW[T] {
	return optimizer.NewAdamW[T](engine, learningRate, beta1, beta2, epsilon, weightDecay)
}
