// Package activations provides activation function layers.
package activations

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BuildFastGelu constructs a FastGelu layer for the registry.
func BuildFastGelu[T tensor.Float](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewFastGelu(engine), nil
}

// BuildSigmoid constructs a Sigmoid activation layer for the registry.
func BuildSigmoid[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewSigmoid(engine, ops), nil
}
