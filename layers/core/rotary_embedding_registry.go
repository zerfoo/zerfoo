package core

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BuildRotaryEmbedding constructs a new RotaryPositionalEmbedding node for the given compute engine.
// It conforms to the layer registry builder signature used by the graph builder.
func BuildRotaryEmbedding[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewRotaryEmbedding(engine), nil
}
