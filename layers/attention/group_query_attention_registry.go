package attention

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BuildGroupQueryAttention constructs a GroupedQueryAttention node for the model builder.
// Unused parameters are accepted to satisfy the common builder signature.
func BuildGroupQueryAttention[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	// Delegate to the proper grouped-query attention builder that
	// reads attributes and parameters to construct the layer.
	return buildGroupedQueryAttention[T](engine, ops, name, params, attributes)
}
