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
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewGroupQueryAttention(engine), nil
}
