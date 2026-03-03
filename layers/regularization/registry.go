package regularization

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// BuildDropout constructs a Dropout node from the provided attributes.
func BuildDropout[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]any,
) (graph.Node[T], error) {
	rate, ok := attributes["rate"].(float64)
	if !ok {
		return nil, fmt.Errorf("Dropout: missing or invalid 'rate' attribute")
	}
	return NewDropout(engine, ops, ops.FromFloat64(rate)), nil
}
