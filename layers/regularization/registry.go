package regularization

import (
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
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

// BuildFeatureDropout constructs a FeatureDropout node from the provided attributes.
func BuildFeatureDropout[T tensor.Float](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]any,
) (graph.Node[T], error) {
	rate, ok := attributes["rate"].(float64)
	if !ok {
		return nil, fmt.Errorf("FeatureDropout: missing or invalid 'rate' attribute")
	}
	return NewFeatureDropout(engine, ops, ops.FromFloat64(rate)), nil
}
