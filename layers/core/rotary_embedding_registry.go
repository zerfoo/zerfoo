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
	name string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	layer := NewRotaryEmbedding(engine)
	if name != "" {
		layer.SetName(name)
	}

	if attributes != nil {
		if v, ok := attributes["rope_base"]; ok {
			switch x := v.(type) {
			case float64:
				layer.base = x
			case float32:
				layer.base = float64(x)
			case int:
				layer.base = float64(x)
			case int64:
				layer.base = float64(x)
			}
		}
		if v, ok := attributes["max_seq_len"]; ok {
			switch x := v.(type) {
			case int:
				layer.maxSeqLen = x
			case int64:
				layer.maxSeqLen = int(x)
			case float64:
				layer.maxSeqLen = int(x)
			case float32:
				layer.maxSeqLen = int(x)
			}
		}
	}

	return layer, nil
}
