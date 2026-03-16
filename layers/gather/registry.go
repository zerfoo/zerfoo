package gather

import (
	"fmt"
	"strings"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BuildGather constructs a Gather layer. For embedding-style nodes whose name
// maps to a known weight parameter, weights are embedded in the layer.
// For "gather from shape" nodes where the indices are constant, the indices
// are embedded in the layer. All other Gather nodes operate as general ONNX
// Gather (axis-0 indexing).
func BuildGather[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	name string,
	params map[string]*graph.Parameter[T],
	attrs map[string]interface{},
) (graph.Node[T], error) {
	// Derive weight patterns from the node name. ONNX node names use "/"
	// separators (e.g. "/model/embed_tokens/Gather") while parameter names
	// use "." separators (e.g. "model.embed_tokens.weight"). Normalize the
	// node name so the pattern matches the parameter.
	normalized := strings.ReplaceAll(strings.TrimPrefix(name, "/"), "/", ".")
	weightPatterns := []string{
		name + ".weight",
		strings.TrimSuffix(name, "/Gather") + ".weight",
		strings.TrimSuffix(normalized, ".Gather") + ".weight",
	}
	for _, pattern := range weightPatterns {
		if param, exists := params[pattern]; exists {
			return NewWithWeights[T](engine, param.Value), nil
		}
	}

	// Check if constant indices were promoted to attributes by zonnx. These
	// appear as []int64 values with keys like "/Constant_output_0".
	for k, v := range attrs {
		if k == "axis" {
			continue
		}
		if ints, ok := v.([]int64); ok {
			intVals := make([]int, len(ints))
			for i, iv := range ints {
				intVals[i] = int(iv)
			}
			idxTensor, err := tensor.New[int]([]int{len(intVals)}, intVals)
			if err != nil {
				return nil, fmt.Errorf("failed to create embedded indices tensor: %w", err)
			}
			return NewWithIndices[T](engine, idxTensor), nil
		}
	}

	// General-purpose Gather: no embedded weights, takes (data, indices) inputs.
	return New[T](engine), nil
}
