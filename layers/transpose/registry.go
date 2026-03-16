package transpose

import (
	"errors"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BuildTranspose constructs a Transpose node, reading the permutation from attributes.
// If perm is absent, the ONNX default (reverse all axes) is applied at forward time.
func BuildTranspose[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	permAttr, ok := attributes["perm"]
	if !ok {
		// ONNX spec: when perm is absent, reverse all axes.
		// Pass nil; resolved at forward time when input rank is known.
		return New(engine, nil), nil
	}

	perm, ok := permAttr.([]any)
	if !ok {
		// If it's not []any, maybe it's already []int64.
		perm64, ok := permAttr.([]int64)
		if !ok {
			return nil, errors.New("attribute 'perm' is not a valid integer slice")
		}

		axes := make([]int, len(perm64))
		for i, v := range perm64 {
			axes[i] = int(v)
		}

		return New(engine, axes), nil
	}

	// convert []any to []int
	axes := make([]int, len(perm))
	for i, v := range perm {
		axes[i] = int(v.(int64))
	}

	return New(engine, axes), nil
}
