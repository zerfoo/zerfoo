package transpose

import (
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	model.RegisterLayer("Transpose", BuildTranspose[float32])
}

func BuildTranspose[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	permAttr, ok := attributes["perm"]
	if !ok {
		return nil, fmt.Errorf("Transpose layer requires a 'perm' attribute")
	}

	perm, ok := permAttr.([]any)
	if !ok {
		// If it's not []any, maybe it's already []int64.
		perm64, ok := permAttr.([]int64)
		if !ok {
			return nil, fmt.Errorf("attribute 'perm' is not a valid integer slice")
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
