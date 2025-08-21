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
	perm, ok := attributes["perm"].([]int)
	if !ok {
		// perm is optional in ONNX, if not present, it's a simple reverse
		return New(engine, nil), nil
	}
	return New(engine, perm), nil
}
