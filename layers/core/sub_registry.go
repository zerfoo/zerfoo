package core

import (
	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	model.RegisterLayer("Sub", BuildSub[float32])
}

func BuildSub[T tensor.Numeric](
	engine compute.Engine[T],
	_ numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	_ map[string]interface{},
) (graph.Node[T], error) {
	return NewSub(engine), nil
}
