package registry

import (
	"fmt"
	"slices"

	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
)

func init() {
	RegisterAll()
	originalLinear, _ := model.GetLayerBuilder[float32]("Linear")
	model.RegisterLayer("Linear", func(e compute.Engine[float32], o numeric.Arithmetic[float32], name string, p map[string]*graph.Parameter[float32], a map[string]any) (graph.Node[float32], error) {
		if weight := p["weights"]; weight != nil {
			return core.NewLinearFromParam(e, weight), nil
		}
		if originalLinear == nil {
			return nil, fmt.Errorf("linear builder unavailable")
		}
		return originalLinear(e, o, name, p, a)
	})

	model.RegisterLayer("ReLU", func(e compute.Engine[float32], o numeric.Arithmetic[float32], _ string, _ map[string]*graph.Parameter[float32], _ map[string]any) (graph.Node[float32], error) {
		return activations.NewReLU(e, o), nil
	})
	model.RegisterLayer("Dense", func(e compute.Engine[float32], o numeric.Arithmetic[float32], _ string, p map[string]*graph.Parameter[float32], _ map[string]any) (graph.Node[float32], error) {
		if p["weights"] == nil || p["bias"] == nil {
			return nil, fmt.Errorf("dense requires weights and bias")
		}
		return core.NewDenseFromParams(core.NewLinearFromParam(e, p["weights"]), core.NewBiasFromParam(e, o, p["bias"])), nil
	})
	model.RegisterLayer("Add", core.BuildAdd[float32])
	model.RegisterLayer("Mul", core.BuildMul[float32])
	model.RegisterLayer("Sub", core.BuildSub[float32])
	model.RegisterLayer("MatMul", core.BuildMatMul[float32])
	model.RegisterLayer("Softmax", activations.BuildSoftmax[float32])
	// These descriptors enable bounded composition. Qualification evidence is
	// attached only after the corresponding suites execute; registered != verified.
	for _, name := range []string{"Linear", "Dense", "ReLU", "Add", "Mul", "Sub", "MatMul", "Softmax"} {
		d := model.ComponentDescriptor{ID: name,
			Version:    1,
			Kind:       "operator",
			Inputs:     []string{"x"},
			Outputs:    []string{"y"},
			Attributes: map[string]model.AttributeSpec{},
			Support: []model.ExecutionSupport{{Operation: "composition",
				Device:    "cpu",
				Precision: "float32",
				Mode:      "eager",
				Status:    "implemented"},
				{Operation: "training",
					Device:    "cpu",
					Precision: "float32",
					Mode:      "eager",
					Status:    "implemented"},
				{Operation: "export",
					Precision: "float32",
					Status:    "implemented"}}}
		switch name {
		case "Linear":
			d.Parameters = []string{"weights"}
		case "Dense":
			d.Parameters = []string{"weights", "bias"}
		case "Add", "Mul", "Sub", "MatMul":
			d.Inputs = []string{"a", "b"}
		}
		// The descriptor is constructed from static valid constants. Any failure
		// leaves this operator unavailable for composition; no false capability.
		if err := model.RegisterComponent(d, compositionShape(name)); err != nil {
			continue
		}
	}
}
func compositionShape(operator string) model.ShapeRule {
	return func(in map[string][]int, p map[string][]int, _ map[string]any) ([]int, error) {

		if operator == "Linear" {
			x, w := in["x"], p["weights"]
			if len(x) != 2 || len(w) != 2 || x[1] != w[0] {
				return nil, fmt.Errorf("linear requires x [batch,in] and weights [in,out]")
			}
			return []int{x[0], w[1]}, nil
		}
		if operator == "Dense" {
			x, w, b := in["x"], p["weights"], p["bias"]
			if len(x) != 2 || len(w) != 2 || len(b) != 1 || x[1] != w[0] || w[1] != b[0] {
				return nil, fmt.Errorf("dense expects x [batch,in], weights [in,out], bias [out]")
			}
			return []int{x[0], w[1]}, nil
		}
		if operator == "Add" || operator == "Mul" || operator == "Sub" {
			if len(in["a"]) != 2 || !slices.Equal(in["a"], in["b"]) {
				return nil, fmt.Errorf("%s requires identical rank-2 shapes; broadcasting is not qualified", operator)
			}
			return slices.Clone(in["a"]), nil
		}
		if operator == "MatMul" {
			a, b := in["a"], in["b"]
			if len(a) != 2 || len(b) != 2 || a[1] != b[0] || a[1] <= 0 {
				return nil, fmt.Errorf("MatMul requires compatible rank-2 shapes")
			}
			return []int{a[0], b[1]}, nil
		}
		if len(in["x"]) != 2 {
			return nil, fmt.Errorf("%s requires rank-2 input", operator)
		}
		return slices.Clone(in["x"]), nil
	}
}
