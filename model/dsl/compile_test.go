package dsl

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func residualDefinition() Definition {
	return Definition{Version: 1, Name: "residual", Inputs: []TensorDef{{Name: "x", DType: "float32", Shape: []int{-1, 2}}}, Parameters: []ParameterDef{{TensorDef: TensorDef{Name: "w", DType: "float32", Shape: []int{2, 2}}, Initializer: "zeros"}, {TensorDef: TensorDef{Name: "b", DType: "float32", Shape: []int{2}}, Initializer: "zeros"}}, Nodes: []NodeDef{
		{Name: "left", Operator: "Dense", Version: 1, Inputs: map[string]Reference{"x": {"x", "value"}}, Parameters: map[string]string{"weights": "w", "bias": "b"}},
		{Name: "right", Operator: "Dense", Version: 1, Inputs: map[string]Reference{"x": {"x", "value"}}, Parameters: map[string]string{"weights": "w", "bias": "b"}},
		{Name: "sum", Operator: "Add", Version: 1, Inputs: map[string]Reference{"a": {"left", "y"}, "b": {"right", "y"}}},
	}, Outputs: []OutputDef{{"branch", Reference{"left", "y"}}, {"combined", Reference{"sum", "y"}}}}
}
func TestCompileResidualSharedParametersAndOutputs(t *testing.T) {
	ctx := context.Background()
	e, err := Compile(ctx, residualDefinition(), compute.NewCPUEngine(numeric.Float32Ops{}), 42)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range e.Parameters() {
		switch p.Name {
		case "w":
			p.Value.SetData([]float32{1, 2, 3, 4})
		case "b":
			p.Value.SetData([]float32{.5, -.5})
		}
	}
	x, err := tensor.New([]int{1, 2}, []float32{2, -1})
	if err != nil {
		t.Fatal(err)
	}
	outputs, err := e.Forward(ctx, map[string]*tensor.TensorNumeric[float32]{"x": x})
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string][]float32{"branch": {-.5, -.5}, "combined": {-1, -1}} {
		for i, actual := range outputs[name].Data() {
			if actual != want[i] {
				t.Fatalf("%s[%d]=%g want %g", name, i, actual, want[i])
			}
		}
	}
	seed, err := tensor.New([]int{1, 2}, []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Backward(ctx, map[string]*tensor.TensorNumeric[float32]{"branch": seed, "combined": seed}); err != nil {
		t.Fatal(err)
	}
	for _, p := range e.Parameters() {
		want := []float32{3, 3}
		if p.Name == "w" {
			want = []float32{6, 6, -3, -3}
		}
		for i, actual := range p.Gradient.Data() {
			if math.Abs(float64(actual-want[i])) > 1e-6 {
				t.Fatalf("shared %s gradient[%d]=%g want %g", p.Name, i, actual, want[i])
			}
		}
	}
	if len(e.Parameters()) != 2 {
		t.Fatal("shared parameters duplicated in optimizer list")
	}
}
func TestDefinitionValidationAndCanonicalIdentity(t *testing.T) {
	original := residualDefinition()
	a, err := Validate(original)
	if err != nil {
		t.Fatal(err)
	}
	original.Nodes[0], original.Nodes[2] = original.Nodes[2], original.Nodes[0]
	b, err := Validate(original)
	if err != nil {
		t.Fatal(err)
	}
	if a.ID != b.ID {
		t.Fatal("node declaration order changed identity")
	}
	for _, name := range []string{"cycle", "extra_port", "bad_shape", "unknown_operator", "unqualified_broadcast", "wrong_version"} {
		t.Run(name, func(t *testing.T) {
			d := residualDefinition()
			switch name {
			case "cycle":
				d.Nodes[0].Inputs["x"] = Reference{"sum", "y"}
			case "extra_port":
				d.Nodes[0].Inputs["ignored"] = Reference{"x", "value"}
			case "bad_shape":
				d.Parameters[0].Shape = []int{3, 2}
			case "unknown_operator":
				d.Nodes[0].Operator = "Unimplemented"
			case "unqualified_broadcast":
				d.Inputs = append(d.Inputs, TensorDef{Name: "extra", DType: "float32", Shape: []int{1, 2}})
				d.Nodes[2].Inputs["b"] = Reference{"extra", "value"}
			case "wrong_version":
				d.Version = 99
			}
			if _, err := Validate(d); err == nil {
				t.Fatal("invalid graph accepted")
			}
		})
	}
}

func TestComposableOperatorValues(t *testing.T) {
	for _, operator := range []string{"Linear", "Dense", "ReLU", "Softmax", "Add", "Mul", "Sub", "MatMul"} {
		t.Run(operator, func(t *testing.T) {
			d := Definition{Version: 1, Name: "operator", Inputs: []TensorDef{{Name: "x", DType: "float32", Shape: []int{1, 2}}}, Nodes: []NodeDef{{Name: "op", Operator: operator, Version: 1, Inputs: map[string]Reference{"x": {"x", "value"}}}}, Outputs: []OutputDef{{"result", Reference{"op", "y"}}}}
			want := []float32{2, 0}
			switch operator {
			case "Linear", "Dense":
				d.Parameters = []ParameterDef{{TensorDef: TensorDef{Name: "w", DType: "float32", Shape: []int{2, 2}}, Initializer: "zeros"}}
				d.Nodes[0].Parameters = map[string]string{"weights": "w"}
				want = []float32{1, 2}
				if operator == "Dense" {
					d.Parameters = append(d.Parameters, ParameterDef{TensorDef: TensorDef{Name: "b", DType: "float32", Shape: []int{2}}, Initializer: "zeros"})
					d.Nodes[0].Parameters["bias"] = "b"
					want = []float32{1.5, 1.5}
				}
			case "Softmax":
				want = []float32{float32(1 / (1 + math.Exp(-3))), float32(1 / (1 + math.Exp(3)))}
			case "Add", "Mul", "Sub", "MatMul":
				shape := []int{1, 2}
				if operator == "MatMul" {
					shape = []int{2, 2}
				}
				d.Inputs = append(d.Inputs, TensorDef{Name: "other", DType: "float32", Shape: shape})
				d.Nodes[0].Inputs = map[string]Reference{"a": {"x", "value"}, "b": {"other", "value"}}
				switch operator {
				case "Add":
					want = []float32{5, 3}
				case "Mul":
					want = []float32{6, -4}
				case "Sub":
					want = []float32{-1, -5}
				case "MatMul":
					want = []float32{1, 2}
				}
			}
			e, err := Compile(context.Background(), d, compute.NewCPUEngine(numeric.Float32Ops{}), 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range e.Parameters() {
				if p.Name == "w" {
					p.Value.SetData([]float32{3, 4, 5, 6})
				} else {
					p.Value.SetData([]float32{.5, -.5})
				}
			}
			x, err := tensor.New([]int{1, 2}, []float32{2, -1})
			if err != nil {
				t.Fatal(err)
			}
			inputs := map[string]*tensor.TensorNumeric[float32]{"x": x}
			if len(d.Inputs) > 1 {
				shape := []int{1, 2}
				values := []float32{3, 4}
				if operator == "MatMul" {
					shape = []int{2, 2}
					values = []float32{3, 4, 5, 6}
				}
				other, err := tensor.New(shape, values)
				if err != nil {
					t.Fatal(err)
				}
				inputs["other"] = other
			}
			out, err := e.Forward(context.Background(), inputs)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range out["result"].Data() {
				if math.Abs(float64(v-want[i])) > 1e-6 {
					t.Fatalf("output[%d]=%g want %g", i, v, want[i])
				}
			}
		})
	}
}

// Compare autograd against a separately evaluated float64 scalar expression,
// including nonlinear branches and repeated uses of the same parameters.
func TestResidualGradientIndependentReference(t *testing.T) {
	d := residualDefinition()
	d.Nodes = append(d.Nodes,
		NodeDef{Name: "relu", Operator: "ReLU", Version: 1, Inputs: map[string]Reference{"x": {"sum", "y"}}},
		NodeDef{Name: "product", Operator: "Mul", Version: 1, Inputs: map[string]Reference{"a": {"relu", "y"}, "b": {"left", "y"}}},
		NodeDef{Name: "difference", Operator: "Sub", Version: 1, Inputs: map[string]Reference{"a": {"product", "y"}, "b": {"right", "y"}}},
		NodeDef{Name: "probability", Operator: "Softmax", Version: 1, Inputs: map[string]Reference{"x": {"difference", "y"}}},
	)
	d.Outputs = []OutputDef{{"result", Reference{"probability", "y"}}}
	ctx := context.Background()
	e, err := Compile(ctx, d, compute.NewCPUEngine(numeric.Float32Ops{}), 0)
	if err != nil {
		t.Fatal(err)
	}
	values := map[string][]float64{"w": {.2, -.1, .3, .4}, "b": {.5, .2}}
	for _, p := range e.Parameters() {
		v := make([]float32, len(values[p.Name]))
		for i, x := range values[p.Name] {
			v[i] = float32(x)
		}
		p.Value.SetData(v)
	}
	x, err := tensor.New([]int{1, 2}, []float32{.7, -.2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.Forward(ctx, map[string]*tensor.TensorNumeric[float32]{"x": x}); err != nil {
		t.Fatal(err)
	}
	seed, err := tensor.New([]int{1, 2}, []float32{.3, -.8})
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Backward(ctx, map[string]*tensor.TensorNumeric[float32]{"result": seed}); err != nil {
		t.Fatal(err)
	}
	reference := func() float64 {
		z := [2]float64{}
		for j := range z {
			a := .7*values["w"][j] - .2*values["w"][2+j] + values["b"][j]
			z[j] = math.Max(0, 2*a)*a - a
		}
		p := 1 / (1 + math.Exp(z[1]-z[0]))
		return .3*p - .8*(1-p)
	}
	const epsilon = 1e-5
	for _, p := range e.Parameters() {
		for i, actual := range p.Gradient.Data() {
			original := values[p.Name][i]
			values[p.Name][i] = original + epsilon
			plus := reference()
			values[p.Name][i] = original - epsilon
			minus := reference()
			values[p.Name][i] = original
			want := (plus - minus) / (2 * epsilon)
			if math.Abs(float64(actual)-want) > 2e-5 {
				t.Fatalf("%s[%d]: gradient=%g independent=%g", p.Name, i, actual, want)
			}
		}
	}
}

func TestLegacyExecutionRejectsIgnoredEdges(t *testing.T) {
	for _, multipleOutputs := range []bool{false, true} {
		g := &ModelGraph{outputs: []string{"out"}, parents: map[string][]string{"out": {"left", "right"}}}
		if multipleOutputs {
			g.outputs = []string{"left", "right"}
			g.parents = nil
		}
		if _, err := g.Build(2, 2); err == nil {
			t.Fatal("legacy Build accepted ignored edges or outputs")
		}
		if _, err := g.BuildTrainable(2, 2); err == nil {
			t.Fatal("legacy training accepted ignored edges or outputs")
		}
	}
}

func TestExecutionRequirementsFailClosed(t *testing.T) {
	d := residualDefinition()
	if _, err := ValidateExecution(d, "training", "cpu", "float32", "eager"); err != nil {
		t.Fatal(err)
	}
	for _, req := range [][4]string{{"training", "cuda", "float32", "eager"}, {"training", "cpu", "float16", "eager"}, {"checkpoint", "cpu", "float32", "eager"}, {"training", "cpu", "float32", "compiled"}} {
		if _, err := ValidateExecution(d, req[0], req[1], req[2], req[3]); err == nil {
			t.Fatalf("unqualified execution accepted: %v", req)
		}
	}
}
