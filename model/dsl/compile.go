package dsl

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// Executable is a definition compiled to existing registered graph nodes.
// Calls are sequential; tensor outputs remain valid until the next invocation.
// Each declared output is retained, and parameter pointers are shared by name.
type Executable struct {
	validated  *Validated
	engine     compute.Engine[float32]
	nodes      map[string]graph.Node[float32]
	graphs     map[string]*graph.Graph[float32]
	params     []*graph.Parameter[float32]
	lastInputs []*tensor.TensorNumeric[float32]
}

// Compile performs complete preflight before allocating parameters or nodes.
func Compile(ctx context.Context, definition Definition, engine compute.Engine[float32], seed uint64) (*Executable, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if engine == nil {
		return nil, fmt.Errorf("dsl: engine required")
	}
	validated, err := Validate(definition)
	if err != nil {
		return nil, err
	}
	e := &Executable{validated: validated, engine: engine, nodes: map[string]graph.Node[float32]{}, graphs: map[string]*graph.Graph[float32]{}}
	builder := graph.NewBuilder(engine)
	for _, input := range validated.Definition.Inputs {
		e.nodes[input.Name+":value"] = builder.Input(input.Shape)
	}
	parameters := map[string]*graph.Parameter[float32]{}
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	for _, def := range validated.Definition.Parameters {
		count, err := shapeElements(def.Shape, false)
		if err != nil {
			return nil, err
		}
		values := make([]float32, count)
		if def.Initializer == "he_normal" {
			scale := math.Sqrt(2 / float64(def.Shape[0]))
			for i := range values {
				values[i] = float32(rng.NormFloat64() * scale)
			}
		}
		value, err := tensor.New(def.Shape, values)
		if err != nil {
			return nil, err
		}
		parameter, err := graph.NewParameter(def.Name, value, tensor.New[float32])
		if err != nil {
			return nil, err
		}
		parameters[def.Name] = parameter
		e.params = append(e.params, parameter)
	}
	for _, def := range validated.Definition.Nodes {
		descriptor, _, _ := model.Component("operator", def.Operator)
		factory, err := model.GetLayerBuilder[float32](def.Operator)
		if err != nil {
			return nil, err
		}
		bound := map[string]*graph.Parameter[float32]{}
		for slot, name := range def.Parameters {
			bound[slot] = parameters[name]
		}
		node, err := factory(engine, engine.Ops(), def.Name, bound, def.Attributes)
		if err != nil {
			return nil, fmt.Errorf("dsl: build %s: %w", def.Name, err)
		}
		inputs := make([]graph.Node[float32], len(descriptor.Inputs))
		for i, port := range descriptor.Inputs {
			inputs[i] = e.nodes[referenceKey(def.Inputs[port])]
		}
		e.nodes[def.Name+":"+descriptor.Outputs[0]] = builder.AddNode(node, inputs...)
	}
	for _, output := range validated.Definition.Outputs {
		g, err := builder.Build(e.nodes[referenceKey(output.Source)])
		if err != nil {
			return nil, err
		}
		e.graphs[output.Name] = g
	}
	return e, nil
}

// Definition returns an owned canonical model definition.
func (e *Executable) Definition() Definition { return CloneDefinition(e.validated.Definition) }

func (e *Executable) ID() string { return e.validated.ID }

// Parameters returns unique parameters in canonical name order.
func (e *Executable) Parameters() []*graph.Parameter[float32] { return slices.Clone(e.params) }

// Forward evaluates all declared outputs from named inputs; every port is bound.
func (e *Executable) Forward(ctx context.Context, inputs map[string]*tensor.TensorNumeric[float32]) (map[string]*tensor.TensorNumeric[float32], error) {
	if len(inputs) != len(e.validated.Definition.Inputs) {
		return nil, fmt.Errorf("dsl: input count mismatch")
	}
	ordered := make([]*tensor.TensorNumeric[float32], len(inputs))
	batch := 0
	for i, input := range e.validated.Definition.Inputs {
		value := inputs[input.Name]
		if value == nil {
			return nil, fmt.Errorf("dsl: missing input %s", input.Name)
		}
		actual := value.Shape()
		if len(actual) != len(input.Shape) {
			return nil, fmt.Errorf("dsl: input rank mismatch")
		}
		for j, want := range input.Shape {
			if want == -1 {
				if actual[j] < 1 || actual[j] > 256 || (batch != 0 && batch != actual[j]) {
					return nil, fmt.Errorf("dsl: dynamic batch mismatch or limit")
				}
				batch = actual[j]
			} else if want != actual[j] {
				return nil, fmt.Errorf("dsl: input shape mismatch for %s", input.Name)
			}
		}
		ordered[i] = value
	}
	first := e.validated.Definition.Outputs[0].Name
	g := e.graphs[first]
	if _, err := g.Forward(ctx, ordered...); err != nil {
		return nil, err
	}
	e.lastInputs = ordered
	result := map[string]*tensor.TensorNumeric[float32]{}
	for _, output := range e.validated.Definition.Outputs {
		value := g.NodeOutput(e.nodes[referenceKey(output.Source)])
		if value == nil {
			return nil, fmt.Errorf("dsl: output %s was not executed", output.Name)
		}
		result[output.Name] = value
	}
	return result, nil
}

// Backward sums gradients from explicitly named outputs and shared parameters.
// Each output uses the same existing graph backward implementation. Deterministic
// nodes are recomputed for each output; stochastic nodes are not qualified here.
func (e *Executable) Backward(ctx context.Context, seeds map[string]*tensor.TensorNumeric[float32]) error {
	if e.lastInputs == nil || len(seeds) == 0 {
		return fmt.Errorf("dsl: forward and at least one output gradient required")
	}
	for name, seed := range seeds {
		if e.graphs[name] == nil || seed == nil {
			return fmt.Errorf("dsl: unknown or nil output gradient %s", name)
		}
	}
	for _, p := range e.params {
		if err := e.engine.Zero(ctx, p.Gradient); err != nil {
			return err
		}
	}
	for _, output := range e.validated.Definition.Outputs {
		seed, ok := seeds[output.Name]
		if !ok {
			continue
		}
		g := e.graphs[output.Name]
		value, err := g.Forward(ctx, e.lastInputs...)
		if err != nil {
			return err
		}
		if !slices.Equal(value.Shape(), seed.Shape()) {
			return fmt.Errorf("dsl: output gradient shape mismatch")
		}
		if err := g.Backward(ctx, types.FullBackprop, seed); err != nil {
			return err
		}
	}
	return nil
}
