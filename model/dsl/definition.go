package dsl

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"sort"

	_ "github.com/zerfoo/zerfoo/layers/registry" // Populate the shared builder/descriptor registry.
	"github.com/zerfoo/zerfoo/model"
)

// Definition is the versioned composition format. V1 supports float32 tensors
// and bounded dynamic batch axes; unsupported constructs fail before allocation.
type Definition struct {
	Version    int            `json:"version"`
	Name       string         `json:"name"`
	Inputs     []TensorDef    `json:"inputs"`
	Parameters []ParameterDef `json:"parameters"`
	Nodes      []NodeDef      `json:"nodes"`
	Outputs    []OutputDef    `json:"outputs"`
}
type TensorDef struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
}
type ParameterDef struct {
	TensorDef
	Initializer string `json:"initializer"`
}
type Reference struct {
	Node string `json:"node"`
	Port string `json:"port"`
}
type NodeDef struct {
	Name       string               `json:"name"`
	Operator   string               `json:"operator"`
	Version    int                  `json:"version"`
	Inputs     map[string]Reference `json:"inputs"`
	Parameters map[string]string    `json:"parameters,omitempty"`
	Attributes map[string]any       `json:"attributes,omitempty"`
}
type OutputDef struct {
	Name   string    `json:"name"`
	Source Reference `json:"source"`
}

// DiagnosticError identifies a failed preflight constraint for an agent to correct.
type DiagnosticError struct {
	Code    string `json:"code"`
	Path    string `json:"path"`
	Message string `json:"message"`
}

func (d *DiagnosticError) Error() string {
	return fmt.Sprintf("dsl: %s at %s: %s", d.Code, d.Path, d.Message)
}
func invalid(code, path, message string) error { return &DiagnosticError{code, path, message} }

const maxDefinitionElements = 1 << 24

var definitionName = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_.-]{0,127}$`)

// Validated holds owned, canonically ordered data and statically inferred shapes.
type Validated struct {
	Definition Definition
	Shapes     map[string][]int
	ID         string
}

func referenceKey(r Reference) string { return r.Node + ":" + r.Port }
func shapeElements(shape []int, dynamic bool) (int, error) {
	if len(shape) < 1 || len(shape) > 4 {
		return 0, fmt.Errorf("rank must be in [1,4]")
	}
	total := 1
	for i, n := range shape {
		if n == -1 && dynamic && i == 0 {
			n = 256
		}
		if n < 1 || n > maxDefinitionElements/total {
			return 0, fmt.Errorf("invalid or excessive tensor shape")
		}
		total *= n
	}
	return total, nil
}

// Validate returns an owned canonical definition without constructing tensors.
func Validate(def Definition) (*Validated, error) {
	raw, err := json.Marshal(def)
	if err != nil {
		return nil, invalid("encoding", "definition", err.Error())
	}
	if len(raw) > 1<<20 {
		return nil, invalid("limit", "definition", "exceeds 1 MiB")
	}
	var d Definition
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, err
	}
	if d.Version != 1 || !definitionName.MatchString(d.Name) {
		return nil, invalid("version_or_name", "definition", "requires version 1 and a valid name")
	}
	if len(d.Inputs) == 0 || len(d.Inputs) > 16 || len(d.Nodes) == 0 || len(d.Nodes) > 256 || len(d.Outputs) == 0 || len(d.Outputs) > 16 || len(d.Parameters) > 512 {
		return nil, invalid("limit", "definition", "invalid input/node/output/parameter counts")
	}
	names := map[string]bool{}
	shapes := map[string][]int{}
	parameters := map[string][]int{}
	total := 0
	for _, input := range d.Inputs {
		if !definitionName.MatchString(input.Name) || names[input.Name] || input.DType != "float32" {
			return nil, invalid("tensor", "inputs", "invalid name, duplicate or unsupported dtype")
		}
		names[input.Name] = true
		count, err := shapeElements(input.Shape, true)
		if err != nil {
			return nil, invalid("shape", input.Name, err.Error())
		}
		total += count
		shapes[input.Name+":value"] = input.Shape
	}
	for _, p := range d.Parameters {
		if !definitionName.MatchString(p.Name) || names[p.Name] || p.DType != "float32" || (p.Initializer != "zeros" && p.Initializer != "he_normal") {
			return nil, invalid("parameter", p.Name, "invalid identity, dtype or initializer")
		}
		names[p.Name] = true
		count, err := shapeElements(p.Shape, false)
		if err != nil {
			return nil, invalid("shape", p.Name, err.Error())
		}
		total += count * 2
		parameters[p.Name] = p.Shape
	}
	nodes := map[string]NodeDef{}
	outputPorts := map[string]string{}
	for _, n := range d.Nodes {
		if !definitionName.MatchString(n.Name) || names[n.Name] {
			return nil, invalid("node", n.Name, "invalid or duplicate name")
		}
		names[n.Name] = true
		nodes[n.Name] = n
		descriptor, _, ok := model.Component("operator", n.Operator)
		if !ok || len(descriptor.Outputs) != 1 {
			return nil, invalid("unsupported_component", n.Name, "a single tensor output contract is required")
		}
		outputPorts[n.Name] = descriptor.Outputs[0]

	}
	pending := slices.Clone(d.Nodes)
	sort.Slice(pending, func(i, j int) bool { return pending[i].Name < pending[j].Name })
	ordered := []NodeDef{}
	usedParameters := map[string]bool{}
	for len(pending) > 0 {
		advanced := false
		next := []NodeDef{}
		for _, n := range pending {
			descriptor, rule, ok := model.Component("operator", n.Operator)
			if !ok || rule == nil || descriptor.Version != n.Version {
				return nil, invalid("unsupported_component", n.Name, n.Operator+" lacks a compatible composition contract")
			}
			if len(n.Inputs) != len(descriptor.Inputs) || len(n.Parameters) != len(descriptor.Parameters) {
				return nil, invalid("ports", n.Name, "input or parameter slot count mismatch")
			}

			attributes, err := model.ValidateAttributes(descriptor, n.Attributes)
			if err != nil {
				return nil, invalid("attributes", n.Name, err.Error())
			}
			n.Attributes = attributes

			inputShapes := map[string][]int{}
			ready := true
			for _, port := range descriptor.Inputs {
				ref, ok := n.Inputs[port]
				if !ok {
					return nil, invalid("ports", n.Name, "missing input "+port)
				}
				shape, ok := shapes[referenceKey(ref)]
				if !ok {
					if _, exists := nodes[ref.Node]; !exists || ref.Port != outputPorts[ref.Node] {
						return nil, invalid("reference", n.Name, "unknown source "+referenceKey(ref))
					}
					ready = false
				}
				inputShapes[port] = shape
			}
			parameterShapes := map[string][]int{}
			for _, slot := range descriptor.Parameters {
				ref, ok := n.Parameters[slot]
				shape, exists := parameters[ref]
				if !ok || !exists {
					return nil, invalid("parameter", n.Name, "missing parameter slot "+slot)
				}
				parameterShapes[slot] = shape
				usedParameters[ref] = true
			}
			if !ready {
				next = append(next, n)
				continue
			}
			shape, err := rule(inputShapes, parameterShapes, n.Attributes)
			if err != nil {
				return nil, invalid("shape", n.Name, err.Error())
			}
			count, err := shapeElements(shape, true)
			if err != nil {
				return nil, invalid("shape", n.Name, err.Error())
			}
			total += count
			if total > maxDefinitionElements {
				return nil, invalid("limit", "definition", "aggregate tensors exceed element budget")
			}
			shapes[n.Name+":"+descriptor.Outputs[0]] = shape
			ordered = append(ordered, n)
			advanced = true
		}
		if !advanced {
			return nil, invalid("cycle", "nodes", "cyclic connections")
		}
		pending = next
	}
	if len(usedParameters) != len(parameters) {
		return nil, invalid("unused_parameter", "parameters", "every parameter must be consumed")
	}
	outputs := map[string]bool{}
	for _, o := range d.Outputs {
		if !definitionName.MatchString(o.Name) || outputs[o.Name] {
			return nil, invalid("output", o.Name, "invalid or duplicate output")
		}
		if _, ok := shapes[referenceKey(o.Source)]; !ok {
			return nil, invalid("reference", o.Name, "unknown output source")
		}
		outputs[o.Name] = true
	}
	d.Nodes = ordered
	sort.Slice(d.Inputs, func(i, j int) bool { return d.Inputs[i].Name < d.Inputs[j].Name })
	sort.Slice(d.Parameters, func(i, j int) bool { return d.Parameters[i].Name < d.Parameters[j].Name })
	sort.Slice(d.Outputs, func(i, j int) bool { return d.Outputs[i].Name < d.Outputs[j].Name })
	raw, err = json.Marshal(d)
	if err != nil {
		return nil, err
	}
	hash := sha256.Sum256(raw)
	return &Validated{Definition: d, Shapes: shapes, ID: hex.EncodeToString(hash[:])}, nil
}

// CloneDefinition copies the serialized definition's supported fields. Attribute
// values are scalars under the v1 schema; nested attributes are rejected.
func CloneDefinition(d Definition) Definition {
	d.Inputs = slices.Clone(d.Inputs)
	for i := range d.Inputs {
		d.Inputs[i].Shape = slices.Clone(d.Inputs[i].Shape)
	}
	d.Parameters = slices.Clone(d.Parameters)
	for i := range d.Parameters {
		d.Parameters[i].Shape = slices.Clone(d.Parameters[i].Shape)
	}
	d.Nodes = slices.Clone(d.Nodes)
	for i := range d.Nodes {
		n := &d.Nodes[i]
		n.Inputs = maps.Clone(n.Inputs)
		n.Parameters = maps.Clone(n.Parameters)
		n.Attributes = maps.Clone(n.Attributes)
	}
	d.Outputs = slices.Clone(d.Outputs)
	return d
}

// ValidateExecution checks an explicit operation/device contract in addition to
// graph grammar. Discovery alone never authorizes a training or device path.
// Implemented support permits execution; it is not a verified-quality verdict.
func ValidateExecution(def Definition, operation, device, precision, mode string) (*Validated, error) {
	v, err := Validate(def)
	if err != nil {
		return nil, err
	}
	for _, n := range v.Definition.Nodes {
		d, _, ok := model.Component("operator", n.Operator)
		supported := false
		if ok {
			for _, s := range d.Support {
				if s.Operation == operation && s.Device == device && s.Precision == precision && s.Mode == mode && (s.Status == "implemented" || s.Status == "verified") {
					supported = true
					break
				}
			}
		}
		if !supported {
			return nil, invalid("unsupported_execution", n.Name, fmt.Sprintf("%s lacks %s on %s/%s/%s", n.Operator, operation, device, precision, mode))
		}
	}
	return v, nil
}
