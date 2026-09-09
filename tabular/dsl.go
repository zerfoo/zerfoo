package tabular

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/zerfoo/zerfoo/model/dsl"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

func validateClassifierDefinition(config ClassifierConfig) error {
	v, err := dsl.Validate(*config.Definition)
	if err != nil {
		return err
	}
	d := v.Definition
	if len(d.Inputs) != 1 || len(d.Outputs) != 1 || !slices.Equal(d.Inputs[0].Shape, []int{-1, config.InputDim}) {
		return fmt.Errorf("tabular: classifier DSL requires one dynamic-batch feature input and one logits output")
	}
	output := d.Outputs[0]
	shape := v.Shapes[output.Source.Node+":"+output.Source.Port]
	if !slices.Equal(shape, []int{-1, config.ClassCount}) {
		return fmt.Errorf("tabular: DSL logits shape does not match classes")
	}
	for _, node := range d.Nodes {
		if node.Name == output.Source.Node && node.Operator == "Softmax" {
			return fmt.Errorf("tabular: classifier output must be logits, not Softmax probabilities")
		}
	}
	return nil
}

// ClassifierDefinition adapts the original linear/MLP configuration into an
// explicit graph. New applications persist this definition, not a second recipe.
func ClassifierDefinition(config ClassifierConfig) (dsl.Definition, error) {
	if config.Definition != nil {
		v, err := dsl.Validate(*config.Definition)
		if err != nil {
			return dsl.Definition{}, err
		}
		return v.Definition, nil
	}
	if err := validateClassifierConfig(config); err != nil {
		return dsl.Definition{}, err
	}
	d := dsl.Definition{Version: 1, Name: "classifier", Inputs: []dsl.TensorDef{{Name: "features", DType: "float32", Shape: []int{-1, config.InputDim}}}}
	dims := append([]int{config.InputDim}, config.HiddenDims...)
	dims = append(dims, config.ClassCount)
	source := dsl.Reference{Node: "features", Port: "value"}
	for i := 0; i+1 < len(dims); i++ {
		name := fmt.Sprintf("layer%d", i)
		d.Parameters = append(d.Parameters, dsl.ParameterDef{TensorDef: dsl.TensorDef{Name: name + ".weights", DType: "float32", Shape: []int{dims[i], dims[i+1]}}, Initializer: "he_normal"}, dsl.ParameterDef{TensorDef: dsl.TensorDef{Name: name + ".bias", DType: "float32", Shape: []int{dims[i+1]}}, Initializer: "zeros"})
		d.Nodes = append(d.Nodes, dsl.NodeDef{Name: name, Operator: "Dense", Version: 1, Inputs: map[string]dsl.Reference{"x": source}, Parameters: map[string]string{"weights": name + ".weights", "bias": name + ".bias"}})
		source = dsl.Reference{Node: name, Port: "y"}
		if i+2 < len(dims) {
			name += "_relu"
			d.Nodes = append(d.Nodes, dsl.NodeDef{Name: name, Operator: "ReLU", Version: 1, Inputs: map[string]dsl.Reference{"x": source}})
			source = dsl.Reference{Node: name, Port: "y"}
		}
	}
	d.Outputs = []dsl.OutputDef{{Name: "logits", Source: source}}
	v, err := dsl.Validate(d)
	if err != nil {
		return dsl.Definition{}, err
	}
	return v.Definition, nil
}
func (c *Classifier[T]) computeDSLGradients(ctx context.Context, rows [][]float64, targets []int) (float64, error) {
	input, err := c.inputTensor(ctx, rows)
	if err != nil {
		return 0, err
	}
	labels, err := c.targetTensor(len(rows), targets)
	if err != nil {
		return 0, err
	}
	logits, err := c.forward(ctx, input)
	if err != nil {
		return 0, err
	}
	objective := loss.NewCrossEntropyLoss[T](c.engine)
	result, err := objective.Forward(ctx, logits, labels)
	if err != nil {
		return 0, err
	}
	value := float64(result.Data()[0])
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, fmt.Errorf("tabular: nonfinite DSL loss")
	}
	one, err := tensor.New([]int{1}, []T{1})
	if err != nil {
		return 0, err
	}
	grads, err := objective.Backward(ctx, types.FullBackprop, one)
	if err != nil {
		return 0, err
	}
	seed, ok := any(grads[0]).(*tensor.TensorNumeric[float32])
	if !ok {
		return 0, fmt.Errorf("tabular: DSL gradient dtype mismatch")
	}
	if err := c.compiled.Backward(ctx, map[string]*tensor.TensorNumeric[float32]{c.config.Definition.Outputs[0].Name: seed}); err != nil {
		return 0, err
	}
	return value, nil
}
