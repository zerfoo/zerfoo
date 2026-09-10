package tabular

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/zerfoo/zerfoo/model/dsl"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestDSLClassifierLearnsAndReloadsResidualGraph(t *testing.T) {
	ctx := context.Background()
	dataset := classifierFixture(t)
	manifest := dataset.Manifest()
	config := ClassifierConfig{InputDim: 4, ClassCount: 3, Labels: manifest.Labels, HiddenDims: []int{16}, Seed: 42}
	definition, err := ClassifierDefinition(config)
	if err != nil {
		t.Fatal(err)
	}
	first := definition.Nodes[0]
	// Add a second invocation of the SAME dense parameters and merge its output.
	other := first
	other.Name = "shared_branch"
	definition.Nodes = append(definition.Nodes, other, dsl.NodeDef{Name: "residual_merge", Operator: "Add", Version: 1, Inputs: map[string]dsl.Reference{"a": {Node: first.Name, Port: "y"}, "b": {Node: other.Name, Port: "y"}}})
	for i := range definition.Nodes {
		if definition.Nodes[i].Operator == "ReLU" {
			definition.Nodes[i].Inputs = map[string]dsl.Reference{"x": {Node: "residual_merge", Port: "y"}}
		}
	}
	config.HiddenDims = nil
	config.Definition = &definition
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	fit, err := FitClassifier(ctx, dataset, config, FitOptions{Epochs: 200, BatchSize: 15, LearningRate: .01, WeightDecay: .0001, Seed: 42}, engine, nil)
	if err != nil {
		t.Fatal(err)
	}
	rows, labels, err := dataset.Partition("validation")
	if err != nil {
		t.Fatal(err)
	}
	before, err := fit.Model.Evaluate(ctx, rows, labels)
	if err != nil {
		t.Fatal(err)
	}
	if before.Accuracy < .85 || fit.Steps != 1200 {
		t.Fatalf("real residual graph failed to learn: %+v steps=%d", before, fit.Steps)
	}
	path := filepath.Join(t.TempDir(), "model")
	id, err := SaveClassifierBundle(ctx, path, fit.Model, dataset, "validation")
	if err != nil {
		t.Fatal(err)
	}
	restored, err := LoadClassifierBundle(ctx, path, id, engine)
	if err != nil {
		t.Fatal(err)
	}
	after, err := restored.Model.Evaluate(ctx, rows, labels)
	if err != nil {
		t.Fatal(err)
	}
	if before.Accuracy != after.Accuracy || before.CrossEntropy != after.CrossEntropy || restored.Manifest.DefinitionSHA256 == "" {
		t.Fatal("custom model did not reload exactly")
	}
	if len(restored.Model.params) != 4 {
		t.Fatal("shared parameters duplicated")
	}
}
