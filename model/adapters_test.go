package model

import (
	"context"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
)

// mockNode is an int variant of a test graph node.
type mockNode struct {
	name        string
	opType      string
	outputShape []int
	attributes  map[string]any
	params      []*graph.Parameter[int]
}

func (m *mockNode) OpType() string                      { return m.opType }
func (m *mockNode) OutputShape() []int                  { return m.outputShape }
func (m *mockNode) Attributes() map[string]any          { return m.attributes }
func (m *mockNode) Parameters() []*graph.Parameter[int] { return m.params }
func (m *mockNode) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[int]) (*tensor.TensorNumeric[int], error) {
	return inputs[0], nil
}
func (m *mockNode) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[int], _ ...*tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
	return []*tensor.TensorNumeric[int]{outputGradient}, nil
}

// mockNodeF32 is a float32 variant of mockNode.
type mockNodeF32 struct {
	name        string
	opType      string
	outputShape []int
	attributes  map[string]any
	params      []*graph.Parameter[float32]
}

func (m *mockNodeF32) OpType() string                          { return m.opType }
func (m *mockNodeF32) OutputShape() []int                      { return m.outputShape }
func (m *mockNodeF32) Attributes() map[string]any              { return m.attributes }
func (m *mockNodeF32) Parameters() []*graph.Parameter[float32] { return m.params }
func (m *mockNodeF32) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	return inputs[0], nil
}
func (m *mockNodeF32) Backward(_ context.Context, _ types.BackwardMode, outputGradient *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return []*tensor.TensorNumeric[float32]{outputGradient}, nil
}

// buildTestModel creates a minimal int Model for backward tests.
func buildTestModel(t *testing.T) *Model[int] {
	t.Helper()
	engine := compute.NewCPUEngine[int](numeric.IntOps{})
	builder := graph.NewBuilder[int](engine)
	inputNode := builder.Input([]int{2, 2})

	paramTensor, err := tensor.New[int]([]int{2, 2}, []int{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("failed to create param tensor: %v", err)
	}
	param, err := graph.NewParameter("test_param", paramTensor, tensor.New[int])
	if err != nil {
		t.Fatalf("failed to create parameter: %v", err)
	}

	node := &mockNode{
		name:        "test_node",
		opType:      "MockOp",
		outputShape: []int{2, 2},
		attributes:  map[string]any{"key": "val"},
		params:      []*graph.Parameter[int]{param},
	}
	builder.AddNode(node, inputNode)

	g, err := builder.Build(node)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}
	return &Model[int]{Graph: g}
}

// buildTestModelF32 creates a minimal float32 Model for round-trip tests.
func buildTestModelF32(t *testing.T) *Model[float32] {
	t.Helper()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	builder := graph.NewBuilder[float32](engine)
	inputNode := builder.Input([]int{2, 2})

	paramTensor, err := tensor.New[float32]([]int{2, 2}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("failed to create param tensor: %v", err)
	}
	param, err := graph.NewParameter("test_param", paramTensor, tensor.New[float32])
	if err != nil {
		t.Fatalf("failed to create parameter: %v", err)
	}

	node := &mockNodeF32{
		name:        "test_node",
		opType:      "MockOp",
		outputShape: []int{2, 2},
		attributes:  map[string]any{"key": "val"},
		params:      []*graph.Parameter[float32]{param},
	}
	builder.AddNode(node, inputNode)

	g, err := builder.Build(node)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}
	return &Model[float32]{Graph: g}
}

// --- StandardModelInstance.Backward tests ---

func TestStandardModelInstance_Backward(t *testing.T) {
	model := buildTestModel(t)
	instance := NewStandardModelInstance(model)
	ctx := context.Background()

	// Forward through the graph directly (Model.Forward requires Embedding).
	input, err := tensor.New[int]([]int{2, 2}, []int{10, 20, 30, 40})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}
	_, err = model.Graph.Forward(ctx, input)
	if err != nil {
		t.Fatalf("Graph.Forward failed: %v", err)
	}

	// Backward with a gradient.
	grad, err := tensor.New[int]([]int{2, 2}, []int{1, 1, 1, 1})
	if err != nil {
		t.Fatalf("failed to create gradient: %v", err)
	}
	err = instance.Backward(ctx, grad)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
}

func TestStandardModelInstance_Backward_NoGraph(t *testing.T) {
	model := &Model[int]{}
	instance := NewStandardModelInstance(model)

	grad, _ := tensor.New[int]([]int{1}, []int{1})
	err := instance.Backward(context.Background(), grad)
	if err == nil {
		t.Fatal("expected error for nil graph")
	}
}

func TestStandardModelInstance_Backward_NoGradient(t *testing.T) {
	model := buildTestModel(t)
	instance := NewStandardModelInstance(model)

	err := instance.Backward(context.Background())
	if err == nil {
		t.Fatal("expected error when no gradient provided")
	}
}

// Verify that FullBackprop mode is passed to graph nodes.
func TestStandardModelInstance_Backward_Mode(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.IntOps{})
	builder := graph.NewBuilder(engine)
	inputNode := builder.Input([]int{2})

	var capturedMode types.BackwardMode
	node := &modeTrackingNode{
		mockNode: mockNode{
			opType:      "TrackingOp",
			outputShape: []int{2},
			params:      []*graph.Parameter[int]{},
		},
		capturedMode: &capturedMode,
	}
	builder.AddNode(node, inputNode)

	g, err := builder.Build(node)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}

	model := &Model[int]{Graph: g}
	instance := NewStandardModelInstance(model)
	ctx := context.Background()

	input, _ := tensor.New([]int{2}, []int{5, 10})
	_, err = g.Forward(ctx, input) // populate memo via graph
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	grad, _ := tensor.New([]int{2}, []int{1, 1})
	err = instance.Backward(ctx, grad)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if capturedMode != types.FullBackprop {
		t.Errorf("expected FullBackprop mode, got %v", capturedMode)
	}
}

// modeTrackingNode captures the BackwardMode passed to its Backward method.
type modeTrackingNode struct {
	mockNode
	capturedMode *types.BackwardMode
}

func (m *modeTrackingNode) Backward(_ context.Context, mode types.BackwardMode, outputGradient *tensor.TensorNumeric[int], _ ...*tensor.TensorNumeric[int]) ([]*tensor.TensorNumeric[int], error) {
	*m.capturedMode = mode
	return []*tensor.TensorNumeric[int]{outputGradient}, nil
}

// --- StandardModelProvider tests ---

func TestStandardModelProvider_CreateModel(t *testing.T) {
	provider := NewStandardModelProvider[float32]()
	ctx := context.Background()

	_, err := provider.CreateModel(ctx, ModelConfig{})
	if err == nil {
		t.Error("expected error from CreateModel (not implemented)")
	}
}

func TestStandardModelProvider_CreateFromGraph(t *testing.T) {
	provider := NewStandardModelProvider[float32]()
	ctx := context.Background()

	model := buildTestModelF32(t)
	config := ModelConfig{
		Version:      "2.0.0",
		TrainingMode: true,
		Extensions:   map[string]interface{}{"key": "val"},
	}

	instance, err := provider.CreateFromGraph(ctx, model.Graph, config)
	if err != nil {
		t.Fatalf("CreateFromGraph failed: %v", err)
	}
	if !instance.IsTraining() {
		t.Error("expected training mode to be set")
	}
	meta := instance.GetMetadata()
	if meta.Extensions["key"] != "val" {
		t.Error("expected extensions to be preserved")
	}
}

func TestStandardModelProvider_Capabilities(t *testing.T) {
	provider := NewStandardModelProvider[float32]()

	caps := provider.GetCapabilities()
	if !caps.SupportsTraining {
		t.Error("expected training support")
	}
	if !caps.SupportsInference {
		t.Error("expected inference support")
	}
	if len(caps.SupportedTypes) == 0 {
		t.Error("expected supported types")
	}

	info := provider.GetProviderInfo()
	if info.Name == "" {
		t.Error("expected non-empty provider name")
	}
}

// --- StandardModelInstance Forward/GetGraph/Parameters tests ---

func TestStandardModelInstance_Forward_NilEmbedding(t *testing.T) {
	// Model.Forward requires Embedding, so Forward on a model without Embedding
	// will panic. We verify the Forward method is callable and returns an error.
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	ctx := context.Background()

	input, err := tensor.New[float32]([]int{2, 2}, []float32{10, 20, 30, 40})
	if err != nil {
		t.Fatalf("failed to create input: %v", err)
	}

	// This should panic because model.Embedding is nil (nil pointer dereference).
	// We recover from the panic to verify the code path is exercised.
	defer func() {
		if r := recover(); r == nil {
			t.Log("Forward did not panic - model may have changed")
		}
	}()

	_, _ = instance.Forward(ctx, input)
}

func TestStandardModelInstance_GetGraph(t *testing.T) {
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)

	g := instance.GetGraph()
	if g == nil {
		t.Error("expected non-nil graph")
	}
}

func TestStandardModelInstance_GetGraph_NilGraph(t *testing.T) {
	model := &Model[float32]{}
	instance := NewStandardModelInstance(model)

	g := instance.GetGraph()
	if g != nil {
		t.Error("expected nil graph")
	}
}

func TestStandardModelInstance_Parameters(t *testing.T) {
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)

	params := instance.Parameters()
	if len(params) == 0 {
		t.Error("expected at least one parameter")
	}
}

func TestStandardModelInstance_Parameters_NilGraph(t *testing.T) {
	model := &Model[float32]{}
	instance := NewStandardModelInstance(model)

	params := instance.Parameters()
	if len(params) != 0 {
		t.Errorf("expected 0 parameters for nil graph, got %d", len(params))
	}
}

func TestStandardModelInstance_Metadata_WithGraph(t *testing.T) {
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)

	meta := instance.GetMetadata()
	if meta.Framework != "zerfoo" {
		t.Errorf("expected framework 'zerfoo', got %q", meta.Framework)
	}
	if meta.Parameters <= 0 {
		t.Error("expected positive parameter count with graph")
	}
	if len(meta.InputShape) == 0 {
		t.Error("expected input shapes to be set")
	}
	if len(meta.OutputShape) == 0 {
		t.Error("expected output shape to be set")
	}
}

// --- BasicModelValidator tests ---

func TestBasicModelValidator_ValidateModel_Valid(t *testing.T) {
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()

	result, err := validator.ValidateModel(ctx, instance)
	if err != nil {
		t.Fatalf("ValidateModel failed: %v", err)
	}
	if !result.IsValid {
		t.Errorf("expected valid model, got errors: %v", result.Errors)
	}
	if result.Summary == "" {
		t.Error("expected non-empty summary")
	}
	if result.Metrics["parameter_count"] == 0 {
		t.Error("expected positive parameter count metric")
	}
}

func TestBasicModelValidator_ValidateModel_NoGraph(t *testing.T) {
	instance := NewMockModelInstance[float32]()
	// MockModelInstance has nil graph
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()

	result, err := validator.ValidateModel(ctx, instance)
	if err != nil {
		t.Fatalf("ValidateModel failed: %v", err)
	}
	if result.IsValid {
		t.Error("expected invalid model (no graph)")
	}
	if len(result.Errors) == 0 {
		t.Error("expected at least one error")
	}
}

func TestBasicModelValidator_ValidateArchitecture_NilGraph(t *testing.T) {
	instance := NewMockModelInstance[float32]()
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()

	err := validator.ValidateArchitecture(ctx, instance)
	if err == nil {
		t.Error("expected error for nil graph")
	}
}

func TestBasicModelValidator_ValidateArchitecture_ValidGraph(t *testing.T) {
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()

	err := validator.ValidateArchitecture(ctx, instance)
	if err != nil {
		t.Errorf("ValidateArchitecture failed for valid model: %v", err)
	}
}

func TestBasicModelValidator_ValidateInputs_DimensionMismatch(t *testing.T) {
	instance := NewMockModelInstance[float32]()
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()

	// MockModelInstance expects inputs with shapes [1,32] and [1,16]
	// Provide input with wrong number of dimensions
	wrongDims, _ := tensor.New[float32]([]int{32}, nil)
	correctInput2, _ := tensor.New[float32]([]int{1, 16}, nil)

	err := validator.ValidateInputs(ctx, instance, wrongDims, correctInput2)
	if err == nil {
		t.Error("expected error for dimension count mismatch")
	}
}

// --- ValidateArchitecture additional edge cases ---

func TestBasicModelValidator_ValidateArchitecture_NoInputs(t *testing.T) {
	// Build a model with graph but force no-inputs scenario
	// We use a mock that has a graph with no inputs
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	builder := graph.NewBuilder[float32](engine)

	// Create a graph with just an output and no real inputs
	node := &mockNodeF32{
		opType:      "MockOp",
		outputShape: []int{1},
		params:      []*graph.Parameter[float32]{},
	}
	// Don't add any input node; just build with node directly
	// This should create a graph with no inputs
	g, err := builder.Build(node)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}

	model := &Model[float32]{Graph: g}
	instance := NewStandardModelInstance(model)
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()

	err = validator.ValidateArchitecture(ctx, instance)
	if err == nil {
		t.Error("expected error for graph with no inputs")
	}
}

func TestBasicModelValidator_ValidateArchitecture_NilParamValue(t *testing.T) {
	engine := compute.NewCPUEngine[float32](&numeric.Float32Ops{})
	builder := graph.NewBuilder[float32](engine)
	inputNode := builder.Input([]int{2})

	// Create parameter with nil value
	nilParam := &graph.Parameter[float32]{Name: "nil_param", Value: nil}
	node := &mockNodeF32{
		opType:      "MockOp",
		outputShape: []int{2},
		params:      []*graph.Parameter[float32]{nilParam},
	}
	builder.AddNode(node, inputNode)

	g, err := builder.Build(node)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}

	model := &Model[float32]{Graph: g}
	instance := NewStandardModelInstance(model)
	validator := NewBasicModelValidator[float32]()
	ctx := context.Background()

	err = validator.ValidateArchitecture(ctx, instance)
	if err == nil {
		t.Error("expected error for parameter with nil value")
	}
}
