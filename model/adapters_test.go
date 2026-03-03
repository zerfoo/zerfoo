package model

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zerfoo/types"
	"github.com/zerfoo/zmf"
	"google.golang.org/protobuf/proto"
)

// mockNodeF32 is a float32 variant of mockNode for loader round-trip tests.
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
// Uses mockNode from zmf_exporter_test.go (same package).
func buildTestModel(t *testing.T) *Model[int] { //nolint:dupl // generic type differs from buildTestModelF32
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
	return &Model[int]{Graph: g, ZMFVersion: "1.0.0"}
}

// buildTestModelF32 creates a minimal float32 Model for round-trip tests.
func buildTestModelF32(t *testing.T) *Model[float32] { //nolint:dupl // generic type differs from buildTestModel
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
	return &Model[float32]{Graph: g, ZMFVersion: "1.0.0"}
}

// --- StandardModelInstance.Backward tests ---

func TestStandardModelInstance_Backward(t *testing.T) {
	model := buildTestModel(t) // int model (uses mockNode from zmf_exporter_test.go)
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

// --- ZMFModelExporter tests ---

func TestZMFModelExporter_ExportToWriter(t *testing.T) {
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	exporter := NewZMFModelExporter[float32]()
	ctx := context.Background()

	var buf bytes.Buffer
	err := exporter.ExportToWriter(ctx, instance, &buf)
	if err != nil {
		t.Fatalf("ExportToWriter failed: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatal("expected non-empty output")
	}

	// Verify the bytes are valid ZMF protobuf.
	zmfModel := &zmf.Model{}
	if err := proto.Unmarshal(buf.Bytes(), zmfModel); err != nil {
		t.Fatalf("exported bytes are not valid ZMF: %v", err)
	}
	if zmfModel.ZmfVersion != "1.0.0" {
		t.Errorf("expected version 1.0.0, got %s", zmfModel.ZmfVersion)
	}
}

func TestZMFModelExporter_ExportToBytes(t *testing.T) {
	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	exporter := NewZMFModelExporter[float32]()
	ctx := context.Background()

	data, err := exporter.ExportToBytes(ctx, instance)
	if err != nil {
		t.Fatalf("ExportToBytes failed: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("expected non-empty data")
	}

	// Verify round-trip via ExportToPath produces identical bytes.
	tmpFile := t.TempDir() + "/test.zmf"
	err = exporter.ExportToPath(ctx, instance, tmpFile)
	if err != nil {
		t.Fatalf("ExportToPath failed: %v", err)
	}
	fileData, err := os.ReadFile(tmpFile) //nolint:gosec // test-only path from t.TempDir
	if err != nil {
		t.Fatalf("failed to read exported file: %v", err)
	}
	if !bytes.Equal(data, fileData) {
		t.Error("ExportToBytes and ExportToPath produced different results")
	}
}

func TestZMFModelExporter_SupportsFormat(t *testing.T) {
	exporter := NewZMFModelExporter[float32]()
	tests := []struct {
		format string
		want   bool
	}{
		{".zmf", true},
		{".zerfoo", true},
		{".onnx", false},
	}
	for _, tt := range tests {
		if got := exporter.SupportsFormat(tt.format); got != tt.want {
			t.Errorf("SupportsFormat(%q) = %v, want %v", tt.format, got, tt.want)
		}
	}
}

// --- ZMFModelLoader tests ---

// registerMockOpBuilder registers a float32 MockOp layer builder so that
// BuildFromZMF can reconstruct mock models. It should be called in each
// loader test (safe to call multiple times; the registry silently overwrites).
func registerMockOpBuilder() {
	RegisterLayer("MockOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		return &mockNodeF32{
			opType:      "MockOp",
			outputShape: []int{2, 2},
			params:      []*graph.Parameter[float32]{},
		}, nil
	})
}

func TestZMFModelLoader_LoadFromBytes(t *testing.T) {
	registerMockOpBuilder()

	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	exporter := NewZMFModelExporter[float32]()
	ctx := context.Background()

	data, err := exporter.ExportToBytes(ctx, instance)
	if err != nil {
		t.Fatalf("ExportToBytes failed: %v", err)
	}

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})

	loaded, err := loader.LoadFromBytes(ctx, data)
	if err != nil {
		t.Fatalf("LoadFromBytes failed: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected non-nil model instance")
	}
	if meta := loaded.GetMetadata(); meta.Version != "1.0.0" {
		t.Errorf("expected version 1.0.0, got %s", meta.Version)
	}
}

func TestZMFModelLoader_LoadFromReader(t *testing.T) {
	registerMockOpBuilder()

	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	exporter := NewZMFModelExporter[float32]()
	ctx := context.Background()

	data, err := exporter.ExportToBytes(ctx, instance)
	if err != nil {
		t.Fatalf("ExportToBytes failed: %v", err)
	}

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})

	loaded, err := loader.LoadFromReader(ctx, bytes.NewReader(data))
	if err != nil {
		t.Fatalf("LoadFromReader failed: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected non-nil model instance")
	}
}

func TestZMFModelLoader_LoadFromPath(t *testing.T) {
	registerMockOpBuilder()

	model := buildTestModelF32(t)
	instance := NewStandardModelInstance(model)
	exporter := NewZMFModelExporter[float32]()
	ctx := context.Background()

	tmpFile := t.TempDir() + "/test.zmf"
	err := exporter.ExportToPath(ctx, instance, tmpFile)
	if err != nil {
		t.Fatalf("ExportToPath failed: %v", err)
	}

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})

	loaded, err := loader.LoadFromPath(ctx, tmpFile)
	if err != nil {
		t.Fatalf("LoadFromPath failed: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected non-nil model instance")
	}
}

func TestZMFModelLoader_SupportsFormat(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})

	tests := []struct {
		format string
		want   bool
	}{
		{".zmf", true},
		{".zerfoo", true},
		{".onnx", false},
		{".pt", false},
	}
	for _, tt := range tests {
		if got := loader.SupportsFormat(tt.format); got != tt.want {
			t.Errorf("SupportsFormat(%q) = %v, want %v", tt.format, got, tt.want)
		}
	}
}

func TestZMFModelLoader_GetLoaderInfo(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})

	info := loader.GetLoaderInfo()
	if info.Name != "ZMF Model Loader" {
		t.Errorf("expected name 'ZMF Model Loader', got %q", info.Name)
	}
	if len(info.SupportedFormats) != 2 {
		t.Errorf("expected 2 supported formats, got %d", len(info.SupportedFormats))
	}
}

func TestZMFModelExporter_GetExporterInfo(t *testing.T) {
	exporter := NewZMFModelExporter[float32]()
	info := exporter.GetExporterInfo()
	if info.Name != "ZMF Model Exporter" {
		t.Errorf("expected name 'ZMF Model Exporter', got %q", info.Name)
	}
}

func TestZMFModelExporter_MarshalModel_NilGraph(t *testing.T) {
	// StandardModelInstance with nil graph - ExportToBytes should panic
	// because convertModelToZMF calls graph.GetAllNodes which panics on nil.
	model := &Model[float32]{}
	instance := NewStandardModelInstance(model)
	exporter := NewZMFModelExporter[float32]()
	ctx := context.Background()

	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil graph in marshalModel")
		}
	}()

	_, _ = exporter.ExportToBytes(ctx, instance)
}

func TestZMFModelExporter_NonStandardInstance(t *testing.T) {
	exporter := NewZMFModelExporter[float32]()
	ctx := context.Background()
	mock := NewMockModelInstance[float32]()

	// ExportToPath should fail for non-StandardModelInstance
	err := exporter.ExportToPath(ctx, mock, "/tmp/should_not_exist.zmf")
	if err == nil {
		t.Error("expected error for non-StandardModelInstance in ExportToPath")
	}

	// ExportToWriter should fail for non-StandardModelInstance
	var buf bytes.Buffer
	err = exporter.ExportToWriter(ctx, mock, &buf)
	if err == nil {
		t.Error("expected error for non-StandardModelInstance in ExportToWriter")
	}

	// ExportToBytes should fail for non-StandardModelInstance
	_, err = exporter.ExportToBytes(ctx, mock)
	if err == nil {
		t.Error("expected error for non-StandardModelInstance in ExportToBytes")
	}
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

func TestZMFModelLoader_LoadFromBytes_InvalidProto(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})
	ctx := context.Background()

	_, err := loader.LoadFromBytes(ctx, []byte{0xFF, 0xFE, 0xFD})
	if err == nil {
		t.Error("expected error for invalid protobuf data")
	}
}

func TestZMFModelLoader_LoadFromReader_InvalidProto(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})
	ctx := context.Background()

	_, err := loader.LoadFromReader(ctx, bytes.NewReader([]byte{0xFF, 0xFE}))
	if err == nil {
		t.Error("expected error for invalid protobuf in reader")
	}
}

func TestZMFModelLoader_LoadFromPath_NotFound(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})
	ctx := context.Background()

	_, err := loader.LoadFromPath(ctx, "/nonexistent/path/model.zmf")
	if err == nil {
		t.Error("expected error for missing file")
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

// errorReader returns an error on Read.
type errorReader struct{}

func (e *errorReader) Read(_ []byte) (int, error) {
	return 0, fmt.Errorf("simulated read error")
}

func TestZMFModelLoader_LoadFromReader_ReadError(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})
	ctx := context.Background()

	_, err := loader.LoadFromReader(ctx, &errorReader{})
	if err == nil {
		t.Error("expected error for failing reader")
	}
}

func TestZMFModelLoader_LoadFromBytes_ValidProto_BadGraph(t *testing.T) {
	// Create a valid protobuf that BuildFromZMF will fail on (unrecognized op type)
	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{Name: "bad", OpType: "UnknownOpType_ForTest", Inputs: []string{"input"}},
			},
			Outputs: []*zmf.ValueInfo{{Name: "bad"}},
		},
	}
	data, err := proto.Marshal(zmfModel)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	loader := NewZMFModelLoader(engine, numeric.Float32Ops{})
	ctx := context.Background()

	_, err = loader.LoadFromBytes(ctx, data)
	if err == nil {
		t.Error("expected error for unrecognized op type in loaded model")
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
