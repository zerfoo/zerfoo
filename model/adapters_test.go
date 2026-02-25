package model

import (
	"bytes"
	"context"
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
func (m *mockNodeF32) Attributes() map[string]any      { return m.attributes }
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
