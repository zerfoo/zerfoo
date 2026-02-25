package model

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
	"github.com/zerfoo/zmf"
)

func TestBuildFromZMF_Int8(t *testing.T) {
	// Create a sample ZMF Model protobuf message with an INT8 parameter.
	sampleModel := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"param1": {
					Dtype: zmf.Tensor_INT8,
					Shape: []int64{2, 2},
					Data:  []byte{1, 2, 3, 4},
				},
			},
			Inputs: []*zmf.ValueInfo{
				{
					Name:  "input",
					Dtype: zmf.Tensor_INT8,
					Shape: []int64{1, 10},
				},
			},
			Outputs: []*zmf.ValueInfo{
				{
					Name:  "input", // just pass through for simplicity
					Dtype: zmf.Tensor_INT8,
					Shape: []int64{1, 10},
				},
			},
		},
	}

	// Create an int8 engine.
	ops := numeric.Int8Ops{}
	engine := compute.NewCPUEngine[int8](ops)

	// Build the graph.
	g, err := BuildFromZMF[int8](engine, ops, sampleModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}

	if g == nil {
		t.Fatal("BuildFromZMF returned a nil graph")
	}
}

func TestBuildFromZMF_NilModel(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := BuildFromZMF[float32](engine, ops, nil)
	if err == nil {
		t.Error("expected error for nil model")
	}
}

func TestBuildFromZMF_NilGraph(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := BuildFromZMF[float32](engine, ops, &zmf.Model{})
	if err == nil {
		t.Error("expected error for nil graph")
	}
}

func TestBuildFromZMF_NoOutputs(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Outputs: []*zmf.ValueInfo{}, // no outputs
		},
	}

	_, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err == nil {
		t.Error("expected error for no outputs")
	}
}

func TestBuildFromZMF_UnknownOpType(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{Name: "bad_node", OpType: "NonExistentOpType_XYZ", Inputs: []string{"input"}},
			},
			Outputs: []*zmf.ValueInfo{
				{Name: "bad_node"},
			},
		},
	}

	_, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err == nil {
		t.Error("expected error for unknown op type")
	}
}

func TestBuildFromZMF_OutputNotFound(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Outputs: []*zmf.ValueInfo{
				{Name: "nonexistent_output"},
			},
		},
	}

	_, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err == nil {
		t.Error("expected error for output node not found")
	}
}

// --- parameterNode tests ---

func TestParameterNode_OpType(t *testing.T) {
	val, _ := tensor.New[float32]([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	pn := &parameterNode[float32]{value: val}

	if pn.OpType() != "Parameter" {
		t.Errorf("expected 'Parameter', got %q", pn.OpType())
	}
}

func TestParameterNode_Attributes(t *testing.T) {
	val, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	pn := &parameterNode[float32]{value: val}

	attrs := pn.Attributes()
	if attrs == nil {
		t.Error("expected non-nil attributes map")
	}
	if len(attrs) != 0 {
		t.Errorf("expected empty attributes map, got %d entries", len(attrs))
	}
}

func TestParameterNode_OutputShape(t *testing.T) {
	val, _ := tensor.New[float32]([]int{3, 4}, []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	pn := &parameterNode[float32]{value: val}

	shape := pn.OutputShape()
	if len(shape) != 2 || shape[0] != 3 || shape[1] != 4 {
		t.Errorf("expected [3 4], got %v", shape)
	}
}

func TestParameterNode_Forward(t *testing.T) {
	val, _ := tensor.New[float32]([]int{2}, []float32{42, 43})
	pn := &parameterNode[float32]{value: val}

	out, err := pn.Forward(context.Background())
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if out != val {
		t.Error("Forward should return the stored value tensor")
	}
}

func TestParameterNode_Backward(t *testing.T) {
	val, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	pn := &parameterNode[float32]{value: val}

	grads, err := pn.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward failed: %v", err)
	}
	if grads != nil {
		t.Error("expected nil gradients from parameterNode.Backward")
	}
}

func TestParameterNode_Parameters(t *testing.T) {
	val, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	pn := &parameterNode[float32]{value: val}

	params := pn.Parameters()
	if params != nil {
		t.Error("expected nil parameters")
	}
}

// --- resolveOutputSuffix tests ---

func TestResolveOutputSuffix_OutputZero(t *testing.T) {
	nodeMap := map[string]graph.Node[float32]{
		"layer_a": &parameterNode[float32]{value: nil},
	}
	result := resolveOutputSuffix("layer_a/output_0", nodeMap)
	if result != "layer_a" {
		t.Errorf("expected 'layer_a', got %q", result)
	}
}

func TestResolveOutputSuffix_ColonZero(t *testing.T) {
	nodeMap := map[string]graph.Node[float32]{
		"layer_b": &parameterNode[float32]{value: nil},
	}
	result := resolveOutputSuffix("layer_b:0", nodeMap)
	if result != "layer_b" {
		t.Errorf("expected 'layer_b', got %q", result)
	}
}

func TestResolveOutputSuffix_OutputOne(t *testing.T) {
	nodeMap := map[string]graph.Node[float32]{
		"layer_c": &parameterNode[float32]{value: nil},
	}
	result := resolveOutputSuffix("layer_c/output_1", nodeMap)
	if result != "layer_c" {
		t.Errorf("expected 'layer_c', got %q", result)
	}
}

func TestResolveOutputSuffix_LayerNormSuffix(t *testing.T) {
	nodeMap := map[string]graph.Node[float32]{
		"base": &parameterNode[float32]{value: nil},
	}
	result := resolveOutputSuffix("base/LayerNorm", nodeMap)
	if result != "base" {
		t.Errorf("expected 'base', got %q", result)
	}
}

func TestResolveOutputSuffix_NoMatch(t *testing.T) {
	nodeMap := map[string]graph.Node[float32]{
		"something": &parameterNode[float32]{value: nil},
	}
	result := resolveOutputSuffix("completely_different", nodeMap)
	if result != "" {
		t.Errorf("expected empty string, got %q", result)
	}
}

func TestResolveOutputSuffix_OutputWithLayerSuffix(t *testing.T) {
	nodeMap := map[string]graph.Node[float32]{
		"base/MatMul": &parameterNode[float32]{value: nil},
	}
	result := resolveOutputSuffix("base/output_0", nodeMap)
	if result != "base/MatMul" {
		t.Errorf("expected 'base/MatMul', got %q", result)
	}
}

// --- convertAttributes tests ---

func TestConvertAttributes_AllTypes(t *testing.T) {
	zmfAttrs := map[string]*zmf.Attribute{
		"float_attr": {Value: &zmf.Attribute_F{F: 3.14}},
		"int_attr":   {Value: &zmf.Attribute_I{I: 42}},
		"str_attr":   {Value: &zmf.Attribute_S{S: "hello"}},
		"ints_attr":  {Value: &zmf.Attribute_Ints{Ints: &zmf.Ints{Val: []int64{1, 2, 3}}}},
		"floats_attr": {Value: &zmf.Attribute_Floats{Floats: &zmf.Floats{Val: []float32{1.0, 2.0}}}},
		"strings_attr": {Value: &zmf.Attribute_Strings{Strings: &zmf.Strings{Val: []string{"a", "b"}}}},
	}

	result := convertAttributes(zmfAttrs)

	if f, ok := result["float_attr"].(float32); !ok || f != 3.14 {
		t.Errorf("float_attr: expected 3.14, got %v", result["float_attr"])
	}
	if i, ok := result["int_attr"].(int); !ok || i != 42 {
		t.Errorf("int_attr: expected 42, got %v", result["int_attr"])
	}
	if s, ok := result["str_attr"].(string); !ok || s != "hello" {
		t.Errorf("str_attr: expected 'hello', got %v", result["str_attr"])
	}
	if ints, ok := result["ints_attr"].([]int64); !ok || len(ints) != 3 {
		t.Errorf("ints_attr: expected []int64 len 3, got %v", result["ints_attr"])
	}
	if floats, ok := result["floats_attr"].([]float32); !ok || len(floats) != 2 {
		t.Errorf("floats_attr: expected []float32 len 2, got %v", result["floats_attr"])
	}
	if strs, ok := result["strings_attr"].([]string); !ok || len(strs) != 2 {
		t.Errorf("strings_attr: expected []string len 2, got %v", result["strings_attr"])
	}
}

func TestConvertAttributes_Empty(t *testing.T) {
	result := convertAttributes(nil)
	if len(result) != 0 {
		t.Errorf("expected empty map for nil attributes, got %d entries", len(result))
	}
}

// --- NewModel test ---

// --- BuildFromZMF additional branch tests ---

// passthroughBuilder returns a LayerBuilder that always creates a passthrough parameterNode.
func passthroughBuilder() LayerBuilder[float32] {
	return func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	}
}

func TestBuildFromZMF_NormalizationTrimsInputs(t *testing.T) {
	tests := []struct {
		name     string
		opType   string
		nodeName string
	}{
		{"SimplifiedLayerNormalization", "SimplifiedLayerNormalization", "sln"},
		{"SkipSimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "skip_sln"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)

			RegisterLayer(tt.opType, passthroughBuilder())
			defer UnregisterLayer(tt.opType)

			zmfModel := &zmf.Model{
				Graph: &zmf.Graph{
					Inputs: []*zmf.ValueInfo{
						{Name: "input", Shape: []int64{1}},
					},
					Nodes: []*zmf.Node{
						{Name: tt.nodeName, OpType: tt.opType, Inputs: []string{"input", "extra"}},
					},
					Outputs: []*zmf.ValueInfo{{Name: tt.nodeName}},
				},
			}

			g, err := BuildFromZMF[float32](engine, ops, zmfModel)
			if err != nil {
				t.Fatalf("BuildFromZMF failed: %v", err)
			}
			if g == nil {
				t.Fatal("expected non-nil graph")
			}
		})
	}
}

func TestBuildFromZMF_EmptyInputName(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestEmptyInputOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestEmptyInputOp")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "node1",
					OpType: "TestEmptyInputOp",
					Inputs: []string{"input", "", "input"}, // empty input name should be filtered
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "node1"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

func TestBuildFromZMF_ParameterAsInput(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestParamInputOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestParamInputOp")

	// Create ZMF model where a node input references a parameter
	paramData := make([]byte, 4)
	paramData[0] = 0x00
	paramData[1] = 0x00
	paramData[2] = 0x80
	paramData[3] = 0x3F // 1.0 in float32

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"weight": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{1},
					Data:  paramData,
				},
			},
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
				{Name: "weight", Shape: []int64{1}}, // parameter also listed as input
			},
			Nodes: []*zmf.Node{
				{
					Name:   "node1",
					OpType: "TestParamInputOp",
					Inputs: []string{"input", "weight"}, // weight is a parameter, not a node
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "node1"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

func TestBuildFromZMF_OutputSuffixResolution(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestOutputSuffOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestOutputSuffOp")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "layer_a",
					OpType: "TestOutputSuffOp",
					Inputs: []string{"input"},
				},
				{
					Name:   "layer_b",
					OpType: "TestOutputSuffOp",
					Inputs: []string{"layer_a/output_0"}, // needs resolveOutputSuffix
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "layer_b"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

func TestBuildFromZMF_OutputResolution(t *testing.T) {
	tests := []struct {
		name       string
		opType     string
		nodeName   string
		outputName string
	}{
		{"suffix_output_0", "TestOutSfxOp", "final_node", "final_node/output_0"},
		{"logits_mapping", "MatMul", "/lm_head/MatMul", "logits"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ops := numeric.Float32Ops{}
			engine := compute.NewCPUEngine[float32](ops)

			RegisterLayer(tt.opType, passthroughBuilder())
			defer UnregisterLayer(tt.opType)

			zmfModel := &zmf.Model{
				Graph: &zmf.Graph{
					Inputs: []*zmf.ValueInfo{
						{Name: "input", Shape: []int64{1}},
					},
					Nodes: []*zmf.Node{
						{Name: tt.nodeName, OpType: tt.opType, Inputs: []string{"input"}},
					},
					Outputs: []*zmf.ValueInfo{{Name: tt.outputName}},
				},
			}

			g, err := BuildFromZMF[float32](engine, ops, zmfModel)
			if err != nil {
				t.Fatalf("BuildFromZMF failed: %v", err)
			}
			if g == nil {
				t.Fatal("expected non-nil graph")
			}
		})
	}
}

func TestBuildFromZMF_NodeWithAttributes(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestAttrOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		attrs map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestAttrOp")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "attr_node",
					OpType: "TestAttrOp",
					Inputs: []string{"input"},
					Attributes: map[string]*zmf.Attribute{
						"epsilon": {Value: &zmf.Attribute_F{F: 1e-5}},
						"axis":    {Value: &zmf.Attribute_I{I: -1}},
					},
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "attr_node"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

func TestBuildFromZMF_InputNodeNotFound(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestNotFoundOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestNotFoundOp")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "node1",
					OpType: "TestNotFoundOp",
					Inputs: []string{"nonexistent_node"}, // this doesn't exist as node or parameter
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "node1"}},
		},
	}

	_, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err == nil {
		t.Error("expected error for missing input node")
	}
}

func TestBuildFromZMF_Reshape_WithShapeParam(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("Reshape", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("Reshape")

	// Shape parameter: [2, 3] as float32 (will be cast to int64)
	shapeData := make([]byte, 8) // 2 float32 values
	// 2.0 in float32: 0x40000000
	shapeData[0] = 0x00
	shapeData[1] = 0x00
	shapeData[2] = 0x00
	shapeData[3] = 0x40
	// 3.0 in float32: 0x40400000
	shapeData[4] = 0x00
	shapeData[5] = 0x00
	shapeData[6] = 0x40
	shapeData[7] = 0x40

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"shape_param": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{2},
					Data:  shapeData,
				},
			},
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{6}},
				{Name: "shape_param", Shape: []int64{2}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "reshape_node",
					OpType: "Reshape",
					Inputs: []string{"input", "shape_param"},
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "reshape_node"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

func TestBuildFromZMF_MatMul_LmHead_EmbedTokens(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("MatMul", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("MatMul")

	// Create embed_tokens parameter [2, 3] - will be transposed to [3, 2]
	embedData := make([]byte, 6*4) // 6 float32 values
	for i := 0; i < 6; i++ {
		// Encode float32 value (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
		val := float32(i + 1)
		bits := math.Float32bits(val)
		binary.LittleEndian.PutUint32(embedData[i*4:], bits)
	}

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Parameters: map[string]*zmf.Tensor{
				"embed_tokens": {
					Dtype: zmf.Tensor_FLOAT32,
					Shape: []int64{2, 3},
					Data:  embedData,
				},
			},
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1, 3}},
				{Name: "embed_tokens", Shape: []int64{2, 3}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "/lm_head/MatMul",
					OpType: "MatMul",
					Inputs: []string{"input", "embed_tokens"},
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "/lm_head/MatMul"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// embeddedWeightsNode implements HasEmbeddedWeights interface.
type embeddedWeightsNode struct {
	parameterNode[float32]
}

func (e *embeddedWeightsNode) HasEmbeddedWeights() bool { return true }

func TestBuildFromZMF_HasEmbeddedWeights(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("GatherOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &embeddedWeightsNode{parameterNode: parameterNode[float32]{value: val}}, nil
	})
	defer UnregisterLayer("GatherOp")

	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{
					Name:   "gather",
					OpType: "GatherOp",
					Inputs: []string{"weights", "input"}, // first input (weights) skipped
				},
			},
			Outputs: []*zmf.ValueInfo{{Name: "gather"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

func TestBuildFromZMF_ExistingNodeNameSkipped(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	RegisterLayer("TestSkipOp", func(
		_ compute.Engine[float32],
		_ numeric.Arithmetic[float32],
		_ string,
		_ map[string]*graph.Parameter[float32],
		_ map[string]any,
	) (graph.Node[float32], error) {
		val, _ := tensor.New[float32]([]int{1}, []float32{1})
		return &parameterNode[float32]{value: val}, nil
	})
	defer UnregisterLayer("TestSkipOp")

	// Node with same name as input should be skipped
	zmfModel := &zmf.Model{
		Graph: &zmf.Graph{
			Inputs: []*zmf.ValueInfo{
				{Name: "input", Shape: []int64{1}},
			},
			Nodes: []*zmf.Node{
				{Name: "input", OpType: "TestSkipOp", Inputs: []string{}}, // same name as input
				{Name: "node2", OpType: "TestSkipOp", Inputs: []string{"input"}},
			},
			Outputs: []*zmf.ValueInfo{{Name: "node2"}},
		},
	}

	g, err := BuildFromZMF[float32](engine, ops, zmfModel)
	if err != nil {
		t.Fatalf("BuildFromZMF failed: %v", err)
	}
	if g == nil {
		t.Fatal("expected non-nil graph")
	}
}

// --- GetLayerBuilder type mismatch test ---

func TestGetLayerBuilder_TypeMismatch(t *testing.T) {
	// Register a layer builder for int, then try to get it as float32
	RegisterLayer("TypeMismatchOp", func(
		_ compute.Engine[int],
		_ numeric.Arithmetic[int],
		_ string,
		_ map[string]*graph.Parameter[int],
		_ map[string]any,
	) (graph.Node[int], error) {
		return nil, nil
	})
	defer UnregisterLayer("TypeMismatchOp")

	// Try to get it as float32 - should fail with type mismatch
	_, err := GetLayerBuilder[float32]("TypeMismatchOp")
	if err == nil {
		t.Error("expected error for type mismatch in GetLayerBuilder")
	}
}

// --- LoadModelFromZMF error test ---

func TestLoadModelFromZMF_FileNotFound(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := LoadModelFromZMF[float32](engine, ops, "/nonexistent/path/model.zmf")
	if err == nil {
		t.Error("expected error for missing file")
	}
}

func TestNewModel(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	builder := graph.NewBuilder[float32](engine)
	inputNode := builder.Input([]int{2, 2})

	node := &mockNodeF32{
		opType:      "MockOp",
		outputShape: []int{2, 2},
		params:      []*graph.Parameter[float32]{},
	}
	builder.AddNode(node, inputNode)

	g, err := builder.Build(node)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}

	m := NewModel[float32](nil, g)
	if m == nil {
		t.Fatal("expected non-nil model")
	}
	if m.Graph != g {
		t.Error("expected model graph to match")
	}
	if m.Embedding != nil {
		t.Error("expected nil embedding")
	}
}

// TestModelForward tests the full Model.Forward path with embedding + graph.
func TestModelForward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	// Create a small token embedding (vocab=4, dim=2)
	embedding, err := embeddings.NewTokenEmbedding[float32](engine, 4, 2)
	if err != nil {
		t.Fatalf("failed to create TokenEmbedding: %v", err)
	}

	// Create a graph that takes embedding output (shape [seqLen, dim]) and passes through
	builder := graph.NewBuilder[float32](engine)
	inputNode := builder.Input([]int{3, 2}) // 3 tokens, dim 2

	passthrough := &mockNodeF32{
		opType:      "Passthrough",
		outputShape: []int{3, 2},
		params:      []*graph.Parameter[float32]{},
	}
	builder.AddNode(passthrough, inputNode)

	g, err := builder.Build(passthrough)
	if err != nil {
		t.Fatalf("failed to build graph: %v", err)
	}

	m := NewModel[float32](embedding, g)
	ctx := context.Background()

	// Token IDs: [0, 1, 2] as float32
	tokenIDs, err := tensor.New[float32]([]int{3}, []float32{0, 1, 2})
	if err != nil {
		t.Fatalf("failed to create token IDs: %v", err)
	}

	output, err := m.Forward(ctx, tokenIDs)
	if err != nil {
		t.Fatalf("Model.Forward failed: %v", err)
	}
	if output == nil {
		t.Fatal("expected non-nil output")
	}
}

// TestModelForward_EmbeddingError tests Model.Forward when embedding fails.
func TestModelForward_EmbeddingError(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	embedding, err := embeddings.NewTokenEmbedding[float32](engine, 4, 2)
	if err != nil {
		t.Fatalf("failed to create TokenEmbedding: %v", err)
	}

	builder := graph.NewBuilder[float32](engine)
	inputNode := builder.Input([]int{1, 2})
	passthrough := &mockNodeF32{
		opType:      "Passthrough",
		outputShape: []int{1, 2},
		params:      []*graph.Parameter[float32]{},
	}
	builder.AddNode(passthrough, inputNode)
	g, _ := builder.Build(passthrough)

	m := NewModel[float32](embedding, g)
	ctx := context.Background()

	// Pass no inputs - should cause embedding error
	_, err = m.Forward(ctx)
	if err == nil {
		t.Error("expected error when embedding receives no inputs")
	}
}
