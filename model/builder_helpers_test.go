package model

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/log"
	"github.com/zerfoo/zerfoo/tensor"
)

// --- maskFromInputNode tests ---

func TestMaskFromInputNode(t *testing.T) {
	node := &maskFromInputNode[float32]{}

	if node.OpType() != "AutoAttentionMask" {
		t.Errorf("OpType = %q, want AutoAttentionMask", node.OpType())
	}
	if node.Attributes() != nil {
		t.Error("expected nil Attributes")
	}
	if node.OutputShape() != nil {
		t.Error("expected nil OutputShape")
	}
	if node.Parameters() != nil {
		t.Error("expected nil Parameters")
	}
}

func TestMaskFromInputNode_Forward(t *testing.T) {
	node := &maskFromInputNode[float32]{}
	input, _ := tensor.New[float32]([]int{2, 5}, make([]float32, 10))

	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if out.Size() != 10 {
		t.Errorf("output size = %d, want 10", out.Size())
	}
	for i, v := range out.Data() {
		if v != 1 {
			t.Errorf("data[%d] = %v, want 1", i, v)
			break
		}
	}
}

func TestMaskFromInputNode_Backward(t *testing.T) {
	node := &maskFromInputNode[float32]{}
	grads, err := node.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if grads != nil {
		t.Error("expected nil gradients")
	}
}

// --- positionIdsNode tests ---

func TestPositionIdsNode(t *testing.T) {
	node := &positionIdsNode[float32]{}

	if node.OpType() != "AutoPositionIds" {
		t.Errorf("OpType = %q, want AutoPositionIds", node.OpType())
	}
	if node.Attributes() != nil {
		t.Error("expected nil Attributes")
	}
	if node.OutputShape() != nil {
		t.Error("expected nil OutputShape")
	}
	if node.Parameters() != nil {
		t.Error("expected nil Parameters")
	}
}

func TestPositionIdsNode_Forward(t *testing.T) {
	node := &positionIdsNode[float32]{}
	// Shape [2, 4]: batch=2, seq_len=4
	input, _ := tensor.New[float32]([]int{2, 4}, make([]float32, 8))

	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	data := out.Data()
	// Each row should be [0, 1, 2, 3]
	expected := []float32{0, 1, 2, 3, 0, 1, 2, 3}
	for i, want := range expected {
		if data[i] != want {
			t.Errorf("data[%d] = %v, want %v", i, data[i], want)
		}
	}
}

func TestPositionIdsNode_Backward(t *testing.T) {
	node := &positionIdsNode[float32]{}
	grads, err := node.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if grads != nil {
		t.Error("expected nil gradients")
	}
}

// --- kvCacheIONode tests ---

func TestZeroKVCacheNode(t *testing.T) {
	node := &kvCacheIONode[float32]{numHeads: 8, headDim: 64}

	if node.OpType() != "AutoKVCacheIO" {
		t.Errorf("OpType = %q, want AutoKVCacheIO", node.OpType())
	}
	if node.Attributes() != nil {
		t.Error("expected nil Attributes")
	}
	if node.OutputShape() != nil {
		t.Error("expected nil OutputShape")
	}
	if node.Parameters() != nil {
		t.Error("expected nil Parameters")
	}
}

func TestZeroKVCacheNode_Forward(t *testing.T) {
	tests := []struct {
		name      string
		numHeads  int
		headDim   int
		input     *tensor.TensorNumeric[float32]
		wantShape []int
	}{
		{
			name:      "with input batch=2",
			numHeads:  4,
			headDim:   32,
			input:     mustTensor(t, []int{2, 10}, make([]float32, 20)),
			wantShape: []int{2, 4, 0, 32},
		},
		{
			name:      "no input defaults batch=1",
			numHeads:  8,
			headDim:   64,
			wantShape: []int{1, 8, 0, 64},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &kvCacheIONode[float32]{numHeads: tt.numHeads, headDim: tt.headDim}
			var inputs []*tensor.TensorNumeric[float32]
			if tt.input != nil {
				inputs = append(inputs, tt.input)
			}

			out, err := node.Forward(context.Background(), inputs...)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}

			shape := out.Shape()
			for i, want := range tt.wantShape {
				if i >= len(shape) || shape[i] != want {
					t.Errorf("shape = %v, want %v", shape, tt.wantShape)
					break
				}
			}
		})
	}
}

func TestZeroKVCacheNode_Backward(t *testing.T) {
	node := &kvCacheIONode[float32]{numHeads: 4, headDim: 32}
	grads, err := node.Backward(context.Background(), 0, nil)
	if err != nil {
		t.Fatalf("Backward: %v", err)
	}
	if grads != nil {
		t.Error("expected nil gradients")
	}
}

// --- resolveParam tests ---

func TestResolveParam(t *testing.T) {
	val, _ := tensor.New[float32]([]int{2}, []float32{1, 2})
	params := map[string]*graph.Parameter[float32]{
		"weight": {Name: "weight", Value: val},
	}

	constVal, _ := tensor.New[float32]([]int{1}, []float32{42})
	nodes := map[string]graph.Node[float32]{
		"const_node": &parameterNode[float32]{value: constVal},
		"other_node": &maskFromInputNode[float32]{},
	}

	tests := []struct {
		name     string
		lookup   string
		wantData float32
		wantNil  bool
	}{
		{"from params", "weight", 1, false},
		{"from const node", "const_node", 42, false},
		{"non-parameterNode", "other_node", 0, true},
		{"not found", "missing", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := resolveParam(tt.lookup, params, nodes)
			if tt.wantNil {
				if result != nil {
					t.Error("expected nil")
				}
				return
			}
			if result == nil {
				t.Fatal("expected non-nil")
			}
			if result.Data()[0] != tt.wantData {
				t.Errorf("data[0] = %v, want %v", result.Data()[0], tt.wantData)
			}
		})
	}
}

// --- isConstantPromotedAttr tests ---

func TestIsConstantPromotedAttr(t *testing.T) {
	tests := []struct {
		key  string
		want bool
	}{
		{"/Constant_output_0", true},
		{"onnx::Gather_919", true},
		{"axis", false},
		{"epsilon", false},
		{"perm", false},
	}

	for _, tt := range tests {
		if got := isConstantPromotedAttr(tt.key); got != tt.want {
			t.Errorf("isConstantPromotedAttr(%q) = %v, want %v", tt.key, got, tt.want)
		}
	}
}

// --- WithGlobalAttributes tests ---

func TestWithGlobalAttributes(t *testing.T) {
	attrs := map[string]interface{}{
		"rope_scaling_type": "linear",
		"rope_scaling_factor": 2.0,
	}

	cfg := buildConfig{}
	opt := WithGlobalAttributes(attrs)
	opt(&cfg)

	if len(cfg.globalAttributes) != 2 {
		t.Errorf("expected 2 global attributes, got %d", len(cfg.globalAttributes))
	}
	if cfg.globalAttributes["rope_scaling_type"] != "linear" {
		t.Error("rope_scaling_type not set")
	}
}

// --- SetLogger tests ---

func TestSetLogger(t *testing.T) {
	// SetLogger with nil should use Nop logger (no panic)
	SetLogger(nil)

	// SetLogger with a real logger
	SetLogger(log.New(nil, log.LevelInfo, log.FormatText))

	// Reset to Nop
	SetLogger(nil)
}

func mustTensor(t *testing.T, shape []int, data []float32) *tensor.TensorNumeric[float32] {
	t.Helper()
	v, err := tensor.New[float32](shape, data)
	if err != nil {
		t.Fatal(err)
	}
	return v
}
