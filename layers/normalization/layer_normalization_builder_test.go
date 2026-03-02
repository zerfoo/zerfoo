package normalization

import (
	"context"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func makeBiasParam(t *testing.T, name string, dim int) *graph.Parameter[float32] {
	t.Helper()
	data := make([]float32, dim)
	val, err := tensor.New[float32]([]int{dim}, data)
	if err != nil {
		t.Fatal(err)
	}
	p, err := graph.NewParameter[float32](name, val, tensor.New[float32])
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBuildLayerNormalization_NameScale(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_scale": makeGainParam(t, "test_scale", 4),
		"test_bias":  makeBiasParam(t, "test_bias", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildLayerNormalization[float32](engine, ops, "test", params, attrs)
	if err != nil {
		t.Fatalf("BuildLayerNormalization failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
	if node.OpType() != "LayerNormalization" {
		t.Errorf("OpType = %q, want LayerNormalization", node.OpType())
	}
}

func TestBuildLayerNormalization_DotWeightBias(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// Path-like name with LayerNormalization suffix -> dot.weight / dot.bias
	params := map[string]*graph.Parameter[float32]{
		"model.layers.0.self_attn_layer_norm.weight": makeGainParam(t, "w", 4),
		"model.layers.0.self_attn_layer_norm.bias":   makeBiasParam(t, "b", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildLayerNormalization[float32](engine, ops, "/model/layers.0/self_attn_layer_norm/LayerNormalization", params, attrs)
	if err != nil {
		t.Fatalf("BuildLayerNormalization dot-weight/bias failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildLayerNormalization_NoScaleNoParams(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	// No matching params - should still succeed with default init
	attrs := map[string]any{"epsilon": float64(1e-6)}
	node, err := BuildLayerNormalization[float32](engine, ops, "test", map[string]*graph.Parameter[float32]{}, attrs)
	if err != nil {
		t.Fatalf("BuildLayerNormalization (no params) failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildLayerNormalization_MissingEpsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := BuildLayerNormalization[float32](engine, ops, "test", map[string]*graph.Parameter[float32]{}, map[string]any{})
	if err == nil {
		t.Error("expected error for missing epsilon")
	}
}

func TestBuildLayerNormalization_Float32Epsilon(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	attrs := map[string]any{"epsilon": float32(1e-6)}
	node, err := BuildLayerNormalization[float32](engine, ops, "test", map[string]*graph.Parameter[float32]{}, attrs)
	if err != nil {
		t.Fatalf("float32 epsilon failed: %v", err)
	}
	if node == nil {
		t.Fatal("returned nil")
	}
}

func TestBuildLayerNormalization_BadEpsilonType(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	_, err := BuildLayerNormalization[float32](engine, ops, "test", map[string]*graph.Parameter[float32]{}, map[string]any{"epsilon": "bad"})
	if err == nil {
		t.Error("expected error for bad epsilon type")
	}
}

func TestBuildLayerNormalization_ForwardPass(t *testing.T) {
	ops := &numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	params := map[string]*graph.Parameter[float32]{
		"test_scale": makeGainParam(t, "test_scale", 4),
		"test_bias":  makeBiasParam(t, "test_bias", 4),
	}
	attrs := map[string]any{"epsilon": float64(1e-6)}

	node, err := BuildLayerNormalization[float32](engine, ops, "test", params, attrs)
	if err != nil {
		t.Fatalf("BuildLayerNormalization failed: %v", err)
	}

	input, _ := tensor.New[float32]([]int{2, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	out, err := node.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if len(out.Shape()) != 2 || out.Shape()[0] != 2 || out.Shape()[1] != 4 {
		t.Errorf("output shape = %v, want [2 4]", out.Shape())
	}
}
