package gather

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

func TestBuildGather_WithWeightParam(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	w, err := tensor.New[float32]([]int{10, 4}, nil)
	if err != nil {
		t.Fatalf("failed to create weight tensor: %v", err)
	}

	params := map[string]*graph.Parameter[float32]{
		"model.embed_tokens.weight": {Value: w},
	}

	node, err := BuildGather(engine, ops, "test", params, nil)
	if err != nil {
		t.Fatalf("BuildGather failed: %v", err)
	}

	g, ok := node.(*Gather[float32])
	if !ok {
		t.Fatalf("BuildGather returned %T, want *Gather[float32]", node)
	}

	if !g.HasEmbeddedWeights() {
		t.Error("expected embedded weights")
	}
}

func TestBuildGather_WithGenericWeightParam(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	w, err := tensor.New[float32]([]int{10, 4}, nil)
	if err != nil {
		t.Fatalf("failed to create weight tensor: %v", err)
	}

	// Use a param name that contains "weight" but doesn't match specific patterns
	params := map[string]*graph.Parameter[float32]{
		"some_layer.weight_data": {Value: w},
	}

	node, err := BuildGather(engine, ops, "test", params, nil)
	if err != nil {
		t.Fatalf("BuildGather failed: %v", err)
	}

	g, ok := node.(*Gather[float32])
	if !ok {
		t.Fatalf("BuildGather returned %T, want *Gather[float32]", node)
	}

	if !g.HasEmbeddedWeights() {
		t.Error("expected embedded weights from generic weight parameter")
	}
}

func TestBuildGather_WithNamedWeightParam(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	w, _ := tensor.New[float32]([]int{5, 3}, nil)

	tests := []struct {
		name      string
		layerName string
		paramName string
	}{
		{"name.weight", "emb", "emb.weight"},
		{"name/Gather trimmed", "emb/Gather", "emb.weight"},
		{"name_weight", "emb", "emb_weight"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			params := map[string]*graph.Parameter[float32]{
				tc.paramName: {Value: w},
			}

			node, err := BuildGather(engine, ops, tc.layerName, params, nil)
			if err != nil {
				t.Fatalf("BuildGather failed: %v", err)
			}

			g := node.(*Gather[float32])
			if !g.HasEmbeddedWeights() {
				t.Error("expected embedded weights")
			}
		})
	}
}

func TestBuildGather_NoParams_CreatesDummy(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	params := map[string]*graph.Parameter[float32]{}

	node, err := BuildGather(engine, ops, "test", params, nil)
	if err != nil {
		t.Fatalf("BuildGather failed: %v", err)
	}

	g, ok := node.(*Gather[float32])
	if !ok {
		t.Fatalf("BuildGather returned %T, want *Gather[float32]", node)
	}

	if !g.HasEmbeddedWeights() {
		t.Error("expected dummy embedded weights")
	}
}
