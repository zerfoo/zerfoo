package gather

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestBuildGather_WithWeightParam(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	w, err := tensor.New[float32]([]int{10, 4}, nil)
	if err != nil {
		t.Fatalf("failed to create weight tensor: %v", err)
	}

	// Use the ONNX-style node name that maps to the param via normalization:
	// "/model/embed_tokens/Gather" -> "model.embed_tokens.Gather"
	// TrimSuffix(".Gather") -> "model.embed_tokens" + ".weight"
	params := map[string]*graph.Parameter[float32]{
		"model.embed_tokens.weight": {Value: w},
	}

	node, err := BuildGather(engine, ops, "/model/embed_tokens/Gather", params, nil)
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

func TestBuildGather_GlobalParamNoMatch(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	w, _ := tensor.New[float32]([]int{10, 4}, nil)

	// A Gather node NOT named for the embedding should NOT pick up the
	// global model.embed_tokens.weight parameter.
	params := map[string]*graph.Parameter[float32]{
		"model.embed_tokens.weight": {Value: w},
	}

	node, err := BuildGather(engine, ops, "/model/Gather", params, nil)
	if err != nil {
		t.Fatalf("BuildGather failed: %v", err)
	}
	g := node.(*Gather[float32])
	if g.HasEmbeddedWeights() {
		t.Error("non-embedding Gather should NOT get global embed_tokens weight")
	}
}

func TestBuildGather_WithGenericWeightParam_NoEmbed(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	w, err := tensor.New[float32]([]int{10, 4}, nil)
	if err != nil {
		t.Fatalf("failed to create weight tensor: %v", err)
	}

	// Generic weight param that doesn't match any embedding pattern should
	// NOT be embedded — the node operates as a general Gather.
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

	if g.HasEmbeddedWeights() {
		t.Error("generic weight param should NOT be embedded")
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

func TestBuildGather_NoParams_GeneralGather(t *testing.T) {
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

	if g.HasEmbeddedWeights() {
		t.Error("no-params Gather should NOT have embedded weights")
	}
}
