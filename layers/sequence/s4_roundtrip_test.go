package sequence

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestS4_ZMFRoundTrip(t *testing.T) {
	engine := makeEngine()
	ops := numeric.Float32Ops{}

	// Create an S4 layer with known parameters.
	original, err := NewS4[float32]("test_s4", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewS4: %v", err)
	}

	// Set parameters to known values.
	origParams := original.Parameters()
	for _, p := range origParams {
		data := p.Value.Data()
		for i := range data {
			data[i] = float32(i)*0.01 + 0.5
		}
	}

	// Build params map as ZMF loading would provide.
	params := make(map[string]*graph.Parameter[float32])
	for _, p := range origParams {
		params[p.Name] = p
	}

	// Call registry builder with the params.
	builder, err := model.GetLayerBuilder[float32]("S4")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}

	restored, err := builder(engine, ops, "test_s4", params, original.Attributes())
	if err != nil {
		t.Fatalf("builder: %v", err)
	}

	// Verify OpType.
	if restored.OpType() != "S4" {
		t.Errorf("OpType = %q, want %q", restored.OpType(), "S4")
	}

	// Verify Attributes.
	origAttrs := original.Attributes()
	restoredAttrs := restored.Attributes()
	for key, want := range origAttrs {
		if got := restoredAttrs[key]; got != want {
			t.Errorf("Attributes[%q] = %v, want %v", key, got, want)
		}
	}

	// Verify Parameters match.
	restoredParams := restored.Parameters()
	if len(restoredParams) != len(origParams) {
		t.Fatalf("Parameters len = %d, want %d", len(restoredParams), len(origParams))
	}

	for i, rp := range restoredParams {
		op := origParams[i]
		if rp.Name != op.Name {
			t.Errorf("param[%d].Name = %q, want %q", i, rp.Name, op.Name)
		}
		rpData := rp.Value.Data()
		opData := op.Value.Data()
		if len(rpData) != len(opData) {
			t.Errorf("param[%d] data len = %d, want %d", i, len(rpData), len(opData))
			continue
		}
		for j := range rpData {
			if rpData[j] != opData[j] {
				t.Errorf("param[%d] data[%d] = %f, want %f", i, j, rpData[j], opData[j])
			}
		}
	}
}

func TestS4_ZMFRoundTrip_WithoutParams(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// When no params are provided, builder should create fresh parameters.
	builder, err := model.GetLayerBuilder[float32]("S4")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}

	attrs := map[string]interface{}{
		"input_dim": 4,
		"state_dim": 8,
	}
	node, err := builder(engine, ops, "fresh_s4", nil, attrs)
	if err != nil {
		t.Fatalf("builder: %v", err)
	}

	if node.OpType() != "S4" {
		t.Errorf("OpType = %q, want %q", node.OpType(), "S4")
	}

	params := node.Parameters()
	if len(params) != 4 {
		t.Fatalf("Parameters len = %d, want 4", len(params))
	}
}
