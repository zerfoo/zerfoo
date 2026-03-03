package core

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestLinear_ZMFRoundTrip(t *testing.T) {
	engine := compute.NewCPUEngine(numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	// Create a Linear layer with known parameters.
	original, err := NewLinear[float32]("test_linear", engine, ops, 4, 8)
	if err != nil {
		t.Fatalf("NewLinear: %v", err)
	}

	// Set weights to known values.
	origParams := original.Parameters()
	origData := origParams[0].Value.Data()
	for i := range origData {
		origData[i] = float32(i) * 0.01
	}

	// Build params map as ZMF loading would provide.
	params := make(map[string]*graph.Parameter[float32])
	for _, p := range origParams {
		params[p.Name] = p
	}

	// Call registry builder with the params.
	builder, err := model.GetLayerBuilder[float32]("Linear")
	if err != nil {
		t.Fatalf("GetLayerBuilder: %v", err)
	}

	restored, err := builder(engine, ops, "test_linear", params, original.Attributes())
	if err != nil {
		t.Fatalf("builder: %v", err)
	}

	// Verify OpType matches.
	if restored.OpType() != original.OpType() {
		t.Errorf("OpType = %q, want %q", restored.OpType(), original.OpType())
	}

	// Verify Attributes match.
	origAttrs := original.Attributes()
	restoredAttrs := restored.Attributes()
	for key, want := range origAttrs {
		if got := restoredAttrs[key]; got != want {
			t.Errorf("Attributes[%q] = %v, want %v", key, got, want)
		}
	}

	// Verify Parameters match (same pointer since we passed the same params).
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
