package multimodal

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func TestProjectionShape(t *testing.T) {
	cfg := ConnectorConfig{VisionDim: 1152, TextDim: 4096}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	conn := NewProjectionConnector[float32](cfg, engine)

	numTokens := 196
	input := make([]float32, numTokens*cfg.VisionDim)
	for i := range input {
		input[i] = float32(i) * 0.0001
	}

	out, err := conn.Project(input, numTokens)
	if err != nil {
		t.Fatalf("Project: %v", err)
	}

	want := numTokens * cfg.TextDim
	if len(out) != want {
		t.Fatalf("output length = %d, want %d (%d tokens x %d dim)", len(out), want, numTokens, cfg.TextDim)
	}
}

func TestProjectionConnectorLoadWeights(t *testing.T) {
	cfg := ConnectorConfig{VisionDim: 4, TextDim: 3}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	conn := NewProjectionConnector[float32](cfg, engine)

	// With zero weights, output should be all zeros.
	input := []float32{1, 0, 0, 0}
	out0, err := conn.Project(input, 1)
	if err != nil {
		t.Fatalf("Project before LoadWeights: %v", err)
	}
	for i, v := range out0 {
		if v != 0 {
			t.Fatalf("before LoadWeights: out[%d] = %f, want 0", i, v)
		}
	}

	// Load identity-like weights: first row = [1, 2, 3].
	weights := make([]float32, cfg.VisionDim*cfg.TextDim)
	weights[0] = 1
	weights[1] = 2
	weights[2] = 3
	if err := conn.LoadWeights(weights); err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}

	out1, err := conn.Project(input, 1)
	if err != nil {
		t.Fatalf("Project after LoadWeights: %v", err)
	}

	// input=[1,0,0,0] x weights should give [1,2,3].
	expected := []float32{1, 2, 3}
	for i, want := range expected {
		if out1[i] != want {
			t.Errorf("after LoadWeights: out[%d] = %f, want %f", i, out1[i], want)
		}
	}
}

func TestConnectorDims(t *testing.T) {
	cfg := ConnectorConfig{VisionDim: 1152, TextDim: 4096}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	conn := NewProjectionConnector[float32](cfg, engine)

	if got := conn.VisionDim(); got != 1152 {
		t.Errorf("VisionDim() = %d, want 1152", got)
	}
	if got := conn.TextDim(); got != 4096 {
		t.Errorf("TextDim() = %d, want 4096", got)
	}
}

func TestProjectionConnectorDefaultWeightKey(t *testing.T) {
	cfg := ConnectorConfig{VisionDim: 4, TextDim: 3}
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	conn := NewProjectionConnector[float32](cfg, engine)

	if conn.cfg.WeightKey != "mm.projector.weight" {
		t.Errorf("default WeightKey = %q, want %q", conn.cfg.WeightKey, "mm.projector.weight")
	}
}
