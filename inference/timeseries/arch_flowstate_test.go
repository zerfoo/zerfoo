package timeseries

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

func TestFlowStateBuild(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := &FlowStateConfig{
		ContextLen:   96,
		ForecastLen:  24,
		NumChannels:  3,
		PatchLen:     16,
		DModel:       32,
		NumSSMLayers: 2,
		DState:       16,
		NumBasis:     8,
		ScaleFactor:  1.0,
	}

	g, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}
	if g == nil {
		t.Fatal("BuildFlowState returned nil graph")
	}
}

func TestFlowStateForward(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := &FlowStateConfig{
		ContextLen:   96,
		ForecastLen:  24,
		NumChannels:  3,
		PatchLen:     16,
		DModel:       32,
		NumSSMLayers: 2,
		DState:       16,
		NumBasis:     8,
		ScaleFactor:  1.0,
	}

	g, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}

	batch := 2
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i%100) * 0.01
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.ForecastLen, cfg.NumChannels}
	if len(got) != len(want) {
		t.Fatalf("output shape rank: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}

	// Verify output contains non-zero values (model produces actual predictions).
	outData := output.Data()
	allZero := true
	for _, v := range outData {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Error("output is all zeros, expected non-zero predictions")
	}
}

func TestFlowStateScaleFactorAdaptation(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	baseCfg := FlowStateConfig{
		ContextLen:   64,
		ForecastLen:  16,
		NumChannels:  2,
		PatchLen:     8,
		DModel:       16,
		NumSSMLayers: 1,
		DState:       8,
		NumBasis:     4,
		ScaleFactor:  1.0,
	}

	batch := 1
	data := make([]float32, batch*baseCfg.ContextLen*baseCfg.NumChannels)
	for i := range data {
		data[i] = float32(i%50) * 0.02
	}
	input, err := tensor.New[float32]([]int{batch, baseCfg.ContextLen, baseCfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	// Run with scale_factor = 1.0 (hourly).
	cfg1 := baseCfg
	cfg1.ScaleFactor = 1.0
	node1, err := newFlowStateNode[float32](&cfg1, engine, numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("newFlowStateNode (sf=1.0): %v", err)
	}
	out1, err := node1.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward (sf=1.0): %v", err)
	}

	// Create a second node with different scale_factor but same weights.
	cfg2 := baseCfg
	cfg2.ScaleFactor = 0.25 // 15-minute
	node2, err := newFlowStateNode[float32](&cfg2, engine, numeric.Float32Ops{})
	if err != nil {
		t.Fatalf("newFlowStateNode (sf=0.25): %v", err)
	}

	// Copy weights from node1 to node2 so only scale_factor differs.
	params1 := node1.Parameters()
	params2 := node2.Parameters()
	if len(params1) != len(params2) {
		t.Fatalf("parameter count mismatch: %d vs %d", len(params1), len(params2))
	}
	for i := range params1 {
		params2[i].Value = params1[i].Value
	}

	out2, err := node2.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward (sf=0.25): %v", err)
	}

	// The outputs should differ because scale_factor changes the Fourier basis
	// evaluation time points.
	d1 := out1.Data()
	d2 := out2.Data()

	var maxDiff float64
	for i := range d1 {
		diff := math.Abs(float64(d1[i] - d2[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	if maxDiff < 1e-6 {
		t.Error("outputs with different scale_factor should differ, but maxDiff < 1e-6")
	}
}

func TestFlowStateConfigValidation(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	valid := FlowStateConfig{
		ContextLen:   96,
		ForecastLen:  24,
		NumChannels:  3,
		PatchLen:     16,
		DModel:       32,
		NumSSMLayers: 2,
		DState:       16,
		NumBasis:     8,
		ScaleFactor:  1.0,
	}

	tests := []struct {
		name   string
		mutate func(*FlowStateConfig)
	}{
		{"nil config", nil},
		{"zero ContextLen", func(c *FlowStateConfig) { c.ContextLen = 0 }},
		{"zero ForecastLen", func(c *FlowStateConfig) { c.ForecastLen = 0 }},
		{"zero NumChannels", func(c *FlowStateConfig) { c.NumChannels = 0 }},
		{"zero PatchLen", func(c *FlowStateConfig) { c.PatchLen = 0 }},
		{"zero DModel", func(c *FlowStateConfig) { c.DModel = 0 }},
		{"zero NumSSMLayers", func(c *FlowStateConfig) { c.NumSSMLayers = 0 }},
		{"zero DState", func(c *FlowStateConfig) { c.DState = 0 }},
		{"zero NumBasis", func(c *FlowStateConfig) { c.NumBasis = 0 }},
		{"zero ScaleFactor", func(c *FlowStateConfig) { c.ScaleFactor = 0 }},
		{"negative ScaleFactor", func(c *FlowStateConfig) { c.ScaleFactor = -1.0 }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cfg *FlowStateConfig
			if tt.mutate != nil {
				c := valid // copy
				tt.mutate(&c)
				cfg = &c
			}
			_, err := BuildFlowState[float32](cfg, engine)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestFlowStateSingleBatch(t *testing.T) {
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})

	cfg := &FlowStateConfig{
		ContextLen:   32,
		ForecastLen:  8,
		NumChannels:  1,
		PatchLen:     8,
		DModel:       16,
		NumSSMLayers: 1,
		DState:       8,
		NumBasis:     4,
		ScaleFactor:  1.0,
	}

	g, err := BuildFlowState[float32](cfg, engine)
	if err != nil {
		t.Fatalf("BuildFlowState: %v", err)
	}

	batch := 1
	data := make([]float32, batch*cfg.ContextLen*cfg.NumChannels)
	for i := range data {
		data[i] = float32(i) * 0.001
	}
	input, err := tensor.New[float32]([]int{batch, cfg.ContextLen, cfg.NumChannels}, data)
	if err != nil {
		t.Fatalf("create input: %v", err)
	}

	output, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	got := output.Shape()
	want := []int{batch, cfg.ForecastLen, cfg.NumChannels}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("output shape[%d]: got %d, want %d", i, got[i], want[i])
		}
	}
}
