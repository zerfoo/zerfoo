package crossasset

import (
	"math"
	"testing"

	"github.com/zerfoo/ztensor/tensor"
)

func TestExtractGPUParams(t *testing.T) {
	cfg := Config{
		NSources:          3,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           2,
		DropoutRate:       0.0,
		LearningRate:      0.001,
	}
	m := NewModel(cfg)

	p, err := extractGPUParams(m)
	if err != nil {
		t.Fatalf("extractGPUParams: %v", err)
	}

	// Input projections: one per source.
	if len(p.inputW) != cfg.NSources {
		t.Errorf("inputW count = %d, want %d", len(p.inputW), cfg.NSources)
	}
	for s := range cfg.NSources {
		shape := p.inputW[s].Shape()
		if shape[0] != cfg.FeaturesPerSource || shape[1] != cfg.DModel {
			t.Errorf("inputW[%d] shape = %v, want [%d, %d]", s, shape, cfg.FeaturesPerSource, cfg.DModel)
		}
		bShape := p.inputB[s].Shape()
		if bShape[0] != 1 || bShape[1] != cfg.DModel {
			t.Errorf("inputB[%d] shape = %v, want [1, %d]", s, bShape, cfg.DModel)
		}
	}

	// Layers.
	if len(p.layers) != cfg.NLayers {
		t.Errorf("layers count = %d, want %d", len(p.layers), cfg.NLayers)
	}
	dm := cfg.DModel
	ffnDim := 4 * dm
	for li, gl := range p.layers {
		checkShape := func(name string, tn *tensor.TensorNumeric[float32], r, c int) {
			t.Helper()
			s := tn.Shape()
			if s[0] != r || s[1] != c {
				t.Errorf("layer[%d].%s shape = %v, want [%d, %d]", li, name, s, r, c)
			}
		}
		checkShape("qW", gl.qW, dm, dm)
		checkShape("kW", gl.kW, dm, dm)
		checkShape("vW", gl.vW, dm, dm)
		checkShape("outW", gl.outW, dm, dm)
		checkShape("ffnW1", gl.ffnW1, dm, ffnDim)
		checkShape("ffnW2", gl.ffnW2, ffnDim, dm)
		checkShape("lnGamma", gl.lnGamma, 1, dm)
		checkShape("lnBeta", gl.lnBeta, 1, dm)
		checkShape("ffnB1", gl.ffnB1, 1, ffnDim)
		checkShape("ffnB2", gl.ffnB2, 1, dm)
		checkShape("ffnGamma", gl.ffnGamma, 1, dm)
		checkShape("ffnBeta", gl.ffnBeta, 1, dm)
	}

	// Head.
	if s := p.headW.Shape(); s[0] != dm || s[1] != 3 {
		t.Errorf("headW shape = %v, want [%d, 3]", s, dm)
	}
	if s := p.headB.Shape(); s[0] != 1 || s[1] != 3 {
		t.Errorf("headB shape = %v, want [1, 3]", s)
	}
}

func TestExtractGPUParams_FloatConversion(t *testing.T) {
	cfg := Config{
		NSources:          2,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           1,
		DropoutRate:       0.0,
		LearningRate:      0.001,
	}
	m := NewModel(cfg)

	p, err := extractGPUParams(m)
	if err != nil {
		t.Fatalf("extractGPUParams: %v", err)
	}

	// Verify float64 -> float32 conversion preserves values within 1e-6.
	for i, v64 := range m.headW {
		v32 := float64(p.headW.Data()[i])
		if diff := math.Abs(v64 - v32); diff > 1e-6 {
			t.Errorf("headW[%d]: float64=%v, float32=%v, diff=%v", i, v64, v32, diff)
		}
	}
}

func TestAllocGrads(t *testing.T) {
	cfg := Config{
		NSources:          3,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           2,
		DropoutRate:       0.0,
		LearningRate:      0.001,
	}
	m := NewModel(cfg)

	p, err := extractGPUParams(m)
	if err != nil {
		t.Fatalf("extractGPUParams: %v", err)
	}

	g, err := allocGrads(p)
	if err != nil {
		t.Fatalf("allocGrads: %v", err)
	}

	// Grads should match param shapes and be all zeros.
	for s := range cfg.NSources {
		gShape, pShape := g.inputW[s].Shape(), p.inputW[s].Shape()
		if gShape[0] != pShape[0] || gShape[1] != pShape[1] {
			t.Errorf("grad inputW[%d] shape %v != param shape %v", s, gShape, pShape)
		}
		for _, v := range g.inputW[s].Data() {
			if v != 0 {
				t.Fatal("grad inputW should be all zeros")
			}
		}
	}

	// Head grad shape matches.
	if gS, pS := g.headW.Shape(), p.headW.Shape(); gS[0] != pS[0] || gS[1] != pS[1] {
		t.Errorf("grad headW shape %v != param shape %v", gS, pS)
	}
}

func TestWriteBackParams(t *testing.T) {
	cfg := Config{
		NSources:          2,
		FeaturesPerSource: 4,
		DModel:            8,
		NHeads:            2,
		NLayers:           1,
		DropoutRate:       0.0,
		LearningRate:      0.001,
	}
	m := NewModel(cfg)

	p, err := extractGPUParams(m)
	if err != nil {
		t.Fatalf("extractGPUParams: %v", err)
	}

	// Modify a GPU param.
	p.headB.Data()[0] = 42.0

	writeBackParams(m, p)

	if math.Abs(m.headB[0]-42.0) > 1e-6 {
		t.Errorf("writeBackParams: headB[0] = %v, want 42.0", m.headB[0])
	}
}
