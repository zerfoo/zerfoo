package parity_test

import (
	"context"
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

// qwen3GGUFPath returns the Qwen 3 GGUF under test, or "" when unset.
//
// Verified against Qwen/Qwen3-0.6B-GGUF (Qwen3-0.6B-Q8_0.gguf,
// sha256 9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031).
func qwen3GGUFPath(t *testing.T) string {
	t.Helper()
	p := os.Getenv("QWEN3_GGUF_PATH")
	if p == "" {
		t.Skip("QWEN3_GGUF_PATH not set; skipping Qwen 3 GGUF parity test")
	}

	return p
}

// TestQwen3GGUFArchitectureDelta pins the three properties that distinguish
// Qwen 3 from Qwen 2 in a real GGUF, so a regression in the loader or the
// builder is caught against actual model metadata rather than a fixture.
func TestQwen3GGUFArchitectureDelta(t *testing.T) {
	path := qwen3GGUFPath(t)

	gm, err := inference.LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}
	cfg := gm.Config

	if cfg.Architecture != "qwen3" {
		t.Fatalf("Architecture = %q, want %q", cfg.Architecture, "qwen3")
	}

	// (1) Head dimension is decoupled from HiddenSize/NumHeads. Qwen3-0.6B has
	// hiddenSize 1024 with 16 query heads but headDim 128, not 64. A loader
	// that ignored attention.key_length would silently build 64-wide heads.
	if cfg.HeadDim <= 0 {
		t.Fatalf("HeadDim = %d, want an explicit head dimension from attention.key_length", cfg.HeadDim)
	}
	if cfg.NumHeads > 0 && cfg.HeadDim == cfg.HiddenSize/cfg.NumHeads {
		t.Logf("note: HeadDim %d coincides with HiddenSize/NumHeads for this size", cfg.HeadDim)
	}

	qW := gm.Tensors["model.layers.0.self_attn.q_proj.weight"]
	if qW == nil {
		t.Fatal("missing model.layers.0.self_attn.q_proj.weight")
	}
	// The Q projection must produce NumHeads*HeadDim features.
	wantQ := cfg.NumHeads * cfg.HeadDim
	shape := qW.Shape()
	foundQ := false
	for _, d := range shape {
		if d == wantQ {
			foundQ = true
		}
	}
	if !foundQ {
		t.Errorf("q_proj shape %v has no dimension equal to NumHeads*HeadDim = %d", shape, wantQ)
	}

	// (2) Per-head QK RMSNorm weights exist on every layer, shaped [HeadDim].
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + strconv.Itoa(i) + ".self_attn."
		for _, name := range []string{"q_norm", "k_norm"} {
			w := gm.Tensors[prefix+name+".weight"]
			if w == nil {
				t.Fatalf("missing %s%s.weight (Qwen 3 requires per-head QK norm)", prefix, name)
			}
			s := w.Shape()
			if len(s) != 1 || s[0] != cfg.HeadDim {
				t.Fatalf("%s%s.weight shape = %v, want [%d]", prefix, name, s, cfg.HeadDim)
			}
		}
	}

	// (3) Qwen 3 dropped Qwen 2's Q/K/V projection biases entirely.
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := "model.layers." + strconv.Itoa(i) + ".self_attn."
		for _, name := range []string{"q_proj", "k_proj", "v_proj"} {
			if w := gm.Tensors[prefix+name+".bias"]; w != nil {
				t.Errorf("unexpected %s%s.bias: Qwen 3 has no attention bias", prefix, name)
			}
		}
	}
}

// TestQwen3GGUFForwardPass builds the graph from the real GGUF and asserts the
// logits are finite and non-degenerate. A builder that wired QK norm wrongly
// typically produces NaNs or a flat/saturated logit distribution.
func TestQwen3GGUFForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	path := qwen3GGUFPath(t)

	gm, err := inference.LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF: %v", err)
	}

	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	g, emb, err := inference.AutoBuild(gm.Tensors, gm.Config, engine)
	if err != nil {
		t.Fatalf("AutoBuild qwen3: %v", err)
	}
	if g == nil || emb == nil {
		t.Fatal("AutoBuild returned a nil graph or embedding")
	}

	tokenIDs := []float32{9707, 11, 1879, 0, 358}
	input, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("build input: %v", err)
	}

	out, err := g.Forward(context.Background(), input)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}

	data := out.Data()
	if len(data) == 0 {
		t.Fatal("forward produced no logits")
	}

	minV, maxV := float32(math.Inf(1)), float32(math.Inf(-1))
	for i, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d is not finite: %v", i, v)
		}
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}

	// A correctly wired model separates its logits. A collapsed graph (for
	// example QK norm applied over the wrong axis) tends toward a constant.
	if maxV-minV < 1.0 {
		t.Fatalf("logit range %v..%v is degenerate; expected meaningful separation", minV, maxV)
	}
	t.Logf("qwen3 forward pass: %d logits, range %.4f..%.4f", len(data), minV, maxV)
}

// TestQwen3GPUParity builds the same real Qwen 3 GGUF on the CPU and GPU
// engines and compares the logits element-by-element. Qwen 3 is the first
// architecture in the tree combining QK RMSNorm with a head dimension that is
// not HiddenSize/NumHeads, so the GPU attention path is exercised in a shape
// combination no existing model covers. Skips when no GPU is present.
func TestQwen3GPUParity(t *testing.T) {
	layerreg.RegisterAll()
	path := qwen3GGUFPath(t)

	ops := numeric.Float32Ops{}

	gmCPU, err := inference.LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF (cpu): %v", err)
	}
	cpuEngine := compute.Engine[float32](compute.NewCPUEngine[float32](ops))
	cpuGraph, _, err := inference.AutoBuild(gmCPU.Tensors, gmCPU.Config, cpuEngine)
	if err != nil {
		t.Fatalf("AutoBuild (cpu): %v", err)
	}

	gpuEngPtr, err := compute.NewGPUEngine[float32](ops, 0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	gpuEngine := compute.Engine[float32](gpuEngPtr)

	// Load a second copy: the builder uploads/transposes weights in place for
	// GPU engines, so the CPU graph must not share tensors with it.
	gmGPU, err := inference.LoadGGUF(path)
	if err != nil {
		t.Fatalf("LoadGGUF (gpu): %v", err)
	}
	gpuGraph, _, err := inference.AutoBuild(gmGPU.Tensors, gmGPU.Config, gpuEngine)
	if err != nil {
		t.Fatalf("AutoBuild (gpu): %v", err)
	}

	tokenIDs := []float32{9707, 11, 1879, 0, 358}
	cpuIn, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("cpu input: %v", err)
	}
	gpuIn, err := tensor.New([]int{1, len(tokenIDs)}, tokenIDs)
	if err != nil {
		t.Fatalf("gpu input: %v", err)
	}

	cpuOut, err := cpuGraph.Forward(context.Background(), cpuIn)
	if err != nil {
		t.Fatalf("cpu forward: %v", err)
	}
	gpuOut, err := gpuGraph.Forward(context.Background(), gpuIn)
	if err != nil {
		t.Fatalf("gpu forward: %v", err)
	}

	cpuData := cpuOut.Data()
	gpuData := gpuOut.Data()
	if len(cpuData) != len(gpuData) {
		t.Fatalf("length mismatch: cpu=%d gpu=%d", len(cpuData), len(gpuData))
	}

	maxDiff, maxIdx := float64(0), 0
	for i := range cpuData {
		d := math.Abs(float64(cpuData[i] - gpuData[i]))
		if d > maxDiff {
			maxDiff, maxIdx = d, i
		}
	}

	// Compare the decisions, not just the numbers: greedy decode only cares
	// which logit is largest.
	argmax := func(v []float32) int {
		best := 0
		for i := range v {
			if v[i] > v[best] {
				best = i
			}
		}

		return best
	}
	cpuTop, gpuTop := argmax(cpuData), argmax(gpuData)

	t.Logf("qwen3 CPU/GPU parity: len=%d maxDiff=%.6e at idx=%d (cpu=%.6f gpu=%.6f)",
		len(cpuData), maxDiff, maxIdx, cpuData[maxIdx], gpuData[maxIdx])
	t.Logf("qwen3 argmax: cpu=%d gpu=%d", cpuTop, gpuTop)

	if cpuTop != gpuTop {
		t.Errorf("CPU and GPU disagree on the greedy token: cpu=%d gpu=%d", cpuTop, gpuTop)
	}
	if maxDiff > 0.1 {
		t.Errorf("maxDiff=%.4f exceeds threshold 0.1", maxDiff)
	}
}
