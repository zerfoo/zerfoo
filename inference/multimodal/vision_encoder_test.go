package multimodal

import (
	"testing"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func testEncoderConfig() EncoderConfig {
	return EncoderConfig{
		HiddenDim: 768,
		NumHeads:  12,
		NumLayers: 12,
		PatchCfg: PatchConfig{
			PatchSize: 16,
			ImageSize: 224,
		},
	}
}

func TestSigLIPEncoderShape(t *testing.T) {
	cfg := testEncoderConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	enc := NewSigLIPEncoder[float32](cfg, engine)

	np := NumPatches(cfg.PatchCfg) // 14*14 = 196
	pd := PatchDim(cfg.PatchCfg)   // 16*16*3 = 768
	patches := make([]float32, np*pd)
	for i := range patches {
		patches[i] = float32(i) * 0.001
	}

	out, err := enc.Encode(patches, cfg.PatchCfg)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}

	want := np * cfg.HiddenDim
	if len(out) != want {
		t.Fatalf("output length = %d, want %d (num_patches=%d * hidden_dim=%d)", len(out), want, np, cfg.HiddenDim)
	}
}

func TestVisionEncoderInterface(t *testing.T) {
	cfg := testEncoderConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	enc := NewSigLIPEncoder[float32](cfg, engine)

	var _ VisionEncoder[float32] = enc
}

func TestEncoderHiddenSize(t *testing.T) {
	cfg := testEncoderConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	enc := NewSigLIPEncoder[float32](cfg, engine)

	if got := enc.HiddenSize(); got != cfg.HiddenDim {
		t.Errorf("HiddenSize() = %d, want %d", got, cfg.HiddenDim)
	}
}

func TestEncoderNumLayers(t *testing.T) {
	cfg := testEncoderConfig()
	engine := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	enc := NewSigLIPEncoder[float32](cfg, engine)

	if got := enc.NumLayers(); got != cfg.NumLayers {
		t.Errorf("NumLayers() = %d, want %d", got, cfg.NumLayers)
	}
}
