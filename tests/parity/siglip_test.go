package parity_test

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/registry"
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/numeric"
	"github.com/zerfoo/zerfoo/tensor"
)

// TestSigLIPForwardPass loads a ZMF-converted SigLIP vision encoder and runs
// a forward pass with a synthetic [1, 3, 224, 224] float32 image tensor.
//
// Assertions:
//   - output shape [1, 196, embedDim] (patch_size=16 gives 14x14=196 patches)
//   - no NaN or Inf values
//
// Skipped when SIGLIP_ZMF_PATH is not set.
func TestSigLIPForwardPass(t *testing.T) {
	zmfPath := os.Getenv("SIGLIP_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("SIGLIP_ZMF_PATH not set; skipping SigLIP forward pass test")
	}

	registry.RegisterAll()

	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	m, err := model.LoadModelFromZMF[float32](eng, ops, zmfPath)
	if err != nil {
		t.Fatalf("LoadModelFromZMF failed: %v", err)
	}
	if m.Graph == nil {
		t.Fatal("model graph is nil")
	}

	// Synthetic normalized image: [1, 3, 224, 224] float32 with values in [-1, 1].
	const (
		batch    = 1
		channels = 3
		height   = 224
		width    = 224
	)
	n := batch * channels * height * width
	imgData := make([]float32, n)
	for i := range imgData {
		// Simple synthetic normalization: (i mod 255) / 127.5 - 1.0
		imgData[i] = float32(i%255)/127.5 - 1.0
	}
	img, err := tensor.New[float32]([]int{batch, channels, height, width}, imgData)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}

	output, err := m.Graph.Forward(context.Background(), img)
	if err != nil {
		t.Fatalf("Graph.Forward failed: %v", err)
	}
	if output == nil {
		t.Fatal("output tensor is nil")
	}

	outShape := output.Shape()
	t.Logf("SigLIP output shape: %v", outShape)

	// For patch_size=16 and 224x224 input: 14*14 = 196 patches.
	const expectedPatches = 196
	if len(outShape) == 3 {
		if outShape[0] != batch {
			t.Errorf("batch dim = %d, want %d", outShape[0], batch)
		}
		if outShape[1] != expectedPatches {
			t.Errorf("patch dim = %d, want %d", outShape[1], expectedPatches)
		}
		t.Logf("embed_dim = %d", outShape[2])
	} else {
		t.Logf("unexpected output rank %d; shape = %v", len(outShape), outShape)
	}

	// Verify no NaN or Inf.
	data := output.Data()
	for i, v := range data {
		f := float64(v)
		if math.IsNaN(f) {
			t.Errorf("output[%d] is NaN", i)
			break
		}
		if math.IsInf(f, 0) {
			t.Errorf("output[%d] is Inf", i)
			break
		}
	}
}

// TestKimiVLConnectorForwardPass loads the Kimi-VL vision-language connector
// and runs a forward pass with synthetic vision embeddings shaped
// [1, 196, embedDim].
//
// Skipped when KIMI_CONNECTOR_ZMF_PATH is not set.
func TestKimiVLConnectorForwardPass(t *testing.T) {
	zmfPath := os.Getenv("KIMI_CONNECTOR_ZMF_PATH")
	if zmfPath == "" {
		t.Skip("KIMI_CONNECTOR_ZMF_PATH not set; skipping Kimi-VL connector test")
	}

	registry.RegisterAll()

	eng := compute.NewCPUEngine[float32](numeric.Float32Ops{})
	ops := numeric.Float32Ops{}

	m, err := model.LoadModelFromZMF[float32](eng, ops, zmfPath)
	if err != nil {
		t.Fatalf("LoadModelFromZMF failed: %v", err)
	}
	if m.Graph == nil {
		t.Fatal("model graph is nil")
	}

	// Synthetic vision embeddings: [1, 196, embedDim].
	// embedDim is not known at test-write time; use 1152 (SigLIP-SO400M default).
	const (
		batch      = 1
		numPatches = 196
		embedDim   = 1152
	)
	n := batch * numPatches * embedDim
	embData := make([]float32, n)
	for i := range embData {
		embData[i] = float32(i%100) / 50.0 // values in [0, 2)
	}
	emb, err := tensor.New[float32]([]int{batch, numPatches, embedDim}, embData)
	if err != nil {
		t.Fatalf("tensor.New failed: %v", err)
	}

	output, err := m.Graph.Forward(context.Background(), emb)
	if err != nil {
		t.Fatalf("Graph.Forward failed: %v", err)
	}
	if output == nil {
		t.Fatal("output tensor is nil")
	}

	outShape := output.Shape()
	t.Logf("Kimi-VL connector output shape: %v", outShape)

	// Expect [1, 196, lmDim].
	if len(outShape) == 3 {
		if outShape[0] != batch {
			t.Errorf("batch dim = %d, want %d", outShape[0], batch)
		}
		if outShape[1] != numPatches {
			t.Errorf("patch dim = %d, want %d", outShape[1], numPatches)
		}
		t.Logf("lm_dim = %d", outShape[2])
	} else {
		t.Logf("unexpected output rank %d; shape = %v", len(outShape), outShape)
	}

	// Verify no NaN or Inf.
	data := output.Data()
	for i, v := range data {
		f := float64(v)
		if math.IsNaN(f) {
			t.Errorf("output[%d] is NaN", i)
			break
		}
		if math.IsInf(f, 0) {
			t.Errorf("output[%d] is Inf", i)
			break
		}
	}
}
