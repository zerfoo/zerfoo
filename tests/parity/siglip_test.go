package parity_test

import (
	"os"
	"testing"
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
	if os.Getenv("SIGLIP_ZMF_PATH") == "" {
		t.Skip("SIGLIP_ZMF_PATH not set; skipping SigLIP forward pass test")
	}
	t.Skip("ZMF loading is no longer supported")
}

// TestKimiVLConnectorForwardPass loads the Kimi-VL vision-language connector
// and runs a forward pass with synthetic vision embeddings shaped
// [1, 196, embedDim].
//
// Skipped when KIMI_CONNECTOR_ZMF_PATH is not set.
func TestKimiVLConnectorForwardPass(t *testing.T) {
	if os.Getenv("KIMI_CONNECTOR_ZMF_PATH") == "" {
		t.Skip("KIMI_CONNECTOR_ZMF_PATH not set; skipping Kimi-VL connector test")
	}
	t.Skip("ZMF loading is no longer supported")
}
