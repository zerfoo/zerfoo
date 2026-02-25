package registry

import (
	"testing"

	"github.com/zerfoo/zerfoo/model"
)

func TestRegisterAll(t *testing.T) {
	RegisterAll()

	expectedOps := []string{
		// Activations
		"FastGelu",
		"Gelu",
		"Tanh",
		// Attention
		"GroupQueryAttention",
		"GlobalAttention",
		// Core
		"Add",
		"Shape",
		"Mul",
		"Sub",
		"Unsqueeze",
		"Cast",
		"Concat",
		"MatMul",
		"Reshape",
		"RotaryEmbedding",
		"SpectralFingerprint",
		"FiLM",
		// Gather
		"Gather",
		// Normalization
		"RMSNorm",
		"SimplifiedLayerNormalization",
		"SkipSimplifiedLayerNormalization",
		// ReduceSum
		"ReduceSum",
		// Transpose
		"Transpose",
	}

	for _, opType := range expectedOps {
		_, err := model.GetLayerBuilder[float32](opType)
		if err != nil {
			t.Errorf("RegisterAll() did not register %q: %v", opType, err)
		}
	}
}

func TestRegisterAll_Idempotent(t *testing.T) {
	// Calling RegisterAll twice should not panic.
	RegisterAll()
	RegisterAll()

	// Verify a builder is still retrievable after double registration.
	_, err := model.GetLayerBuilder[float32]("Gather")
	if err != nil {
		t.Errorf("after double RegisterAll(), Gather not found: %v", err)
	}
}
