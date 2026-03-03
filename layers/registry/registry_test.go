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
		"Sigmoid",
		"Softmax",
		"Erf",
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
		"Slice",
		"Pad",
		"TopK",
		"Conv",
		"GlobalAveragePool",
		"Resize",
		"MoEGate",
		"MixtureOfExperts",
		// Gather
		"Gather",
		// Normalization
		"RMSNorm",
		"LayerNormalization",
		"SimplifiedLayerNormalization",
		"SkipSimplifiedLayerNormalization",
		"BatchNormalization",
		// ReduceSum
		"ReduceSum",
		// Transpose
		"Transpose",
		// Regularization
		"Dropout",
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
