package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var gemma3nConfig = modelParityConfig{
	Name:           "Gemma 3n",
	ZMFEnvVar:      "GEMMA3N_ZMF_PATH",
	ModelDirEnvVar: "GEMMA3N_MODEL_DIR",
	ModelID:        "gemma3n",
	MinVocabSize:   256000, // Gemma 3n vocab: 262144 (same tokenizer as Gemma 3)
}

func TestGemma3nForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, gemma3nConfig)
}

func TestGemma3nGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, gemma3nConfig)
}

func TestGemma3nGeneration(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, gemma3nConfig)
}
