package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var mixtralConfig = modelParityConfig{
	Name:           "Mixtral",
	ZMFEnvVar:      "MIXTRAL_ZMF_PATH",
	ModelDirEnvVar: "MIXTRAL_MODEL_DIR",
	ModelID:        "mixtral",
	MinVocabSize:   30000, // Mixtral vocab: 32000 (same tokenizer as Mistral)
}

func TestMixtralForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, mixtralConfig)
}

func TestMixtralGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, mixtralConfig)
}

func TestMixtralGeneration(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, mixtralConfig)
}
