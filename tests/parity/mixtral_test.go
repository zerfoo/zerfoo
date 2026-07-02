package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var mixtralConfig = testutil.ModelParityConfig{
	Name:           "Mixtral",
	ZMFEnvVar:      "MIXTRAL_ZMF_PATH",
	ModelDirEnvVar: "MIXTRAL_MODEL_DIR",
	ModelID:        "mixtral",
	MinVocabSize:   30000, // Mixtral vocab: 32000 (same tokenizer as Mistral)
}

func TestMixtralForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, mixtralConfig)
}

func TestMixtralGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, mixtralConfig)
}

func TestMixtralGeneration(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, mixtralConfig)
}
