package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var mistralConfig = testutil.ModelParityConfig{
	Name:           "Mistral",
	ZMFEnvVar:      "MISTRAL_ZMF_PATH",
	ModelDirEnvVar: "MISTRAL_MODEL_DIR",
	ModelID:        "mistral",
	MinVocabSize:   30000, // Mistral 7B vocab: 32000
}

func TestMistralForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, mistralConfig)
}

func TestMistralGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, mistralConfig)
}

func TestMistralGeneration(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, mistralConfig)
}
