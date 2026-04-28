package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var gemma3Config = testutil.ModelParityConfig{
	Name:           "Gemma 3",
	ZMFEnvVar:      "GEMMA3_ZMF_PATH",
	ModelDirEnvVar: "GEMMA3_MODEL_DIR",
	ModelID:        "gemma-3",
	MinVocabSize:   256000, // Gemma 3 vocab: 262144
}

func TestGemma3ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, gemma3Config)
}

func TestGemma3GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, gemma3Config)
}

func TestGemma3Generation(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, gemma3Config)
}
