package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var llama4Config = testutil.ModelParityConfig{
	Name:           "Llama 4",
	ZMFEnvVar:      "LLAMA4_ZMF_PATH",
	ModelDirEnvVar: "LLAMA4_MODEL_DIR",
	ModelID:        "llama4",
	MinVocabSize:   100000, // Llama 4 vocab: 128256
}

func TestLlama4ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, llama4Config)
}

func TestLlama4GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, llama4Config)
}

func TestLlama4Generation(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, llama4Config)
}
