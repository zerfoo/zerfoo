package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var llama4Config = modelParityConfig{
	Name:           "Llama 4",
	ZMFEnvVar:      "LLAMA4_ZMF_PATH",
	ModelDirEnvVar: "LLAMA4_MODEL_DIR",
	ModelID:        "llama4",
	MinVocabSize:   100000, // Llama 4 vocab: 128256
}

func TestLlama4ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, llama4Config)
}

func TestLlama4GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, llama4Config)
}

func TestLlama4Generation(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, llama4Config)
}
