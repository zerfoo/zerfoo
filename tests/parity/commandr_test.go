package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var commandRConfig = modelParityConfig{
	Name:           "Command R",
	ZMFEnvVar:      "COMMANDR_ZMF_PATH",
	ModelDirEnvVar: "COMMANDR_MODEL_DIR",
	ModelID:        "command-r",
	MinVocabSize:   200000, // Command R vocab: 256000
}

func TestCommandRForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, commandRConfig)
}

func TestCommandRGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, commandRConfig)
}

func TestCommandRGeneration(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, commandRConfig)
}
