package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var commandRConfig = testutil.ModelParityConfig{
	Name:           "Command R",
	ZMFEnvVar:      "COMMANDR_ZMF_PATH",
	ModelDirEnvVar: "COMMANDR_MODEL_DIR",
	ModelID:        "command-r",
	MinVocabSize:   200000, // Command R vocab: 256000
}

func TestCommandRForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, commandRConfig)
}

func TestCommandRGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, commandRConfig)
}

func TestCommandRGeneration(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, commandRConfig)
}
