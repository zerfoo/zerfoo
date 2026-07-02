package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var falconConfig = testutil.ModelParityConfig{
	Name:           "Falcon",
	ZMFEnvVar:      "FALCON_ZMF_PATH",
	ModelDirEnvVar: "FALCON_MODEL_DIR",
	ModelID:        "falcon",
	MinVocabSize:   60000, // Falcon vocab: 65024
}

func TestFalconForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, falconConfig)
}

func TestFalconGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, falconConfig)
}

func TestFalconGeneration(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, falconConfig)
}
