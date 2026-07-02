package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var phi4Config = testutil.ModelParityConfig{
	Name:           "Phi-3",
	ZMFEnvVar:      "PHI4_ZMF_PATH",
	ModelDirEnvVar: "PHI4_MODEL_DIR",
	ModelID:        "phi-3",
	MinVocabSize:   30000, // Phi-3-mini vocab: 32064
}

func TestPhi4ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, phi4Config)
}

func TestPhi4GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, phi4Config)
}

func TestPhi4Generation(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, phi4Config)
}
