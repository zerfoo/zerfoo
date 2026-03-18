package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var falconConfig = modelParityConfig{
	Name:           "Falcon",
	ZMFEnvVar:      "FALCON_ZMF_PATH",
	ModelDirEnvVar: "FALCON_MODEL_DIR",
	ModelID:        "falcon",
	MinVocabSize:   60000, // Falcon vocab: 65024
}

func TestFalconForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, falconConfig)
}

func TestFalconGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, falconConfig)
}

func TestFalconGeneration(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, falconConfig)
}
