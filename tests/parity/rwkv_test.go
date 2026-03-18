package parity_test

import (
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var rwkvConfig = modelParityConfig{
	Name:           "RWKV",
	ZMFEnvVar:      "RWKV_ZMF_PATH",
	ModelDirEnvVar: "RWKV_MODEL_DIR",
	ModelID:        "rwkv",
	MinVocabSize:   60000, // RWKV vocab: 65536
}

func TestRWKVForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	runModelForwardPass(t, rwkvConfig)
}

func TestRWKVGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	runModelGreedyDecode(t, rwkvConfig)
}

func TestRWKVGeneration(t *testing.T) {
	layerreg.RegisterAll()
	runModelGeneration(t, rwkvConfig)
}
