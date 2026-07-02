package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var rwkvConfig = testutil.ModelParityConfig{
	Name:           "RWKV",
	ZMFEnvVar:      "RWKV_ZMF_PATH",
	ModelDirEnvVar: "RWKV_MODEL_DIR",
	ModelID:        "rwkv",
	MinVocabSize:   60000, // RWKV vocab: 65536
}

func TestRWKVForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, rwkvConfig)
}

func TestRWKVGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, rwkvConfig)
}

func TestRWKVGeneration(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, rwkvConfig)
}
