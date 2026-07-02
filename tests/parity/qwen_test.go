package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var qwenConfig = testutil.ModelParityConfig{
	Name:           "Qwen 2.5",
	ZMFEnvVar:      "QWEN25_ZMF_PATH",
	ModelDirEnvVar: "QWEN25_MODEL_DIR",
	ModelID:        "qwen-2.5",
	MinVocabSize:   150000, // Qwen 2.5 vocab: 151936
}

func TestQwen25ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, qwenConfig)
}

func TestQwen25GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, qwenConfig)
}

func TestQwen25Generation(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, qwenConfig)
}
