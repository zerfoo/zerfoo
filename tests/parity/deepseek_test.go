package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var deepseekV3Config = testutil.ModelParityConfig{
	Name:           "DeepSeek-V3",
	ZMFEnvVar:      "DEEPSEEK_ZMF_PATH",
	ModelDirEnvVar: "DEEPSEEK_MODEL_DIR",
	ModelID:        "deepseek-v3",
	MinVocabSize:   100000, // DeepSeek V3 vocab: 129280
}

func TestDeepSeekV3ForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, deepseekV3Config)
}

func TestDeepSeekV3GreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, deepseekV3Config)
}

func TestDeepSeekV3Generation(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, deepseekV3Config)
}
