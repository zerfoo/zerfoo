package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

var gemma3nConfig = testutil.ModelParityConfig{
	Name:           "Gemma 3n",
	ZMFEnvVar:      "GEMMA3N_ZMF_PATH",
	ModelDirEnvVar: "GEMMA3N_MODEL_DIR",
	ModelID:        "gemma3n",
	MinVocabSize:   256000, // Gemma 3n vocab: 262144 (same tokenizer as Gemma 3)
}

func TestGemma3nForwardPass(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelForwardPass(t, gemma3nConfig)
}

func TestGemma3nGreedyDecode(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGreedyDecode(t, gemma3nConfig)
}

func TestGemma3nGeneration(t *testing.T) {
	layerreg.RegisterAll()
	testutil.RunModelGeneration(t, gemma3nConfig)
}
