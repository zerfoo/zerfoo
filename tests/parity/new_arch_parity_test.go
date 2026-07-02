package parity_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/tests/parity/testutil"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

// TestNewArchParity runs parity tests for all 6 new architectures added in Wave 2:
// Llama 4, Gemma 3n, Command R, Falcon, Mixtral, and RWKV.
//
// Each architecture is tested with forward pass, greedy decode, and generation
// subtests. Tests skip when the corresponding model directory env var is not set.
//
// Environment variables required per architecture:
//
//	LLAMA4_MODEL_DIR   or LLAMA4_ZMF_PATH
//	GEMMA3N_MODEL_DIR  or GEMMA3N_ZMF_PATH
//	COMMANDR_MODEL_DIR or COMMANDR_ZMF_PATH
//	FALCON_MODEL_DIR   or FALCON_ZMF_PATH
//	MIXTRAL_MODEL_DIR  or MIXTRAL_ZMF_PATH
//	RWKV_MODEL_DIR     or RWKV_ZMF_PATH
func TestNewArchParity(t *testing.T) {
	layerreg.RegisterAll()

	archs := []testutil.ModelParityConfig{
		llama4Config,
		gemma3nConfig,
		commandRConfig,
		falconConfig,
		mixtralConfig,
		rwkvConfig,
	}

	for _, arch := range archs {
		t.Run(arch.Name+"/forward_pass", func(t *testing.T) {
			testutil.RunModelForwardPass(t, arch)
		})
		t.Run(arch.Name+"/greedy_decode", func(t *testing.T) {
			testutil.RunModelGreedyDecode(t, arch)
		})
		t.Run(arch.Name+"/generation", func(t *testing.T) {
			testutil.RunModelGeneration(t, arch)
		})
	}
}
