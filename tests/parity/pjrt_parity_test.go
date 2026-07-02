//go:build pjrt_test

// PJRT CPU parity tests (T126.1.1, E126).
//
// These tests are gated behind the `pjrt_test` build tag because they
// require external artifacts that are not present in default CI:
//
//  1. A PJRT CPU plugin shared library (e.g. pjrt_c_api_cpu_plugin.so),
//     pointed to by the PJRT_CPU_PLUGIN env var.
//  2. A locally-resident Gemma 3 1B GGUF model directory, pointed to by
//     GEMMA3_MODEL_DIR (same convention used by gemma3_test.go).
//
// Run with:
//
//	go test -tags pjrt_test -run TestPJRTCPUParity -count=1 ./tests/parity/...
//
// Acceptance per docs/plan.md E126/T126.1.1: first-token logits emitted via
// the PJRT CPU path match those produced by the native Engine CPU path
// within absolute tolerance 1e-4.
//
// STATUS: scaffolded but blocked. The framework does not yet expose a
// deterministic pre-sampling logit accessor on inference.Model; the public
// Generate / GenerateStream APIs all sample tokens internally. Without a
// logits hook we cannot perform a numerical parity assertion. See
// tests/parity/README.md for the follow-up plan.
package parity_test

import (
	"os"
	"testing"

	layerreg "github.com/zerfoo/zerfoo/layers/registry"
)

// pjrtTolerance is the absolute tolerance for per-element logit comparison
// between the native CPU engine and the PJRT CPU plugin. Codified here so
// the value is reachable when the test body is wired up.
const pjrtTolerance = 1e-4

// pjrtParityCase describes one model fixture under parity comparison.
type pjrtParityCase struct {
	name           string
	modelID        string
	modelDirEnvVar string
	prompt         string
}

// pjrtParityCases lists the fixtures the parity matrix should cover. Kept
// table-driven so additional models slot in without touching control flow.
var pjrtParityCases = []pjrtParityCase{
	{
		name:           "gemma3_1b",
		modelID:        "gemma-3",
		modelDirEnvVar: "GEMMA3_MODEL_DIR",
		prompt:         "The capital of France is",
	},
}

// TestPJRTCPUParity is the entry point for T126.1.1. The body intentionally
// fails fast with a Skip until both the PJRT plugin and a public
// inference.Model logits accessor are available; this preserves the build
// under -tags pjrt_test without claiming false coverage.
func TestPJRTCPUParity(t *testing.T) {
	layerreg.RegisterAll()

	pluginPath := os.Getenv("PJRT_CPU_PLUGIN")
	if pluginPath == "" {
		t.Skip("PJRT_CPU_PLUGIN not set; skipping PJRT CPU parity tests")
	}
	if _, err := os.Stat(pluginPath); err != nil {
		t.Skipf("PJRT_CPU_PLUGIN %q not accessible: %v", pluginPath, err)
	}

	t.Skip("blocked: inference.Model lacks a deterministic first-token logits accessor; " +
		"see tests/parity/README.md for the follow-up needed before parity assertions can run")

	for _, tc := range pjrtParityCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			modelDir := os.Getenv(tc.modelDirEnvVar)
			if modelDir == "" {
				t.Skipf("%s not set; skipping", tc.modelDirEnvVar)
			}
			// When a logits hook lands, replace this body with:
			//   nativeLogits := <load without WithPJRT> -> first-token logits
			//   pjrtLogits   := <load with    WithPJRT(plugin)> -> first-token logits
			//   compare element-wise with pjrtTolerance.
			_ = pjrtTolerance
		})
	}
}
