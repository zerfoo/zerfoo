// Package hrm_test contains tests for the HRM layers.
package hrm_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/hrm"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestNewHModule(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 32
	numHeads := 2

	attn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create attention: %v", err)
	}

	hModule, err := hrm.NewHModule[float32](engine, ops, modelDim, ffnDim, attn)
	if err != nil {
		t.Fatalf("failed to create HModule: %v", err)
	}

	if hModule == nil {
		t.Fatal("HModule is nil")
	}

	if hModule.Block == nil {
		t.Error("HModule.Block is nil")
	}

	if hModule.HiddenState == nil {
		t.Error("HModule.HiddenState is nil")
	}

	if len(hModule.Parameters()) == 0 {
		t.Error("HModule has no parameters")
	}
}
