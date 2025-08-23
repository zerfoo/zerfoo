// Package hrm_test contains tests for the HRM layers.
package hrm_test

import (
	"testing"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/hrm"
	"github.com/zerfoo/zerfoo/numeric"
)

func TestNewLModule(t *testing.T) {
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	modelDim := 16
	ffnDim := 32
	numHeads := 2

	attn, err := attention.NewGlobalAttention[float32](engine, ops, modelDim, numHeads, numHeads)
	if err != nil {
		t.Fatalf("failed to create attention: %v", err)
	}

	lModule, err := hrm.NewLModule[float32](engine, ops, modelDim, ffnDim, attn)
	if err != nil {
		t.Fatalf("failed to create LModule: %v", err)
	}

	if lModule == nil {
		t.Fatal("LModule is nil")
	}

	if lModule.Block == nil {
		t.Error("LModule.Block is nil")
	}

	if lModule.HiddenState == nil {
		t.Error("LModule.HiddenState is nil")
	}

	if len(lModule.Parameters()) == 0 {
		t.Error("LModule has no parameters")
	}
}
