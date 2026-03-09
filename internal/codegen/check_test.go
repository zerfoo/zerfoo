package codegen

import (
	"testing"

	"github.com/zerfoo/zerfoo/graph"
)

func TestCheckSupportAllSupported(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "Add"},
		{OpName: "Mul"},
		{OpName: "RMSNorm"},
	}
	unsupported := CheckSupport(instructions)
	if len(unsupported) != 0 {
		t.Errorf("expected no unsupported ops, got %v", unsupported)
	}
}

func TestCheckSupportWithUnsupported(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "Add"},
		{OpName: "FancyOp"},
		{OpName: "FancyOp"}, // duplicate
		{OpName: "AnotherOp"},
	}
	unsupported := CheckSupport(instructions)
	if len(unsupported) != 2 {
		t.Errorf("expected 2 unsupported ops, got %v", unsupported)
	}
}
