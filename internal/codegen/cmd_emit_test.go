package codegen

import (
	"fmt"
	"testing"

	"github.com/zerfoo/ztensor/graph"
)

func TestEmitMegakernelAllOpsSupported(t *testing.T) {
	// Verify every op that could appear in a Gemma 3 instruction tape
	// has a registered emitter.
	allOps := []string{
		"Add", "Sub", "Mul", "Div", "Pow",
		"Exp", "Log", "Sqrt", "Rsqrt", "Tanh",
		"Neg", "Abs", "Silu",
		"AddScalar", "MulScalar", "SubScalar", "DivScalar", "PowScalar",
		"RMSNorm", "Softmax",
		"MatMul", "MatMulNBits",
		"Gather", "Concat", "Reshape", "Transpose",
	}

	var unsupported []string
	for _, op := range allOps {
		if !Supported(op) {
			unsupported = append(unsupported, op)
		}
	}
	if len(unsupported) > 0 {
		t.Errorf("unsupported ops: %v", unsupported)
	}
}

func TestEmitCheckRealOps(t *testing.T) {
	// This test verifies that if we encounter ops from a real model,
	// the emitter reports clearly which ones are unsupported.
	realOps := []string{
		"Gather", "RMSNorm", "MatMulNBits", "MulScalar",
		"Softmax", "MatMul", "Add", "Silu", "Mul",
		"Reshape", "Transpose",
	}

	for _, op := range realOps {
		meta := graph.InstructionMeta{
			OpName:    op,
			InputIdx:  []int{0, 1},
			OutputIdx: 2,
		}
		slots := []SlotInfo{{Shape: []int{1, 2048}}, {Shape: []int{1, 2048}}}
		_, err := Emit(meta, slots)
		if err != nil {
			t.Errorf("real model op %q: %v", op, err)
		}
	}
	t.Logf("all %d real model ops have emitters", len(realOps))
	fmt.Printf("Supported ops: %d total\n", len(realOps))
}
