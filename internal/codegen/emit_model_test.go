package codegen

import (
	"strings"
	"testing"

	"github.com/zerfoo/ztensor/graph"
)

func TestEmitMegakernelLargeModel(t *testing.T) {
	// Simulate a simplified Gemma 3 instruction tape (~20 instructions).
	// Real model has ~650 but this tests the emitter handles multi-layer models.
	instructions := []graph.InstructionMeta{
		// Gather embedding
		{OpName: "Gather", InputIdx: []int{0, 1}, OutputIdx: 2},
		// Layer 1: RMSNorm -> MatMul (Q) -> MatMul (K) -> MatMul (V)
		{OpName: "RMSNorm", InputIdx: []int{2, 3}, OutputIdx: 4},
		{OpName: "MatMulNBits", InputIdx: []int{5, 4}, OutputIdx: 6},  // Q
		{OpName: "MatMulNBits", InputIdx: []int{7, 4}, OutputIdx: 8},  // K
		{OpName: "MatMulNBits", InputIdx: []int{9, 4}, OutputIdx: 10}, // V
		// Attention: Mul(Q, scale) -> Softmax -> MatMul(attn, V)
		{OpName: "MulScalar", InputIdx: []int{6}, OutputIdx: 11},
		{OpName: "Softmax", InputIdx: []int{11}, OutputIdx: 12},
		{OpName: "MatMul", InputIdx: []int{12, 10}, OutputIdx: 13},
		// Post-attention: MatMul(O) -> Add(residual)
		{OpName: "MatMulNBits", InputIdx: []int{14, 13}, OutputIdx: 15},
		{OpName: "Add", InputIdx: []int{2, 15}, OutputIdx: 16},
		// FFN: RMSNorm -> MatMul(gate) -> Silu -> Mul -> MatMul(down) -> Add
		{OpName: "RMSNorm", InputIdx: []int{16, 17}, OutputIdx: 18},
		{OpName: "MatMulNBits", InputIdx: []int{19, 18}, OutputIdx: 20}, // gate
		{OpName: "MatMulNBits", InputIdx: []int{21, 18}, OutputIdx: 22}, // up
		{OpName: "Silu", InputIdx: []int{20}, OutputIdx: 23},
		{OpName: "Mul", InputIdx: []int{23, 22}, OutputIdx: 24},
		{OpName: "MatMulNBits", InputIdx: []int{25, 24}, OutputIdx: 26}, // down
		{OpName: "Add", InputIdx: []int{16, 26}, OutputIdx: 27},
		// Final: RMSNorm -> MatMul (logits)
		{OpName: "RMSNorm", InputIdx: []int{27, 28}, OutputIdx: 29},
		{OpName: "MatMul", InputIdx: []int{30, 29}, OutputIdx: 31},
	}

	// Create slot shapes.
	numSlots := 32
	slotShapes := make([][]int, numSlots)
	for i := range slotShapes {
		slotShapes[i] = []int{1, 2048}
	}

	// Frozen slots: embedding table, weight matrices, norm weights.
	frozenIdxs := []int{0, 3, 5, 7, 9, 14, 17, 19, 21, 25, 28, 30}
	frozen := make([]FrozenSlotMeta, len(frozenIdxs))
	for i, idx := range frozenIdxs {
		frozen[i] = FrozenSlotMeta{SlotIdx: idx}
	}

	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		FrozenSlots:  frozen,
		InputSlots:   []int{1}, // token index
		OutputSlot:   31,       // logits
	}

	code, err := EmitMegakernel(cfg)
	if err != nil {
		t.Fatalf("EmitMegakernel: %v", err)
	}

	// Verify all 19 instructions are present.
	if count := strings.Count(code, "// ["); count != len(instructions) {
		t.Errorf("instruction comments: got %d, want %d", count, len(instructions))
	}

	// Verify frozen slot defines.
	if !strings.Contains(code, "#define frozen_") {
		t.Error("missing frozen #define mappings")
	}

	// Verify __global__ declaration.
	if !strings.Contains(code, "__global__ void megakernel") {
		t.Error("missing __global__ void megakernel")
	}

	// Verify workspace-based slot declarations.
	if !strings.Contains(code, "workspace +") {
		t.Error("missing workspace-based slot declarations")
	}

	// Verify launch wrapper.
	if !strings.Contains(code, "launch_megakernel") {
		t.Error("missing launch_megakernel wrapper")
	}

	// Verify key ops.
	if !strings.Contains(code, "dev_rmsnorm") {
		t.Error("missing dev_rmsnorm call")
	}
	if !strings.Contains(code, "dev_gemv_q4") {
		t.Error("missing dev_gemv_q4 call")
	}
	if !strings.Contains(code, "dev_softmax") {
		t.Error("missing dev_softmax call")
	}

	t.Logf("Generated %d bytes of CUDA code for %d instructions", len(code), len(instructions))
}
