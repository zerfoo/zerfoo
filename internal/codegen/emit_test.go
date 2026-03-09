package codegen

import (
	"strings"
	"testing"

	"github.com/zerfoo/zerfoo/graph"
)

func TestEmitMegakernel(t *testing.T) {
	// Simple graph: slot0 (input) + slot1 (frozen) -> slot2 (output)
	instructions := []graph.InstructionMeta{
		{OpName: "Add", InputIdx: []int{0, 1}, OutputIdx: 2},
	}
	slotShapes := [][]int{
		{1, 4}, // input
		{1, 4}, // frozen weight
		{1, 4}, // output
	}
	frozen := []FrozenSlotMeta{
		{SlotIdx: 1},
	}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		FrozenSlots:  frozen,
		InputSlots:   []int{0},
		OutputSlot:   2,
	}

	code, err := EmitMegakernel(cfg)
	if err != nil {
		t.Fatalf("EmitMegakernel: %v", err)
	}

	checks := []string{
		"__global__",
		"megakernel",
		"slot_0",
		"frozen_1",
		"slot_2",
		"+",
		"launch_megakernel",
		"extern \"C\"",
		"cudaDeviceSynchronize",
		"workspace",
	}
	for _, want := range checks {
		if !strings.Contains(code, want) {
			t.Errorf("generated code missing %q", want)
		}
	}
}

func TestEmitMegakernelUnsupportedOp(t *testing.T) {
	instructions := []graph.InstructionMeta{
		{OpName: "FancyNewOp", InputIdx: []int{0}, OutputIdx: 1},
	}
	slotShapes := [][]int{{1, 4}, {1, 4}}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		InputSlots:   []int{0},
		OutputSlot:   1,
	}

	_, err := EmitMegakernel(cfg)
	if err == nil {
		t.Fatal("expected error for unsupported op")
	}
	if !strings.Contains(err.Error(), "unsupported") {
		t.Errorf("error should mention 'unsupported': %v", err)
	}
}

func TestEmitMegakernelMultiOp(t *testing.T) {
	// input -> Mul(input, frozen) -> Exp -> output
	instructions := []graph.InstructionMeta{
		{OpName: "Mul", InputIdx: []int{0, 1}, OutputIdx: 2},
		{OpName: "Exp", InputIdx: []int{2}, OutputIdx: 3},
	}
	slotShapes := [][]int{
		{1, 2048}, // input
		{1, 2048}, // frozen weight
		{1, 2048}, // intermediate
		{1, 2048}, // output
	}
	frozen := []FrozenSlotMeta{{SlotIdx: 1}}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		FrozenSlots:  frozen,
		InputSlots:   []int{0},
		OutputSlot:   3,
	}

	code, err := EmitMegakernel(cfg)
	if err != nil {
		t.Fatalf("EmitMegakernel: %v", err)
	}

	// Both ops should appear in order.
	mulIdx := strings.Index(code, "slot_2[tid] = slot_0[tid] * slot_1[tid]")
	expIdx := strings.Index(code, "expf(slot_2[tid])")
	if mulIdx < 0 {
		t.Error("missing Mul instruction")
	}
	if expIdx < 0 {
		t.Error("missing Exp instruction")
	}
	if mulIdx >= 0 && expIdx >= 0 && mulIdx >= expIdx {
		t.Error("Mul should appear before Exp")
	}

	// Verify launch wrapper present.
	if !strings.Contains(code, "launch_megakernel") {
		t.Error("missing launch_megakernel wrapper")
	}
}

func TestEmitMegakernelKVCache(t *testing.T) {
	// Graph: input -> KVCacheAppendK(layer=0) -> KVCacheGetK(layer=0) -> output
	instructions := []graph.InstructionMeta{
		{OpName: "KVCacheAppendK", InputIdx: []int{0, 0}, OutputIdx: 1},
		{OpName: "KVCacheGetK", InputIdx: []int{0}, OutputIdx: 2},
		{OpName: "KVCacheSeqLen", InputIdx: nil, OutputIdx: 3},
	}
	slotShapes := [][]int{
		{8, 128}, // input K data
		{8, 128}, // append output
		{8, 128}, // get output
		{1},      // seq_len
	}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   slotShapes,
		InputSlots:   []int{0},
		OutputSlot:   2,
		NumKVLayers:  26,
	}

	code, err := EmitMegakernel(cfg)
	if err != nil {
		t.Fatalf("EmitMegakernel: %v", err)
	}

	// KV cache kernel args should be present.
	checks := []string{
		"float** __restrict__ kv_k",
		"float** __restrict__ kv_v",
		"int seq_pos",
		"int kv_seq_len",
		"dev_kv_append",
		"kv_k[0]",
		"kv_seq_len",
		"seq_len_3",
	}
	for _, want := range checks {
		if !strings.Contains(code, want) {
			t.Errorf("generated code missing %q", want)
		}
	}

	// Launch wrapper should also pass KV args.
	if !strings.Contains(code, "float** kv_k") {
		t.Error("launch wrapper missing kv_k parameter")
	}
	if !strings.Contains(code, "float** kv_v") {
		t.Error("launch wrapper missing kv_v parameter")
	}
	if !strings.Contains(code, "kv_k, kv_v, seq_pos, kv_seq_len") {
		t.Error("launch wrapper kernel call missing KV args")
	}
}

func TestEmitMegakernelNoKVCache(t *testing.T) {
	// When NumKVLayers is 0, no KV cache args should appear.
	instructions := []graph.InstructionMeta{
		{OpName: "Add", InputIdx: []int{0, 1}, OutputIdx: 2},
	}
	cfg := MegakernelConfig{
		Instructions: instructions,
		SlotShapes:   [][]int{{1, 4}, {1, 4}, {1, 4}},
		FrozenSlots:  []FrozenSlotMeta{{SlotIdx: 1}},
		InputSlots:   []int{0},
		OutputSlot:   2,
		NumKVLayers:  0,
	}

	code, err := EmitMegakernel(cfg)
	if err != nil {
		t.Fatalf("EmitMegakernel: %v", err)
	}

	if strings.Contains(code, "kv_k") {
		t.Error("NumKVLayers=0 should not emit kv_k")
	}
	if strings.Contains(code, "kv_v") {
		t.Error("NumKVLayers=0 should not emit kv_v")
	}
	if strings.Contains(code, "seq_pos") {
		t.Error("NumKVLayers=0 should not emit seq_pos")
	}
	if strings.Contains(code, "kv_seq_len") {
		t.Error("NumKVLayers=0 should not emit kv_seq_len")
	}
}

func TestComputeWorkspaceLayout(t *testing.T) {
	cfg := MegakernelConfig{
		Instructions: []graph.InstructionMeta{
			{OpName: "Add", InputIdx: []int{0, 1}, OutputIdx: 2},
		},
		SlotShapes:  [][]int{{1, 4}, {1, 4}, {1, 4}},
		FrozenSlots: []FrozenSlotMeta{{SlotIdx: 1}},
		InputSlots:  []int{0},
		OutputSlot:  2,
	}

	layout := ComputeWorkspaceLayout(cfg)

	// Slot 1 is frozen, should NOT be in workspace.
	if _, ok := layout.SlotOffsets[1]; ok {
		t.Error("frozen slot 1 should not be in workspace")
	}

	// Slots 0 and 2 should be in workspace.
	if _, ok := layout.SlotOffsets[0]; !ok {
		t.Error("slot 0 should be in workspace")
	}
	if _, ok := layout.SlotOffsets[2]; !ok {
		t.Error("slot 2 should be in workspace")
	}

	// Total size = 4 (slot0) + 4 (slot2) = 8.
	if layout.TotalSize != 8 {
		t.Errorf("total size: got %d, want 8", layout.TotalSize)
	}

	if layout.OutputSize != 4 {
		t.Errorf("output size: got %d, want 4", layout.OutputSize)
	}
}
