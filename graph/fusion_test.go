package graph

import (
	"context"
	"math"
	"testing"

	"github.com/zerfoo/zerfoo/tensor"
)

// buildRMSNormPattern creates a synthetic instruction list representing the
// ONNX decomposed RMSNorm pattern:
//
//	Pow(x,2) → ReduceMean(axis=-1,keepDims=true) → Add(eps) → Sqrt → Div(x,sqrt) → Mul(norm,weight)
//
// Returns instructions, frozenIdx, slots, slotShapes.
func buildRMSNormPattern(xSlot, exponentSlot, epsilonSlot, weightSlot, startSlot int, epsilon float32) ([]Instruction[float32], []int, []*tensor.TensorNumeric[float32], [][]int) {
	// Slot assignments for intermediates:
	powOut := startSlot
	meanOut := startSlot + 1
	addOut := startSlot + 2
	sqrtOut := startSlot + 3
	divOut := startSlot + 4
	mulOut := startSlot + 5

	numSlots := mulOut + 1

	slots := make([]*tensor.TensorNumeric[float32], numSlots)
	slotShapes := make([][]int, numSlots)

	// Create frozen tensors.
	expT, _ := tensor.New[float32]([]int{1}, []float32{2.0})
	epsT, _ := tensor.New[float32]([]int{1}, []float32{epsilon})
	weightT, _ := tensor.New[float32]([]int{4}, []float32{1.0, 1.0, 1.0, 1.0})

	slots[exponentSlot] = expT
	slots[epsilonSlot] = epsT
	slots[weightSlot] = weightT

	slotShapes[xSlot] = []int{1, 4}
	slotShapes[exponentSlot] = []int{1}
	slotShapes[epsilonSlot] = []int{1}
	slotShapes[weightSlot] = []int{4}

	frozenIdx := []int{exponentSlot, epsilonSlot, weightSlot}

	noop := func(_ context.Context, _ []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	}

	instructions := []Instruction[float32]{
		{Forward: noop, InputIdx: []int{xSlot, exponentSlot}, OutputIdx: powOut, OpName: "Pow"},
		{Forward: noop, InputIdx: []int{powOut}, OutputIdx: meanOut, OpName: "ReduceMean", ExtraArgs: map[string]any{"axis": -1, "keepDims": true}},
		{Forward: noop, InputIdx: []int{meanOut, epsilonSlot}, OutputIdx: addOut, OpName: "Add"},
		{Forward: noop, InputIdx: []int{addOut}, OutputIdx: sqrtOut, OpName: "Sqrt"},
		{Forward: noop, InputIdx: []int{xSlot, sqrtOut}, OutputIdx: divOut, OpName: "Div"},
		{Forward: noop, InputIdx: []int{divOut, weightSlot}, OutputIdx: mulOut, OpName: "Mul"},
	}

	return instructions, frozenIdx, slots, slotShapes
}

func TestFuseRMSNorm_SinglePattern(t *testing.T) {
	instructions, frozenIdx, slots, slotShapes := buildRMSNormPattern(0, 1, 2, 3, 4, 1e-5)

	got, gotFrozen, _, _ := FuseRMSNorm(instructions, frozenIdx, slots, slotShapes)

	if len(got) != 1 {
		t.Fatalf("expected 1 fused instruction, got %d", len(got))
	}
	if got[0].OpName != "FusedRMSNorm" {
		t.Fatalf("expected FusedRMSNorm, got %s", got[0].OpName)
	}
	// Output slot should be the same as the original Mul output.
	if got[0].OutputIdx != 9 {
		t.Fatalf("expected output slot 9, got %d", got[0].OutputIdx)
	}
	// Inputs should be [xSlot, weightSlot].
	if len(got[0].InputIdx) != 2 {
		t.Fatalf("expected 2 inputs, got %d", len(got[0].InputIdx))
	}
	if got[0].InputIdx[0] != 0 {
		t.Fatalf("expected x input slot 0, got %d", got[0].InputIdx[0])
	}
	if got[0].InputIdx[1] != 3 {
		t.Fatalf("expected weight input slot 3, got %d", got[0].InputIdx[1])
	}
	// Frozen indices should be preserved.
	if len(gotFrozen) != len(frozenIdx) {
		t.Fatalf("expected %d frozen slots, got %d", len(frozenIdx), len(gotFrozen))
	}
}

func TestFuseRMSNorm_MultiplePatterns(t *testing.T) {
	// Build two RMSNorm patterns in sequence.
	instr1, frozen1, slots1, shapes1 := buildRMSNormPattern(0, 1, 2, 3, 4, 1e-5)
	// Second pattern starts at slot 10 (after the first pattern's mulOut=9).
	instr2, frozen2, slots2, shapes2 := buildRMSNormPattern(9, 11, 12, 13, 14, 1e-6)

	// Merge: extend slots/shapes to cover both patterns.
	maxSlot := 20
	slots := make([]*tensor.TensorNumeric[float32], maxSlot)
	shapes := make([][]int, maxSlot)
	copy(slots, slots1)
	copy(shapes, shapes1)
	for i, s := range slots2 {
		if s != nil && i < maxSlot {
			slots[i] = s
		}
	}
	for i, s := range shapes2 {
		if s != nil && i < maxSlot {
			shapes[i] = s
		}
	}

	instructions := append(instr1, instr2...)
	frozenIdx := append(frozen1, frozen2...)

	got, _, _, _ := FuseRMSNorm(instructions, frozenIdx, slots, shapes)

	if len(got) != 2 {
		t.Fatalf("expected 2 fused instructions, got %d", len(got))
	}
	for i, inst := range got {
		if inst.OpName != "FusedRMSNorm" {
			t.Fatalf("instruction %d: expected FusedRMSNorm, got %s", i, inst.OpName)
		}
	}
}

func TestFuseRMSNorm_NoPattern(t *testing.T) {
	noop := func(_ context.Context, _ []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	}
	instructions := []Instruction[float32]{
		{Forward: noop, InputIdx: []int{0, 1}, OutputIdx: 2, OpName: "MatMul"},
		{Forward: noop, InputIdx: []int{2, 3}, OutputIdx: 4, OpName: "Add"},
	}
	slots := make([]*tensor.TensorNumeric[float32], 5)
	shapes := make([][]int, 5)

	got, _, _, _ := FuseRMSNorm(instructions, nil, slots, shapes)

	if len(got) != 2 {
		t.Fatalf("expected 2 instructions (no fusion), got %d", len(got))
	}
}

func TestFuseRMSNorm_DifferentXSlots(t *testing.T) {
	// Pattern where Div's first input != Pow's first input.
	// This is valid in ONNX models where Cast ops sit between x and Div.
	// The fusion should still match (connectivity is valid).
	noop := func(_ context.Context, _ []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	}
	expT, _ := tensor.New[float32]([]int{1}, []float32{2.0})
	epsT, _ := tensor.New[float32]([]int{1}, []float32{1e-5})
	weightT, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})

	slots := make([]*tensor.TensorNumeric[float32], 12)
	slots[1] = expT
	slots[2] = epsT
	slots[3] = weightT
	shapes := make([][]int, 12)
	frozenIdx := []int{1, 2, 3}

	instructions := []Instruction[float32]{
		{Forward: noop, InputIdx: []int{0, 1}, OutputIdx: 4, OpName: "Pow"},
		{Forward: noop, InputIdx: []int{4}, OutputIdx: 5, OpName: "ReduceMean", ExtraArgs: map[string]any{"axis": -1, "keepDims": true}},
		{Forward: noop, InputIdx: []int{5, 2}, OutputIdx: 6, OpName: "Add"},
		{Forward: noop, InputIdx: []int{6}, OutputIdx: 7, OpName: "Sqrt"},
		// Div's first input is 10, NOT 0 (Pow's first input) — valid with Cast.
		{Forward: noop, InputIdx: []int{10, 7}, OutputIdx: 8, OpName: "Div"},
		{Forward: noop, InputIdx: []int{8, 3}, OutputIdx: 9, OpName: "Mul"},
	}

	got, _, _, _ := FuseRMSNorm(instructions, frozenIdx, slots, shapes)

	// Should fuse: connectivity Pow→ReduceMean→Add→Sqrt→Div→Mul is valid.
	if len(got) != 1 {
		t.Fatalf("expected 1 fused instruction, got %d", len(got))
	}
	if got[0].OpName != "FusedRMSNorm" {
		t.Fatalf("expected FusedRMSNorm, got %s", got[0].OpName)
	}
}

func TestFuseRMSNorm_BrokenChain(t *testing.T) {
	// Pattern where Sqrt is replaced with a different op → should NOT fuse.
	noop := func(_ context.Context, _ []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	}
	expT, _ := tensor.New[float32]([]int{1}, []float32{2.0})
	epsT, _ := tensor.New[float32]([]int{1}, []float32{1e-5})
	weightT, _ := tensor.New[float32]([]int{4}, []float32{1, 1, 1, 1})

	slots := make([]*tensor.TensorNumeric[float32], 12)
	slots[1] = expT
	slots[2] = epsT
	slots[3] = weightT
	shapes := make([][]int, 12)
	frozenIdx := []int{1, 2, 3}

	instructions := []Instruction[float32]{
		{Forward: noop, InputIdx: []int{0, 1}, OutputIdx: 4, OpName: "Pow"},
		{Forward: noop, InputIdx: []int{4}, OutputIdx: 5, OpName: "ReduceMean", ExtraArgs: map[string]any{"axis": -1, "keepDims": true}},
		{Forward: noop, InputIdx: []int{5, 2}, OutputIdx: 6, OpName: "Add"},
		// NOT Sqrt — broken chain.
		{Forward: noop, InputIdx: []int{6}, OutputIdx: 7, OpName: "Exp"},
		{Forward: noop, InputIdx: []int{0, 7}, OutputIdx: 8, OpName: "Div"},
		{Forward: noop, InputIdx: []int{8, 3}, OutputIdx: 9, OpName: "Mul"},
	}

	got, _, _, _ := FuseRMSNorm(instructions, frozenIdx, slots, shapes)

	if len(got) != 6 {
		t.Fatalf("expected 6 instructions (no fusion due to broken chain), got %d", len(got))
	}
}

func TestFuseRMSNorm_PreservesNonPatternOps(t *testing.T) {
	// Add a MatMul before and after the RMSNorm pattern.
	noop := func(_ context.Context, _ []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	}

	instructions, frozenIdx, slots, shapes := buildRMSNormPattern(0, 1, 2, 3, 4, 1e-5)

	// Prepend a MatMul and append an Add.
	pre := Instruction[float32]{Forward: noop, InputIdx: []int{0, 1}, OutputIdx: 10, OpName: "MatMul"}
	post := Instruction[float32]{Forward: noop, InputIdx: []int{9, 10}, OutputIdx: 11, OpName: "Add"}

	allInstructions := make([]Instruction[float32], 0, 8)
	allInstructions = append(allInstructions, pre)
	allInstructions = append(allInstructions, instructions...)
	allInstructions = append(allInstructions, post)

	// Extend slots/shapes.
	for len(slots) <= 11 {
		slots = append(slots, nil)
		shapes = append(shapes, nil)
	}

	got, _, _, _ := FuseRMSNorm(allInstructions, frozenIdx, slots, shapes)

	// Should be: MatMul + FusedRMSNorm + Add = 3.
	if len(got) != 3 {
		t.Fatalf("expected 3 instructions (MatMul + FusedRMSNorm + Add), got %d", len(got))
	}
	if got[0].OpName != "MatMul" {
		t.Fatalf("expected MatMul at index 0, got %s", got[0].OpName)
	}
	if got[1].OpName != "FusedRMSNorm" {
		t.Fatalf("expected FusedRMSNorm at index 1, got %s", got[1].OpName)
	}
	if got[2].OpName != "Add" {
		t.Fatalf("expected Add at index 2, got %s", got[2].OpName)
	}
}

func TestFuseRMSNorm_Correctness(t *testing.T) {
	// Verify the fused instruction produces correct output.
	x, _ := tensor.New[float32]([]int{1, 4}, []float32{1.0, 2.0, 3.0, 4.0})
	weight, _ := tensor.New[float32]([]int{4}, []float32{1.0, 1.0, 1.0, 1.0})
	eps := float32(1e-5)

	// Compute expected: x / sqrt(mean(x^2) + eps) * weight
	// x^2 = [1, 4, 9, 16], mean = 7.5, sqrt(7.5 + 1e-5) ≈ 2.7386...
	// normalized = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
	meanSq := float32(7.5)
	rms := float32(math.Sqrt(float64(meanSq + eps)))
	expected := []float32{1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms}

	instructions, frozenIdx, slots, shapes := buildRMSNormPattern(0, 1, 2, 3, 4, eps)
	got, _, _, _ := FuseRMSNorm(instructions, frozenIdx, slots, shapes)

	if len(got) != 1 || got[0].OpName != "FusedRMSNorm" {
		t.Fatal("fusion did not produce expected FusedRMSNorm instruction")
	}

	result, err := got[0].Forward(context.Background(), []*tensor.TensorNumeric[float32]{x, weight})
	if err != nil {
		t.Fatalf("FusedRMSNorm Forward failed: %v", err)
	}

	resultData := result.Data()
	if len(resultData) != 4 {
		t.Fatalf("expected 4 elements, got %d", len(resultData))
	}
	for i, want := range expected {
		if diff := math.Abs(float64(resultData[i] - want)); diff > 1e-4 {
			t.Errorf("element %d: got %f, want %f (diff %f)", i, resultData[i], want, diff)
		}
	}
}

func TestFuseRMSNorm_TooFewInstructions(t *testing.T) {
	noop := func(_ context.Context, _ []*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
		return nil, nil
	}
	instructions := []Instruction[float32]{
		{Forward: noop, InputIdx: []int{0}, OutputIdx: 1, OpName: "Sqrt"},
	}
	slots := make([]*tensor.TensorNumeric[float32], 2)
	shapes := make([][]int, 2)

	got, _, _, _ := FuseRMSNorm(instructions, nil, slots, shapes)
	if len(got) != 1 {
		t.Fatalf("expected 1 instruction unchanged, got %d", len(got))
	}
}
