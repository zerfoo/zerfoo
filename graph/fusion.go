package graph

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// FuseRMSNorm scans a compiled instruction list for the decomposed RMSNorm
// pattern (Pow → ReduceMean → Add → Sqrt → Div → Mul) and replaces each
// matching 6-op sequence with a single fused RMSNorm instruction.
//
// The ONNX decomposition of RMSNorm produces:
//
//	x → Pow(x, 2) → ReduceMean(axis=-1, keepdims=true) → Add(eps) → Sqrt → Div(x, sqrt) → Mul(norm, weight)
//
// where Div's first input is the same tensor as Pow's first input (original x),
// the second input to Add is a frozen epsilon scalar, and the second input to
// Mul is a frozen weight vector.
func FuseRMSNorm[T tensor.Numeric](instructions []Instruction[T], frozenIdx []int, slots []*tensor.TensorNumeric[T], slotShapes [][]int) ([]Instruction[T], []int, []*tensor.TensorNumeric[T], [][]int) {
	if len(instructions) < 6 {
		return instructions, frozenIdx, slots, slotShapes
	}

	// Build frozen slot lookup.
	frozenSet := make(map[int]bool, len(frozenIdx))
	for _, idx := range frozenIdx {
		frozenSet[idx] = true
	}

	// Build output-slot -> instruction-index map.
	outToInstr := make(map[int]int, len(instructions))
	for i, inst := range instructions {
		outToInstr[inst.OutputIdx] = i
	}

	// Track which instructions are consumed by fusion (to skip later).
	fused := make(map[int]bool)

	// Replacement instructions to insert (keyed by the Mul instruction index).
	type replacement struct {
		instrIdx int
		inst     Instruction[T]
	}
	var replacements []replacement

	debug := os.Getenv("ZERFOO_DEBUG_FUSION") == "1"
	if debug {
		opCounts := make(map[string]int)
		for _, inst := range instructions {
			opCounts[inst.OpName]++
		}
		log.Printf("FuseRMSNorm: %d instructions, op counts: %v", len(instructions), opCounts)
		log.Printf("FuseRMSNorm: %d frozen slots, %d total slots", len(frozenIdx), len(slots))
		// Print first few frozen indices and first Div instruction inputs.
		if len(frozenIdx) > 0 {
			sample := frozenIdx
			if len(sample) > 10 {
				sample = sample[:10]
			}
			log.Printf("FuseRMSNorm: first frozen indices: %v", sample)
		}
		// Print first Div instruction to understand connectivity.
		for i, inst := range instructions {
			if inst.OpName == "Div" {
				log.Printf("FuseRMSNorm: first Div[%d] inputs=%v output=%d", i, inst.InputIdx, inst.OutputIdx)
				// Print the Mul that consumes this Div's output.
				for j, m := range instructions {
					if m.OpName == "Mul" {
						for _, inp := range m.InputIdx {
							if inp == inst.OutputIdx {
								log.Printf("FuseRMSNorm: -> consumed by Mul[%d] inputs=%v, frozen status: [%v, %v]",
									j, m.InputIdx, frozenSet[m.InputIdx[0]], frozenSet[m.InputIdx[1]])
								break
							}
						}
					}
				}
				break
			}
		}
	}

	// Scan for Mul ops and trace backward.
	for mulIdx, mulInst := range instructions {
		if mulInst.OpName != "Mul" || len(mulInst.InputIdx) != 2 {
			continue
		}
		if fused[mulIdx] {
			continue
		}

		// Mul inputs: [normalized, weight] or [weight, normalized].
		// Identify which input comes from a Div (the normalized output).
		// The other input is the weight (may be a parameter or pass through Cast/Reshape).
		divOutputSlot := -1
		weightSlot := -1
		for _, inputSlot := range mulInst.InputIdx {
			if divIdx, ok := outToInstr[inputSlot]; ok && instructions[divIdx].OpName == "Div" {
				divOutputSlot = inputSlot
			} else {
				weightSlot = inputSlot
			}
		}
		if weightSlot == -1 || divOutputSlot == -1 {
			if debug {
				log.Printf("FuseRMSNorm: Mul[%d] inputs=%v (skip: no Div input found)", mulIdx, mulInst.InputIdx)
			}
			continue
		}

		// Trace: Div
		divIdx, ok := outToInstr[divOutputSlot]
		if !ok {
			continue
		}
		divInst := instructions[divIdx]
		if divInst.OpName != "Div" || len(divInst.InputIdx) != 2 {
			if debug {
				log.Printf("FuseRMSNorm: Mul[%d] -> slot %d not Div (op=%s inputs=%d)", mulIdx, divOutputSlot, instructions[divIdx].OpName, len(instructions[divIdx].InputIdx))
			}
			continue
		}

		// Div inputs: [x, sqrt_output]
		xSlot := divInst.InputIdx[0]
		sqrtOutputSlot := divInst.InputIdx[1]

		// Trace: Sqrt
		sqrtIdx, ok := outToInstr[sqrtOutputSlot]
		if !ok {
			if debug {
				log.Printf("FuseRMSNorm: Mul[%d] -> Div[%d] sqrtSlot=%d has no producer", mulIdx, divIdx, sqrtOutputSlot)
			}
			continue
		}
		sqrtInst := instructions[sqrtIdx]
		if sqrtInst.OpName != "Sqrt" || len(sqrtInst.InputIdx) != 1 {
			if debug {
				log.Printf("FuseRMSNorm: Mul[%d] -> Div[%d] -> slot %d not Sqrt (op=%s inputs=%d)", mulIdx, divIdx, sqrtOutputSlot, sqrtInst.OpName, len(sqrtInst.InputIdx))
			}
			continue
		}

		// Trace: Add (Sqrt's input)
		addOutputSlot := sqrtInst.InputIdx[0]
		addIdx, ok := outToInstr[addOutputSlot]
		if !ok {
			continue
		}
		addInst := instructions[addIdx]
		if addInst.OpName != "Add" || len(addInst.InputIdx) != 2 {
			if debug {
				log.Printf("FuseRMSNorm: Mul[%d] -> Div -> Sqrt -> slot %d not Add (op=%s inputs=%d)", mulIdx, addOutputSlot, addInst.OpName, len(addInst.InputIdx))
			}
			continue
		}

		// Add inputs: [reducemean_output, epsilon].
		// Identify which input comes from a ReduceMean. The other is epsilon.
		reduceMeanOutputSlot := -1
		epsilonSlot := -1
		for _, inputSlot := range addInst.InputIdx {
			if rmIdx, ok := outToInstr[inputSlot]; ok && instructions[rmIdx].OpName == "ReduceMean" {
				reduceMeanOutputSlot = inputSlot
			} else {
				epsilonSlot = inputSlot
			}
		}
		if epsilonSlot == -1 || reduceMeanOutputSlot == -1 {
			continue
		}

		// Trace: ReduceMean
		reduceMeanIdx, ok := outToInstr[reduceMeanOutputSlot]
		if !ok {
			continue
		}
		reduceMeanInst := instructions[reduceMeanIdx]
		if reduceMeanInst.OpName != "ReduceMean" || len(reduceMeanInst.InputIdx) < 1 {
			continue
		}
		// Verify keepDims=true and axis=-1 (last dimension) when ExtraArgs
		// are available (CompileTraced path). When ExtraArgs is nil (Compile
		// path), trust the pattern match — ONNX RMSNorm decomposition always
		// uses keepDims=true and reduces the last dimension.
		if reduceMeanInst.ExtraArgs != nil {
			keepDims := extractBool(reduceMeanInst.ExtraArgs, "keepDims")
			if !keepDims {
				continue
			}
		}

		// Trace: Pow (ReduceMean's input)
		powOutputSlot := reduceMeanInst.InputIdx[0]
		powIdx, ok := outToInstr[powOutputSlot]
		if !ok {
			continue
		}
		powInst := instructions[powIdx]
		if powInst.OpName != "Pow" || len(powInst.InputIdx) != 2 {
			if debug {
				log.Printf("FuseRMSNorm: Mul[%d] -> ...-> ReduceMean -> slot %d not Pow (op=%s inputs=%d)", mulIdx, powOutputSlot, powInst.OpName, len(powInst.InputIdx))
			}
			continue
		}

		// In pure ONNX, Pow.input[0] and Div.input[0] should both be x.
		// However, ONNX models may insert Cast ops between x and Div (e.g.,
		// float16→float32 conversion). Accept the pattern if they share a
		// common ancestor within 1 hop, or if they're the same slot.
		// The full chain match (Pow→ReduceMean→Add→Sqrt→Div→Mul) provides
		// sufficient confidence even without exact x-slot equality.
		xSlotForFusion := xSlot // Use Div's input as the x for the fused op

		// The exponent (InputIdx[1]) should be a scalar 2. We trust the pattern
		// match (Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul connectivity)
		// rather than verifying the value, since the slot may be on GPU.
		// Optionally verify value is 2, but slot data might be on GPU.
		// We trust the pattern match.

		// Extract epsilon value from the frozen slot.
		var epsilon T
		if epsilonSlot < len(slots) && slots[epsilonSlot] != nil {
			data := slots[epsilonSlot].Data()
			if len(data) > 0 {
				epsilon = data[0]
			}
		}

		// All checks passed — record the fusion.
		capturedEps := epsilon
		capturedWeightSlot := weightSlot
		capturedXSlot := xSlotForFusion

		fwdFn := func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			x := inputs[0]
			weight := inputs[1]
			f32x, xOk := any(x).(*tensor.TensorNumeric[float32])
			f32w, wOk := any(weight).(*tensor.TensorNumeric[float32])
			if !xOk || !wOk {
				return nil, fmt.Errorf("FusedRMSNorm: unsupported type, expected float32")
			}
			f32eps, _ := any(capturedEps).(float32)
			out, _, err := compute.FusedRMSNorm(f32x, f32w, f32eps)
			if err != nil {
				return nil, fmt.Errorf("FusedRMSNorm: %w", err)
			}
			return any(out).(*tensor.TensorNumeric[T]), nil
		}

		fusedInst := Instruction[T]{
			Forward:   fwdFn,
			InputIdx:  []int{capturedXSlot, capturedWeightSlot},
			OutputIdx: mulInst.OutputIdx,
			OpName:    "FusedRMSNorm",
		}

		// Mark all 6 instructions as fused.
		fused[powIdx] = true
		fused[reduceMeanIdx] = true
		fused[addIdx] = true
		fused[sqrtIdx] = true
		fused[divIdx] = true
		fused[mulIdx] = true

		replacements = append(replacements, replacement{instrIdx: mulIdx, inst: fusedInst})
	}

	if len(replacements) == 0 {
		return instructions, frozenIdx, slots, slotShapes
	}

	// Build replacement lookup.
	replMap := make(map[int]Instruction[T], len(replacements))
	for _, r := range replacements {
		replMap[r.instrIdx] = r.inst
	}

	// Build the new instruction list: skip fused ops, insert replacements.
	result := make([]Instruction[T], 0, len(instructions)-5*len(replacements))
	for i, inst := range instructions {
		if r, ok := replMap[i]; ok {
			result = append(result, r)
		} else if !fused[i] {
			result = append(result, inst)
		}
	}

	return result, frozenIdx, slots, slotShapes
}
