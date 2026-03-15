package graph

import (
	"context"
	"fmt"

	"github.com/zerfoo/zerfoo/compute"
	"github.com/zerfoo/zerfoo/tensor"
)

// FuseRMSNorm scans a compiled instruction list for the decomposed RMSNorm
// pattern (Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul) and replaces each
// matching 6-op sequence with a single fused RMSNorm instruction.
//
// The ONNX decomposition of RMSNorm produces:
//
//	x -> Pow(x, 2) -> ReduceMean(axis=-1, keepdims=true) -> Add(eps) -> Sqrt -> Div(x, sqrt) -> Mul(norm, weight)
//
// ONNX models may insert Cast ops between x and Div, so Pow.input[0] and
// Div.input[0] may differ. The fusion uses connectivity-based matching rather
// than exact slot equality.
func FuseRMSNorm[T tensor.Numeric](instructions []Instruction[T], frozenIdx []int, slots []*tensor.TensorNumeric[T], slotShapes [][]int, engine ...compute.Engine[T]) ([]Instruction[T], []int, []*tensor.TensorNumeric[T], [][]int) {
	var eng compute.Engine[T]
	if len(engine) > 0 {
		eng = engine[0]
	}
	if len(instructions) < 6 {
		return instructions, frozenIdx, slots, slotShapes
	}

	// Build output-slot -> instruction-index map.
	outToInstr := make(map[int]int, len(instructions))
	for i, inst := range instructions {
		outToInstr[inst.OutputIdx] = i
	}

	// Track which instructions are consumed by fusion.
	fused := make(map[int]bool)

	type replacement struct {
		instrIdx int
		inst     Instruction[T]
	}
	var replacements []replacement

	// Scan for Mul ops and trace backward through the chain.
	for mulIdx, mulInst := range instructions {
		if mulInst.OpName != "Mul" || len(mulInst.InputIdx) != 2 || fused[mulIdx] {
			continue
		}

		// Identify which Mul input comes from a Div (normalized output).
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
			continue
		}

		// Trace: Div
		divIdx := outToInstr[divOutputSlot]
		divInst := instructions[divIdx]
		if divInst.OpName != "Div" || len(divInst.InputIdx) != 2 {
			continue
		}
		xSlot := divInst.InputIdx[0]
		sqrtOutputSlot := divInst.InputIdx[1]

		// Trace: Sqrt
		sqrtIdx, ok := outToInstr[sqrtOutputSlot]
		if !ok {
			continue
		}
		sqrtInst := instructions[sqrtIdx]
		if sqrtInst.OpName != "Sqrt" || len(sqrtInst.InputIdx) != 1 {
			continue
		}

		// Trace: Add
		addOutputSlot := sqrtInst.InputIdx[0]
		addIdx, ok := outToInstr[addOutputSlot]
		if !ok {
			continue
		}
		addInst := instructions[addIdx]
		if addInst.OpName != "Add" || len(addInst.InputIdx) != 2 {
			continue
		}

		// Identify which Add input comes from ReduceMean.
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
		reduceMeanIdx := outToInstr[reduceMeanOutputSlot]
		reduceMeanInst := instructions[reduceMeanIdx]
		if reduceMeanInst.OpName != "ReduceMean" || len(reduceMeanInst.InputIdx) < 1 {
			continue
		}
		if reduceMeanInst.ExtraArgs != nil {
			if !extractBool(reduceMeanInst.ExtraArgs, "keepDims") {
				continue
			}
		}

		// Trace: Pow
		powOutputSlot := reduceMeanInst.InputIdx[0]
		powIdx, ok := outToInstr[powOutputSlot]
		if !ok {
			continue
		}
		powInst := instructions[powIdx]
		if powInst.OpName != "Pow" || len(powInst.InputIdx) != 2 {
			continue
		}

		// Extract epsilon value.
		var epsilon T
		if epsilonSlot < len(slots) && slots[epsilonSlot] != nil {
			data := slots[epsilonSlot].Data()
			if len(data) > 0 {
				epsilon = data[0]
			}
		}

		// Build fused Forward function.
		capturedEps := epsilon
		capturedEng := eng
		fwdFn := func(ctx context.Context, inputs []*tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
			x := inputs[0]
			weight := inputs[1]
			if x == nil || len(x.Shape()) == 0 {
				return nil, fmt.Errorf("FusedRMSNorm: nil or empty input tensor")
			}
			f32x, xOk := any(x).(*tensor.TensorNumeric[float32])
			f32w, wOk := any(weight).(*tensor.TensorNumeric[float32])
			if !xOk || !wOk {
				return nil, fmt.Errorf("FusedRMSNorm: unsupported type, expected float32")
			}
			f32eps, _ := any(capturedEps).(float32)
			// GPU path via FusedRMSNormer interface.
			if capturedEng != nil {
				if fused, ok := any(capturedEng).(compute.FusedRMSNormer); ok {
					out, _, err := fused.FusedRMSNormGPU(f32x, f32w, f32eps)
					if err == nil {
						return any(out).(*tensor.TensorNumeric[T]), nil
					}
				}
			}
			// CPU fallback.
			out, _, err := compute.FusedRMSNorm(f32x, f32w, f32eps)
			if err != nil {
				return nil, fmt.Errorf("FusedRMSNorm: %w", err)
			}
			return any(out).(*tensor.TensorNumeric[T]), nil
		}

		fusedInst := Instruction[T]{
			Forward:   fwdFn,
			InputIdx:  []int{xSlot, weightSlot},
			OutputIdx: mulInst.OutputIdx,
			OpName:    "FusedRMSNorm",
		}

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

	replMap := make(map[int]Instruction[T], len(replacements))
	for _, r := range replacements {
		replMap[r.instrIdx] = r.inst
	}

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
