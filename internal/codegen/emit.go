package codegen

import (
	"fmt"
	"sort"
	"strings"

	"github.com/zerfoo/ztensor/graph"
)

// FrozenSlotMeta describes a frozen (constant/weight) slot for the emitter.
type FrozenSlotMeta struct {
	SlotIdx int
}

// MegakernelConfig holds all information needed to emit a megakernel .cu file.
type MegakernelConfig struct {
	Instructions []graph.InstructionMeta
	SlotShapes   [][]int
	FrozenSlots  []FrozenSlotMeta
	InputSlots   []int
	OutputSlot   int
	NumKVLayers  int // number of KV cache layers (0 = no KV cache)
}

// WorkspaceLayout describes the memory layout for megakernel slot buffers.
// Frozen slots are NOT in the workspace -- they have their own pointer array.
type WorkspaceLayout struct {
	SlotOffsets  map[int]int // slot index -> offset in workspace (element count)
	TotalSize    int         // total workspace size in elements
	InputOffset  int         // offset of first input slot
	OutputOffset int         // offset of output slot
	OutputSize   int         // size of output slot in elements
}

// slotSize returns the total number of elements for a slot shape.
func slotSize(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// ComputeWorkspaceLayout computes the workspace memory layout for the
// megakernel. Frozen slots are excluded since they use a separate pointer
// array. The layout is deterministic (sorted by slot index).
func ComputeWorkspaceLayout(cfg MegakernelConfig) WorkspaceLayout {
	frozenSet := make(map[int]bool, len(cfg.FrozenSlots))
	for _, f := range cfg.FrozenSlots {
		frozenSet[f.SlotIdx] = true
	}

	// Collect non-frozen used slots.
	usedSlots := make(map[int]bool)
	for _, inst := range cfg.Instructions {
		for _, idx := range inst.InputIdx {
			if !frozenSet[idx] {
				usedSlots[idx] = true
			}
		}
		if !frozenSet[inst.OutputIdx] {
			usedSlots[inst.OutputIdx] = true
		}
	}
	for _, idx := range cfg.InputSlots {
		if !frozenSet[idx] {
			usedSlots[idx] = true
		}
	}
	if !frozenSet[cfg.OutputSlot] {
		usedSlots[cfg.OutputSlot] = true
	}

	// Sort slot indices for deterministic layout.
	var indices []int
	for idx := range usedSlots {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	offsets := make(map[int]int, len(indices))
	total := 0
	for _, idx := range indices {
		offsets[idx] = total
		size := 1
		if idx < len(cfg.SlotShapes) && cfg.SlotShapes[idx] != nil {
			if s := slotSize(cfg.SlotShapes[idx]); s > 0 {
				size = s
			}
		}
		total += size
	}

	inputOffset := 0
	if len(cfg.InputSlots) > 0 {
		if off, ok := offsets[cfg.InputSlots[0]]; ok {
			inputOffset = off
		}
	}

	outputOffset := offsets[cfg.OutputSlot]
	outputSize := 1
	if cfg.OutputSlot < len(cfg.SlotShapes) && cfg.SlotShapes[cfg.OutputSlot] != nil {
		if s := slotSize(cfg.SlotShapes[cfg.OutputSlot]); s > 0 {
			outputSize = s
		}
	}

	return WorkspaceLayout{
		SlotOffsets:  offsets,
		TotalSize:    total,
		InputOffset:  inputOffset,
		OutputOffset: outputOffset,
		OutputSize:   outputSize,
	}
}

// EmitMegakernel generates a complete CUDA .cu source string from the
// compiled instruction tape. Returns an error if any op is unsupported.
//
// The generated kernel uses a flat workspace buffer where each slot occupies
// a contiguous region. Frozen slots (model weights) are passed as a separate
// pointer array. A host-callable launch wrapper is emitted for dlopen/dlsym.
func EmitMegakernel(cfg MegakernelConfig) (string, error) {
	var b strings.Builder

	layout := ComputeWorkspaceLayout(cfg)

	frozenSet := make(map[int]bool, len(cfg.FrozenSlots))
	for _, f := range cfg.FrozenSlots {
		frozenSet[f.SlotIdx] = true
	}

	// Header.
	b.WriteString("#include <cuda_runtime.h>\n")
	b.WriteString("#include \"megakernel_ops.cu\"\n\n")

	// Frozen slot defines: map slot index to frozen array position.
	for i, f := range cfg.FrozenSlots {
		fmt.Fprintf(&b, "#define frozen_%d (frozen[%d])\n", f.SlotIdx, i)
	}
	if len(cfg.FrozenSlots) > 0 {
		b.WriteString("\n")
	}

	// Kernel function.
	b.WriteString("__global__ void megakernel(\n")
	b.WriteString("    float* __restrict__ workspace,\n")
	b.WriteString("    const float* const* __restrict__ frozen,\n")
	b.WriteString("    int pos,\n")
	if cfg.NumKVLayers > 0 {
		b.WriteString("    float** __restrict__ kv_k,\n")
		b.WriteString("    float** __restrict__ kv_v,\n")
		b.WriteString("    int seq_pos,\n")
		b.WriteString("    int kv_seq_len,\n")
	}
	b.WriteString("    int num_elements\n")
	b.WriteString(") {\n")
	b.WriteString("  int tid = blockIdx.x * blockDim.x + threadIdx.x;\n")
	b.WriteString("  if (tid >= num_elements) return;\n\n")

	// Slot pointer declarations (non-frozen slots in workspace).
	var slotIndices []int
	for idx := range layout.SlotOffsets {
		slotIndices = append(slotIndices, idx)
	}
	sort.Ints(slotIndices)

	for _, idx := range slotIndices {
		offset := layout.SlotOffsets[idx]
		fmt.Fprintf(&b, "  float* slot_%d = workspace + %d;\n", idx, offset)
	}
	b.WriteString("\n")

	// Emit instructions.
	for i, inst := range cfg.Instructions {
		inputs := make([]SlotInfo, len(inst.InputIdx))
		for j, idx := range inst.InputIdx {
			if idx < len(cfg.SlotShapes) && cfg.SlotShapes[idx] != nil {
				inputs[j] = SlotInfo{Shape: cfg.SlotShapes[idx]}
			}
		}

		code, err := Emit(inst, inputs)
		if err != nil {
			return "", fmt.Errorf("instruction %d (%s): %w", i, inst.OpName, err)
		}
		fmt.Fprintf(&b, "  // [%d] %s\n", i, inst.OpName)
		b.WriteString(code)
		b.WriteString("\n")
	}

	b.WriteString("}\n\n")

	// Host-callable launch wrapper (extern "C" for dlopen/dlsym).
	b.WriteString("extern \"C\" int launch_megakernel(\n")
	b.WriteString("    float* workspace,\n")
	b.WriteString("    const float* const* frozen,\n")
	b.WriteString("    int pos,\n")
	if cfg.NumKVLayers > 0 {
		b.WriteString("    float** kv_k,\n")
		b.WriteString("    float** kv_v,\n")
		b.WriteString("    int seq_pos,\n")
		b.WriteString("    int kv_seq_len,\n")
	}
	b.WriteString("    int num_elements\n")
	b.WriteString(") {\n")
	b.WriteString("  int grid = (num_elements + 255) / 256;\n")
	if cfg.NumKVLayers > 0 {
		b.WriteString("  megakernel<<<grid, 256>>>(workspace, frozen, pos, kv_k, kv_v, seq_pos, kv_seq_len, num_elements);\n")
	} else {
		b.WriteString("  megakernel<<<grid, 256>>>(workspace, frozen, pos, num_elements);\n")
	}
	b.WriteString("  return (int)cudaDeviceSynchronize();\n")
	b.WriteString("}\n")

	return b.String(), nil
}
