package generate

import (
	"strings"

	"github.com/zerfoo/ztensor/graph"
)

// kvCacheOpNames lists the op names that indicate KV cache usage.
var kvCacheOpNames = []string{
	"KVCacheAppendK",
	"KVCacheAppendV",
	"KVCacheGetK",
	"KVCacheGetV",
}

// detectKVCacheOps returns true if any instruction in the tape is a KV cache operation.
func detectKVCacheOps(instructions []graph.InstructionMeta) bool {
	for _, inst := range instructions {
		for _, name := range kvCacheOpNames {
			if inst.OpName == name {
				return true
			}
		}
	}
	return false
}

// extractKVCacheDims scans KV cache append ops to determine numLayers, numHeads,
// and headDim. numLayers is max(layer index) + 1. numHeads and headDim are
// extracted from the input tensor shape of the first KVCacheAppendK op.
func extractKVCacheDims(instructions []graph.InstructionMeta, slotShapes [][]int) (numLayers, numHeads, headDim int) {
	maxLayer := -1
	for _, inst := range instructions {
		if !strings.HasPrefix(inst.OpName, "KVCacheAppend") {
			continue
		}
		layer, ok := extractLayerIndex(inst)
		if !ok {
			continue
		}
		if layer > maxLayer {
			maxLayer = layer
		}
		// Extract numHeads and headDim from the first append op's input shape.
		if numHeads == 0 && len(inst.InputIdx) > 0 {
			slotIdx := inst.InputIdx[0]
			if slotIdx < len(slotShapes) && len(slotShapes[slotIdx]) >= 2 {
				shape := slotShapes[slotIdx]
				// Shape is [batch, numHeads, headDim] or [numHeads, headDim].
				headDim = shape[len(shape)-1]
				numHeads = shape[len(shape)-2]
			}
		}
	}
	if maxLayer >= 0 {
		numLayers = maxLayer + 1
	}
	return
}

// extractLayerIndex extracts the "layer" value from an instruction's ExtraArgs.
func extractLayerIndex(inst graph.InstructionMeta) (int, bool) {
	if inst.ExtraArgs == nil {
		return 0, false
	}
	v, ok := inst.ExtraArgs["layer"]
	if !ok {
		return 0, false
	}
	switch layer := v.(type) {
	case int:
		return layer, true
	case float64:
		return int(layer), true
	default:
		return 0, false
	}
}
