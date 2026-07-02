package inference

import "fmt"

// LayerType classifies a transformer layer's attention mode.
type LayerType int

const (
	// LayerTypeSliding denotes a sliding-window attention layer.
	LayerTypeSliding LayerType = iota
	// LayerTypeGlobal denotes a full (global) attention layer.
	LayerTypeGlobal
)

// ResolveKVDonor returns the donor layer index for a shared-KV layer.
//
// Per HuggingFace transformers modeling_gemma4.py lines 1148-1160, a
// shared-KV layer reuses K/V from the last non-shared layer of the same
// attention type (sliding vs global).
//
// Preconditions: layerIdx >= firstSharedIdx, 0 <= firstSharedIdx <= len(layerTypes),
// len(layerTypes) > layerIdx.
// Panics if any precondition is violated (caller bug).
func ResolveKVDonor(layerIdx, firstSharedIdx int, layerTypes []LayerType) int {
	if firstSharedIdx < 0 || firstSharedIdx > len(layerTypes) {
		panic(fmt.Sprintf("ResolveKVDonor: firstSharedIdx %d out of range [0,%d]", firstSharedIdx, len(layerTypes)))
	}
	if layerIdx < firstSharedIdx {
		panic(fmt.Sprintf("ResolveKVDonor: layerIdx %d < firstSharedIdx %d", layerIdx, firstSharedIdx))
	}
	if layerIdx >= len(layerTypes) {
		panic(fmt.Sprintf("ResolveKVDonor: layerIdx %d >= len(layerTypes) %d", layerIdx, len(layerTypes)))
	}
	want := layerTypes[layerIdx]
	for j := layerIdx - 1; j >= 0; j-- {
		if j < firstSharedIdx && layerTypes[j] == want {
			return j
		}
	}
	panic(fmt.Sprintf("ResolveKVDonor: no donor found for layerIdx %d (firstSharedIdx=%d)", layerIdx, firstSharedIdx))
}
