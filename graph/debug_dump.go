package graph

import (
	"fmt"
	"os"

	"github.com/zerfoo/zerfoo/tensor"
)

// debugDumpEnabled is true when ZERFOO_DEBUG_DUMP=1 is set.
var debugDumpEnabled = os.Getenv("ZERFOO_DEBUG_DUMP") == "1"

// debugDumpN is the number of float32 values to dump per checkpoint.
const debugDumpN = 8

// debugDumpCheckpoints lists the instruction op names that trigger a dump.
// For per-layer ops (GQA, FFN), only the first occurrence is dumped (layer 0).
// For RMSNorm, all occurrences are dumped so the final one can be identified.
var debugDumpCheckpoints = map[string]bool{
	"EmbeddingLookup":       true,
	"GroupedQueryAttention": true,
	"FFN":                   true,
	"RMSNorm":               true,
	"LMHead":                true,
}

// debugDumpOnlyFirst lists ops where only the first occurrence should be dumped.
var debugDumpOnlyFirst = map[string]bool{
	"EmbeddingLookup":       true,
	"GroupedQueryAttention": true,
	"FFN":                   true,
	"LMHead":                true,
}

// debugDumpSeen tracks how many times each op has been seen in the current token.
var debugDumpSeen map[string]int

// debugDumpTensor prints the first N float32 values from a tensor to stderr.
func debugDumpTensor(label string, t *tensor.TensorNumeric[float32]) {
	if t == nil {
		fmt.Fprintf(os.Stderr, "[DEBUG_DUMP] %s: nil tensor\n", label)
		return
	}

	totalElems := 1
	for _, d := range t.Shape() {
		totalElems *= d
	}
	n := debugDumpN
	if n > totalElems {
		n = totalElems
	}
	if n <= 0 {
		return
	}

	vals := make([]float32, n)

	if gs, ok := t.GetStorage().(*tensor.GPUStorage[float32]); ok {
		view := gs.SubSlice(0, n)
		if err := view.CopyTo(vals); err != nil {
			fmt.Fprintf(os.Stderr, "[DEBUG_DUMP] %s: D2H copy failed: %v\n", label, err)
			return
		}
	} else {
		data := t.Data()
		copy(vals, data[:n])
	}

	fmt.Fprintf(os.Stderr, "[DEBUG_DUMP] %s shape=%v first_%d: %v\n", label, t.Shape(), n, vals)
}
