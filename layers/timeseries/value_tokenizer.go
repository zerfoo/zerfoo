package timeseries

import (
	"fmt"
	"math"
	"sort"
)

// ValueTokenizer maps continuous float values to discrete bin indices and back.
//
// Chronos (Ansari et al., 2024) tokenizes time-series values into discrete
// bins whose edges are learned during pre-training and stored in the model
// config. This tokenizer performs:
//
//   - Tokenize: map a float value to the index of the bin that contains it.
//     Values below the first edge map to bin 0; values at or above the last
//     edge map to the last bin (numBins-1).
//
//   - Detokenize: map a bin index back to the bin center, computed as the
//     midpoint of the bin's lower and upper edges. The first and last bins
//     use the nearest interior edge width for extrapolation.
type ValueTokenizer struct {
	// edges are the sorted bin boundaries. For N bins there are N+1 edges.
	edges []float64

	// centers are precomputed bin centers (one per bin).
	centers []float64

	numBins int
}

// NewValueTokenizer creates a tokenizer from the given bin edges.
//
// edges must contain at least 2 elements (defining at least 1 bin) and must
// be sorted in strictly ascending order.
func NewValueTokenizer(edges []float64) (*ValueTokenizer, error) {
	if len(edges) < 2 {
		return nil, fmt.Errorf("ValueTokenizer: need at least 2 edges, got %d", len(edges))
	}
	for i := 1; i < len(edges); i++ {
		if edges[i] <= edges[i-1] {
			return nil, fmt.Errorf("ValueTokenizer: edges must be strictly ascending, edges[%d]=%g <= edges[%d]=%g",
				i, edges[i], i-1, edges[i-1])
		}
	}

	numBins := len(edges) - 1
	sorted := make([]float64, len(edges))
	copy(sorted, edges)

	centers := make([]float64, numBins)
	for i := 0; i < numBins; i++ {
		centers[i] = (sorted[i] + sorted[i+1]) / 2.0
	}

	return &ValueTokenizer{
		edges:   sorted,
		centers: centers,
		numBins: numBins,
	}, nil
}

// NumBins returns the number of discrete bins.
func (vt *ValueTokenizer) NumBins() int {
	return vt.numBins
}

// Edges returns a copy of the bin edges.
func (vt *ValueTokenizer) Edges() []float64 {
	out := make([]float64, len(vt.edges))
	copy(out, vt.edges)
	return out
}

// Centers returns a copy of the precomputed bin centers.
func (vt *ValueTokenizer) Centers() []float64 {
	out := make([]float64, len(vt.centers))
	copy(out, vt.centers)
	return out
}

// Tokenize maps a continuous value to its bin index.
//
// The bin index is determined by binary search over the edges. A value v
// falls into bin i if edges[i] <= v < edges[i+1]. Values below edges[0]
// map to bin 0; values >= edges[numBins] map to bin numBins-1.
func (vt *ValueTokenizer) Tokenize(v float64) int {
	if math.IsNaN(v) {
		return 0
	}
	// sort.SearchFloat64s returns the smallest index i where edges[i] >= v.
	idx := sort.SearchFloat64s(vt.edges, v)

	// v is below all edges or exactly at the first edge.
	if idx == 0 {
		return 0
	}

	// If v exactly matches an edge, it belongs to the bin starting at that
	// edge (bin idx), unless it's the last edge where we clamp to the
	// final bin.
	if idx < len(vt.edges) && vt.edges[idx] == v {
		if idx >= vt.numBins {
			return vt.numBins - 1
		}
		return idx
	}

	// v falls strictly between edges[idx-1] and edges[idx] (or above all edges).
	bin := idx - 1
	if bin >= vt.numBins {
		return vt.numBins - 1
	}
	return bin
}

// TokenizeBatch maps a slice of values to bin indices.
func (vt *ValueTokenizer) TokenizeBatch(values []float64) []int {
	out := make([]int, len(values))
	for i, v := range values {
		out[i] = vt.Tokenize(v)
	}
	return out
}

// Detokenize maps a bin index back to the bin center value.
//
// If the index is out of range it is clamped to [0, numBins-1].
func (vt *ValueTokenizer) Detokenize(bin int) float64 {
	if bin < 0 {
		bin = 0
	}
	if bin >= vt.numBins {
		bin = vt.numBins - 1
	}
	return vt.centers[bin]
}

// DetokenizeBatch maps a slice of bin indices back to bin center values.
func (vt *ValueTokenizer) DetokenizeBatch(bins []int) []float64 {
	out := make([]float64, len(bins))
	for i, b := range bins {
		out[i] = vt.Detokenize(b)
	}
	return out
}
