package main

import (
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/zerfoo/zerfoo/inference"
)

// rowDiffStats holds summary statistics of per-row L2 differences between two
// equally-shaped [vocab, cols] float32 matrices.
type rowDiffStats struct {
	Rows   int       // number of rows compared
	Cols   int       // columns per row
	P50    float32   // median L2
	P95    float32   // 95th percentile L2
	P99    float32   // 99th percentile L2
	P100   float32   // max L2 (p100)
	TopK   []rowDiff // largest-L2 rows, descending
	Scoped bool      // true when only a prefix of rows was compared (scoped run)
}

// rowDiff records a single row's L2 distance.
type rowDiff struct {
	Row int
	L2  float32
}

// perRowL2 computes the L2 norm of (a - b) for every row i in [0, rows),
// assuming both slices are laid out as row-major [rows, cols] float32 matrices.
// Returns one L2 value per row.
func perRowL2(a, b []float32, rows, cols int) ([]float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("perRowL2: rows=%d cols=%d must be positive", rows, cols)
	}
	need := rows * cols
	if len(a) < need || len(b) < need {
		return nil, fmt.Errorf("perRowL2: need %d elements, got len(a)=%d len(b)=%d", need, len(a), len(b))
	}
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		base := r * cols
		var sumSq float64
		for c := 0; c < cols; c++ {
			d := float64(a[base+c]) - float64(b[base+c])
			sumSq += d * d
		}
		out[r] = float32(math.Sqrt(sumSq))
	}
	return out, nil
}

// summarizeRowL2 converts a slice of per-row L2 values into percentile stats
// and a top-K largest-error list. topK must be >= 0; if 0 no rows are returned.
// The caller retains ownership of l2; this function does not modify it.
func summarizeRowL2(l2 []float32, cols, topK int) rowDiffStats {
	n := len(l2)
	stats := rowDiffStats{Rows: n, Cols: cols}
	if n == 0 || topK < 0 {
		return stats
	}
	sorted := make([]float32, n)
	copy(sorted, l2)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	stats.P50 = percentile(sorted, 0.50)
	stats.P95 = percentile(sorted, 0.95)
	stats.P99 = percentile(sorted, 0.99)
	stats.P100 = sorted[n-1]

	if topK == 0 {
		return stats
	}
	if topK > n {
		topK = n
	}
	// Build (row, L2) pairs and sort descending by L2, ascending by row on ties.
	pairs := make([]rowDiff, n)
	for i, v := range l2 {
		pairs[i] = rowDiff{Row: i, L2: v}
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].L2 != pairs[j].L2 {
			return pairs[i].L2 > pairs[j].L2
		}
		return pairs[i].Row < pairs[j].Row
	})
	stats.TopK = pairs[:topK]
	return stats
}

// percentile returns the p-th quantile (0..1) of a pre-sorted ascending slice
// using the nearest-rank method: index = ceil(p * n) - 1, clamped to [0, n-1].
func percentile(sorted []float32, p float64) float32 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[n-1]
	}
	idx := int(math.Ceil(p*float64(n))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	return sorted[idx]
}

// writeRowDiffReport prints a human-readable summary of the row-diff statistics
// to w. The caller controls output destination so tests can capture it.
func writeRowDiffReport(w io.Writer, label string, stats rowDiffStats) {
	fmt.Fprintf(w, "ple-embed-diff[%s]: rows=%d cols=%d scoped=%v\n",
		label, stats.Rows, stats.Cols, stats.Scoped)
	fmt.Fprintf(w, "ple-embed-diff[%s]: L2 p50=%.6g p95=%.6g p99=%.6g p100=%.6g\n",
		label, stats.P50, stats.P95, stats.P99, stats.P100)
	for i, rd := range stats.TopK {
		fmt.Fprintf(w, "ple-embed-diff[%s]: top[%02d] row=%d L2=%.6g\n",
			label, i+1, rd.Row, rd.L2)
	}
}

// runPLEEmbedDiff loads `model.ple_embed_tokens.weight` from both a Q4 and a
// Q8 GGUF, dequantizes each to F32 via the standard tensor.Data() path, and
// emits per-row L2 summary statistics + top-20 worst rows. If maxRows > 0 and
// the tensor has more rows than maxRows, only the first maxRows are compared
// (for use with the scoped R99.2.2.A ablation) and the output flags scoped=true.
func runPLEEmbedDiff(q4Path, q8Path string, maxRows int, w io.Writer) error {
	const tensorName = "model.ple_embed_tokens.weight"
	const topK = 20

	fmt.Fprintf(w, "ple-embed-diff: q4=%s\n", q4Path)
	fmt.Fprintf(w, "ple-embed-diff: q8=%s\n", q8Path)

	mdlQ4, err := inference.LoadGGUF(q4Path)
	if err != nil {
		return fmt.Errorf("load q4 GGUF: %w", err)
	}
	mdlQ8, err := inference.LoadGGUF(q8Path)
	if err != nil {
		return fmt.Errorf("load q8 GGUF: %w", err)
	}

	tQ4, ok := mdlQ4.Tensors[tensorName]
	if !ok {
		return fmt.Errorf("q4 GGUF missing tensor %q", tensorName)
	}
	tQ8, ok := mdlQ8.Tensors[tensorName]
	if !ok {
		return fmt.Errorf("q8 GGUF missing tensor %q", tensorName)
	}

	shapeQ4 := tQ4.Shape()
	shapeQ8 := tQ8.Shape()
	if len(shapeQ4) != 2 || len(shapeQ8) != 2 {
		return fmt.Errorf("expected rank-2 tensor, got shapes q4=%v q8=%v", shapeQ4, shapeQ8)
	}
	if shapeQ4[0] != shapeQ8[0] || shapeQ4[1] != shapeQ8[1] {
		return fmt.Errorf("shape mismatch: q4=%v q8=%v", shapeQ4, shapeQ8)
	}
	rows, cols := shapeQ4[0], shapeQ4[1]
	fmt.Fprintf(w, "ple-embed-diff: tensor=%s shape=[%d,%d]\n", tensorName, rows, cols)

	scoped := false
	if maxRows > 0 && maxRows < rows {
		fmt.Fprintf(w, "ple-embed-diff: scoped run -- comparing first %d of %d rows (R99.2.2.A)\n", maxRows, rows)
		rows = maxRows
		scoped = true
	}

	dataQ4 := tQ4.Data()
	dataQ8 := tQ8.Data()

	l2, err := perRowL2(dataQ4, dataQ8, rows, cols)
	if err != nil {
		return err
	}
	stats := summarizeRowL2(l2, cols, topK)
	stats.Scoped = scoped
	writeRowDiffReport(w, "q4_vs_q8", stats)
	return nil
}
