package main

import (
	"bytes"
	"math"
	"strings"
	"testing"
)

func TestPerRowL2_KnownDeltas(t *testing.T) {
	// 4 rows x 3 cols. Construct (a - b) so row L2 norms are
	// row 0: ||(1, 0, 0)|| = 1
	// row 1: ||(3, 4, 0)|| = 5
	// row 2: ||(0, 0, 0)|| = 0
	// row 3: ||(1, 2, 2)|| = 3
	a := []float32{
		1, 0, 0,
		3, 4, 0,
		7, 7, 7,
		1, 2, 2,
	}
	b := []float32{
		0, 0, 0,
		0, 0, 0,
		7, 7, 7,
		0, 0, 0,
	}
	l2, err := perRowL2(a, b, 4, 3)
	if err != nil {
		t.Fatalf("perRowL2: %v", err)
	}
	want := []float32{1, 5, 0, 3}
	if len(l2) != len(want) {
		t.Fatalf("len(l2)=%d want %d", len(l2), len(want))
	}
	for i := range want {
		if diff := math.Abs(float64(l2[i] - want[i])); diff > 1e-5 {
			t.Errorf("row %d: got %.6g want %.6g", i, l2[i], want[i])
		}
	}
}

func TestPerRowL2_ShapeErrors(t *testing.T) {
	if _, err := perRowL2(nil, nil, 0, 3); err == nil {
		t.Error("expected error for rows=0")
	}
	if _, err := perRowL2([]float32{1}, []float32{1}, 1, 0); err == nil {
		t.Error("expected error for cols=0")
	}
	if _, err := perRowL2([]float32{1, 2}, []float32{1, 2, 3}, 1, 3); err == nil {
		t.Error("expected error for short input")
	}
}

func TestSummarizeRowL2_PercentilesAndTopK(t *testing.T) {
	// 10 values chosen so percentile indices are obvious with nearest-rank:
	// sorted: [1 2 3 4 5 6 7 8 9 10].
	// p50: ceil(0.50*10)-1 = 4 -> value 5
	// p95: ceil(0.95*10)-1 = 9 -> value 10 (same as p100 at n=10; expected)
	// p99: ceil(0.99*10)-1 = 9 -> value 10
	// p100: value 10
	l2 := []float32{10, 1, 7, 3, 9, 5, 2, 8, 4, 6}
	stats := summarizeRowL2(l2, 8960, 3)
	if stats.Rows != 10 {
		t.Errorf("Rows = %d, want 10", stats.Rows)
	}
	if stats.Cols != 8960 {
		t.Errorf("Cols = %d, want 8960", stats.Cols)
	}
	if stats.P50 != 5 {
		t.Errorf("P50 = %v, want 5", stats.P50)
	}
	if stats.P95 != 10 {
		t.Errorf("P95 = %v, want 10", stats.P95)
	}
	if stats.P99 != 10 {
		t.Errorf("P99 = %v, want 10", stats.P99)
	}
	if stats.P100 != 10 {
		t.Errorf("P100 = %v, want 10", stats.P100)
	}
	// Top 3 by L2 descending: values 10, 9, 8 at original rows 0, 4, 7.
	wantTop := []rowDiff{{0, 10}, {4, 9}, {7, 8}}
	if len(stats.TopK) != len(wantTop) {
		t.Fatalf("len(TopK) = %d, want %d", len(stats.TopK), len(wantTop))
	}
	for i, rd := range stats.TopK {
		if rd.Row != wantTop[i].Row || rd.L2 != wantTop[i].L2 {
			t.Errorf("TopK[%d] = {row=%d, L2=%v}, want {row=%d, L2=%v}",
				i, rd.Row, rd.L2, wantTop[i].Row, wantTop[i].L2)
		}
	}
}

func TestSummarizeRowL2_TopKClampedToLength(t *testing.T) {
	// Request 50 rows but only 3 are present -- must return exactly 3.
	stats := summarizeRowL2([]float32{0.5, 2.0, 1.0}, 4, 50)
	if len(stats.TopK) != 3 {
		t.Fatalf("len(TopK) = %d, want 3", len(stats.TopK))
	}
	wantOrder := []int{1, 2, 0} // L2=2.0, 1.0, 0.5
	for i, rd := range stats.TopK {
		if rd.Row != wantOrder[i] {
			t.Errorf("TopK[%d].Row = %d, want %d", i, rd.Row, wantOrder[i])
		}
	}
}

func TestSummarizeRowL2_ZeroTopK(t *testing.T) {
	stats := summarizeRowL2([]float32{1, 2, 3, 4, 5}, 2, 0)
	if stats.TopK != nil && len(stats.TopK) != 0 {
		t.Errorf("TopK must be empty when topK=0, got %v", stats.TopK)
	}
	if stats.P100 != 5 {
		t.Errorf("P100 = %v, want 5", stats.P100)
	}
}

func TestSummarizeRowL2_TiedL2SortsByRow(t *testing.T) {
	// Two rows share the same max L2 -- the smaller row index must come first.
	stats := summarizeRowL2([]float32{2, 2, 1}, 1, 2)
	if len(stats.TopK) != 2 {
		t.Fatalf("len(TopK) = %d, want 2", len(stats.TopK))
	}
	if stats.TopK[0].Row != 0 || stats.TopK[1].Row != 1 {
		t.Errorf("tie-break wrong: got rows %d, %d; want 0, 1",
			stats.TopK[0].Row, stats.TopK[1].Row)
	}
}

func TestSummarizeRowL2_EmptyInput(t *testing.T) {
	stats := summarizeRowL2(nil, 4, 10)
	if stats.Rows != 0 || stats.P100 != 0 || len(stats.TopK) != 0 {
		t.Errorf("empty summary mishandled: %+v", stats)
	}
}

func TestWriteRowDiffReport_ContainsExpectedFields(t *testing.T) {
	stats := summarizeRowL2([]float32{0.1, 0.3, 0.2}, 8960, 2)
	stats.Scoped = true
	var buf bytes.Buffer
	writeRowDiffReport(&buf, "q4_vs_q8", stats)
	out := buf.String()

	for _, want := range []string{
		"q4_vs_q8",
		"rows=3",
		"cols=8960",
		"scoped=true",
		"p50=",
		"p95=",
		"p99=",
		"p100=",
		"top[01]",
		"top[02]",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q; full output:\n%s", want, out)
		}
	}
	// Top entries must list row indices of the two largest values: row 1 (0.3) then row 2 (0.2).
	if !strings.Contains(out, "row=1") || !strings.Contains(out, "row=2") {
		t.Errorf("top rows missing from output:\n%s", out)
	}
}
