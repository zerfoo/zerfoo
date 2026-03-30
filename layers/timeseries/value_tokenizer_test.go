package timeseries

import (
	"math"
	"testing"
)

func TestNewValueTokenizer_Valid(t *testing.T) {
	edges := []float64{0.0, 1.0, 2.0, 3.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}
	if got := vt.NumBins(); got != 3 {
		t.Errorf("NumBins = %d, want 3", got)
	}
}

func TestNewValueTokenizer_InvalidArgs(t *testing.T) {
	tests := []struct {
		name  string
		edges []float64
	}{
		{"too few edges", []float64{1.0}},
		{"empty edges", nil},
		{"non-ascending", []float64{1.0, 0.5, 2.0}},
		{"duplicate edges", []float64{1.0, 1.0, 2.0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewValueTokenizer(tt.edges)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestValueTokenizer_Tokenize(t *testing.T) {
	// 4 bins: [0,1), [1,2), [2,3), [3,4)
	edges := []float64{0.0, 1.0, 2.0, 3.0, 4.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}

	tests := []struct {
		name string
		val  float64
		want int
	}{
		{"below range", -1.0, 0},
		{"at first edge", 0.0, 0},
		{"mid first bin", 0.5, 0},
		{"at second edge", 1.0, 1},
		{"mid second bin", 1.5, 1},
		{"at third edge", 2.0, 2},
		{"mid third bin", 2.5, 2},
		{"at fourth edge", 3.0, 3},
		{"mid fourth bin", 3.5, 3},
		{"at last edge", 4.0, 3},
		{"above range", 5.0, 3},
		{"NaN", math.NaN(), 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := vt.Tokenize(tt.val)
			if got != tt.want {
				t.Errorf("Tokenize(%g) = %d, want %d", tt.val, got, tt.want)
			}
		})
	}
}

func TestValueTokenizer_Detokenize(t *testing.T) {
	// 3 bins: [0,1), [1,2), [2,3) with centers 0.5, 1.5, 2.5
	edges := []float64{0.0, 1.0, 2.0, 3.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}

	tests := []struct {
		bin  int
		want float64
	}{
		{0, 0.5},
		{1, 1.5},
		{2, 2.5},
		{-1, 0.5},  // clamped to 0
		{10, 2.5},  // clamped to numBins-1
	}

	for _, tt := range tests {
		got := vt.Detokenize(tt.bin)
		if math.Abs(got-tt.want) > 1e-10 {
			t.Errorf("Detokenize(%d) = %g, want %g", tt.bin, got, tt.want)
		}
	}
}

func TestValueTokenizer_RoundTrip(t *testing.T) {
	// Verify round-trip: tokenize then detokenize stays within bin width.
	edges := []float64{-2.0, -1.0, 0.0, 1.0, 2.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}

	values := []float64{-1.5, -0.3, 0.0, 0.7, 1.9, -3.0, 5.0}
	for _, v := range values {
		bin := vt.Tokenize(v)
		center := vt.Detokenize(bin)

		// The bin width is 1.0 for all bins here. The maximum error from
		// tokenize→detokenize is half the bin width (0.5) for values
		// inside the range. For out-of-range values the error can be larger
		// but the bin assignment is still correct (clamped to boundary bins).
		if v >= edges[0] && v <= edges[len(edges)-1] {
			binWidth := edges[bin+1] - edges[bin]
			tol := binWidth/2.0 + 1e-10
			if math.Abs(v-center) > tol {
				t.Errorf("round-trip(%g): bin=%d, center=%g, error=%g > tol=%g",
					v, bin, center, math.Abs(v-center), tol)
			}
		}
	}
}

func TestValueTokenizer_TokenizeBatch(t *testing.T) {
	edges := []float64{0.0, 1.0, 2.0, 3.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}

	values := []float64{0.5, 1.5, 2.5}
	got := vt.TokenizeBatch(values)
	want := []int{0, 1, 2}
	if len(got) != len(want) {
		t.Fatalf("TokenizeBatch length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("TokenizeBatch[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestValueTokenizer_DetokenizeBatch(t *testing.T) {
	edges := []float64{0.0, 1.0, 2.0, 3.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}

	bins := []int{0, 1, 2}
	got := vt.DetokenizeBatch(bins)
	want := []float64{0.5, 1.5, 2.5}
	if len(got) != len(want) {
		t.Fatalf("DetokenizeBatch length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-10 {
			t.Errorf("DetokenizeBatch[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func TestValueTokenizer_EdgesAndCenters(t *testing.T) {
	edges := []float64{0.0, 1.0, 3.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}

	gotEdges := vt.Edges()
	if len(gotEdges) != 3 {
		t.Fatalf("Edges length = %d, want 3", len(gotEdges))
	}
	for i, e := range edges {
		if gotEdges[i] != e {
			t.Errorf("Edges[%d] = %g, want %g", i, gotEdges[i], e)
		}
	}

	// Verify edges are a copy (mutation doesn't affect tokenizer).
	gotEdges[0] = 999.0
	if vt.edges[0] == 999.0 {
		t.Error("Edges returned internal slice, not a copy")
	}

	centers := vt.Centers()
	wantCenters := []float64{0.5, 2.0}
	for i, c := range wantCenters {
		if math.Abs(centers[i]-c) > 1e-10 {
			t.Errorf("Centers[%d] = %g, want %g", i, centers[i], c)
		}
	}
}

func TestValueTokenizer_UnevenBins(t *testing.T) {
	// Non-uniform bin widths: [0,0.1), [0.1,0.5), [0.5,10)
	edges := []float64{0.0, 0.1, 0.5, 10.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}

	tests := []struct {
		val  float64
		want int
	}{
		{0.05, 0},
		{0.1, 1},
		{0.3, 1},
		{0.5, 2},
		{5.0, 2},
	}
	for _, tt := range tests {
		got := vt.Tokenize(tt.val)
		if got != tt.want {
			t.Errorf("Tokenize(%g) = %d, want %d", tt.val, got, tt.want)
		}
	}

	// Round-trip within bin width tolerance.
	for _, tt := range tests {
		bin := vt.Tokenize(tt.val)
		center := vt.Detokenize(bin)
		binWidth := edges[bin+1] - edges[bin]
		tol := binWidth/2.0 + 1e-10
		if math.Abs(tt.val-center) > tol {
			t.Errorf("round-trip(%g): center=%g, error=%g > tol=%g",
				tt.val, center, math.Abs(tt.val-center), tol)
		}
	}
}

func TestValueTokenizer_SingleBin(t *testing.T) {
	edges := []float64{0.0, 1.0}
	vt, err := NewValueTokenizer(edges)
	if err != nil {
		t.Fatalf("NewValueTokenizer: %v", err)
	}
	if vt.NumBins() != 1 {
		t.Errorf("NumBins = %d, want 1", vt.NumBins())
	}

	// Everything maps to bin 0.
	for _, v := range []float64{-1.0, 0.0, 0.5, 1.0, 2.0} {
		if got := vt.Tokenize(v); got != 0 {
			t.Errorf("Tokenize(%g) = %d, want 0", v, got)
		}
	}
	if got := vt.Detokenize(0); math.Abs(got-0.5) > 1e-10 {
		t.Errorf("Detokenize(0) = %g, want 0.5", got)
	}
}
