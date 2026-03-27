package transmla

import (
	"math"
	"testing"
)

func TestTruncatedSVD_Roundtrip(t *testing.T) {
	// A simple 3×2 matrix with known SVD properties.
	a := [][]float64{
		{3, 0},
		{0, 4},
		{0, 0},
	}
	svd, err := TruncatedSVD(a, 2)
	if err != nil {
		t.Fatalf("TruncatedSVD: %v", err)
	}

	// Reconstruct A from U·diag(S)·Vt and check.
	m, n := len(a), len(a[0])
	rank := len(svd.S)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var val float64
			for r := 0; r < rank; r++ {
				val += svd.U[i][r] * svd.S[r] * svd.Vt[r][j]
			}
			if diff := math.Abs(val - a[i][j]); diff > 1e-10 {
				t.Errorf("reconstruct[%d][%d] = %g, want %g (diff %g)", i, j, val, a[i][j], diff)
			}
		}
	}

	// Singular values should be 4 and 3 (descending).
	wantS := []float64{4, 3}
	for i, want := range wantS {
		if diff := math.Abs(svd.S[i] - want); diff > 1e-10 {
			t.Errorf("S[%d] = %g, want %g", i, svd.S[i], want)
		}
	}
}

func TestTruncatedSVD_Truncation(t *testing.T) {
	// 3×3 matrix, truncate to rank 1.
	a := [][]float64{
		{1, 0, 0},
		{0, 2, 0},
		{0, 0, 3},
	}
	svd, err := TruncatedSVD(a, 1)
	if err != nil {
		t.Fatalf("TruncatedSVD: %v", err)
	}
	if len(svd.S) != 1 {
		t.Fatalf("expected 1 singular value, got %d", len(svd.S))
	}
	if diff := math.Abs(svd.S[0] - 3); diff > 1e-10 {
		t.Errorf("S[0] = %g, want 3", svd.S[0])
	}
}

func TestTruncatedSVD_Errors(t *testing.T) {
	tests := []struct {
		name   string
		matrix [][]float64
		rank   int
	}{
		{"empty", [][]float64{}, 1},
		{"zero columns", [][]float64{{}}, 1},
		{"ragged", [][]float64{{1, 2}, {3}}, 1},
		{"rank zero", [][]float64{{1}}, 0},
		{"rank too large", [][]float64{{1, 2}}, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := TruncatedSVD(tc.matrix, tc.rank)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestDecomposeKVProjection_RoundTrip(t *testing.T) {
	// W_K (2×4) and W_V (2×4), full rank decomposition (rank 4).
	wK := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	}
	wV := [][]float64{
		{9, 10, 11, 12},
		{13, 14, 15, 16},
	}

	wDKV, wUK, wUV, err := DecomposeKVProjection(wK, wV, 4)
	if err != nil {
		t.Fatalf("DecomposeKVProjection: %v", err)
	}

	// Reconstruct W_K ≈ wUK · wDKV^T and check.
	dModel := len(wK[0])
	rank := len(wDKV[0])

	for i := 0; i < len(wK); i++ {
		for j := 0; j < dModel; j++ {
			var val float64
			for r := 0; r < rank; r++ {
				val += wUK[i][r] * wDKV[j][r]
			}
			if diff := math.Abs(val - wK[i][j]); diff > 1e-8 {
				t.Errorf("wK_reconstructed[%d][%d] = %g, want %g (diff %g)", i, j, val, wK[i][j], diff)
			}
		}
	}

	// Reconstruct W_V ≈ wUV · wDKV^T and check.
	for i := 0; i < len(wV); i++ {
		for j := 0; j < dModel; j++ {
			var val float64
			for r := 0; r < rank; r++ {
				val += wUV[i][r] * wDKV[j][r]
			}
			if diff := math.Abs(val - wV[i][j]); diff > 1e-8 {
				t.Errorf("wV_reconstructed[%d][%d] = %g, want %g (diff %g)", i, j, val, wV[i][j], diff)
			}
		}
	}
}

func TestDecomposeKVProjection_LowRank(t *testing.T) {
	// Create a rank-2 matrix embedded in 4×4 so rank-2 truncation is exact.
	wK := [][]float64{
		{1, 2, 3, 4},
		{2, 4, 6, 8},
	}
	wV := [][]float64{
		{3, 6, 9, 12},
		{4, 8, 12, 16},
	}

	wDKV, wUK, wUV, err := DecomposeKVProjection(wK, wV, 1)
	if err != nil {
		t.Fatalf("DecomposeKVProjection: %v", err)
	}

	// This is a rank-1 matrix (all rows are multiples of [1,2,3,4]),
	// so rank-1 decomposition should be nearly exact.
	original := append(wK, wV...)
	relErr := ReconstructionError(original, wDKV, wUK, wUV)
	if relErr > 1e-10 {
		t.Errorf("reconstruction error for rank-1 matrix = %g, want < 1e-10", relErr)
	}
}

func TestReconstructionError_DecreasesWithRank(t *testing.T) {
	// A general 4×3 matrix.
	a := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 10},
		{11, 13, 14},
	}
	wK := a[:2]
	wV := a[2:]

	var prevErr float64 = math.Inf(1)
	for rank := 1; rank <= 3; rank++ {
		wDKV, wUK, wUV, err := DecomposeKVProjection(wK, wV, rank)
		if err != nil {
			t.Fatalf("rank %d: %v", rank, err)
		}
		relErr := ReconstructionError(a, wDKV, wUK, wUV)
		if relErr > prevErr+1e-12 {
			t.Errorf("rank %d error %g > rank %d error %g", rank, relErr, rank-1, prevErr)
		}
		prevErr = relErr
	}
	// At full rank (3), error should be near zero.
	if prevErr > 1e-10 {
		t.Errorf("full rank error = %g, want < 1e-10", prevErr)
	}
}

func TestReconstructionError_ZeroMatrix(t *testing.T) {
	original := [][]float64{
		{0, 0},
		{0, 0},
	}
	wDKV := [][]float64{{0}, {0}}
	wUK := [][]float64{{0}}
	wUV := [][]float64{{0}}
	got := ReconstructionError(original, wDKV, wUK, wUV)
	if got != 0 {
		t.Errorf("zero matrix error = %g, want 0", got)
	}
}

func TestDecomposeKVProjection_Errors(t *testing.T) {
	tests := []struct {
		name string
		wK   [][]float64
		wV   [][]float64
		rank int
	}{
		{"empty wK", [][]float64{}, [][]float64{{1}}, 1},
		{"empty wV", [][]float64{{1}}, [][]float64{}, 1},
		{"mismatched cols", [][]float64{{1, 2}}, [][]float64{{3}}, 1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, _, _, err := DecomposeKVProjection(tc.wK, tc.wV, tc.rank)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestSVD_WideMatrix(t *testing.T) {
	// 2×4 matrix (wider than tall).
	a := [][]float64{
		{1, 2, 3, 4},
		{5, 6, 7, 8},
	}
	svd, err := TruncatedSVD(a, 2)
	if err != nil {
		t.Fatalf("TruncatedSVD: %v", err)
	}

	// Reconstruct and verify.
	for i := 0; i < 2; i++ {
		for j := 0; j < 4; j++ {
			var val float64
			for r := 0; r < 2; r++ {
				val += svd.U[i][r] * svd.S[r] * svd.Vt[r][j]
			}
			if diff := math.Abs(val - a[i][j]); diff > 1e-10 {
				t.Errorf("reconstruct[%d][%d] = %g, want %g", i, j, val, a[i][j])
			}
		}
	}
}
