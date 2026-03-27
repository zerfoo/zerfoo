// Package transmla converts standard multi-head attention (MHA) weights
// into multi-head latent attention (MLA) form via truncated SVD.
//
// The conversion decomposes the concatenated key-value projection matrix
// W_KV = [W_K; W_V] into three smaller matrices:
//
//	wDKV  (d_model × rank)   — shared down-projection
//	wUK   (rank × d_k)       — key up-projection
//	wUV   (rank × d_v)       — value up-projection
//
// such that W_KV ≈ [wUK; wUV] · wDKV^T, reducing KV cache size from
// (d_k + d_v) to rank per token.
package transmla

import (
	"errors"
	"math"
)

// SVDResult holds the full or truncated singular value decomposition of a
// matrix A = U · diag(S) · Vt.
type SVDResult struct {
	// U is an m×k matrix of left singular vectors.
	U [][]float64
	// S is a length-k vector of singular values in descending order.
	S []float64
	// Vt is a k×n matrix of right singular vectors (transposed).
	Vt [][]float64
}

// TruncatedSVD computes a rank-r truncated SVD of an m×n matrix using the
// one-sided Jacobi method. The full SVD is computed first, then truncated
// to the top-r singular values/vectors.
//
// The input matrix is a row-major m×n slice of slices. All rows must have
// the same length. Rank must be positive and at most min(m, n).
func TruncatedSVD(matrix [][]float64, rank int) (*SVDResult, error) {
	m := len(matrix)
	if m == 0 {
		return nil, errors.New("transmla: empty matrix")
	}
	n := len(matrix[0])
	if n == 0 {
		return nil, errors.New("transmla: matrix has zero columns")
	}
	for i := range matrix {
		if len(matrix[i]) != n {
			return nil, errors.New("transmla: ragged matrix")
		}
	}
	minDim := m
	if n < minDim {
		minDim = n
	}
	if rank <= 0 || rank > minDim {
		return nil, errors.New("transmla: rank out of range")
	}

	// Compute A^T A (n×n) for the one-sided Jacobi approach.
	// We work with the Gram matrix and extract singular values/vectors.
	u, s, vt := jacobiSVD(matrix, m, n)

	// Truncate to the requested rank.
	for i := range u {
		u[i] = u[i][:rank]
	}
	s = s[:rank]
	vt = vt[:rank]

	return &SVDResult{U: u, S: s, Vt: vt}, nil
}

// DecomposeKVProjection takes key and value projection weight matrices and
// decomposes their vertical concatenation [W_K; W_V] via truncated SVD.
//
// W_K is (d_k × d_model) and W_V is (d_v × d_model). The concatenated
// matrix is ((d_k+d_v) × d_model).
//
// Returns:
//   - wDKV: (d_model × rank) shared down-projection
//   - wUK:  (d_k × rank)     key up-projection
//   - wUV:  (d_v × rank)     value up-projection
//
// such that W_K ≈ wUK · wDKV^T and W_V ≈ wUV · wDKV^T.
func DecomposeKVProjection(wK, wV [][]float64, rank int) (wDKV, wUK, wUV [][]float64, err error) {
	if len(wK) == 0 || len(wV) == 0 {
		return nil, nil, nil, errors.New("transmla: empty projection matrix")
	}
	dModel := len(wK[0])
	if dModel == 0 {
		return nil, nil, nil, errors.New("transmla: zero-width projection matrix")
	}
	for _, row := range wV {
		if len(row) != dModel {
			return nil, nil, nil, errors.New("transmla: W_K and W_V must have same number of columns")
		}
	}

	dK := len(wK)
	dV := len(wV)

	// Concatenate vertically: wKV = [W_K; W_V], shape (dK+dV) × dModel.
	wKV := make([][]float64, dK+dV)
	for i := 0; i < dK; i++ {
		row := make([]float64, dModel)
		copy(row, wK[i])
		wKV[i] = row
	}
	for i := 0; i < dV; i++ {
		row := make([]float64, dModel)
		copy(row, wV[i])
		wKV[dK+i] = row
	}

	// SVD: wKV = U · diag(S) · Vt
	// wKV is (dK+dV) × dModel.
	svd, err := TruncatedSVD(wKV, rank)
	if err != nil {
		return nil, nil, nil, err
	}

	// Reconstruct: wKV ≈ (U · diag(S)) · Vt
	// Let wDKV^T = Vt (rank × dModel), so wDKV = Vt^T (dModel × rank).
	// Let [wUK; wUV] = U · diag(S).
	//
	// wDKV: dModel × rank (transpose of Vt).
	wDKV = make([][]float64, dModel)
	for j := 0; j < dModel; j++ {
		row := make([]float64, rank)
		for r := 0; r < rank; r++ {
			row[r] = svd.Vt[r][j]
		}
		wDKV[j] = row
	}

	// U · diag(S) split into key and value parts.
	wUK = make([][]float64, dK)
	for i := 0; i < dK; i++ {
		row := make([]float64, rank)
		for r := 0; r < rank; r++ {
			row[r] = svd.U[i][r] * svd.S[r]
		}
		wUK[i] = row
	}
	wUV = make([][]float64, dV)
	for i := 0; i < dV; i++ {
		row := make([]float64, rank)
		for r := 0; r < rank; r++ {
			row[r] = svd.U[dK+i][r] * svd.S[r]
		}
		wUV[i] = row
	}

	return wDKV, wUK, wUV, nil
}

// ReconstructionError computes the relative Frobenius norm error between the
// original concatenated [W_K; W_V] matrix and its low-rank reconstruction
// from wDKV, wUK, wUV: reconstructed = [wUK; wUV] · wDKV^T.
//
// Returns ||original - reconstructed||_F / ||original||_F.
func ReconstructionError(original [][]float64, wDKV, wUK, wUV [][]float64) float64 {
	m := len(original)
	if m == 0 {
		return 0
	}
	n := len(original[0])
	dK := len(wUK)

	// Combine [wUK; wUV] into a single (m × rank) matrix.
	combined := make([][]float64, m)
	for i := 0; i < dK; i++ {
		combined[i] = wUK[i]
	}
	for i := dK; i < m; i++ {
		combined[i] = wUV[i-dK]
	}

	rank := len(wDKV[0])
	var errNorm, origNorm float64
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			// reconstructed[i][j] = sum_r combined[i][r] * wDKV[j][r]
			var val float64
			for r := 0; r < rank; r++ {
				val += combined[i][r] * wDKV[j][r]
			}
			diff := original[i][j] - val
			errNorm += diff * diff
			origNorm += original[i][j] * original[i][j]
		}
	}
	if origNorm == 0 {
		return 0
	}
	return math.Sqrt(errNorm / origNorm)
}

// jacobiSVD computes the full SVD of an m×n matrix A using one-sided Jacobi
// rotations. Returns U (m×k), S (k), Vt (k×n) with k = min(m,n),
// singular values in descending order.
//
// For wide matrices (m < n), the algorithm transposes A, computes the SVD of
// A^T (which is tall), and swaps U/V in the result.
func jacobiSVD(a [][]float64, m, n int) ([][]float64, []float64, [][]float64) {
	// For wide matrices, compute SVD(A^T) and transpose the result.
	// SVD(A^T) = U' S V'^T means A = V' S U'^T, so U=V', Vt=U'^T.
	if m < n {
		at := make([][]float64, n)
		for j := 0; j < n; j++ {
			at[j] = make([]float64, m)
			for i := 0; i < m; i++ {
				at[j][i] = a[i][j]
			}
		}
		uT, sT, vtT := jacobiSVDTall(at, n, m)
		// A = V' S U'^T → U = V'^T transposed back, Vt = U'^T
		k := m // min(m,n) = m since m < n
		u := make([][]float64, m)
		for i := 0; i < m; i++ {
			u[i] = make([]float64, k)
			for j := 0; j < k; j++ {
				u[i][j] = vtT[j][i]
			}
		}
		vt := make([][]float64, k)
		for i := 0; i < k; i++ {
			vt[i] = make([]float64, n)
			for j := 0; j < n; j++ {
				vt[i][j] = uT[j][i]
			}
		}
		return u, sT, vt
	}
	return jacobiSVDTall(a, m, n)
}

// jacobiSVDTall computes the SVD of a tall (m >= n) matrix using one-sided
// Jacobi rotations on columns. Returns U (m×n), S (n), Vt (n×n).
func jacobiSVDTall(a [][]float64, m, n int) ([][]float64, []float64, [][]float64) {
	// Copy A into a working matrix.
	w := make([][]float64, m)
	for i := 0; i < m; i++ {
		w[i] = make([]float64, n)
		copy(w[i], a[i])
	}

	// V accumulates the right rotations, starts as I_n.
	v := eye(n)

	// One-sided Jacobi: apply rotations to columns of W (and V) to
	// diagonalise W^T W.
	const maxSweeps = 100
	const tol = 1e-15

	for sweep := 0; sweep < maxSweeps; sweep++ {
		converged := true
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				// Compute 2×2 sub-problem for columns p, q.
				var app, aqq, apq float64
				for i := 0; i < m; i++ {
					app += w[i][p] * w[i][p]
					aqq += w[i][q] * w[i][q]
					apq += w[i][p] * w[i][q]
				}
				if math.Abs(apq) <= tol*math.Sqrt(app*aqq) {
					continue
				}
				converged = false

				// Compute Jacobi rotation angle.
				tau := (aqq - app) / (2 * apq)
				var t float64
				if tau >= 0 {
					t = 1.0 / (tau + math.Sqrt(1+tau*tau))
				} else {
					t = -1.0 / (-tau + math.Sqrt(1+tau*tau))
				}
				c := 1.0 / math.Sqrt(1+t*t)
				s := t * c

				// Rotate columns p, q in W.
				for i := 0; i < m; i++ {
					wp := w[i][p]
					wq := w[i][q]
					w[i][p] = c*wp - s*wq
					w[i][q] = s*wp + c*wq
				}
				// Rotate columns p, q in V.
				for i := 0; i < n; i++ {
					vp := v[i][p]
					vq := v[i][q]
					v[i][p] = c*vp - s*vq
					v[i][q] = s*vp + c*vq
				}
			}
		}
		if converged {
			break
		}
	}

	// Extract singular values as column norms of W, and normalise columns
	// to get U.
	sigma := make([]float64, n)
	uMat := make([][]float64, m)
	for i := 0; i < m; i++ {
		uMat[i] = make([]float64, n)
	}

	for j := 0; j < n; j++ {
		var norm float64
		for i := 0; i < m; i++ {
			norm += w[i][j] * w[i][j]
		}
		norm = math.Sqrt(norm)
		sigma[j] = norm
		if norm > 0 {
			for i := 0; i < m; i++ {
				uMat[i][j] = w[i][j] / norm
			}
		}
	}

	// Sort singular values descending and permute U, V columns accordingly.
	for i := 0; i < n-1; i++ {
		maxIdx := i
		for j := i + 1; j < n; j++ {
			if sigma[j] > sigma[maxIdx] {
				maxIdx = j
			}
		}
		if maxIdx != i {
			sigma[i], sigma[maxIdx] = sigma[maxIdx], sigma[i]
			for r := 0; r < m; r++ {
				uMat[r][i], uMat[r][maxIdx] = uMat[r][maxIdx], uMat[r][i]
			}
			for r := 0; r < n; r++ {
				v[r][i], v[r][maxIdx] = v[r][maxIdx], v[r][i]
			}
		}
	}

	// Build Vt (n×n) from V (n×n)^T.
	vt := make([][]float64, n)
	for i := 0; i < n; i++ {
		vt[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			vt[i][j] = v[j][i]
		}
	}

	return uMat, sigma, vt
}

// eye returns an n×n identity matrix.
func eye(n int) [][]float64 {
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
		m[i][i] = 1
	}
	return m
}
