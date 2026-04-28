package causal

import "math"

// correlationMatrix computes the Pearson correlation matrix for data shaped
// [n_samples][n_variables].
func correlationMatrix(data [][]float64) [][]float64 {
	n := len(data)
	p := len(data[0])

	// Compute means.
	mean := make([]float64, p)
	for i := 0; i < n; i++ {
		for j := 0; j < p; j++ {
			mean[j] += data[i][j]
		}
	}
	for j := 0; j < p; j++ {
		mean[j] /= float64(n)
	}

	// Compute covariance matrix.
	cov := make([][]float64, p)
	for i := range cov {
		cov[i] = make([]float64, p)
	}
	for s := 0; s < n; s++ {
		for i := 0; i < p; i++ {
			di := data[s][i] - mean[i]
			for j := i; j < p; j++ {
				cov[i][j] += di * (data[s][j] - mean[j])
			}
		}
	}
	for i := 0; i < p; i++ {
		for j := i; j < p; j++ {
			cov[i][j] /= float64(n - 1)
			cov[j][i] = cov[i][j]
		}
	}

	// Convert to correlation.
	corr := make([][]float64, p)
	for i := range corr {
		corr[i] = make([]float64, p)
		corr[i][i] = 1.0
	}
	for i := 0; i < p; i++ {
		si := math.Sqrt(cov[i][i])
		for j := i + 1; j < p; j++ {
			sj := math.Sqrt(cov[j][j])
			if si > 0 && sj > 0 {
				corr[i][j] = cov[i][j] / (si * sj)
				corr[j][i] = corr[i][j]
			}
		}
	}
	return corr
}

// partialCorrelation computes the partial correlation between variables x and
// y given conditioning set z, using recursive formula on the correlation
// matrix.
func partialCorrelation(corr [][]float64, x, y int, z []int) float64 {
	if len(z) == 0 {
		return corr[x][y]
	}

	// Recursive: r(x,y|z) = (r(x,y|z\last) - r(x,last|z\last)*r(y,last|z\last)) /
	//   sqrt((1-r(x,last|z\last)^2)*(1-r(y,last|z\last)^2))
	last := z[len(z)-1]
	rest := z[:len(z)-1]

	rxyRest := partialCorrelation(corr, x, y, rest)
	rxlRest := partialCorrelation(corr, x, last, rest)
	rylRest := partialCorrelation(corr, y, last, rest)

	denom := math.Sqrt((1 - rxlRest*rxlRest) * (1 - rylRest*rylRest))
	if denom < 1e-15 {
		return 0
	}
	return (rxyRest - rxlRest*rylRest) / denom
}

// fisherZTest tests whether the partial correlation r is significantly
// different from zero given n samples and k conditioning variables. It returns
// true if the variables are conditionally independent at significance level
// alpha.
func fisherZTest(r float64, n, k int, alpha float64) bool {
	// Fisher z-transform: z = 0.5 * ln((1+r)/(1-r))
	// Under H0 (independence), z ~ N(0, 1/(n-k-3))
	dof := n - k - 3
	if dof < 1 {
		return false // not enough samples
	}

	// Clamp r to avoid infinities.
	if r > 0.9999 {
		r = 0.9999
	}
	if r < -0.9999 {
		r = -0.9999
	}

	z := 0.5 * math.Log((1+r)/(1-r))
	stat := math.Abs(z) * math.Sqrt(float64(dof))

	// Two-sided test: compare |stat| with z_{1-alpha/2}.
	threshold := normalQuantile(1 - alpha/2)
	return stat < threshold
}

// normalQuantile approximates the quantile function of the standard normal
// distribution using the rational approximation by Abramowitz & Stegun.
func normalQuantile(p float64) float64 {
	// Handles p in (0,1). Uses the Beasley-Springer-Moro algorithm.
	if p <= 0 || p >= 1 {
		return 0
	}
	if p == 0.5 {
		return 0
	}

	t := p
	if t > 0.5 {
		t = 1 - t
	}
	s := math.Sqrt(-2 * math.Log(t))

	// Rational approximation.
	const (
		c0 = 2.515517
		c1 = 0.802853
		c2 = 0.010328
		d1 = 1.432788
		d2 = 0.189269
		d3 = 0.001308
	)
	q := s - (c0+c1*s+c2*s*s)/(1+d1*s+d2*s*s+d3*s*s*s)

	if p < 0.5 {
		return -q
	}
	return q
}
