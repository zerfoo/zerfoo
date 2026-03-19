package regime

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Config controls HMM training behaviour.
type Config struct {
	// NStates is the number of hidden states (regimes). Must be >= 2.
	NStates int

	// MaxIterations is the maximum number of Baum-Welch EM iterations.
	// Default: 100.
	MaxIterations int

	// Tolerance is the log-likelihood convergence threshold. Training stops
	// when the absolute change in log-likelihood is below this value.
	// Default: 1e-6.
	Tolerance float64

	// Seed for random initialisation. A value of 0 uses a fixed default seed.
	Seed int64
}

// HMM is a Hidden Markov Model with Gaussian emissions for regime
// classification. Hidden states represent distinct regimes (e.g. trending,
// mean-reverting, volatile, crash).
type HMM struct {
	nStates int

	// Initial state probabilities π[i] = P(s₁ = i).
	Initial []float64

	// Transition matrix A[i][j] = P(sₜ₊₁ = j | sₜ = i).
	Transition [][]float64

	// Emission parameters: Gaussian mean and variance per state.
	Means     []float64
	Variances []float64
}

// NewHMM creates a new HMM with the given configuration. Parameters are
// initialised randomly.
func NewHMM(cfg Config) (*HMM, error) {
	if cfg.NStates < 2 {
		return nil, fmt.Errorf("regime: NStates must be >= 2, got %d", cfg.NStates)
	}

	seed := cfg.Seed
	if seed == 0 {
		seed = 42
	}
	rng := rand.New(rand.NewSource(seed))

	n := cfg.NStates
	h := &HMM{
		nStates:    n,
		Initial:    make([]float64, n),
		Transition: make([][]float64, n),
		Means:      make([]float64, n),
		Variances:  make([]float64, n),
	}

	// Random initial probabilities (Dirichlet-like via exponentials).
	sum := 0.0
	for i := 0; i < n; i++ {
		h.Initial[i] = rng.ExpFloat64()
		sum += h.Initial[i]
	}
	for i := 0; i < n; i++ {
		h.Initial[i] /= sum
	}

	// Random transition matrix (row-stochastic).
	for i := 0; i < n; i++ {
		h.Transition[i] = make([]float64, n)
		s := 0.0
		for j := 0; j < n; j++ {
			h.Transition[i][j] = rng.ExpFloat64()
			s += h.Transition[i][j]
		}
		for j := 0; j < n; j++ {
			h.Transition[i][j] /= s
		}
	}

	// Spread initial means across the observation range (will be adjusted
	// during training). Use distinct offsets so states don't collapse.
	for i := 0; i < n; i++ {
		h.Means[i] = float64(i) - float64(n-1)/2.0 + 0.1*rng.NormFloat64()
		h.Variances[i] = 1.0 + rng.Float64()
	}

	return h, nil
}

// NStates returns the number of hidden states.
func (h *HMM) NStates() int { return h.nStates }

// gaussianPDF returns the probability density of x under N(mu, variance).
func gaussianPDF(x, mu, variance float64) float64 {
	d := x - mu
	return math.Exp(-0.5*d*d/variance) / math.Sqrt(2*math.Pi*variance)
}

// forward computes the scaled forward variables and per-step scaling factors.
// alpha[t][i] is the scaled forward probability at time t in state i.
// scales[t] normalises alpha at each step to prevent underflow.
// The log-likelihood of the observation sequence is sum(log(scales[t])).
func (h *HMM) forward(obs []float64) (alpha [][]float64, scales []float64) {
	T := len(obs)
	n := h.nStates

	alpha = make([][]float64, T)
	scales = make([]float64, T)

	// t = 0
	alpha[0] = make([]float64, n)
	for i := 0; i < n; i++ {
		alpha[0][i] = h.Initial[i] * gaussianPDF(obs[0], h.Means[i], h.Variances[i])
		scales[0] += alpha[0][i]
	}
	if scales[0] == 0 {
		scales[0] = 1e-300
	}
	for i := 0; i < n; i++ {
		alpha[0][i] /= scales[0]
	}

	// t = 1..T-1
	for t := 1; t < T; t++ {
		alpha[t] = make([]float64, n)
		for j := 0; j < n; j++ {
			s := 0.0
			for i := 0; i < n; i++ {
				s += alpha[t-1][i] * h.Transition[i][j]
			}
			alpha[t][j] = s * gaussianPDF(obs[t], h.Means[j], h.Variances[j])
			scales[t] += alpha[t][j]
		}
		if scales[t] == 0 {
			scales[t] = 1e-300
		}
		for j := 0; j < n; j++ {
			alpha[t][j] /= scales[t]
		}
	}
	return
}

// backward computes the scaled backward variables using the same scaling
// factors from the forward pass.
func (h *HMM) backward(obs []float64, scales []float64) [][]float64 {
	T := len(obs)
	n := h.nStates

	beta := make([][]float64, T)
	beta[T-1] = make([]float64, n)
	for i := 0; i < n; i++ {
		beta[T-1][i] = 1.0
	}

	for t := T - 2; t >= 0; t-- {
		beta[t] = make([]float64, n)
		for i := 0; i < n; i++ {
			s := 0.0
			for j := 0; j < n; j++ {
				s += h.Transition[i][j] * gaussianPDF(obs[t+1], h.Means[j], h.Variances[j]) * beta[t+1][j]
			}
			beta[t][i] = s / scales[t+1]
		}
	}
	return beta
}

// Fit trains the HMM on the observation sequence using the Baum-Welch algorithm.
// It returns the final log-likelihood and number of iterations performed.
func (h *HMM) Fit(obs []float64, cfg Config) (logLikelihood float64, iterations int, err error) {
	if len(obs) < 2 {
		return 0, 0, fmt.Errorf("regime: observation sequence must have at least 2 elements")
	}

	maxIter := cfg.MaxIterations
	if maxIter <= 0 {
		maxIter = 100
	}
	tol := cfg.Tolerance
	if tol <= 0 {
		tol = 1e-6
	}

	// Initialise emission means from data quantiles so states start near
	// distinct regions of the observation distribution.
	h.initEmissionsFromData(obs)

	T := len(obs)
	n := h.nStates
	prevLL := math.Inf(-1)

	for iter := 0; iter < maxIter; iter++ {
		// E-step: forward-backward.
		alpha, scales := h.forward(obs)
		beta := h.backward(obs, scales)

		// Log-likelihood from scaling factors.
		ll := 0.0
		for t := 0; t < T; t++ {
			ll += math.Log(scales[t])
		}

		// Convergence check.
		if iter > 0 && math.Abs(ll-prevLL) < tol {
			return ll, iter + 1, nil
		}
		prevLL = ll

		// Compute gamma[t][i] = P(sₜ = i | O, λ).
		gamma := make([][]float64, T)
		for t := 0; t < T; t++ {
			gamma[t] = make([]float64, n)
			s := 0.0
			for i := 0; i < n; i++ {
				gamma[t][i] = alpha[t][i] * beta[t][i]
				s += gamma[t][i]
			}
			if s > 0 {
				for i := 0; i < n; i++ {
					gamma[t][i] /= s
				}
			}
		}

		// Compute xi[t][i][j] = P(sₜ = i, sₜ₊₁ = j | O, λ).
		xi := make([][][]float64, T-1)
		for t := 0; t < T-1; t++ {
			xi[t] = make([][]float64, n)
			s := 0.0
			for i := 0; i < n; i++ {
				xi[t][i] = make([]float64, n)
				for j := 0; j < n; j++ {
					xi[t][i][j] = alpha[t][i] * h.Transition[i][j] *
						gaussianPDF(obs[t+1], h.Means[j], h.Variances[j]) * beta[t+1][j]
					s += xi[t][i][j]
				}
			}
			if s > 0 {
				for i := 0; i < n; i++ {
					for j := 0; j < n; j++ {
						xi[t][i][j] /= s
					}
				}
			}
		}

		// M-step: re-estimate parameters.
		// Initial probabilities.
		for i := 0; i < n; i++ {
			h.Initial[i] = gamma[0][i]
		}

		// Transition matrix.
		for i := 0; i < n; i++ {
			gammaSum := 0.0
			for t := 0; t < T-1; t++ {
				gammaSum += gamma[t][i]
			}
			for j := 0; j < n; j++ {
				xiSum := 0.0
				for t := 0; t < T-1; t++ {
					xiSum += xi[t][i][j]
				}
				if gammaSum > 0 {
					h.Transition[i][j] = xiSum / gammaSum
				}
			}
		}

		// Emission parameters (Gaussian mean and variance).
		for i := 0; i < n; i++ {
			gammaSum := 0.0
			meanNum := 0.0
			for t := 0; t < T; t++ {
				gammaSum += gamma[t][i]
				meanNum += gamma[t][i] * obs[t]
			}
			if gammaSum > 0 {
				h.Means[i] = meanNum / gammaSum
				varNum := 0.0
				for t := 0; t < T; t++ {
					d := obs[t] - h.Means[i]
					varNum += gamma[t][i] * d * d
				}
				v := varNum / gammaSum
				if v < 1e-6 {
					v = 1e-6
				}
				h.Variances[i] = v
			}
		}

		iterations = iter + 1
		logLikelihood = ll
	}

	return logLikelihood, iterations, nil
}

// initEmissionsFromData sets initial emission means to data percentiles so
// that distinct regimes start in distinct regions of the observation space.
func (h *HMM) initEmissionsFromData(obs []float64) {
	n := h.nStates
	T := len(obs)

	// Sort a copy of the data to pick percentile-based initial means.
	sorted := make([]float64, T)
	copy(sorted, obs)
	sortFloat64s(sorted)

	for i := 0; i < n; i++ {
		// Pick the value at the (i+0.5)/n quantile.
		frac := (float64(i) + 0.5) / float64(n)
		idx := int(frac * float64(T-1))
		if idx >= T {
			idx = T - 1
		}
		h.Means[i] = sorted[idx]
	}

	// Initialise variances: use the variance of each state's nearest
	// segment of sorted data, falling back to global variance.
	dataVar := 0.0
	mean := 0.0
	for _, v := range obs {
		mean += v
	}
	mean /= float64(T)
	for _, v := range obs {
		d := v - mean
		dataVar += d * d
	}
	dataVar /= float64(T)
	if dataVar < 1e-6 {
		dataVar = 1.0
	}
	segSize := T / n
	if segSize < 2 {
		segSize = 2
	}
	for i := 0; i < n; i++ {
		start := i * (T / n)
		end := start + segSize
		if end > T {
			end = T
		}
		segMean := 0.0
		for _, v := range sorted[start:end] {
			segMean += v
		}
		segMean /= float64(end - start)
		segVar := 0.0
		for _, v := range sorted[start:end] {
			d := v - segMean
			segVar += d * d
		}
		segVar /= float64(end - start)
		if segVar < 1e-6 {
			segVar = dataVar / float64(n)
		}
		h.Variances[i] = segVar
	}
}

// sortFloat64s sorts a slice of float64 in ascending order.
func sortFloat64s(a []float64) {
	sort.Float64s(a)
}

// Viterbi returns the most likely hidden state sequence for the given
// observations using the Viterbi algorithm.
func (h *HMM) Viterbi(obs []float64) ([]int, error) {
	if len(obs) == 0 {
		return nil, fmt.Errorf("regime: observation sequence must not be empty")
	}

	T := len(obs)
	n := h.nStates

	// delta[t][i] = max log-probability of any path ending in state i at time t.
	delta := make([][]float64, T)
	psi := make([][]int, T)

	// t = 0
	delta[0] = make([]float64, n)
	psi[0] = make([]int, n)
	for i := 0; i < n; i++ {
		lp := math.Log(h.Initial[i]) + logGaussianPDF(obs[0], h.Means[i], h.Variances[i])
		delta[0][i] = lp
	}

	// t = 1..T-1
	for t := 1; t < T; t++ {
		delta[t] = make([]float64, n)
		psi[t] = make([]int, n)
		for j := 0; j < n; j++ {
			best := math.Inf(-1)
			bestI := 0
			for i := 0; i < n; i++ {
				v := delta[t-1][i] + math.Log(h.Transition[i][j])
				if v > best {
					best = v
					bestI = i
				}
			}
			delta[t][j] = best + logGaussianPDF(obs[t], h.Means[j], h.Variances[j])
			psi[t][j] = bestI
		}
	}

	// Backtrack.
	states := make([]int, T)
	best := math.Inf(-1)
	for i := 0; i < n; i++ {
		if delta[T-1][i] > best {
			best = delta[T-1][i]
			states[T-1] = i
		}
	}
	for t := T - 2; t >= 0; t-- {
		states[t] = psi[t+1][states[t+1]]
	}

	return states, nil
}

// Predict returns the most likely regime (hidden state index) for the final
// observation in the sequence, along with the posterior probabilities for
// each state.
func (h *HMM) Predict(obs []float64) (state int, probs []float64, err error) {
	if len(obs) == 0 {
		return 0, nil, fmt.Errorf("regime: observation sequence must not be empty")
	}

	alpha, _ := h.forward(obs)
	T := len(obs)
	n := h.nStates

	probs = make([]float64, n)
	s := 0.0
	for i := 0; i < n; i++ {
		probs[i] = alpha[T-1][i]
		s += probs[i]
	}
	if s > 0 {
		for i := 0; i < n; i++ {
			probs[i] /= s
		}
	}

	best := -1.0
	for i := 0; i < n; i++ {
		if probs[i] > best {
			best = probs[i]
			state = i
		}
	}
	return state, probs, nil
}

// logGaussianPDF returns the log probability density of x under N(mu, variance).
func logGaussianPDF(x, mu, variance float64) float64 {
	d := x - mu
	return -0.5*math.Log(2*math.Pi*variance) - 0.5*d*d/variance
}

// LogLikelihood computes the log-likelihood of the observation sequence under
// the current model parameters.
func (h *HMM) LogLikelihood(obs []float64) (float64, error) {
	if len(obs) == 0 {
		return 0, fmt.Errorf("regime: observation sequence must not be empty")
	}
	_, scales := h.forward(obs)
	ll := 0.0
	for _, s := range scales {
		ll += math.Log(s)
	}
	return ll, nil
}
