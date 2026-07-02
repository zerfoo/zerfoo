// Package automl provides automated machine learning utilities including
// Bayesian hyperparameter optimization.
package automl

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"sync"
)

// HParam describes a single hyperparameter to optimize.
type HParam struct {
	Name  string
	Min   float64
	Max   float64
	IsLog bool // if true, optimize in log space
}

// Trial records one evaluation of a hyperparameter configuration.
type Trial struct {
	ID     int
	Params map[string]float64
	Score  float64
	Done   bool
}

// BayesianOptimizer performs Bayesian hyperparameter optimization using
// random exploration followed by Expected Improvement over a surrogate.
type BayesianOptimizer struct {
	mu     sync.Mutex
	params []HParam
	rng    *rand.Rand
	trials []Trial
	nextID int
}

// NewBayesianOptimizer creates a new optimizer for the given hyperparameters.
func NewBayesianOptimizer(params []HParam, seed int64) *BayesianOptimizer {
	return &BayesianOptimizer{
		params: params,
		rng:    rand.New(rand.NewPCG(uint64(seed), uint64(seed>>1|1))),
		trials: make([]Trial, 0),
	}
}

const explorationTrials = 5
const eiCandidates = 1000

// Suggest returns the next trial ID and a suggested parameter configuration.
// The first few trials use random sampling; subsequent trials use Expected
// Improvement approximation over random candidates.
func (bo *BayesianOptimizer) Suggest() (int, map[string]float64) {
	bo.mu.Lock()
	defer bo.mu.Unlock()

	id := bo.nextID
	bo.nextID++

	var params map[string]float64

	completed := bo.completedTrials()
	if len(completed) < explorationTrials {
		params = bo.randomSample()
	} else {
		params = bo.suggestEI(completed)
	}

	bo.trials = append(bo.trials, Trial{
		ID:     id,
		Params: params,
		Done:   false,
	})

	return id, params
}

// Report records the score for a completed trial.
func (bo *BayesianOptimizer) Report(trialID int, score float64) error {
	bo.mu.Lock()
	defer bo.mu.Unlock()

	for i := range bo.trials {
		if bo.trials[i].ID == trialID {
			if bo.trials[i].Done {
				return fmt.Errorf("automl: trial %d already reported", trialID)
			}
			bo.trials[i].Score = score
			bo.trials[i].Done = true
			return nil
		}
	}
	return fmt.Errorf("automl: unknown trial %d", trialID)
}

// BestTrial returns the completed trial with the highest score.
// If no trials have been completed, ok is false.
func (bo *BayesianOptimizer) BestTrial() (Trial, bool) {
	bo.mu.Lock()
	defer bo.mu.Unlock()

	var best Trial
	found := false
	for _, t := range bo.trials {
		if t.Done && (!found || t.Score > best.Score) {
			best = t
			found = true
		}
	}
	return best, found
}

// Trials returns all trials in creation order.
func (bo *BayesianOptimizer) Trials() []Trial {
	bo.mu.Lock()
	defer bo.mu.Unlock()

	out := make([]Trial, len(bo.trials))
	copy(out, bo.trials)
	return out
}

func (bo *BayesianOptimizer) completedTrials() []Trial {
	var out []Trial
	for _, t := range bo.trials {
		if t.Done {
			out = append(out, t)
		}
	}
	return out
}

func (bo *BayesianOptimizer) randomSample() map[string]float64 {
	params := make(map[string]float64, len(bo.params))
	for _, hp := range bo.params {
		params[hp.Name] = bo.sampleParam(hp)
	}
	return params
}

func (bo *BayesianOptimizer) sampleParam(hp HParam) float64 {
	if hp.IsLog {
		logMin := math.Log(hp.Min)
		logMax := math.Log(hp.Max)
		return math.Exp(logMin + bo.rng.Float64()*(logMax-logMin))
	}
	return hp.Min + bo.rng.Float64()*(hp.Max-hp.Min)
}

// suggestEI generates random candidates and picks the one with the highest
// Expected Improvement. The surrogate is a simple distance-weighted
// interpolation of observed scores (simplified kriging).
func (bo *BayesianOptimizer) suggestEI(completed []Trial) map[string]float64 {
	// Find the best score so far.
	bestScore := math.Inf(-1)
	for _, t := range completed {
		if t.Score > bestScore {
			bestScore = t.Score
		}
	}

	// Compute score statistics for normalization.
	mean, std := scoreStats(completed)
	if std < 1e-12 {
		std = 1.0
	}

	bestCandidate := bo.randomSample()
	bestEI := math.Inf(-1)

	for i := 0; i < eiCandidates; i++ {
		candidate := bo.randomSample()
		ei := bo.expectedImprovement(candidate, completed, bestScore, mean, std)
		if ei > bestEI {
			bestEI = ei
			bestCandidate = candidate
		}
	}

	return bestCandidate
}

// expectedImprovement estimates EI for a candidate point using a
// distance-weighted surrogate model.
func (bo *BayesianOptimizer) expectedImprovement(
	candidate map[string]float64,
	completed []Trial,
	bestScore, scoreMean, scoreStd float64,
) float64 {
	// Compute distance-weighted prediction (surrogate mean and variance).
	mu, sigma := bo.surrogate(candidate, completed, scoreMean, scoreStd)

	if sigma < 1e-12 {
		// No uncertainty — EI is zero unless mu > bestScore.
		if mu > bestScore {
			return mu - bestScore
		}
		return 0
	}

	// Standard EI formula: EI = (mu - f*) * Phi(z) + sigma * phi(z)
	// where z = (mu - f*) / sigma
	z := (mu - bestScore) / sigma
	return (mu-bestScore)*normCDF(z) + sigma*normPDF(z)
}

// surrogate computes a distance-weighted prediction (mean and std) for the
// candidate point based on completed trials.
func (bo *BayesianOptimizer) surrogate(
	candidate map[string]float64,
	completed []Trial,
	scoreMean, scoreStd float64,
) (float64, float64) {
	type distScore struct {
		dist  float64
		score float64
	}

	ds := make([]distScore, len(completed))
	for i, t := range completed {
		ds[i] = distScore{
			dist:  bo.normalizedDistance(candidate, t.Params),
			score: t.Score,
		}
	}

	// Sort by distance.
	sort.Slice(ds, func(i, j int) bool { return ds[i].dist < ds[j].dist })

	// Use inverse-distance weighting with a lengthscale.
	lengthscale := 0.1
	var weightSum float64
	var weightedScore float64

	for _, d := range ds {
		w := math.Exp(-d.dist * d.dist / (2 * lengthscale * lengthscale))
		weightSum += w
		weightedScore += w * d.score
	}

	if weightSum < 1e-12 {
		return scoreMean, scoreStd
	}

	mu := weightedScore / weightSum

	// Estimate variance from weighted residuals.
	var weightedVar float64
	for _, d := range ds {
		w := math.Exp(-d.dist * d.dist / (2 * lengthscale * lengthscale))
		diff := d.score - mu
		weightedVar += w * diff * diff
	}
	sigma := math.Sqrt(weightedVar / weightSum)

	// Add a distance-based uncertainty term: farther points are more uncertain.
	minDist := ds[0].dist
	sigma += scoreStd * minDist

	return mu, sigma
}

// normalizedDistance computes the Euclidean distance between two parameter
// configurations, with each dimension normalized to [0, 1].
func (bo *BayesianOptimizer) normalizedDistance(a, b map[string]float64) float64 {
	var sumSq float64
	for _, hp := range bo.params {
		var na, nb float64
		if hp.IsLog {
			logMin := math.Log(hp.Min)
			logMax := math.Log(hp.Max)
			span := logMax - logMin
			if span < 1e-12 {
				continue
			}
			na = (math.Log(a[hp.Name]) - logMin) / span
			nb = (math.Log(b[hp.Name]) - logMin) / span
		} else {
			span := hp.Max - hp.Min
			if span < 1e-12 {
				continue
			}
			na = (a[hp.Name] - hp.Min) / span
			nb = (b[hp.Name] - hp.Min) / span
		}
		diff := na - nb
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq)
}

func scoreStats(trials []Trial) (mean, std float64) {
	n := float64(len(trials))
	if n == 0 {
		return 0, 1
	}
	var sum float64
	for _, t := range trials {
		sum += t.Score
	}
	mean = sum / n

	var varSum float64
	for _, t := range trials {
		d := t.Score - mean
		varSum += d * d
	}
	std = math.Sqrt(varSum / n)
	return mean, std
}

// normPDF returns the standard normal probability density function.
func normPDF(x float64) float64 {
	return math.Exp(-x*x/2) / math.Sqrt(2*math.Pi)
}

// normCDF approximates the standard normal cumulative distribution function.
func normCDF(x float64) float64 {
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}
