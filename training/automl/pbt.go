package automl

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Agent represents a single member of the PBT population. Each agent
// maintains its own hyperparameter configuration, model weights (opaque
// to PBT), and current performance score.
type Agent struct {
	ID      int
	Params  map[string]float64
	Weights []float64
	Score   float64
}

// PBTConfig configures population-based training.
type PBTConfig struct {
	// Params defines the hyperparameter search space.
	Params []HParam

	// PopulationSize is the number of agents (default 8).
	PopulationSize int

	// Generations is the number of exploit-explore cycles (default 10).
	Generations int

	// Objective evaluates a hyperparameter config and optional model weights,
	// returning (score, updated_weights). Higher scores are better.
	// Weights may be nil on first call; the function should initialize them.
	Objective func(params map[string]float64, weights []float64) (float64, []float64)

	// Seed for the random number generator.
	Seed int64

	// ExploitFrac is the fraction of the population considered top/bottom
	// for exploit (default 0.2, meaning top 20% replaces bottom 20%).
	ExploitFrac float64

	// PerturbFactors defines the multiplicative perturbation range [low, high]
	// applied during exploration (default [0.8, 1.2]).
	PerturbFactors [2]float64
}

// PBTResult contains the outcome of a PBT run.
type PBTResult struct {
	BestAgent       Agent
	FinalPopulation []Agent
	Generations     int
}

// PBT implements Population-Based Training (Jaderberg et al. 2017).
type PBT struct {
	config PBTConfig
	rng    *rand.Rand
}

// newPBTRNG creates an RNG from a seed, exported for test use.
func newPBTRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewPCG(uint64(seed), uint64(seed>>1|1)))
}

// NewPBT creates a PBT runner with the given configuration.
func NewPBT(config PBTConfig) *PBT {
	if config.PopulationSize <= 0 {
		config.PopulationSize = 8
	}
	if config.Generations <= 0 {
		config.Generations = 10
	}
	if config.ExploitFrac <= 0 {
		config.ExploitFrac = 0.2
	}
	if config.PerturbFactors == [2]float64{} {
		config.PerturbFactors = [2]float64{0.8, 1.2}
	}

	return &PBT{
		config: config,
		rng:    newPBTRNG(config.Seed),
	}
}

// Run executes the PBT loop and returns the result.
func (pbt *PBT) Run() PBTResult {
	pop := pbt.initPopulation()

	// Evaluate initial population.
	for i := range pop {
		pop[i].Score, pop[i].Weights = pbt.config.Objective(pop[i].Params, pop[i].Weights)
	}

	for gen := 0; gen < pbt.config.Generations; gen++ {
		pbt.exploitAndExplore(pop)

		// Re-evaluate all agents with their (possibly updated) params and weights.
		for i := range pop {
			pop[i].Score, pop[i].Weights = pbt.config.Objective(pop[i].Params, pop[i].Weights)
		}
	}

	best := pbt.findBest(pop)

	result := PBTResult{
		BestAgent:       best,
		FinalPopulation: make([]Agent, len(pop)),
		Generations:     pbt.config.Generations,
	}
	copy(result.FinalPopulation, pop)
	return result
}

func (pbt *PBT) initPopulation() []Agent {
	pop := make([]Agent, pbt.config.PopulationSize)
	for i := range pop {
		params := make(map[string]float64, len(pbt.config.Params))
		for _, hp := range pbt.config.Params {
			if hp.IsLog {
				logMin := math.Log(hp.Min)
				logMax := math.Log(hp.Max)
				params[hp.Name] = math.Exp(logMin + pbt.rng.Float64()*(logMax-logMin))
			} else {
				params[hp.Name] = hp.Min + pbt.rng.Float64()*(hp.Max-hp.Min)
			}
		}
		pop[i] = Agent{
			ID:     i,
			Params: params,
		}
	}
	return pop
}

// exploitAndExplore replaces the bottom fraction with copies of the top
// fraction (exploit), then perturbs the copied hyperparameters (explore).
func (pbt *PBT) exploitAndExplore(pop []Agent) {
	n := len(pop)
	k := int(math.Max(1, math.Round(float64(n)*pbt.config.ExploitFrac)))

	// Sort by score descending.
	sorted := make([]int, n)
	for i := range sorted {
		sorted[i] = i
	}
	sort.Slice(sorted, func(i, j int) bool {
		return pop[sorted[i]].Score > pop[sorted[j]].Score
	})

	topIndices := sorted[:k]
	bottomIndices := sorted[n-k:]

	// Replace each bottom agent with a copy of a random top agent.
	for _, bi := range bottomIndices {
		ti := topIndices[pbt.rng.IntN(k)]

		// Copy weights.
		if pop[ti].Weights != nil {
			w := make([]float64, len(pop[ti].Weights))
			copy(w, pop[ti].Weights)
			pop[bi].Weights = w
		}

		// Copy params then perturb.
		for key, val := range pop[ti].Params {
			pop[bi].Params[key] = val
		}
		pbt.perturb(pop[bi].Params)
	}
}

// perturb applies multiplicative perturbation to each hyperparameter,
// clamping to the defined bounds.
func (pbt *PBT) perturb(params map[string]float64) {
	low := pbt.config.PerturbFactors[0]
	high := pbt.config.PerturbFactors[1]

	for _, hp := range pbt.config.Params {
		// Randomly choose low or high factor.
		factor := low
		if pbt.rng.Float64() < 0.5 {
			factor = high
		}

		if hp.IsLog {
			// Perturb in log space.
			logVal := math.Log(params[hp.Name])
			logVal *= factor
			val := math.Exp(logVal)
			val = math.Max(hp.Min, math.Min(hp.Max, val))
			params[hp.Name] = val
		} else {
			val := params[hp.Name] * factor
			val = math.Max(hp.Min, math.Min(hp.Max, val))
			params[hp.Name] = val
		}
	}
}

func (pbt *PBT) findBest(pop []Agent) Agent {
	best := pop[0]
	for _, a := range pop[1:] {
		if a.Score > best.Score {
			best = a
		}
	}
	return best
}
