package nas

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

// ObjectiveDirection specifies whether an objective should be minimized or maximized.
type ObjectiveDirection int

const (
	// Minimize indicates a lower value is better (e.g., latency, model size).
	Minimize ObjectiveDirection = iota
	// Maximize indicates a higher value is better (e.g., accuracy).
	Maximize
)

// Objective defines a single optimization objective.
type Objective struct {
	// Name is a human-readable identifier for the objective.
	Name string
	// Direction specifies whether to minimize or maximize this objective.
	Direction ObjectiveDirection
	// Evaluate scores a cell architecture on this objective.
	Evaluate func(Cell) float64
}

// MultiObjectiveConfig holds configuration for multi-objective NAS using NSGA-II.
type MultiObjectiveConfig struct {
	// PopulationSize is the number of candidate architectures per generation.
	PopulationSize int
	// Generations is the number of evolutionary generations to run.
	Generations int
	// CrossoverRate is the probability of crossover between two parents [0, 1].
	CrossoverRate float64
	// MutationRate is the probability of mutating each edge's operation [0, 1].
	MutationRate float64
	// Seed for reproducibility. Zero means non-deterministic.
	Seed uint64
}

// Solution represents a single candidate solution evaluated on all objectives.
type Solution struct {
	// Cell is the architecture this solution represents.
	Cell Cell
	// Scores holds the objective value for each objective, in the same order
	// as the objectives slice passed to SearchMultiObjective.
	Scores []float64
	// Rank is the non-domination rank (0 = Pareto front, 1 = second front, etc.).
	Rank int
	// CrowdingDistance measures how isolated this solution is in objective space.
	CrowdingDistance float64
}

// ParetoFrontier holds the set of non-dominated solutions discovered by the
// multi-objective search, along with the full final population.
type ParetoFrontier struct {
	// Front contains only the rank-0 (non-dominated) solutions.
	Front []Solution
	// Population contains the entire final population.
	Population []Solution
	// Generations is the number of generations that were executed.
	Generations int
}

// SearchMultiObjective runs an NSGA-II multi-objective evolutionary search over
// the given search space. It returns the Pareto frontier of non-dominated solutions.
func SearchMultiObjective(space *SearchSpace, objectives []Objective, config MultiObjectiveConfig) (*ParetoFrontier, error) {
	if space == nil {
		return nil, errors.New("nas: SearchMultiObjective requires a non-nil SearchSpace")
	}
	if len(objectives) < 2 {
		return nil, errors.New("nas: SearchMultiObjective requires at least 2 objectives")
	}
	if config.PopulationSize < 4 {
		return nil, errors.New("nas: PopulationSize must be at least 4")
	}
	if config.Generations < 1 {
		return nil, errors.New("nas: Generations must be at least 1")
	}
	if config.CrossoverRate < 0 || config.CrossoverRate > 1 {
		return nil, errors.New("nas: CrossoverRate must be in [0, 1]")
	}
	if config.MutationRate < 0 || config.MutationRate > 1 {
		return nil, errors.New("nas: MutationRate must be in [0, 1]")
	}

	var rng *rand.Rand
	if config.Seed != 0 {
		rng = rand.New(rand.NewPCG(config.Seed, 0))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}

	// Initialize population.
	pop := make([]Solution, config.PopulationSize)
	for i := range pop {
		pop[i] = evaluateCell(space.Sample(rng), objectives)
	}
	pop = nsga2Select(pop, objectives, config.PopulationSize)

	for range config.Generations {
		// Create offspring via tournament selection, crossover, and mutation.
		offspring := make([]Solution, config.PopulationSize)
		for i := range offspring {
			p1 := tournamentSelect(pop, rng)
			p2 := tournamentSelect(pop, rng)
			child := crossover(p1.Cell, p2.Cell, space, config.CrossoverRate, rng)
			child = mutate(child, space, config.MutationRate, rng)
			offspring[i] = evaluateCell(child, objectives)
		}

		// Combine parent and offspring (mu+lambda) and select next generation.
		combined := make([]Solution, 0, 2*config.PopulationSize)
		combined = append(combined, pop...)
		combined = append(combined, offspring...)
		pop = nsga2Select(combined, objectives, config.PopulationSize)
	}

	// Extract rank-0 front.
	var front []Solution
	for _, s := range pop {
		if s.Rank == 0 {
			front = append(front, s)
		}
	}

	return &ParetoFrontier{
		Front:       front,
		Population:  pop,
		Generations: config.Generations,
	}, nil
}

// evaluateCell scores a cell on all objectives and returns a Solution.
func evaluateCell(cell Cell, objectives []Objective) Solution {
	scores := make([]float64, len(objectives))
	for i, obj := range objectives {
		scores[i] = obj.Evaluate(cell)
	}
	return Solution{Cell: cell, Scores: scores}
}

// dominates returns true if solution a dominates solution b:
// a is at least as good on all objectives and strictly better on at least one.
func dominates(a, b Solution, objectives []Objective) bool {
	strictlyBetter := false
	for i, obj := range objectives {
		var aBetter, bBetter bool
		if obj.Direction == Minimize {
			aBetter = a.Scores[i] < b.Scores[i]
			bBetter = b.Scores[i] < a.Scores[i]
		} else {
			aBetter = a.Scores[i] > b.Scores[i]
			bBetter = b.Scores[i] > a.Scores[i]
		}
		if bBetter {
			return false
		}
		if aBetter {
			strictlyBetter = true
		}
	}
	return strictlyBetter
}

// nonDominatedSort performs fast non-dominated sorting (NSGA-II).
// Returns fronts where fronts[0] is the Pareto front. Sets Rank on each solution.
func nonDominatedSort(pop []Solution, objectives []Objective) [][]int {
	n := len(pop)
	dominationCount := make([]int, n)
	dominated := make([][]int, n)

	var frontIndices []int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if dominates(pop[i], pop[j], objectives) {
				dominated[i] = append(dominated[i], j)
				dominationCount[j]++
			} else if dominates(pop[j], pop[i], objectives) {
				dominated[j] = append(dominated[j], i)
				dominationCount[i]++
			}
		}
	}

	for i := 0; i < n; i++ {
		if dominationCount[i] == 0 {
			frontIndices = append(frontIndices, i)
			pop[i].Rank = 0
		}
	}

	var fronts [][]int
	rank := 0
	for len(frontIndices) > 0 {
		fronts = append(fronts, frontIndices)
		var nextFront []int
		for _, i := range frontIndices {
			for _, j := range dominated[i] {
				dominationCount[j]--
				if dominationCount[j] == 0 {
					pop[j].Rank = rank + 1
					nextFront = append(nextFront, j)
				}
			}
		}
		frontIndices = nextFront
		rank++
	}
	return fronts
}

// assignCrowdingDistance computes crowding distance for solutions at the given indices.
func assignCrowdingDistance(pop []Solution, indices []int, objectives []Objective) {
	n := len(indices)
	if n <= 2 {
		for _, idx := range indices {
			pop[idx].CrowdingDistance = math.Inf(1)
		}
		return
	}

	for _, idx := range indices {
		pop[idx].CrowdingDistance = 0
	}

	// Work on a copy so sorting for one objective doesn't affect another.
	idxCopy := make([]int, n)
	for m := range objectives {
		copy(idxCopy, indices)
		sort.Slice(idxCopy, func(i, j int) bool {
			return pop[idxCopy[i]].Scores[m] < pop[idxCopy[j]].Scores[m]
		})

		pop[idxCopy[0]].CrowdingDistance = math.Inf(1)
		pop[idxCopy[n-1]].CrowdingDistance = math.Inf(1)

		objRange := pop[idxCopy[n-1]].Scores[m] - pop[idxCopy[0]].Scores[m]
		if objRange < 1e-30 {
			continue
		}

		for i := 1; i < n-1; i++ {
			pop[idxCopy[i]].CrowdingDistance += (pop[idxCopy[i+1]].Scores[m] - pop[idxCopy[i-1]].Scores[m]) / objRange
		}
	}
}

// nsga2Select selects the best popSize solutions using NSGA-II:
// prefer lower rank, then higher crowding distance for partial fronts.
func nsga2Select(combined []Solution, objectives []Objective, popSize int) []Solution {
	fronts := nonDominatedSort(combined, objectives)

	var selected []Solution
	for _, frontIndices := range fronts {
		assignCrowdingDistance(combined, frontIndices, objectives)
		if len(selected)+len(frontIndices) <= popSize {
			for _, idx := range frontIndices {
				selected = append(selected, combined[idx])
			}
		} else {
			remaining := popSize - len(selected)
			sort.Slice(frontIndices, func(i, j int) bool {
				return combined[frontIndices[i]].CrowdingDistance > combined[frontIndices[j]].CrowdingDistance
			})
			for k := 0; k < remaining; k++ {
				selected = append(selected, combined[frontIndices[k]])
			}
			break
		}
	}

	return selected
}

// tournamentSelect picks two random individuals and returns the better one
// (lower rank, or higher crowding distance if tied).
func tournamentSelect(pop []Solution, rng *rand.Rand) Solution {
	i := rng.IntN(len(pop))
	j := rng.IntN(len(pop))
	if pop[i].Rank < pop[j].Rank {
		return pop[i]
	}
	if pop[j].Rank < pop[i].Rank {
		return pop[j]
	}
	if pop[i].CrowdingDistance >= pop[j].CrowdingDistance {
		return pop[i]
	}
	return pop[j]
}

// crossover performs uniform crossover on edges between two parent cells.
func crossover(p1, p2 Cell, space *SearchSpace, rate float64, rng *rand.Rand) Cell {
	if rng.Float64() > rate || len(p1.Edges) != len(p2.Edges) {
		return p1
	}
	edges := make([]Edge, len(p1.Edges))
	for i := range edges {
		if rng.Float64() < 0.5 {
			edges[i] = p1.Edges[i]
		} else {
			edges[i] = p2.Edges[i]
		}
	}
	return Cell{NumNodes: space.NumNodes, Edges: edges}
}

// mutate randomly replaces edge operations with probability rate per edge.
func mutate(cell Cell, space *SearchSpace, rate float64, rng *rand.Rand) Cell {
	edges := make([]Edge, len(cell.Edges))
	copy(edges, cell.Edges)
	for i := range edges {
		if rng.Float64() < rate {
			edges[i].Op = space.Ops[rng.IntN(len(space.Ops))]
		}
	}
	return Cell{NumNodes: cell.NumNodes, Edges: edges}
}

// String returns a human-readable representation of a Solution.
func (s Solution) String() string {
	return fmt.Sprintf("Solution{Rank:%d Scores:%v Crowding:%.4f}", s.Rank, s.Scores, s.CrowdingDistance)
}
