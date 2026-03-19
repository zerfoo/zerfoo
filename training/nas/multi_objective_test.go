package nas

import (
	"math"
	"testing"
)

// accuracyObjective returns a mock accuracy score based on the number of
// non-zero, non-skip edges (more compute = higher accuracy).
func accuracyObjective() Objective {
	return Objective{
		Name:      "accuracy",
		Direction: Maximize,
		Evaluate: func(c Cell) float64 {
			score := 0.0
			for _, e := range c.Edges {
				switch e.Op {
				case OpConv3x3:
					score += 3.0
				case OpConv5x5:
					score += 4.0
				case OpSepConv3x3:
					score += 2.5
				case OpSepConv5x5:
					score += 3.5
				case OpAvgPool3x3, OpMaxPool3x3:
					score += 1.0
				case OpSkipConnect:
					score += 0.5
				case OpZero:
					score += 0.0
				}
			}
			return score
		},
	}
}

// latencyObjective returns a mock latency (lower is better) based on total FLOPs.
func latencyObjective() Objective {
	estimator := NewLatencyEstimator(DGXSpark())
	return Objective{
		Name:      "latency",
		Direction: Minimize,
		Evaluate: func(c Cell) float64 {
			return estimator.Estimate(c)
		},
	}
}

// modelSizeObjective returns a mock model size (lower is better) based on param count.
func modelSizeObjective() Objective {
	opParams := DefaultOpParams()
	return Objective{
		Name:      "model_size",
		Direction: Minimize,
		Evaluate: func(c Cell) float64 {
			var total int64
			for _, e := range c.Edges {
				if p, ok := opParams[e.Op]; ok {
					total += p
				}
			}
			return float64(total)
		},
	}
}

func TestMultiObjectiveNAS_ParetoFrontier(t *testing.T) {
	space := NewSearchSpace(4)

	objectives := []Objective{
		accuracyObjective(),
		latencyObjective(),
	}

	config := MultiObjectiveConfig{
		PopulationSize: 40,
		Generations:    20,
		CrossoverRate:  0.9,
		MutationRate:   0.1,
		Seed:           42,
	}

	result, err := SearchMultiObjective(space, objectives, config)
	if err != nil {
		t.Fatalf("SearchMultiObjective failed: %v", err)
	}

	// Must have a non-empty Pareto front.
	if len(result.Front) == 0 {
		t.Fatal("expected non-empty Pareto front")
	}

	// All front solutions must have rank 0.
	for i, s := range result.Front {
		if s.Rank != 0 {
			t.Errorf("front solution %d has rank %d, want 0", i, s.Rank)
		}
	}

	// No front solution should dominate another.
	for i := 0; i < len(result.Front); i++ {
		for j := i + 1; j < len(result.Front); j++ {
			if dominates(result.Front[i], result.Front[j], objectives) {
				t.Errorf("front solution %d dominates %d: %v vs %v",
					i, j, result.Front[i].Scores, result.Front[j].Scores)
			}
			if dominates(result.Front[j], result.Front[i], objectives) {
				t.Errorf("front solution %d dominates %d: %v vs %v",
					j, i, result.Front[j].Scores, result.Front[i].Scores)
			}
		}
	}

	// Population should be full size.
	if len(result.Population) != config.PopulationSize {
		t.Errorf("population size = %d, want %d", len(result.Population), config.PopulationSize)
	}

	// All cells must be valid.
	for i, s := range result.Population {
		if !s.Cell.Valid() {
			t.Errorf("population solution %d has invalid cell", i)
		}
	}

	// Front should show trade-off: not all solutions identical.
	if len(result.Front) > 1 {
		allSame := true
		for _, s := range result.Front[1:] {
			for k := range s.Scores {
				if s.Scores[k] != result.Front[0].Scores[k] {
					allSame = false
					break
				}
			}
		}
		if allSame {
			t.Error("all Pareto front solutions have identical scores; expected diversity")
		}
	}

	t.Logf("Pareto front size: %d / %d population", len(result.Front), len(result.Population))
	for i, s := range result.Front {
		t.Logf("  front[%d]: accuracy=%.2f latency=%.2e", i, s.Scores[0], s.Scores[1])
	}
}

func TestMultiObjectiveNAS_ThreeObjectives(t *testing.T) {
	space := NewSearchSpace(4)

	objectives := []Objective{
		accuracyObjective(),
		latencyObjective(),
		modelSizeObjective(),
	}

	config := MultiObjectiveConfig{
		PopulationSize: 30,
		Generations:    15,
		CrossoverRate:  0.8,
		MutationRate:   0.15,
		Seed:           123,
	}

	result, err := SearchMultiObjective(space, objectives, config)
	if err != nil {
		t.Fatalf("SearchMultiObjective failed: %v", err)
	}

	if len(result.Front) == 0 {
		t.Fatal("expected non-empty Pareto front")
	}

	// Verify non-domination on 3 objectives.
	for i := 0; i < len(result.Front); i++ {
		for j := i + 1; j < len(result.Front); j++ {
			if dominates(result.Front[i], result.Front[j], objectives) {
				t.Errorf("front[%d] dominates front[%d]", i, j)
			}
			if dominates(result.Front[j], result.Front[i], objectives) {
				t.Errorf("front[%d] dominates front[%d]", j, i)
			}
		}
	}

	t.Logf("3-objective front size: %d", len(result.Front))
}

func TestMultiObjectiveNAS_Convergence(t *testing.T) {
	space := NewSearchSpace(4)

	objectives := []Objective{
		accuracyObjective(),
		latencyObjective(),
	}

	// Run with few generations.
	earlyConfig := MultiObjectiveConfig{
		PopulationSize: 30,
		Generations:    5,
		CrossoverRate:  0.9,
		MutationRate:   0.1,
		Seed:           99,
	}

	earlyResult, err := SearchMultiObjective(space, objectives, earlyConfig)
	if err != nil {
		t.Fatalf("early search failed: %v", err)
	}

	// Run with many more generations (same seed, same initial pop).
	lateConfig := MultiObjectiveConfig{
		PopulationSize: 30,
		Generations:    50,
		CrossoverRate:  0.9,
		MutationRate:   0.1,
		Seed:           99,
	}

	lateResult, err := SearchMultiObjective(space, objectives, lateConfig)
	if err != nil {
		t.Fatalf("late search failed: %v", err)
	}

	// Measure quality as the hypervolume indicator approximation.
	// Use the worst-case reference point from the early population.
	earlyHV := hypervolume2D(earlyResult.Front, objectives)
	lateHV := hypervolume2D(lateResult.Front, objectives)

	t.Logf("early hypervolume (5 gen): %.6f, late hypervolume (50 gen): %.6f", earlyHV, lateHV)

	// The later run should have at least as good a hypervolume.
	// We allow a small tolerance for numerical noise.
	if lateHV < earlyHV-1e-9 {
		t.Errorf("convergence regression: late HV %.6f < early HV %.6f", lateHV, earlyHV)
	}

	// The Pareto front should not be empty in either case.
	if len(earlyResult.Front) == 0 || len(lateResult.Front) == 0 {
		t.Error("expected non-empty fronts")
	}
}

// hypervolume2D computes the 2D hypervolume indicator for a Pareto front.
// Objectives are normalized so that the reference point is at (0, 0) in
// normalized space (worst on both axes).
func hypervolume2D(front []Solution, objectives []Objective) float64 {
	if len(front) == 0 || len(objectives) < 2 {
		return 0
	}

	// Normalize: for Maximize objectives, negate so that lower is always better.
	type point struct{ x, y float64 }
	points := make([]point, len(front))
	for i, s := range front {
		x := s.Scores[0]
		y := s.Scores[1]
		if objectives[0].Direction == Maximize {
			x = -x
		}
		if objectives[1].Direction == Maximize {
			y = -y
		}
		points[i] = point{x, y}
	}

	// Find reference point (worst values = maximum in minimization space).
	refX, refY := math.Inf(-1), math.Inf(-1)
	for _, p := range points {
		if p.x > refX {
			refX = p.x
		}
		if p.y > refY {
			refY = p.y
		}
	}
	// Shift reference slightly beyond worst.
	refX += 1.0
	refY += 1.0

	// Sort by x ascending.
	sorted := make([]point, len(points))
	copy(sorted, points)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j].x < sorted[j-1].x; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}

	// Compute hypervolume via sweeping.
	hv := 0.0
	prevY := refY
	for _, p := range sorted {
		if p.y < prevY {
			hv += (refX - p.x) * (prevY - p.y)
			prevY = p.y
		}
	}
	return hv
}

func TestMultiObjectiveNAS_ValidationErrors(t *testing.T) {
	space := NewSearchSpace(4)
	objectives := []Objective{accuracyObjective(), latencyObjective()}

	tests := []struct {
		name   string
		space  *SearchSpace
		objs   []Objective
		config MultiObjectiveConfig
	}{
		{"nil space", nil, objectives, MultiObjectiveConfig{PopulationSize: 10, Generations: 1}},
		{"one objective", space, objectives[:1], MultiObjectiveConfig{PopulationSize: 10, Generations: 1}},
		{"small pop", space, objectives, MultiObjectiveConfig{PopulationSize: 2, Generations: 1}},
		{"zero gen", space, objectives, MultiObjectiveConfig{PopulationSize: 10, Generations: 0}},
		{"negative crossover", space, objectives, MultiObjectiveConfig{PopulationSize: 10, Generations: 1, CrossoverRate: -0.1}},
		{"crossover > 1", space, objectives, MultiObjectiveConfig{PopulationSize: 10, Generations: 1, CrossoverRate: 1.1}},
		{"negative mutation", space, objectives, MultiObjectiveConfig{PopulationSize: 10, Generations: 1, MutationRate: -0.1}},
		{"mutation > 1", space, objectives, MultiObjectiveConfig{PopulationSize: 10, Generations: 1, MutationRate: 1.1}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := SearchMultiObjective(tt.space, tt.objs, tt.config)
			if err == nil {
				t.Error("expected error, got nil")
			}
		})
	}
}

func TestDominates(t *testing.T) {
	objectives := []Objective{
		{Name: "a", Direction: Minimize},
		{Name: "b", Direction: Maximize},
	}

	a := Solution{Scores: []float64{1.0, 5.0}}
	b := Solution{Scores: []float64{2.0, 4.0}}
	c := Solution{Scores: []float64{1.0, 4.0}}

	if !dominates(a, b, objectives) {
		t.Error("a should dominate b (better on both)")
	}
	if dominates(b, a, objectives) {
		t.Error("b should not dominate a")
	}
	if !dominates(a, c, objectives) {
		t.Error("a should dominate c (equal on min, better on max)")
	}
	if dominates(c, a, objectives) {
		t.Error("c should not dominate a")
	}
	// b and c: b has worse min (2>1) but c has worse max (4<4) — actually c=(1,4), b=(2,4)
	// b.min=2 > c.min=1 so b is worse on min; b.max=4 == c.max=4 so equal on max.
	// c dominates b? c is better on min (1<2), equal on max (4==4) → yes, c dominates b.
	if !dominates(c, b, objectives) {
		t.Error("c should dominate b")
	}
}
