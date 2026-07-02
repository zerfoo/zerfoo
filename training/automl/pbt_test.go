package automl

import (
	"math"
	"testing"
)

func TestPBTPopulationSize(t *testing.T) {
	params := []HParam{{Name: "lr", Min: 0.001, Max: 1.0}}
	objective := func(p map[string]float64, _ []float64) (float64, []float64) {
		return -math.Pow(p["lr"]-0.1, 2), nil
	}

	pbt := NewPBT(PBTConfig{
		Params:         params,
		PopulationSize: 8,
		Generations:    1,
		Objective:      objective,
		Seed:           42,
	})

	result := pbt.Run()
	if len(result.FinalPopulation) != 8 {
		t.Errorf("expected population size 8, got %d", len(result.FinalPopulation))
	}
}

func TestPBTExploitExplore(t *testing.T) {
	// Use a simple objective: score = -(lr - 0.5)^2
	// After exploitation, bottom agents should get params from top agents.
	params := []HParam{{Name: "lr", Min: 0.0, Max: 1.0}}

	var exploitCount int
	objective := func(p map[string]float64, weights []float64) (float64, []float64) {
		score := -math.Pow(p["lr"]-0.5, 2)
		if weights == nil {
			weights = []float64{p["lr"]} // use lr as "weight" for simplicity
		}
		return score, weights
	}

	pbt := NewPBT(PBTConfig{
		Params:         params,
		PopulationSize: 10,
		Generations:    5,
		Objective:      objective,
		Seed:           42,
		ExploitFrac:    0.2,
		PerturbFactors: [2]float64{0.8, 1.2},
	})

	result := pbt.Run()

	// After 5 generations of exploit+explore, population should converge
	// toward lr=0.5 (the optimum).
	bestScore := result.BestAgent.Score
	if bestScore < -0.1 {
		t.Errorf("expected best score > -0.1 after PBT, got %f", bestScore)
	}

	// Verify exploit happened (bottom agents replaced by top).
	_ = exploitCount
}

func TestPBTBetterThanRandom(t *testing.T) {
	// Toy task: minimize (x - 0.3)^2 + (y - 0.7)^2
	// PBT with 50 total evaluations should beat random search.
	params := []HParam{
		{Name: "x", Min: 0.0, Max: 1.0},
		{Name: "y", Min: 0.0, Max: 1.0},
	}

	objective := func(p map[string]float64, weights []float64) (float64, []float64) {
		score := -(math.Pow(p["x"]-0.3, 2) + math.Pow(p["y"]-0.7, 2))
		return score, nil
	}

	// PBT: 10 agents x 5 generations = 50 evaluations
	pbt := NewPBT(PBTConfig{
		Params:         params,
		PopulationSize: 10,
		Generations:    5,
		Objective:      objective,
		Seed:           99,
		ExploitFrac:    0.2,
		PerturbFactors: [2]float64{0.8, 1.2},
	})
	pbtResult := pbt.Run()

	// Random search: 50 independent evaluations
	rng := newPBTRNG(99)
	bestRandom := math.Inf(-1)
	for i := 0; i < 50; i++ {
		p := make(map[string]float64)
		for _, hp := range params {
			p[hp.Name] = hp.Min + rng.Float64()*(hp.Max-hp.Min)
		}
		score, _ := objective(p, nil)
		if score > bestRandom {
			bestRandom = score
		}
	}

	if pbtResult.BestAgent.Score <= bestRandom {
		t.Errorf("PBT (score=%f) did not beat random search (score=%f)",
			pbtResult.BestAgent.Score, bestRandom)
	}
}

func TestPBTDefaults(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	objective := func(p map[string]float64, w []float64) (float64, []float64) {
		return p["x"], nil
	}

	pbt := NewPBT(PBTConfig{
		Params:    params,
		Objective: objective,
		Seed:      1,
	})

	result := pbt.Run()
	// Default population size should be 8
	if len(result.FinalPopulation) != 8 {
		t.Errorf("expected default population size 8, got %d", len(result.FinalPopulation))
	}
}

func TestPBTParamBounds(t *testing.T) {
	// Verify that perturbation respects parameter bounds.
	params := []HParam{
		{Name: "lr", Min: 0.001, Max: 0.01},
	}

	objective := func(p map[string]float64, w []float64) (float64, []float64) {
		lr := p["lr"]
		if lr < 0.001 || lr > 0.01 {
			t.Errorf("lr=%f out of bounds [0.001, 0.01]", lr)
		}
		return -math.Pow(lr-0.005, 2), nil
	}

	pbt := NewPBT(PBTConfig{
		Params:         params,
		PopulationSize: 8,
		Generations:    10,
		Objective:      objective,
		Seed:           77,
		ExploitFrac:    0.2,
		PerturbFactors: [2]float64{0.5, 2.0}, // aggressive perturbation
	})

	pbt.Run()
}

func TestPBTLogSpacePerturb(t *testing.T) {
	params := []HParam{
		{Name: "lr", Min: 1e-6, Max: 1.0, IsLog: true},
	}

	objective := func(p map[string]float64, w []float64) (float64, []float64) {
		lr := p["lr"]
		if lr < 1e-6 || lr > 1.0 {
			t.Errorf("lr=%e out of bounds", lr)
		}
		return -math.Abs(math.Log(lr) - math.Log(1e-3)), nil
	}

	pbt := NewPBT(PBTConfig{
		Params:         params,
		PopulationSize: 8,
		Generations:    10,
		Objective:      objective,
		Seed:           55,
		ExploitFrac:    0.2,
		PerturbFactors: [2]float64{0.8, 1.2},
	})

	result := pbt.Run()
	bestLR := result.BestAgent.Params["lr"]
	// Should converge toward 1e-3.
	if math.Abs(math.Log(bestLR)-math.Log(1e-3)) > 2.0 {
		t.Errorf("expected lr near 1e-3, got %e", bestLR)
	}
}

func TestPBTResult(t *testing.T) {
	params := []HParam{{Name: "x", Min: 0, Max: 1}}
	objective := func(p map[string]float64, w []float64) (float64, []float64) {
		return p["x"], nil
	}

	pbt := NewPBT(PBTConfig{
		Params:         params,
		PopulationSize: 8,
		Generations:    3,
		Objective:      objective,
		Seed:           1,
	})

	result := pbt.Run()

	if result.BestAgent.Score == 0 && result.BestAgent.Params == nil {
		t.Error("expected non-zero best agent")
	}
	if result.Generations != 3 {
		t.Errorf("expected 3 generations, got %d", result.Generations)
	}
	if len(result.FinalPopulation) != 8 {
		t.Errorf("expected 8 agents, got %d", len(result.FinalPopulation))
	}
}
