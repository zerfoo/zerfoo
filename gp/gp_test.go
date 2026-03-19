package gp

import (
	"math"
	"math/rand"
	"testing"
)

func TestEvolve_SimpleFitness(t *testing.T) {
	// Target function: f(x) = x^2 + x + 1
	// We provide Add, Mul, and variable x plus constants.
	primitives := []Primitive{
		AddPrimitive(),
		MulPrimitive(),
		SubPrimitive(),
		VariablePrimitive("x", 0),
		ConstantPrimitive("1", 1),
		ConstantPrimitive("2", 2),
	}

	// Fitness: mean squared error over sample points.
	target := func(x float64) float64 { return x*x + x + 1 }
	samples := make([]float64, 21)
	for i := range samples {
		samples[i] = -1.0 + float64(i)*0.1
	}

	fitnessFunc := func(p *Program) float64 {
		mse := 0.0
		for _, x := range samples {
			predicted := p.Evaluate([]float64{x})
			diff := predicted - target(x)
			mse += diff * diff
		}
		return mse / float64(len(samples))
	}

	config := GPConfig{
		PopulationSize: 500,
		MaxGenerations: 100,
		MaxDepth:       4,
		CrossoverRate:  0.9,
		MutationRate:   0.1,
		TournamentSize: 5,
		Seed:           42,
	}

	prog, err := Evolve(primitives, fitnessFunc, config)
	if err != nil {
		t.Fatalf("Evolve returned error: %v", err)
	}
	if prog == nil {
		t.Fatal("Evolve returned nil program")
	}

	// Check that the evolved program has reasonable fitness.
	// With these settings it should get MSE < 1.0 at minimum.
	if prog.Fitness > 1.0 {
		t.Errorf("expected fitness < 1.0, got %f (program: %s)", prog.Fitness, prog)
	}

	// Verify Evaluate works on new inputs.
	result := prog.Evaluate([]float64{0})
	if math.IsNaN(result) || math.IsInf(result, 0) {
		t.Errorf("program evaluation returned non-finite value: %f", result)
	}
}

func TestEvolve_TreeOperations(t *testing.T) {
	primitives := []Primitive{
		AddPrimitive(),
		MulPrimitive(),
		VariablePrimitive("x", 0),
		ConstantPrimitive("1", 1),
	}

	// Simple fitness: just measure distance from x * x at x=2.
	fitnessFunc := func(p *Program) float64 {
		val := p.Evaluate([]float64{2})
		return math.Abs(val - 4)
	}

	config := GPConfig{
		PopulationSize: 50,
		MaxGenerations: 10,
		MaxDepth:       3,
		CrossoverRate:  0.9,
		MutationRate:   0.5, // high mutation to exercise mutation paths
		TournamentSize: 3,
		Seed:           99,
	}

	prog, err := Evolve(primitives, fitnessFunc, config)
	if err != nil {
		t.Fatalf("Evolve returned error: %v", err)
	}
	if prog == nil {
		t.Fatal("Evolve returned nil program")
	}
	if prog.Root == nil {
		t.Fatal("evolved program has nil root")
	}

	// Verify tree depth is within bounds.
	if d := prog.Root.depth(); d > config.MaxDepth {
		t.Errorf("tree depth %d exceeds max depth %d", d, config.MaxDepth)
	}

	// Verify tree has at least one node.
	if s := prog.Root.size(); s < 1 {
		t.Errorf("tree size %d is invalid", s)
	}

	// Verify String does not panic and produces output.
	str := prog.String()
	if str == "" {
		t.Error("program String() returned empty string")
	}
}

func TestEvolve_Crossover(t *testing.T) {
	// Build two known trees and verify crossover produces a valid tree.
	add := AddPrimitive()
	mul := MulPrimitive()
	x := VariablePrimitive("x", 0)
	one := ConstantPrimitive("1", 1)

	// Tree A: (Add x 1)
	treeA := &Node{Prim: add, Children: []*Node{
		{Prim: x},
		{Prim: one},
	}}
	// Tree B: (Mul x x)
	treeB := &Node{Prim: mul, Children: []*Node{
		{Prim: x},
		{Prim: x},
	}}

	// Run crossover many times and verify result is always valid.
	rng := newTestRNG(123)
	for i := 0; i < 50; i++ {
		a := treeA.clone()
		b := treeB.clone()
		result := crossover(a, b, rng)
		if result == nil {
			t.Fatal("crossover returned nil")
		}
		if result.size() < 1 {
			t.Errorf("crossover produced tree with size %d", result.size())
		}
		// Should evaluate without panic.
		result.Evaluate([]float64{3.0})
	}
}

func TestEvolve_Mutation(t *testing.T) {
	add := AddPrimitive()
	x := VariablePrimitive("x", 0)
	one := ConstantPrimitive("1", 1)

	tree := &Node{Prim: add, Children: []*Node{
		{Prim: x},
		{Prim: one},
	}}

	rng := newTestRNG(456)
	gen := &generator{
		functions: []Primitive{add},
		terminals: []Primitive{x, one},
		maxDepth:  3,
		rng:       rng,
	}

	for i := 0; i < 50; i++ {
		root := tree.clone()
		result := mutate(root, gen, rng)
		if result == nil {
			t.Fatal("mutate returned nil")
		}
		if result.size() < 1 {
			t.Errorf("mutate produced tree with size %d", result.size())
		}
		result.Evaluate([]float64{2.0})
	}
}

func TestEvolve_ValidationErrors(t *testing.T) {
	dummy := func(p *Program) float64 { return 0 }

	t.Run("no primitives", func(t *testing.T) {
		_, err := Evolve(nil, dummy, GPConfig{})
		if err == nil {
			t.Error("expected error for empty primitives")
		}
	})

	t.Run("no terminals", func(t *testing.T) {
		_, err := Evolve([]Primitive{AddPrimitive()}, dummy, GPConfig{})
		if err == nil {
			t.Error("expected error for no terminals")
		}
	})

	t.Run("no functions", func(t *testing.T) {
		_, err := Evolve([]Primitive{VariablePrimitive("x", 0)}, dummy, GPConfig{})
		if err == nil {
			t.Error("expected error for no functions")
		}
	})
}

func TestNode_Clone(t *testing.T) {
	add := AddPrimitive()
	x := VariablePrimitive("x", 0)

	original := &Node{Prim: add, Children: []*Node{
		{Prim: x},
		{Prim: x},
	}}

	cloned := original.clone()

	// Modifying clone should not affect original.
	cloned.Children[0] = &Node{Prim: ConstantPrimitive("5", 5)}

	origVal := original.Evaluate([]float64{3.0})
	clonedVal := cloned.Evaluate([]float64{3.0})

	if origVal == clonedVal {
		t.Error("modifying clone affected original tree")
	}
	if origVal != 6.0 {
		t.Errorf("original should evaluate to 6.0, got %f", origVal)
	}
}

func newTestRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
