package gp

import (
	"fmt"
	"math"
	"math/rand"
)

// Arity indicates how many child arguments a primitive requires.
// Terminals have arity 0; functions have arity >= 1.
type Arity int

// Primitive represents a function or terminal node in an expression tree.
type Primitive struct {
	// Name is a human-readable label (e.g. "Add", "x0", "const").
	Name string
	// Arity is the number of child arguments. Terminals have arity 0.
	Arity Arity
	// Func evaluates this primitive given its child values.
	// For terminals, args is empty.
	Func func(args []float64) float64
}

// Node is a single node in an expression tree.
type Node struct {
	Prim     Primitive
	Children []*Node
}

// Evaluate recursively computes the value of the expression tree rooted at n,
// binding variable terminals to the provided inputs.
func (n *Node) Evaluate(inputs []float64) float64 {
	if n.Prim.Arity == 0 {
		return n.Prim.Func(inputs)
	}
	args := make([]float64, len(n.Children))
	for i, child := range n.Children {
		args[i] = child.Evaluate(inputs)
	}
	return n.Prim.Func(args)
}

// depth returns the maximum depth of the tree rooted at n.
func (n *Node) depth() int {
	if len(n.Children) == 0 {
		return 0
	}
	max := 0
	for _, c := range n.Children {
		d := c.depth()
		if d > max {
			max = d
		}
	}
	return max + 1
}

// size returns the total number of nodes in the tree.
func (n *Node) size() int {
	s := 1
	for _, c := range n.Children {
		s += c.size()
	}
	return s
}

// clone returns a deep copy of the tree.
func (n *Node) clone() *Node {
	c := &Node{Prim: n.Prim}
	if len(n.Children) > 0 {
		c.Children = make([]*Node, len(n.Children))
		for i, child := range n.Children {
			c.Children[i] = child.clone()
		}
	}
	return c
}

// String returns a human-readable S-expression for the tree.
func (n *Node) String() string {
	if n.Prim.Arity == 0 {
		return n.Prim.Name
	}
	s := "(" + n.Prim.Name
	for _, c := range n.Children {
		s += " " + c.String()
	}
	return s + ")"
}

// Program represents an evolved expression tree.
type Program struct {
	Root    *Node
	Fitness float64
}

// Evaluate computes the program output for the given input vector.
func (p *Program) Evaluate(inputs []float64) float64 {
	return p.Root.Evaluate(inputs)
}

// String returns the S-expression representation of the program.
func (p *Program) String() string {
	return p.Root.String()
}

// FitnessFunc evaluates how well a program solves the target task.
// Lower values indicate better fitness (minimisation).
type FitnessFunc func(p *Program) float64

// GPConfig controls the evolutionary process.
type GPConfig struct {
	// PopulationSize is the number of individuals per generation. Default: 100.
	PopulationSize int
	// MaxGenerations is the maximum number of evolutionary generations. Default: 50.
	MaxGenerations int
	// MaxDepth is the maximum allowed tree depth. Default: 5.
	MaxDepth int
	// CrossoverRate is the probability of applying crossover. Default: 0.9.
	CrossoverRate float64
	// MutationRate is the probability of applying mutation. Default: 0.1.
	MutationRate float64
	// TournamentSize is the number of individuals in tournament selection. Default: 3.
	TournamentSize int
	// Seed sets the random number generator seed. 0 means non-deterministic.
	Seed int64
}

func (c GPConfig) withDefaults() GPConfig {
	if c.PopulationSize <= 0 {
		c.PopulationSize = 100
	}
	if c.MaxGenerations <= 0 {
		c.MaxGenerations = 50
	}
	if c.MaxDepth <= 0 {
		c.MaxDepth = 5
	}
	if c.CrossoverRate <= 0 {
		c.CrossoverRate = 0.9
	}
	if c.MutationRate <= 0 {
		c.MutationRate = 0.1
	}
	if c.TournamentSize <= 0 {
		c.TournamentSize = 3
	}
	return c
}

// Evolve runs a generational genetic programming loop to find a program
// that minimises the given fitness function. It returns the best program
// found across all generations.
func Evolve(primitives []Primitive, fitness FitnessFunc, config GPConfig) (*Program, error) {
	if len(primitives) == 0 {
		return nil, fmt.Errorf("gp: at least one primitive is required")
	}

	// Separate functions from terminals.
	var functions, terminals []Primitive
	for _, p := range primitives {
		if p.Arity == 0 {
			terminals = append(terminals, p)
		} else {
			functions = append(functions, p)
		}
	}
	if len(terminals) == 0 {
		return nil, fmt.Errorf("gp: at least one terminal (arity 0) is required")
	}
	if len(functions) == 0 {
		return nil, fmt.Errorf("gp: at least one function (arity > 0) is required")
	}

	cfg := config.withDefaults()
	var rng *rand.Rand
	if cfg.Seed != 0 {
		rng = rand.New(rand.NewSource(cfg.Seed))
	} else {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}

	gen := &generator{
		functions: functions,
		terminals: terminals,
		maxDepth:  cfg.MaxDepth,
		rng:       rng,
	}

	// Initialise population using ramped half-and-half.
	pop := make([]*Program, cfg.PopulationSize)
	for i := range pop {
		var root *Node
		if i%2 == 0 {
			root = gen.full(cfg.MaxDepth)
		} else {
			root = gen.grow(cfg.MaxDepth)
		}
		p := &Program{Root: root}
		p.Fitness = fitness(p)
		pop[i] = p
	}

	best := pop[0]
	for _, p := range pop[1:] {
		if p.Fitness < best.Fitness {
			best = p
		}
	}

	for g := 0; g < cfg.MaxGenerations; g++ {
		next := make([]*Program, cfg.PopulationSize)

		// Elitism: carry the best individual forward.
		next[0] = &Program{Root: best.Root.clone(), Fitness: best.Fitness}

		for i := 1; i < cfg.PopulationSize; i++ {
			parent1 := tournamentSelect(pop, cfg.TournamentSize, rng)
			child := parent1.Root.clone()

			if rng.Float64() < cfg.CrossoverRate {
				parent2 := tournamentSelect(pop, cfg.TournamentSize, rng)
				child = crossover(child, parent2.Root.clone(), rng)
			}

			if rng.Float64() < cfg.MutationRate {
				child = mutate(child, gen, rng)
			}

			// Enforce depth limit.
			if child.depth() > cfg.MaxDepth {
				child = gen.grow(cfg.MaxDepth)
			}

			p := &Program{Root: child}
			p.Fitness = fitness(p)
			next[i] = p
		}

		pop = next
		for _, p := range pop {
			if p.Fitness < best.Fitness {
				best = p
			}
		}

		// Perfect fitness: stop early.
		if best.Fitness == 0 {
			break
		}
	}

	return &Program{Root: best.Root.clone(), Fitness: best.Fitness}, nil
}

// generator handles random tree creation.
type generator struct {
	functions []Primitive
	terminals []Primitive
	maxDepth  int
	rng       *rand.Rand
}

// full creates a tree where every branch reaches the specified depth.
func (g *generator) full(depth int) *Node {
	if depth == 0 {
		t := g.terminals[g.rng.Intn(len(g.terminals))]
		return &Node{Prim: t}
	}
	f := g.functions[g.rng.Intn(len(g.functions))]
	n := &Node{Prim: f, Children: make([]*Node, int(f.Arity))}
	for i := range n.Children {
		n.Children[i] = g.full(depth - 1)
	}
	return n
}

// grow creates a tree where branches may terminate early.
func (g *generator) grow(depth int) *Node {
	all := append(g.functions, g.terminals...)
	if depth == 0 {
		t := g.terminals[g.rng.Intn(len(g.terminals))]
		return &Node{Prim: t}
	}
	p := all[g.rng.Intn(len(all))]
	if p.Arity == 0 {
		return &Node{Prim: p}
	}
	n := &Node{Prim: p, Children: make([]*Node, int(p.Arity))}
	for i := range n.Children {
		n.Children[i] = g.grow(depth - 1)
	}
	return n
}

// tournamentSelect picks the best individual from a random subset of the population.
func tournamentSelect(pop []*Program, size int, rng *rand.Rand) *Program {
	best := pop[rng.Intn(len(pop))]
	for i := 1; i < size; i++ {
		candidate := pop[rng.Intn(len(pop))]
		if candidate.Fitness < best.Fitness {
			best = candidate
		}
	}
	return best
}

// collectNodes returns pointers to all nodes in the tree along with their depths.
type nodeRef struct {
	parent *Node
	index  int // child index within parent; -1 for root
	node   *Node
	depth  int
}

func collectNodes(root *Node) []nodeRef {
	var refs []nodeRef
	var walk func(n *Node, parent *Node, idx, depth int)
	walk = func(n *Node, parent *Node, idx, depth int) {
		refs = append(refs, nodeRef{parent: parent, index: idx, node: n, depth: depth})
		for i, c := range n.Children {
			walk(c, n, i, depth+1)
		}
	}
	walk(root, nil, -1, 0)
	return refs
}

// crossover performs subtree crossover between two trees.
func crossover(a, b *Node, rng *rand.Rand) *Node {
	nodesA := collectNodes(a)
	nodesB := collectNodes(b)

	// Pick a random node in a (crossover point).
	refA := nodesA[rng.Intn(len(nodesA))]
	// Pick a random node in b (donor subtree).
	refB := nodesB[rng.Intn(len(nodesB))]

	if refA.parent == nil {
		// Replace entire tree.
		return refB.node
	}
	refA.parent.Children[refA.index] = refB.node
	return a
}

// mutate replaces a random subtree with a newly generated one.
func mutate(root *Node, gen *generator, rng *rand.Rand) *Node {
	nodes := collectNodes(root)
	ref := nodes[rng.Intn(len(nodes))]

	remaining := gen.maxDepth - ref.depth
	if remaining < 0 {
		remaining = 0
	}
	newSubtree := gen.grow(remaining)

	if ref.parent == nil {
		return newSubtree
	}
	ref.parent.Children[ref.index] = newSubtree
	return root
}

// Common primitives for convenience.

// AddPrimitive returns a Primitive that computes a + b.
func AddPrimitive() Primitive {
	return Primitive{Name: "Add", Arity: 2, Func: func(args []float64) float64 {
		return args[0] + args[1]
	}}
}

// SubPrimitive returns a Primitive that computes a - b.
func SubPrimitive() Primitive {
	return Primitive{Name: "Sub", Arity: 2, Func: func(args []float64) float64 {
		return args[0] - args[1]
	}}
}

// MulPrimitive returns a Primitive that computes a * b.
func MulPrimitive() Primitive {
	return Primitive{Name: "Mul", Arity: 2, Func: func(args []float64) float64 {
		return args[0] * args[1]
	}}
}

// ProtectedDivPrimitive returns a Primitive that computes a / b, returning 1 when b is near zero.
func ProtectedDivPrimitive() Primitive {
	return Primitive{Name: "Div", Arity: 2, Func: func(args []float64) float64 {
		if math.Abs(args[1]) < 1e-10 {
			return 1
		}
		return args[0] / args[1]
	}}
}

// SinPrimitive returns a Primitive that computes sin(a).
func SinPrimitive() Primitive {
	return Primitive{Name: "Sin", Arity: 1, Func: func(args []float64) float64 {
		return math.Sin(args[0])
	}}
}

// CosPrimitive returns a Primitive that computes cos(a).
func CosPrimitive() Primitive {
	return Primitive{Name: "Cos", Arity: 1, Func: func(args []float64) float64 {
		return math.Cos(args[0])
	}}
}

// VariablePrimitive returns a terminal that reads inputs[index].
func VariablePrimitive(name string, index int) Primitive {
	return Primitive{Name: name, Arity: 0, Func: func(args []float64) float64 {
		if index < len(args) {
			return args[index]
		}
		return 0
	}}
}

// ConstantPrimitive returns a terminal that always produces the given value.
func ConstantPrimitive(name string, value float64) Primitive {
	return Primitive{Name: name, Arity: 0, Func: func(_ []float64) float64 {
		return value
	}}
}
