package causal

import "fmt"

// Edge represents a directed causal relationship where From causes To.
type Edge struct {
	From int
	To   int
}

// CausalGraph holds the result of causal structure discovery.
type CausalGraph struct {
	Nodes []string
	Edges []Edge
}

// AdjacencyMatrix returns a directed adjacency matrix where adj[i][j] is true
// when there is a directed edge from node i to node j.
func (g *CausalGraph) AdjacencyMatrix() [][]bool {
	n := len(g.Nodes)
	adj := make([][]bool, n)
	for i := range adj {
		adj[i] = make([]bool, n)
	}
	for _, e := range g.Edges {
		adj[e.From][e.To] = true
	}
	return adj
}

// Parents returns the indices of all parent nodes of the given node.
func (g *CausalGraph) Parents(node int) []int {
	var parents []int
	for _, e := range g.Edges {
		if e.To == node {
			parents = append(parents, e.From)
		}
	}
	return parents
}

// Children returns the indices of all child nodes of the given node.
func (g *CausalGraph) Children(node int) []int {
	var children []int
	for _, e := range g.Edges {
		if e.From == node {
			children = append(children, e.To)
		}
	}
	return children
}

// IsDAG returns true if the graph is a valid directed acyclic graph.
func (g *CausalGraph) IsDAG() bool {
	n := len(g.Nodes)
	adj := g.AdjacencyMatrix()

	// Kahn's algorithm: topological sort via in-degree counting.
	inDeg := make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if adj[i][j] {
				inDeg[j]++
			}
		}
	}
	queue := make([]int, 0, n)
	for i, d := range inDeg {
		if d == 0 {
			queue = append(queue, i)
		}
	}
	visited := 0
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		visited++
		for j := 0; j < n; j++ {
			if adj[node][j] {
				inDeg[j]--
				if inDeg[j] == 0 {
					queue = append(queue, j)
				}
			}
		}
	}
	return visited == n
}

// DiscoverConfig controls the behaviour of DiscoverGraph.
type DiscoverConfig struct {
	// Alpha is the significance level for conditional independence tests.
	// Default: 0.05.
	Alpha float64

	// MaxConditioningSet is the maximum size of conditioning sets explored
	// during skeleton discovery. A value of 0 means no limit.
	MaxConditioningSet int
}

// DiscoverGraph learns a causal DAG from observational data using the PC
// algorithm. data is shaped [n_samples][n_variables] and varNames provides a
// label for each variable column.
func DiscoverGraph(data [][]float64, varNames []string, config DiscoverConfig) (*CausalGraph, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("causal: data must have at least one sample")
	}
	nVars := len(data[0])
	if nVars < 2 {
		return nil, fmt.Errorf("causal: need at least 2 variables, got %d", nVars)
	}
	if len(varNames) != nVars {
		return nil, fmt.Errorf("causal: varNames length %d does not match data width %d", len(varNames), nVars)
	}
	for i, row := range data {
		if len(row) != nVars {
			return nil, fmt.Errorf("causal: row %d has %d columns, expected %d", i, len(row), nVars)
		}
	}
	nSamples := len(data)
	if nSamples <= nVars {
		return nil, fmt.Errorf("causal: need more samples (%d) than variables (%d)", nSamples, nVars)
	}

	alpha := config.Alpha
	if alpha <= 0 {
		alpha = 0.05
	}
	maxCond := config.MaxConditioningSet

	// Compute correlation matrix once.
	corr := correlationMatrix(data)

	// Phase 1: skeleton discovery.
	skeleton, sepSets := discoverSkeleton(nVars, nSamples, corr, alpha, maxCond)

	// Phase 2: orient v-structures.
	directed := orientVStructures(nVars, skeleton, sepSets)

	// Phase 3: apply Meek's rules.
	applyMeekRules(nVars, skeleton, directed)

	// Build result graph. For edges that were oriented (v-structures or Meek's
	// rules), use the discovered direction. For remaining undirected edges,
	// orient them in index order (i -> j where i < j) to produce a valid DAG.
	g := &CausalGraph{Nodes: make([]string, nVars)}
	copy(g.Nodes, varNames)

	for i := 0; i < nVars; i++ {
		for j := i + 1; j < nVars; j++ {
			if !skeleton[i][j] {
				continue
			}
			if directed[i][j] {
				g.Edges = append(g.Edges, Edge{From: i, To: j})
			} else if directed[j][i] {
				g.Edges = append(g.Edges, Edge{From: j, To: i})
			} else {
				// Undirected edge: orient i -> j (lower index to higher).
				g.Edges = append(g.Edges, Edge{From: i, To: j})
			}
		}
	}

	return g, nil
}
