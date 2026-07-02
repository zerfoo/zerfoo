// Package nas implements Neural Architecture Search for the Zerfoo ML framework.
//
// The search space is defined as a directed acyclic graph (DAG) of cells,
// where each cell contains nodes connected by edges. Each edge carries an
// operation type (e.g., convolution, pooling, skip connection). The search
// space can be sampled randomly or enumerated exhaustively for small spaces.
package nas

import "math/rand/v2"

// OpType represents an operation type that can be placed on a cell edge.
type OpType string

const (
	OpConv3x3     OpType = "conv_3x3"
	OpConv5x5     OpType = "conv_5x5"
	OpSepConv3x3  OpType = "sep_conv_3x3"
	OpSepConv5x5  OpType = "sep_conv_5x5"
	OpAvgPool3x3  OpType = "avg_pool_3x3"
	OpMaxPool3x3  OpType = "max_pool_3x3"
	OpSkipConnect OpType = "skip_connect"
	OpZero        OpType = "zero"
)

// AllOps returns the default set of all 8 operation types.
func AllOps() []OpType {
	return []OpType{
		OpConv3x3, OpConv5x5, OpSepConv3x3, OpSepConv5x5,
		OpAvgPool3x3, OpMaxPool3x3, OpSkipConnect, OpZero,
	}
}

// Edge represents a directed edge in a cell DAG, connecting node From to node
// To with operation Op. Edges must satisfy From < To to ensure acyclicity.
type Edge struct {
	From int
	To   int
	Op   OpType
}

// Cell represents a single architecture cell as a DAG of edges between nodes.
type Cell struct {
	NumNodes int
	Edges    []Edge
}

// Valid reports whether the cell is a valid DAG: every edge must have From < To
// (which guarantees no cycles), and node indices must be in [0, NumNodes).
func (c Cell) Valid() bool {
	if c.NumNodes < 2 {
		return false
	}
	for _, e := range c.Edges {
		if e.From >= e.To {
			return false
		}
		if e.From < 0 || e.To >= c.NumNodes {
			return false
		}
	}
	return true
}

// SearchSpace defines the space of possible cell architectures. It is
// parameterized by the number of nodes and the set of candidate operations.
type SearchSpace struct {
	NumNodes int
	Ops      []OpType
}

// NewSearchSpace creates a search space with the given number of nodes and all
// 8 default operation types.
func NewSearchSpace(numNodes int) *SearchSpace {
	return &SearchSpace{
		NumNodes: numNodes,
		Ops:      AllOps(),
	}
}

// NewSearchSpaceWithOps creates a search space with the given number of nodes
// and a custom set of operation types.
func NewSearchSpaceWithOps(numNodes int, ops []OpType) *SearchSpace {
	return &SearchSpace{
		NumNodes: numNodes,
		Ops:      ops,
	}
}

// numEdges returns the number of edges in a cell: numNodes*(numNodes-1)/2.
func (s *SearchSpace) numEdges() int {
	return s.NumNodes * (s.NumNodes - 1) / 2
}

// NumCells returns the total number of possible cell architectures:
// len(Ops)^numEdges where numEdges = numNodes*(numNodes-1)/2.
func (s *SearchSpace) NumCells() int64 {
	numOps := int64(len(s.Ops))
	numEdges := s.numEdges()
	result := int64(1)
	for range numEdges {
		result *= numOps
	}
	return result
}

// Sample randomly samples a valid cell architecture from the search space.
func (s *SearchSpace) Sample(rng *rand.Rand) Cell {
	edges := make([]Edge, 0, s.numEdges())
	for i := range s.NumNodes {
		for j := i + 1; j < s.NumNodes; j++ {
			op := s.Ops[rng.IntN(len(s.Ops))]
			edges = append(edges, Edge{From: i, To: j, Op: op})
		}
	}
	return Cell{NumNodes: s.NumNodes, Edges: edges}
}

// Enumerate returns up to maxCells cell architectures by exhaustive
// enumeration. The cells are generated in lexicographic order over the
// operation assignments to edges.
func (s *SearchSpace) Enumerate(maxCells int) []Cell {
	numEdges := s.numEdges()
	total := s.NumCells()
	if int64(maxCells) < total {
		total = int64(maxCells)
	}

	// Build the list of (from, to) pairs in canonical order.
	type pair struct{ from, to int }
	pairs := make([]pair, 0, numEdges)
	for i := range s.NumNodes {
		for j := i + 1; j < s.NumNodes; j++ {
			pairs = append(pairs, pair{i, j})
		}
	}

	numOps := len(s.Ops)
	// opIdx tracks the current operation index for each edge position.
	opIdx := make([]int, numEdges)

	cells := make([]Cell, 0, total)
	for range total {
		edges := make([]Edge, numEdges)
		for k, p := range pairs {
			edges[k] = Edge{From: p.from, To: p.to, Op: s.Ops[opIdx[k]]}
		}
		cells = append(cells, Cell{NumNodes: s.NumNodes, Edges: edges})

		// Increment the multi-digit counter (little-endian).
		for k := numEdges - 1; k >= 0; k-- {
			opIdx[k]++
			if opIdx[k] < numOps {
				break
			}
			opIdx[k] = 0
		}
	}
	return cells
}
