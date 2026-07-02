package nas

import (
	"math/rand/v2"
	"testing"
)

func TestSearchSpaceSample(t *testing.T) {
	ss := NewSearchSpace(4)
	rng := rand.New(rand.NewPCG(42, 99))

	for i := range 100 {
		cell := ss.Sample(rng)
		if !cell.Valid() {
			t.Fatalf("sample %d: cell is not valid: %+v", i, cell)
		}
		wantEdges := ss.numEdges()
		if len(cell.Edges) != wantEdges {
			t.Fatalf("sample %d: got %d edges, want %d", i, len(cell.Edges), wantEdges)
		}
	}
}

func TestSearchSpaceEnumerate(t *testing.T) {
	// 2 nodes, 2 ops → 1 edge → 2 cells.
	ops := []OpType{OpConv3x3, OpSkipConnect}
	ss := NewSearchSpaceWithOps(2, ops)

	cells := ss.Enumerate(100)
	if len(cells) != 2 {
		t.Fatalf("got %d cells, want 2", len(cells))
	}
	for i, c := range cells {
		if !c.Valid() {
			t.Fatalf("cell %d is not valid: %+v", i, c)
		}
	}

	// Verify the two cells have different ops on the single edge.
	if cells[0].Edges[0].Op == cells[1].Edges[0].Op {
		t.Fatal("enumerated cells should have different ops")
	}

	// 3 nodes, 2 ops → 3 edges → 2^3 = 8 cells.
	ss3 := NewSearchSpaceWithOps(3, ops)
	cells3 := ss3.Enumerate(100)
	if len(cells3) != 8 {
		t.Fatalf("got %d cells, want 8", len(cells3))
	}
	for i, c := range cells3 {
		if !c.Valid() {
			t.Fatalf("cell %d is not valid: %+v", i, c)
		}
	}
}

func TestNumCells(t *testing.T) {
	tests := []struct {
		name     string
		numNodes int
		numOps   int
		want     int64
	}{
		{"2 nodes, 8 ops", 2, 8, 8},      // 1 edge, 8^1
		{"3 nodes, 8 ops", 3, 8, 512},    // 3 edges, 8^3
		{"4 nodes, 8 ops", 4, 8, 262144}, // 6 edges, 8^6
		{"3 nodes, 2 ops", 3, 2, 8},      // 3 edges, 2^3
		{"2 nodes, 2 ops", 2, 2, 2},      // 1 edge, 2^1
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ops := AllOps()[:tt.numOps]
			ss := NewSearchSpaceWithOps(tt.numNodes, ops)
			got := ss.NumCells()
			if got != tt.want {
				t.Errorf("NumCells() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestCellValid(t *testing.T) {
	t.Run("valid cell", func(t *testing.T) {
		c := Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 0, To: 1, Op: OpConv3x3},
				{From: 0, To: 2, Op: OpSkipConnect},
				{From: 1, To: 2, Op: OpMaxPool3x3},
			},
		}
		if !c.Valid() {
			t.Fatal("expected valid cell")
		}
	})

	t.Run("cycle (from >= to)", func(t *testing.T) {
		c := Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 1, To: 0, Op: OpConv3x3}, // from > to → cycle
			},
		}
		if c.Valid() {
			t.Fatal("expected invalid cell with from > to")
		}
	})

	t.Run("self-loop (from == to)", func(t *testing.T) {
		c := Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 1, To: 1, Op: OpConv3x3},
			},
		}
		if c.Valid() {
			t.Fatal("expected invalid cell with self-loop")
		}
	})

	t.Run("node out of range", func(t *testing.T) {
		c := Cell{
			NumNodes: 3,
			Edges: []Edge{
				{From: 0, To: 5, Op: OpConv3x3},
			},
		}
		if c.Valid() {
			t.Fatal("expected invalid cell with out-of-range node")
		}
	})

	t.Run("too few nodes", func(t *testing.T) {
		c := Cell{NumNodes: 1}
		if c.Valid() {
			t.Fatal("expected invalid cell with < 2 nodes")
		}
	})
}
