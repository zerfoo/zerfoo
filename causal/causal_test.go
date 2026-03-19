package causal

import (
	"math"
	"math/rand"
	"testing"
)

// generateChainData generates data from the DAG: X0 -> X1 -> X2.
// X0 ~ N(0,1), X1 = 0.8*X0 + noise, X2 = 0.8*X1 + noise.
func generateChainData(n int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		x0 := rng.NormFloat64()
		x1 := 0.8*x0 + 0.3*rng.NormFloat64()
		x2 := 0.8*x1 + 0.3*rng.NormFloat64()
		data[i] = []float64{x0, x1, x2}
	}
	return data
}

// generateForkData generates data from the DAG: X0 <- X1 -> X2 (fork).
// X1 ~ N(0,1), X0 = 0.8*X1 + noise, X2 = 0.8*X1 + noise.
func generateForkData(n int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		x1 := rng.NormFloat64()
		x0 := 0.8*x1 + 0.3*rng.NormFloat64()
		x2 := 0.8*x1 + 0.3*rng.NormFloat64()
		data[i] = []float64{x0, x1, x2}
	}
	return data
}

// generateColliderData generates data from the DAG: X0 -> X1 <- X2 (collider).
// X0 ~ N(0,1), X2 ~ N(0,1), X1 = 0.8*X0 + 0.8*X2 + noise.
func generateColliderData(n int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		x0 := rng.NormFloat64()
		x2 := rng.NormFloat64()
		x1 := 0.8*x0 + 0.8*x2 + 0.3*rng.NormFloat64()
		data[i] = []float64{x0, x1, x2}
	}
	return data
}

func TestDiscoverGraph_KnownStructure(t *testing.T) {
	t.Run("chain skeleton X0-X1-X2", func(t *testing.T) {
		// In a chain X0->X1->X2, the PC algorithm correctly identifies the
		// skeleton (X0-X1, X1-X2, no X0-X2 edge) but cannot orient edges
		// because the chain is Markov-equivalent to X0<-X1->X2 (fork).
		data := generateChainData(5000, 42)
		g, err := DiscoverGraph(data, []string{"X0", "X1", "X2"}, DiscoverConfig{Alpha: 0.05})
		if err != nil {
			t.Fatalf("DiscoverGraph error: %v", err)
		}
		if !g.IsDAG() {
			t.Fatal("result is not a DAG")
		}

		// Verify skeleton: edges between (0,1) and (1,2), no edge (0,2).
		skeleton := make(map[[2]int]bool)
		for _, e := range g.Edges {
			skeleton[[2]int{e.From, e.To}] = true
		}
		has01 := skeleton[[2]int{0, 1}] || skeleton[[2]int{1, 0}]
		has12 := skeleton[[2]int{1, 2}] || skeleton[[2]int{2, 1}]
		has02 := skeleton[[2]int{0, 2}] || skeleton[[2]int{2, 0}]

		if !has01 {
			t.Error("missing skeleton edge between X0 and X1")
		}
		if !has12 {
			t.Error("missing skeleton edge between X1 and X2")
		}
		if has02 {
			t.Error("spurious edge between X0 and X2")
		}
	})

	t.Run("collider X0->X1<-X2", func(t *testing.T) {
		// A collider is uniquely identifiable by the PC algorithm: X0 and X2
		// are marginally independent but become dependent when conditioning
		// on X1. The v-structure X0->X1<-X2 must be recovered.
		data := generateColliderData(5000, 42)
		g, err := DiscoverGraph(data, []string{"X0", "X1", "X2"}, DiscoverConfig{Alpha: 0.05})
		if err != nil {
			t.Fatalf("DiscoverGraph error: %v", err)
		}
		if !g.IsDAG() {
			t.Fatal("result is not a DAG")
		}

		wantEdges := map[[2]int]bool{
			{0, 1}: true,
			{2, 1}: true,
		}
		gotEdges := make(map[[2]int]bool)
		for _, e := range g.Edges {
			gotEdges[[2]int{e.From, e.To}] = true
		}
		for edge := range wantEdges {
			if !gotEdges[edge] {
				t.Errorf("missing expected edge %d -> %d", edge[0], edge[1])
			}
		}
		for _, e := range g.Edges {
			key := [2]int{e.From, e.To}
			if !wantEdges[key] {
				t.Errorf("unexpected edge %d -> %d", e.From, e.To)
			}
		}
	})
}

func TestDiscoverGraph_DAGConstraint(t *testing.T) {
	// Generate several random DAGs and verify the output is always a DAG.
	seeds := []int64{1, 7, 42, 100, 999}
	for _, seed := range seeds {
		data := generateChainData(2000, seed)
		g, err := DiscoverGraph(data, []string{"A", "B", "C"}, DiscoverConfig{})
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if !g.IsDAG() {
			t.Errorf("seed %d: output is not a DAG", seed)
		}
	}

	// Also test with fork and collider structures.
	for _, gen := range []func(int, int64) [][]float64{generateForkData, generateColliderData} {
		for _, seed := range seeds {
			data := gen(2000, seed)
			g, err := DiscoverGraph(data, []string{"A", "B", "C"}, DiscoverConfig{})
			if err != nil {
				t.Fatalf("seed %d: %v", seed, err)
			}
			if !g.IsDAG() {
				t.Errorf("seed %d: output is not a DAG", seed)
			}
		}
	}
}

func TestDiscoverGraph_Errors(t *testing.T) {
	tests := []struct {
		name     string
		data     [][]float64
		varNames []string
		wantErr  string
	}{
		{
			name:     "empty data",
			data:     nil,
			varNames: nil,
			wantErr:  "at least one sample",
		},
		{
			name:     "single variable",
			data:     [][]float64{{1}, {2}},
			varNames: []string{"X"},
			wantErr:  "at least 2 variables",
		},
		{
			name:     "varNames mismatch",
			data:     [][]float64{{1, 2}, {3, 4}},
			varNames: []string{"X"},
			wantErr:  "does not match",
		},
		{
			name:     "jagged rows",
			data:     [][]float64{{1, 2}, {3}},
			varNames: []string{"X", "Y"},
			wantErr:  "columns",
		},
		{
			name:     "too few samples",
			data:     [][]float64{{1, 2, 3}},
			varNames: []string{"X", "Y", "Z"},
			wantErr:  "more samples",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := DiscoverGraph(tt.data, tt.varNames, DiscoverConfig{})
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !contains(err.Error(), tt.wantErr) {
				t.Errorf("error %q does not contain %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestCausalGraph_AdjacencyMatrix(t *testing.T) {
	g := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{0, 1}, {1, 2}},
	}
	adj := g.AdjacencyMatrix()
	if !adj[0][1] || !adj[1][2] {
		t.Error("expected edges missing in adjacency matrix")
	}
	if adj[1][0] || adj[0][2] {
		t.Error("unexpected edges in adjacency matrix")
	}
}

func TestCausalGraph_ParentsChildren(t *testing.T) {
	g := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{0, 1}, {2, 1}},
	}
	parents := g.Parents(1)
	if len(parents) != 2 {
		t.Fatalf("expected 2 parents, got %d", len(parents))
	}
	children := g.Children(0)
	if len(children) != 1 || children[0] != 1 {
		t.Errorf("expected children [1], got %v", children)
	}
	if len(g.Children(1)) != 0 {
		t.Errorf("node 1 should have no children")
	}
}

func TestCausalGraph_IsDAG(t *testing.T) {
	dag := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{0, 1}, {1, 2}},
	}
	if !dag.IsDAG() {
		t.Error("expected DAG, got cycle")
	}

	cyclic := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{0, 1}, {1, 2}, {2, 0}},
	}
	if cyclic.IsDAG() {
		t.Error("expected cycle, got DAG")
	}
}

func TestCorrelationMatrix(t *testing.T) {
	data := generateChainData(10000, 42)
	corr := correlationMatrix(data)

	// X0 and X1 should be highly correlated.
	if corr[0][1] < 0.7 {
		t.Errorf("corr(X0,X1) = %f, want > 0.7", corr[0][1])
	}
	// Diagonal should be 1.
	for i := 0; i < 3; i++ {
		if math.Abs(corr[i][i]-1.0) > 1e-10 {
			t.Errorf("corr[%d][%d] = %f, want 1.0", i, i, corr[i][i])
		}
	}
	// Symmetry.
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(corr[i][j]-corr[j][i]) > 1e-10 {
				t.Errorf("corr not symmetric: corr[%d][%d]=%f != corr[%d][%d]=%f", i, j, corr[i][j], j, i, corr[j][i])
			}
		}
	}
}

func TestNormalQuantile(t *testing.T) {
	// z_{0.975} should be approximately 1.96.
	q := normalQuantile(0.975)
	if math.Abs(q-1.96) > 0.01 {
		t.Errorf("normalQuantile(0.975) = %f, want ~1.96", q)
	}
	// z_{0.5} should be 0.
	q = normalQuantile(0.5)
	if math.Abs(q) > 1e-10 {
		t.Errorf("normalQuantile(0.5) = %f, want 0", q)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchSubstr(s, substr)
}

func searchSubstr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
