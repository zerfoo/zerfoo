package causal

import (
	"math"
	"math/rand"
	"testing"
)

// generateLinearChainData generates data from A -> B -> C with known
// coefficients: B = 0.8*A + noise, C = 0.6*B + noise.
func generateLinearChainData(n int, rng *rand.Rand) [][]float64 {
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		a := rng.NormFloat64()
		b := 0.8*a + 0.1*rng.NormFloat64()
		c := 0.6*b + 0.1*rng.NormFloat64()
		data[i] = []float64{a, b, c}
	}
	return data
}

// generateConfounderData generates data where Z confounds X and Y:
// Z -> X, Z -> Y, X -> Y. X = 0.5*Z + noise, Y = 0.7*X + 0.4*Z + noise.
func generateConfounderData(n int, rng *rand.Rand) [][]float64 {
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		z := rng.NormFloat64()
		x := 0.5*z + 0.1*rng.NormFloat64()
		y := 0.7*x + 0.4*z + 0.1*rng.NormFloat64()
		data[i] = []float64{z, x, y}
	}
	return data
}

func TestIntervene_SimpleChain(t *testing.T) {
	// Graph: A(0) -> B(1) -> C(2)
	// B = 0.8*A + noise, C = 0.6*B + noise
	// do(B = 2.0) should:
	//   - set B = 2.0 (breaking A -> B link)
	//   - predict C ≈ 0.6 * 2.0 = 1.2
	//   - NOT include A in effects (A is upstream, not downstream)
	graph := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{From: 0, To: 1}, {From: 1, To: 2}},
	}

	rng := rand.New(rand.NewSource(42))
	data := generateLinearChainData(10000, rng)

	pred, err := Intervene(graph, Intervention{
		Variable: "B",
		Value:    2.0,
		Data:     data,
	})
	if err != nil {
		t.Fatalf("Intervene error: %v", err)
	}

	if pred.IntervenedVariable != "B" {
		t.Errorf("IntervenedVariable = %q, want %q", pred.IntervenedVariable, "B")
	}
	if pred.IntervenedValue != 2.0 {
		t.Errorf("IntervenedValue = %f, want 2.0", pred.IntervenedValue)
	}

	// Should have exactly one downstream effect: C.
	if len(pred.Effects) != 1 {
		t.Fatalf("expected 1 effect, got %d: %+v", len(pred.Effects), pred.Effects)
	}

	effect := pred.Effects[0]
	if effect.Variable != "C" {
		t.Errorf("effect variable = %q, want %q", effect.Variable, "C")
	}

	// C ≈ 0.6 * 2.0 = 1.2 (with intercept near 0 since data is zero-mean).
	expectedC := 0.6 * 2.0
	if math.Abs(effect.Value-expectedC) > 0.15 {
		t.Errorf("C effect = %f, want ≈ %f (tolerance 0.15)", effect.Value, expectedC)
	}
}

func TestIntervene_SimpleChain_RootIntervention(t *testing.T) {
	// Graph: A(0) -> B(1) -> C(2)
	// do(A = 1.0) should propagate through B and C.
	graph := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{From: 0, To: 1}, {From: 1, To: 2}},
	}

	rng := rand.New(rand.NewSource(42))
	data := generateLinearChainData(10000, rng)

	pred, err := Intervene(graph, Intervention{
		Variable: "A",
		Value:    1.0,
		Data:     data,
	})
	if err != nil {
		t.Fatalf("Intervene error: %v", err)
	}

	// Expect 2 downstream effects: B and C.
	if len(pred.Effects) != 2 {
		t.Fatalf("expected 2 effects, got %d", len(pred.Effects))
	}

	// B ≈ 0.8 * 1.0 = 0.8
	// C ≈ 0.6 * 0.8 = 0.48
	for _, eff := range pred.Effects {
		switch eff.Variable {
		case "B":
			if math.Abs(eff.Value-0.8) > 0.15 {
				t.Errorf("B effect = %f, want ≈ 0.8", eff.Value)
			}
		case "C":
			if math.Abs(eff.Value-0.48) > 0.15 {
				t.Errorf("C effect = %f, want ≈ 0.48", eff.Value)
			}
		default:
			t.Errorf("unexpected effect variable: %q", eff.Variable)
		}
	}
}

func TestIntervene_Confounder(t *testing.T) {
	// Graph: Z(0) -> X(1), Z(0) -> Y(2), X(1) -> Y(2)
	// Z is a confounder of X and Y.
	// X = 0.5*Z + noise, Y = 0.7*X + 0.4*Z + noise
	//
	// Naive correlation would overestimate causal effect of X on Y because
	// Z confounds both. The do-calculus intervention do(X = 1.0) should:
	//   - Cut Z -> X (mutilate graph)
	//   - Set X = 1.0
	//   - Predict Y using both X=1.0 and Z=mean(Z)≈0
	//   - Y ≈ 0.7*1.0 + 0.4*0.0 ≈ 0.7
	graph := &CausalGraph{
		Nodes: []string{"Z", "X", "Y"},
		Edges: []Edge{
			{From: 0, To: 1}, // Z -> X
			{From: 0, To: 2}, // Z -> Y
			{From: 1, To: 2}, // X -> Y
		},
	}

	rng := rand.New(rand.NewSource(42))
	data := generateConfounderData(10000, rng)

	pred, err := Intervene(graph, Intervention{
		Variable: "X",
		Value:    1.0,
		Data:     data,
	})
	if err != nil {
		t.Fatalf("Intervene error: %v", err)
	}

	// Only Y should be downstream of X.
	if len(pred.Effects) != 1 {
		t.Fatalf("expected 1 effect, got %d: %+v", len(pred.Effects), pred.Effects)
	}

	effect := pred.Effects[0]
	if effect.Variable != "Y" {
		t.Errorf("effect variable = %q, want %q", effect.Variable, "Y")
	}

	// Under do(X=1): Y = 0.7*1.0 + 0.4*mean(Z) + intercept.
	// mean(Z) ≈ 0, intercept ≈ 0, so Y ≈ 0.7.
	// The key test: if we DIDN'T handle confounding, naive regression of Y on X
	// alone would give a coefficient > 0.7 (biased by Z). The structural
	// equation approach correctly decomposes the effect.
	if math.Abs(effect.Value-0.7) > 0.15 {
		t.Errorf("Y effect = %f, want ≈ 0.7 (tolerance 0.15)", effect.Value)
	}
}

func TestIntervene_LeafIntervention(t *testing.T) {
	// Intervening on a leaf node should produce no downstream effects.
	graph := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{From: 0, To: 1}, {From: 1, To: 2}},
	}

	rng := rand.New(rand.NewSource(42))
	data := generateLinearChainData(10000, rng)

	pred, err := Intervene(graph, Intervention{
		Variable: "C",
		Value:    5.0,
		Data:     data,
	})
	if err != nil {
		t.Fatalf("Intervene error: %v", err)
	}

	if len(pred.Effects) != 0 {
		t.Errorf("expected 0 effects for leaf intervention, got %d", len(pred.Effects))
	}
}

func TestIntervene_Errors(t *testing.T) {
	graph := &CausalGraph{
		Nodes: []string{"A", "B"},
		Edges: []Edge{{From: 0, To: 1}},
	}
	data := [][]float64{{1, 2}, {3, 4}, {5, 6}}

	tests := []struct {
		name    string
		graph   *CausalGraph
		interv  Intervention
		wantErr string
	}{
		{
			name:    "nil graph",
			graph:   nil,
			interv:  Intervention{Variable: "A", Value: 1.0, Data: data},
			wantErr: "must not be nil",
		},
		{
			name:    "variable not found",
			graph:   graph,
			interv:  Intervention{Variable: "Z", Value: 1.0, Data: data},
			wantErr: "not found",
		},
		{
			name:    "empty data",
			graph:   graph,
			interv:  Intervention{Variable: "A", Value: 1.0, Data: nil},
			wantErr: "must not be empty",
		},
		{
			name:  "data width mismatch",
			graph: graph,
			interv: Intervention{
				Variable: "A",
				Value:    1.0,
				Data:     [][]float64{{1, 2, 3}},
			},
			wantErr: "columns",
		},
		{
			name: "cyclic graph",
			graph: &CausalGraph{
				Nodes: []string{"A", "B"},
				Edges: []Edge{{From: 0, To: 1}, {From: 1, To: 0}},
			},
			interv:  Intervention{Variable: "A", Value: 1.0, Data: data},
			wantErr: "must be a DAG",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Intervene(tt.graph, tt.interv)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !contains(err.Error(), tt.wantErr) {
				t.Errorf("error %q does not contain %q", err.Error(), tt.wantErr)
			}
		})
	}
}

func TestIntervene_MultipleChildren(t *testing.T) {
	// Graph: A(0) -> B(1), A(0) -> C(2)
	// B = 0.5*A + noise, C = 0.3*A + noise
	// do(A = 2.0) should predict B ≈ 1.0 and C ≈ 0.6
	graph := &CausalGraph{
		Nodes: []string{"A", "B", "C"},
		Edges: []Edge{{From: 0, To: 1}, {From: 0, To: 2}},
	}

	rng := rand.New(rand.NewSource(42))
	n := 10000
	data := make([][]float64, n)
	for i := 0; i < n; i++ {
		a := rng.NormFloat64()
		b := 0.5*a + 0.1*rng.NormFloat64()
		c := 0.3*a + 0.1*rng.NormFloat64()
		data[i] = []float64{a, b, c}
	}

	pred, err := Intervene(graph, Intervention{
		Variable: "A",
		Value:    2.0,
		Data:     data,
	})
	if err != nil {
		t.Fatalf("Intervene error: %v", err)
	}

	if len(pred.Effects) != 2 {
		t.Fatalf("expected 2 effects, got %d", len(pred.Effects))
	}

	for _, eff := range pred.Effects {
		switch eff.Variable {
		case "B":
			if math.Abs(eff.Value-1.0) > 0.15 {
				t.Errorf("B effect = %f, want ≈ 1.0", eff.Value)
			}
		case "C":
			if math.Abs(eff.Value-0.6) > 0.15 {
				t.Errorf("C effect = %f, want ≈ 0.6", eff.Value)
			}
		default:
			t.Errorf("unexpected effect variable: %q", eff.Variable)
		}
	}
}
