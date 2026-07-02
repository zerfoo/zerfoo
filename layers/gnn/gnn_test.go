package gnn

import (
	"math"
	"testing"
)

// triangleGraph returns a 3-node triangle adjacency matrix and random features.
func triangleGraph(featureDim int) ([][]float64, [][]float64) {
	adj := [][]float64{
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 0},
	}
	features := [][]float64{
		{1.0, 0.0, 0.0, 0.5},
		{0.0, 1.0, 0.0, 0.5},
		{0.0, 0.0, 1.0, 0.5},
	}
	// Trim or pad features to featureDim.
	for i := range features {
		if len(features[i]) > featureDim {
			features[i] = features[i][:featureDim]
		}
		for len(features[i]) < featureDim {
			features[i] = append(features[i], 0.1)
		}
	}
	return adj, features
}

func TestGCN_Forward(t *testing.T) {
	tests := []struct {
		name       string
		config     GCNConfig
		nNodes     int
		inputDim   int
		wantOutDim int
	}{
		{
			name: "single hidden layer",
			config: GCNConfig{
				InputDim:   4,
				HiddenDims: []int{8},
				OutputDim:  3,
			},
			nNodes:     3,
			inputDim:   4,
			wantOutDim: 3,
		},
		{
			name: "two hidden layers",
			config: GCNConfig{
				InputDim:   4,
				HiddenDims: []int{8, 6},
				OutputDim:  2,
			},
			nNodes:     3,
			inputDim:   4,
			wantOutDim: 2,
		},
		{
			name: "no hidden layers",
			config: GCNConfig{
				InputDim:  4,
				OutputDim: 5,
			},
			nNodes:     3,
			inputDim:   4,
			wantOutDim: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gcn := NewGCN(tt.config)
			adj, feat := triangleGraph(tt.inputDim)
			out, err := gcn.Forward(adj, feat)
			if err != nil {
				t.Fatalf("Forward() error: %v", err)
			}
			if len(out) != tt.nNodes {
				t.Errorf("output rows = %d, want %d", len(out), tt.nNodes)
			}
			if len(out[0]) != tt.wantOutDim {
				t.Errorf("output cols = %d, want %d", len(out[0]), tt.wantOutDim)
			}
			// Verify non-zero output.
			hasNonZero := false
			for _, row := range out {
				for _, v := range row {
					if v != 0 {
						hasNonZero = true
					}
				}
			}
			if !hasNonZero {
				t.Error("all outputs are zero")
			}
		})
	}
}

func TestGCN_Forward_Errors(t *testing.T) {
	gcn := NewGCN(GCNConfig{InputDim: 4, OutputDim: 2})

	t.Run("empty adjacency", func(t *testing.T) {
		_, err := gcn.Forward([][]float64{}, [][]float64{})
		if err == nil {
			t.Error("expected error for empty adjacency")
		}
	})

	t.Run("mismatched features", func(t *testing.T) {
		adj := [][]float64{{0, 1}, {1, 0}}
		feat := [][]float64{{1, 2, 3, 4}} // only 1 row, need 2
		_, err := gcn.Forward(adj, feat)
		if err == nil {
			t.Error("expected error for mismatched features")
		}
	})

	t.Run("wrong feature dim", func(t *testing.T) {
		adj := [][]float64{{0, 1}, {1, 0}}
		feat := [][]float64{{1, 2}, {3, 4}} // dim 2, need 4
		_, err := gcn.Forward(adj, feat)
		if err == nil {
			t.Error("expected error for wrong feature dim")
		}
	})
}

func TestGCN_Train(t *testing.T) {
	gcn := NewGCN(GCNConfig{
		InputDim:     4,
		HiddenDims:   []int{8},
		OutputDim:    3,
		LearningRate: 0.01,
	})
	adj, feat := triangleGraph(4)
	labels := []int{0, 1, 2}

	// Get initial predictions.
	outBefore, err := gcn.Forward(adj, feat)
	if err != nil {
		t.Fatalf("Forward() before train: %v", err)
	}

	err = gcn.Train(adj, feat, labels, TrainConfig{Epochs: 50})
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	outAfter, err := gcn.Forward(adj, feat)
	if err != nil {
		t.Fatalf("Forward() after train: %v", err)
	}

	// Verify outputs changed after training.
	changed := false
	for i := range outBefore {
		for j := range outBefore[i] {
			if outBefore[i][j] != outAfter[i][j] {
				changed = true
			}
		}
	}
	if !changed {
		t.Error("outputs did not change after training")
	}
}

func TestGAT_Forward(t *testing.T) {
	tests := []struct {
		name       string
		config     GATConfig
		nNodes     int
		inputDim   int
		wantOutDim int
	}{
		{
			name: "single head",
			config: GATConfig{
				InputDim:  4,
				HiddenDim: 8,
				OutputDim: 3,
				NHeads:    1,
			},
			nNodes:     3,
			inputDim:   4,
			wantOutDim: 3,
		},
		{
			name: "multi head",
			config: GATConfig{
				InputDim:  4,
				HiddenDim: 6,
				OutputDim: 2,
				NHeads:    4,
			},
			nNodes:     3,
			inputDim:   4,
			wantOutDim: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gat := NewGAT(tt.config)
			adj, feat := triangleGraph(tt.inputDim)
			out, err := gat.Forward(adj, feat)
			if err != nil {
				t.Fatalf("Forward() error: %v", err)
			}
			if len(out) != tt.nNodes {
				t.Errorf("output rows = %d, want %d", len(out), tt.nNodes)
			}
			if len(out[0]) != tt.wantOutDim {
				t.Errorf("output cols = %d, want %d", len(out[0]), tt.wantOutDim)
			}
			hasNonZero := false
			for _, row := range out {
				for _, v := range row {
					if v != 0 {
						hasNonZero = true
					}
				}
			}
			if !hasNonZero {
				t.Error("all outputs are zero")
			}
		})
	}
}

func TestGAT_AttentionMask(t *testing.T) {
	gat := NewGAT(GATConfig{
		InputDim:  4,
		HiddenDim: 8,
		OutputDim: 3,
		NHeads:    2,
	})

	// Path graph: 0-1-2 (no edge between 0 and 2).
	adj := [][]float64{
		{0, 1, 0},
		{1, 0, 1},
		{0, 1, 0},
	}
	feat := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}

	alpha, err := gat.AttentionMask(adj, feat)
	if err != nil {
		t.Fatalf("AttentionMask() error: %v", err)
	}

	// Check shape.
	if len(alpha) != 3 || len(alpha[0]) != 3 {
		t.Fatalf("attention shape = [%d][%d], want [3][3]", len(alpha), len(alpha[0]))
	}

	// Attention for non-neighbors (excluding self-loops) should be zero.
	// Node 0 and node 2 are not connected, so alpha[0][2] and alpha[2][0] should be 0.
	if alpha[0][2] != 0 {
		t.Errorf("alpha[0][2] = %f, want 0 (non-neighbor)", alpha[0][2])
	}
	if alpha[2][0] != 0 {
		t.Errorf("alpha[2][0] = %f, want 0 (non-neighbor)", alpha[2][0])
	}

	// Attention weights should sum to 1 for each node (over neighbors + self).
	for i, row := range alpha {
		var sum float64
		for _, v := range row {
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-6 {
			t.Errorf("attention sum for node %d = %f, want 1.0", i, sum)
		}
	}

	// Self-attention should be non-zero (self-loops are added).
	for i := range alpha {
		if alpha[i][i] == 0 {
			t.Errorf("alpha[%d][%d] = 0, want non-zero (self-loop)", i, i)
		}
	}

	// Neighbors should have non-zero attention.
	if alpha[0][1] == 0 {
		t.Error("alpha[0][1] = 0, want non-zero (neighbor)")
	}
	if alpha[1][0] == 0 {
		t.Error("alpha[1][0] = 0, want non-zero (neighbor)")
	}
	if alpha[1][2] == 0 {
		t.Error("alpha[1][2] = 0, want non-zero (neighbor)")
	}
}

func TestGAT_Train(t *testing.T) {
	gat := NewGAT(GATConfig{
		InputDim:     4,
		HiddenDim:    8,
		OutputDim:    3,
		NHeads:       2,
		LearningRate: 0.01,
	})
	adj, feat := triangleGraph(4)
	labels := []int{0, 1, 2}

	outBefore, err := gat.Forward(adj, feat)
	if err != nil {
		t.Fatalf("Forward() before train: %v", err)
	}

	err = gat.Train(adj, feat, labels, TrainConfig{Epochs: 50})
	if err != nil {
		t.Fatalf("Train() error: %v", err)
	}

	outAfter, err := gat.Forward(adj, feat)
	if err != nil {
		t.Fatalf("Forward() after train: %v", err)
	}

	changed := false
	for i := range outBefore {
		for j := range outBefore[i] {
			if outBefore[i][j] != outAfter[i][j] {
				changed = true
			}
		}
	}
	if !changed {
		t.Error("outputs did not change after training")
	}
}

func TestNormalizeAdjacency(t *testing.T) {
	adj := [][]float64{
		{0, 1, 0},
		{1, 0, 1},
		{0, 1, 0},
	}
	norm := normalizeAdjacency(adj)

	// After adding self-loops: degrees are [2, 3, 2].
	// D^{-1/2} = diag(1/sqrt(2), 1/sqrt(3), 1/sqrt(2))
	// Check symmetry.
	for i := range norm {
		for j := range norm[i] {
			if math.Abs(norm[i][j]-norm[j][i]) > 1e-10 {
				t.Errorf("norm[%d][%d]=%f != norm[%d][%d]=%f", i, j, norm[i][j], j, i, norm[j][i])
			}
		}
	}

	// Check self-loop value for node 0: (1/sqrt(2))^2 * 1 = 0.5.
	if math.Abs(norm[0][0]-0.5) > 1e-10 {
		t.Errorf("norm[0][0] = %f, want 0.5", norm[0][0])
	}

	// Check edge (0,1): 1/(sqrt(2)*sqrt(3)).
	want01 := 1.0 / math.Sqrt(6)
	if math.Abs(norm[0][1]-want01) > 1e-10 {
		t.Errorf("norm[0][1] = %f, want %f", norm[0][1], want01)
	}
}
