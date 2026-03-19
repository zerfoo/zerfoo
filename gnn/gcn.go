package gnn

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// GCNConfig configures a Graph Convolutional Network.
type GCNConfig struct {
	InputDim     int
	HiddenDims   []int
	OutputDim    int
	DropoutRate  float64
	LearningRate float64
}

// GCN is a multi-layer Graph Convolutional Network.
type GCN struct {
	config  GCNConfig
	weights [][][]float64 // weights[layer][in][out]
	biases  [][]float64   // biases[layer][out]
}

// NewGCN creates a GCN with Xavier-initialized weights.
func NewGCN(config GCNConfig) *GCN {
	dims := make([]int, 0, len(config.HiddenDims)+2)
	dims = append(dims, config.InputDim)
	dims = append(dims, config.HiddenDims...)
	dims = append(dims, config.OutputDim)

	g := &GCN{config: config}
	for i := 0; i < len(dims)-1; i++ {
		g.weights = append(g.weights, xavierMatrix(dims[i], dims[i+1]))
		g.biases = append(g.biases, make([]float64, dims[i+1]))
	}
	return g
}

// Forward performs a forward pass through the GCN.
// adjacency is [n_nodes][n_nodes] and features is [n_nodes][input_dim].
// Returns node embeddings of shape [n_nodes][output_dim].
func (g *GCN) Forward(adjacency, features [][]float64) ([][]float64, error) {
	n := len(adjacency)
	if n == 0 {
		return nil, fmt.Errorf("gnn: empty adjacency matrix")
	}
	if len(features) != n {
		return nil, fmt.Errorf("gnn: features rows %d != adjacency rows %d", len(features), n)
	}
	if len(features[0]) != g.config.InputDim {
		return nil, fmt.Errorf("gnn: feature dim %d != config InputDim %d", len(features[0]), g.config.InputDim)
	}

	normAdj := normalizeAdjacency(adjacency)
	h := features
	for i, w := range g.weights {
		h = matMul(normAdj, h)
		h = matMul(h, w)
		h = addBias(h, g.biases[i])
		if i < len(g.weights)-1 {
			h = reluMatrix(h)
		}
	}
	return h, nil
}

// TrainConfig configures a training run.
type TrainConfig struct {
	Epochs int
}

// Train trains the GCN using gradient descent with cross-entropy loss.
func (g *GCN) Train(adjacency, features [][]float64, labels []int, config TrainConfig) error {
	n := len(adjacency)
	if len(features) != n || len(labels) != n {
		return fmt.Errorf("gnn: mismatched sizes: adjacency=%d features=%d labels=%d", n, len(features), len(labels))
	}

	normAdj := normalizeAdjacency(adjacency)
	lr := g.config.LearningRate
	if lr == 0 {
		lr = 0.01
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Forward pass, saving activations.
		activations := make([][][]float64, len(g.weights)+1)
		activations[0] = features
		h := features
		for i, w := range g.weights {
			h = matMul(normAdj, h)
			h = matMul(h, w)
			h = addBias(h, g.biases[i])
			if i < len(g.weights)-1 {
				h = reluMatrix(h)
				if g.config.DropoutRate > 0 {
					h = dropout(h, g.config.DropoutRate)
				}
			}
			activations[i+1] = h
		}

		// Softmax + cross-entropy gradient at output layer.
		probs := softmaxMatrix(h)
		dOut := make([][]float64, n)
		for i := range dOut {
			dOut[i] = make([]float64, len(probs[i]))
			copy(dOut[i], probs[i])
			if labels[i] < len(dOut[i]) {
				dOut[i][labels[i]] -= 1.0
			}
			for j := range dOut[i] {
				dOut[i][j] /= float64(n)
			}
		}

		// Backpropagation through layers.
		delta := dOut
		for i := len(g.weights) - 1; i >= 0; i-- {
			aH := matMul(normAdj, activations[i])
			// dW = aH^T * delta
			dW := matMulTransposeA(aH, delta)
			// dBias = sum of delta over nodes.
			dB := make([]float64, len(g.biases[i]))
			for row := range delta {
				for col := range delta[row] {
					dB[col] += delta[row][col]
				}
			}

			// Propagate gradient: delta_prev = (normAdj * delta * W^T)
			if i > 0 {
				delta = matMul(delta, transposeMatrix(g.weights[i]))
				delta = matMul(transposeMatrix(normAdj), delta)
				// ReLU derivative.
				for r := range delta {
					for c := range delta[r] {
						if activations[i][r][c] <= 0 {
							delta[r][c] = 0
						}
					}
				}
			}

			// Update weights and biases.
			for r := range g.weights[i] {
				for c := range g.weights[i][r] {
					g.weights[i][r][c] -= lr * dW[r][c]
				}
			}
			for c := range g.biases[i] {
				g.biases[i][c] -= lr * dB[c]
			}
		}
	}
	return nil
}

// normalizeAdjacency computes D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
// where A_tilde = A + I (add self-loops).
func normalizeAdjacency(adj [][]float64) [][]float64 {
	n := len(adj)
	aTilde := make([][]float64, n)
	for i := range aTilde {
		aTilde[i] = make([]float64, n)
		copy(aTilde[i], adj[i])
		aTilde[i][i] += 1.0 // self-loop
	}

	// Compute degree vector and D^{-1/2}.
	dInvSqrt := make([]float64, n)
	for i := range aTilde {
		var deg float64
		for _, v := range aTilde[i] {
			deg += v
		}
		if deg > 0 {
			dInvSqrt[i] = 1.0 / math.Sqrt(deg)
		}
	}

	// D^{-1/2} A_tilde D^{-1/2}
	norm := make([][]float64, n)
	for i := range norm {
		norm[i] = make([]float64, n)
		for j := range norm[i] {
			norm[i][j] = dInvSqrt[i] * aTilde[i][j] * dInvSqrt[j]
		}
	}
	return norm
}

func xavierMatrix(rows, cols int) [][]float64 {
	scale := math.Sqrt(2.0 / float64(rows+cols))
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = rand.NormFloat64() * scale
		}
	}
	return m
}

func matMul(a, b [][]float64) [][]float64 {
	rows := len(a)
	if rows == 0 {
		return nil
	}
	inner := len(a[0])
	cols := len(b[0])
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
		for k := 0; k < inner; k++ {
			aik := a[i][k]
			for j := 0; j < cols; j++ {
				out[i][j] += aik * b[k][j]
			}
		}
	}
	return out
}

func matMulTransposeA(a, b [][]float64) [][]float64 {
	// Computes a^T * b where a is [m][k] and b is [m][n], result is [k][n].
	k := len(a[0])
	n := len(b[0])
	m := len(a)
	out := make([][]float64, k)
	for i := range out {
		out[i] = make([]float64, n)
		for p := 0; p < m; p++ {
			api := a[p][i]
			for j := 0; j < n; j++ {
				out[i][j] += api * b[p][j]
			}
		}
	}
	return out
}

func transposeMatrix(m [][]float64) [][]float64 {
	rows := len(m)
	cols := len(m[0])
	t := make([][]float64, cols)
	for j := range t {
		t[j] = make([]float64, rows)
		for i := range m {
			t[j][i] = m[i][j]
		}
	}
	return t
}

func addBias(m [][]float64, bias []float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			out[i][j] = m[i][j] + bias[j]
		}
	}
	return out
}

func reluMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			if m[i][j] > 0 {
				out[i][j] = m[i][j]
			}
		}
	}
	return out
}

func softmaxMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = make([]float64, len(m[i]))
		max := m[i][0]
		for _, v := range m[i] {
			if v > max {
				max = v
			}
		}
		var sum float64
		for j, v := range m[i] {
			out[i][j] = math.Exp(v - max)
			sum += out[i][j]
		}
		for j := range out[i] {
			out[i][j] /= sum
		}
	}
	return out
}

func dropout(m [][]float64, rate float64) [][]float64 {
	out := make([][]float64, len(m))
	scale := 1.0 / (1.0 - rate)
	for i := range m {
		out[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			if rand.Float64() >= rate {
				out[i][j] = m[i][j] * scale
			}
		}
	}
	return out
}
