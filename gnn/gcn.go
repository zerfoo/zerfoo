package gnn

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// cpuEngine is a package-level CPU engine used by matrix helpers.
var cpuEngine = compute.NewCPUEngine[float64](numeric.Float64Ops{})

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

// toTensor2D converts a [][]float64 matrix to a 2D tensor.
func toTensor2D(m [][]float64) *tensor.TensorNumeric[float64] {
	rows := len(m)
	cols := len(m[0])
	flat := make([]float64, rows*cols)
	for i, row := range m {
		copy(flat[i*cols:], row)
	}
	t, _ := tensor.New[float64]([]int{rows, cols}, flat)
	return t
}

// fromTensor2D converts a 2D tensor back to [][]float64.
func fromTensor2D(t *tensor.TensorNumeric[float64]) [][]float64 {
	shape := t.Shape()
	rows, cols := shape[0], shape[1]
	data := t.Data()
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, cols)
		copy(out[i], data[i*cols:(i+1)*cols])
	}
	return out
}

func matMul(a, b [][]float64) [][]float64 {
	if len(a) == 0 {
		return nil
	}
	ta := toTensor2D(a)
	tb := toTensor2D(b)
	result, err := cpuEngine.MatMul(context.Background(), ta, tb)
	if err != nil {
		panic("gnn: matMul engine error: " + err.Error())
	}
	return fromTensor2D(result)
}

func matMulTransposeA(a, b [][]float64) [][]float64 {
	// Computes a^T * b by transposing a first, then using engine MatMul.
	ta := toTensor2D(a)
	tb := toTensor2D(b)
	taT, err := cpuEngine.Transpose(context.Background(), ta, []int{1, 0})
	if err != nil {
		panic("gnn: transpose engine error: " + err.Error())
	}
	result, err := cpuEngine.MatMul(context.Background(), taT, tb)
	if err != nil {
		panic("gnn: matMulTransposeA engine error: " + err.Error())
	}
	return fromTensor2D(result)
}

func transposeMatrix(m [][]float64) [][]float64 {
	t := toTensor2D(m)
	result, err := cpuEngine.Transpose(context.Background(), t, []int{1, 0})
	if err != nil {
		panic("gnn: transpose engine error: " + err.Error())
	}
	return fromTensor2D(result)
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
	t := toTensor2D(m)
	// Softmax along axis 1 (columns within each row).
	result, err := cpuEngine.Softmax(context.Background(), t, 1)
	if err != nil {
		panic("gnn: softmax engine error: " + err.Error())
	}
	return fromTensor2D(result)
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
