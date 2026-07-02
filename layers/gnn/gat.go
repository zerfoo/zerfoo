package gnn

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// GATConfig configures a Graph Attention Network.
type GATConfig struct {
	InputDim     int
	HiddenDim    int
	OutputDim    int
	NHeads       int
	DropoutRate  float64
	LearningRate float64
}

// GAT is a multi-head Graph Attention Network.
type GAT struct {
	config GATConfig
	// Per-head parameters for the first attention layer.
	wHeads [][][]float64 // [head][input_dim][hidden_dim]
	aLeft  [][]float64   // [head][hidden_dim] — attention vector left
	aRight [][]float64   // [head][hidden_dim] — attention vector right
	// Output projection: from NHeads*HiddenDim to OutputDim.
	wOut [][]float64 // [NHeads*HiddenDim][OutputDim]
	bOut []float64   // [OutputDim]
}

// NewGAT creates a GAT with Xavier-initialized weights.
func NewGAT(config GATConfig) *GAT {
	if config.NHeads <= 0 {
		config.NHeads = 1
	}
	g := &GAT{config: config}
	g.wHeads = make([][][]float64, config.NHeads)
	g.aLeft = make([][]float64, config.NHeads)
	g.aRight = make([][]float64, config.NHeads)
	for h := 0; h < config.NHeads; h++ {
		g.wHeads[h] = xavierMatrix(config.InputDim, config.HiddenDim)
		g.aLeft[h] = xavierVector(config.HiddenDim)
		g.aRight[h] = xavierVector(config.HiddenDim)
	}
	concatDim := config.NHeads * config.HiddenDim
	g.wOut = xavierMatrix(concatDim, config.OutputDim)
	g.bOut = make([]float64, config.OutputDim)
	return g
}

// Forward performs a forward pass through the GAT.
// adjacency is [n_nodes][n_nodes] and features is [n_nodes][input_dim].
// Returns node embeddings of shape [n_nodes][output_dim].
func (g *GAT) Forward(adjacency, features [][]float64) ([][]float64, error) {
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

	concat := g.multiHeadAttention(adjacency, features)
	out := matMul(concat, g.wOut)
	out = addBias(out, g.bOut)
	return out, nil
}

// AttentionMask returns the learned attention coefficients for each head.
// Returns [n_nodes][n_nodes] averaged over all heads. Coefficients are zero
// for non-neighbors and sum to 1 over each node's neighborhood.
func (g *GAT) AttentionMask(adjacency, features [][]float64) ([][]float64, error) {
	n := len(adjacency)
	if n == 0 {
		return nil, fmt.Errorf("gnn: empty adjacency matrix")
	}
	if len(features) != n {
		return nil, fmt.Errorf("gnn: features rows %d != adjacency rows %d", len(features), n)
	}

	// Add self-loops for attention computation.
	adjSelf := addSelfLoops(adjacency)

	avg := makeMatrix(n, n)
	for h := 0; h < g.config.NHeads; h++ {
		alpha := g.computeAttention(adjSelf, features, h)
		for i := range avg {
			for j := range avg[i] {
				avg[i][j] += alpha[i][j] / float64(g.config.NHeads)
			}
		}
	}
	return avg, nil
}

// Train trains the GAT using gradient descent with cross-entropy loss.
func (g *GAT) Train(adjacency, features [][]float64, labels []int, config TrainConfig) error {
	n := len(adjacency)
	if len(features) != n || len(labels) != n {
		return fmt.Errorf("gnn: mismatched sizes: adjacency=%d features=%d labels=%d", n, len(features), len(labels))
	}

	lr := g.config.LearningRate
	if lr == 0 {
		lr = 0.01
	}

	for epoch := 0; epoch < config.Epochs; epoch++ {
		// Forward pass.
		concat := g.multiHeadAttention(adjacency, features)
		logits := matMul(concat, g.wOut)
		logits = addBias(logits, g.bOut)
		probs := softmaxMatrix(logits)

		// Output gradient.
		dLogits := make([][]float64, n)
		for i := range dLogits {
			dLogits[i] = make([]float64, g.config.OutputDim)
			copy(dLogits[i], probs[i])
			if labels[i] < len(dLogits[i]) {
				dLogits[i][labels[i]] -= 1.0
			}
			for j := range dLogits[i] {
				dLogits[i][j] /= float64(n)
			}
		}

		// Gradient for wOut and bOut.
		dWOut := matMulTransposeA(concat, dLogits)
		dBOut := make([]float64, g.config.OutputDim)
		for i := range dLogits {
			for j := range dLogits[i] {
				dBOut[j] += dLogits[i][j]
			}
		}

		// Gradient flowing back to concat.
		dConcat := matMul(dLogits, transposeMatrix(g.wOut))

		// Update wOut and bOut.
		for r := range g.wOut {
			for c := range g.wOut[r] {
				g.wOut[r][c] -= lr * dWOut[r][c]
			}
		}
		for c := range g.bOut {
			g.bOut[c] -= lr * dBOut[c]
		}

		// Update per-head weights using approximate gradient.
		// dConcat is [n][NHeads*HiddenDim], split into per-head [n][HiddenDim].
		adjSelf := addSelfLoops(adjacency)
		hd := g.config.HiddenDim
		for head := 0; head < g.config.NHeads; head++ {
			// Extract this head's gradient slice.
			dHead := make([][]float64, n)
			for i := range dHead {
				dHead[i] = dConcat[i][head*hd : (head+1)*hd]
			}

			// Projected features for this head.
			wh := matMul(features, g.wHeads[head]) // [n][hd]

			// Attention weights for this head.
			alpha := g.computeAttention(adjSelf, features, head)

			// Gradient of attention output w.r.t. wh:
			// output_i = sum_j alpha_ij * wh_j
			// d(loss)/d(wh_j) = sum_i alpha_ij * dHead_i
			dWh := matMul(transposeMatrix(alpha), dHead) // [n][hd]

			// dW_head = features^T * dWh
			dW := matMulTransposeA(features, dWh)
			for r := range g.wHeads[head] {
				for c := range g.wHeads[head][r] {
					g.wHeads[head][r][c] -= lr * dW[r][c]
				}
			}

			// Approximate gradient for attention vectors.
			for i := range wh {
				for d := 0; d < hd; d++ {
					g.aLeft[head][d] -= lr * dHead[i][d] * wh[i][d] / float64(n)
					g.aRight[head][d] -= lr * dHead[i][d] * wh[i][d] / float64(n)
				}
			}
		}
	}
	return nil
}

func (g *GAT) multiHeadAttention(adjacency, features [][]float64) [][]float64 {
	n := len(features)
	adjSelf := addSelfLoops(adjacency)
	hd := g.config.HiddenDim

	concat := make([][]float64, n)
	for i := range concat {
		concat[i] = make([]float64, g.config.NHeads*hd)
	}

	for h := 0; h < g.config.NHeads; h++ {
		alpha := g.computeAttention(adjSelf, features, h)
		wh := matMul(features, g.wHeads[h])
		out := matMul(alpha, wh)
		for i := range out {
			copy(concat[i][h*hd:(h+1)*hd], out[i])
		}
	}
	return concat
}

func (g *GAT) computeAttention(adjSelf [][]float64, features [][]float64, head int) [][]float64 {
	n := len(features)
	hd := g.config.HiddenDim
	wh := matMul(features, g.wHeads[head]) // [n][hd]

	// Compute attention scores e_ij = LeakyReLU(aLeft^T Wh_i + aRight^T Wh_j).
	leftScores := make([]float64, n)
	rightScores := make([]float64, n)
	for i := 0; i < n; i++ {
		for d := 0; d < hd; d++ {
			leftScores[i] += g.aLeft[head][d] * wh[i][d]
			rightScores[i] += g.aRight[head][d] * wh[i][d]
		}
	}

	alpha := make([][]float64, n)
	for i := range alpha {
		alpha[i] = make([]float64, n)
		maxVal := math.Inf(-1)
		for j := 0; j < n; j++ {
			if adjSelf[i][j] == 0 {
				continue
			}
			e := leakyReLU(leftScores[i]+rightScores[j], 0.2)
			alpha[i][j] = e
			if e > maxVal {
				maxVal = e
			}
		}
		// Masked softmax over neighbors.
		var sum float64
		for j := 0; j < n; j++ {
			if adjSelf[i][j] == 0 {
				alpha[i][j] = 0
				continue
			}
			alpha[i][j] = math.Exp(alpha[i][j] - maxVal)
			sum += alpha[i][j]
		}
		if sum > 0 {
			for j := range alpha[i] {
				alpha[i][j] /= sum
			}
		}
	}
	return alpha
}

func addSelfLoops(adj [][]float64) [][]float64 {
	n := len(adj)
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, n)
		copy(out[i], adj[i])
		out[i][i] = 1.0
	}
	return out
}

func makeMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
	}
	return m
}

func leakyReLU(x, negSlope float64) float64 {
	if x > 0 {
		return x
	}
	return negSlope * x
}

func xavierVector(n int) []float64 {
	scale := math.Sqrt(2.0 / float64(n))
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.NormFloat64() * scale
	}
	return v
}
