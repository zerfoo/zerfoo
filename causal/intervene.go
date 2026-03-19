package causal

import (
	"fmt"
	"math"
)

// Intervention specifies a do-calculus intervention: set a variable to a fixed
// value. Data is required to estimate the linear causal coefficients.
type Intervention struct {
	// Variable is the name of the variable to intervene on (must exist in the graph).
	Variable string

	// Value is the value to set the variable to under do(Variable = Value).
	Value float64

	// Data is the observational dataset used to estimate causal coefficients,
	// shaped [n_samples][n_variables] matching the graph's node ordering.
	Data [][]float64
}

// Effect holds the estimated causal effect on a single variable.
type Effect struct {
	// Variable is the name of the affected variable.
	Variable string

	// Value is the predicted value under the intervention.
	Value float64
}

// Prediction holds the result of a causal intervention.
type Prediction struct {
	// IntervenedVariable is the variable that was set.
	IntervenedVariable string

	// IntervenedValue is the value it was set to.
	IntervenedValue float64

	// Effects contains the predicted values for all downstream variables.
	Effects []Effect
}

// Intervene performs a do-calculus intervention on a causal graph: do(Variable = Value).
// It estimates linear causal coefficients from the provided observational data,
// constructs the mutilated graph (removing incoming edges to the intervened
// variable), and propagates the intervention value through downstream variables
// in topological order.
func Intervene(graph *CausalGraph, intervention Intervention) (*Prediction, error) {
	if graph == nil {
		return nil, fmt.Errorf("causal: graph must not be nil")
	}
	if !graph.IsDAG() {
		return nil, fmt.Errorf("causal: graph must be a DAG")
	}

	// Find the intervened variable index.
	targetIdx := -1
	for i, name := range graph.Nodes {
		if name == intervention.Variable {
			targetIdx = i
			break
		}
	}
	if targetIdx < 0 {
		return nil, fmt.Errorf("causal: variable %q not found in graph", intervention.Variable)
	}

	// Validate data.
	if len(intervention.Data) == 0 {
		return nil, fmt.Errorf("causal: intervention data must not be empty")
	}
	nVars := len(graph.Nodes)
	for i, row := range intervention.Data {
		if len(row) != nVars {
			return nil, fmt.Errorf("causal: data row %d has %d columns, expected %d", i, len(row), nVars)
		}
	}

	// Estimate linear coefficients: for each node, regress it on its parents.
	// coefficients[child][parent] = beta
	intercepts := make([]float64, nVars)
	coefficients := make([]map[int]float64, nVars)
	for i := 0; i < nVars; i++ {
		coefficients[i] = make(map[int]float64)
	}

	for child := 0; child < nVars; child++ {
		parents := graph.Parents(child)
		if len(parents) == 0 {
			// Root node: intercept = mean.
			intercepts[child] = columnMean(intervention.Data, child)
			continue
		}
		betas, intercept := olsRegress(intervention.Data, child, parents)
		intercepts[child] = intercept
		for i, p := range parents {
			coefficients[child][p] = betas[i]
		}
	}

	// Topological sort of the graph.
	order := topologicalSort(graph)

	// Propagate values through the mutilated graph.
	// In the mutilated graph, incoming edges to targetIdx are removed.
	values := make([]float64, nVars)

	// For root nodes not downstream of intervention, use their observed means.
	for _, idx := range order {
		if idx == targetIdx {
			values[idx] = intervention.Value
			continue
		}

		// In the mutilated graph, parents of targetIdx are cut.
		// For all other nodes, use their structural equation.
		parents := graph.Parents(idx)
		val := intercepts[idx]
		for _, p := range parents {
			// In the mutilated graph, edges into targetIdx are removed,
			// but edges into other nodes remain unchanged.
			val += coefficients[idx][p] * values[p]
		}
		values[idx] = val
	}

	// Collect downstream effects (nodes reachable from targetIdx).
	downstream := reachable(graph, targetIdx)

	var effects []Effect
	for _, idx := range order {
		if idx == targetIdx {
			continue
		}
		if downstream[idx] {
			effects = append(effects, Effect{
				Variable: graph.Nodes[idx],
				Value:    values[idx],
			})
		}
	}

	return &Prediction{
		IntervenedVariable: intervention.Variable,
		IntervenedValue:    intervention.Value,
		Effects:            effects,
	}, nil
}

// topologicalSort returns the nodes of the graph in topological order using
// Kahn's algorithm.
func topologicalSort(g *CausalGraph) []int {
	n := len(g.Nodes)
	adj := g.AdjacencyMatrix()

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

	var order []int
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		order = append(order, node)
		for j := 0; j < n; j++ {
			if adj[node][j] {
				inDeg[j]--
				if inDeg[j] == 0 {
					queue = append(queue, j)
				}
			}
		}
	}
	return order
}

// reachable returns a set of all nodes reachable from start via directed edges.
func reachable(g *CausalGraph, start int) map[int]bool {
	visited := make(map[int]bool)
	stack := []int{start}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, child := range g.Children(node) {
			if !visited[child] {
				visited[child] = true
				stack = append(stack, child)
			}
		}
	}
	return visited
}

// columnMean returns the mean of the specified column in the dataset.
func columnMean(data [][]float64, col int) float64 {
	sum := 0.0
	for _, row := range data {
		sum += row[col]
	}
	return sum / float64(len(data))
}

// olsRegress fits a simple OLS regression: y = intercept + sum(beta_i * x_i).
// It returns the coefficient for each parent and the intercept.
func olsRegress(data [][]float64, target int, predictors []int) (betas []float64, intercept float64) {
	n := len(data)
	p := len(predictors)

	if p == 0 {
		intercept = columnMean(data, target)
		return nil, intercept
	}

	// For single predictor, use simple regression formula.
	if p == 1 {
		pred := predictors[0]
		meanX := columnMean(data, pred)
		meanY := columnMean(data, target)

		var ssxy, ssxx float64
		for _, row := range data {
			dx := row[pred] - meanX
			dy := row[target] - meanY
			ssxy += dx * dy
			ssxx += dx * dx
		}

		if ssxx < 1e-15 {
			return []float64{0}, meanY
		}
		beta := ssxy / ssxx
		return []float64{beta}, meanY - beta*meanX
	}

	// Multiple predictors: normal equations (X'X)^{-1} X'y.
	// Build X matrix with intercept column.
	cols := p + 1

	// Compute X'X and X'y.
	xtx := make([][]float64, cols)
	for i := range xtx {
		xtx[i] = make([]float64, cols)
	}
	xty := make([]float64, cols)

	for s := 0; s < n; s++ {
		y := data[s][target]
		// First column is intercept (1).
		xty[0] += y
		xtx[0][0] += 1
		for j := 0; j < p; j++ {
			xj := data[s][predictors[j]]
			xtx[0][j+1] += xj
			xtx[j+1][0] += xj
			xty[j+1] += xj * y
			for k := 0; k < p; k++ {
				xtx[j+1][k+1] += data[s][predictors[k]] * xj
			}
		}
	}

	// Solve via Gaussian elimination with partial pivoting.
	coeffs := solveLinearSystem(xtx, xty)
	if coeffs == nil {
		// Singular matrix: fall back to zero coefficients.
		intercept = columnMean(data, target)
		betas = make([]float64, p)
		return betas, intercept
	}

	intercept = coeffs[0]
	betas = make([]float64, p)
	copy(betas, coeffs[1:])
	return betas, intercept
}

// solveLinearSystem solves Ax = b via Gaussian elimination with partial
// pivoting. Returns nil if the matrix is singular.
func solveLinearSystem(a [][]float64, b []float64) []float64 {
	n := len(b)

	// Augmented matrix.
	aug := make([][]float64, n)
	for i := range aug {
		aug[i] = make([]float64, n+1)
		copy(aug[i], a[i])
		aug[i][n] = b[i]
	}

	for col := 0; col < n; col++ {
		// Partial pivoting.
		maxVal := math.Abs(aug[col][col])
		maxRow := col
		for row := col + 1; row < n; row++ {
			if v := math.Abs(aug[row][col]); v > maxVal {
				maxVal = v
				maxRow = row
			}
		}
		if maxVal < 1e-12 {
			return nil
		}
		aug[col], aug[maxRow] = aug[maxRow], aug[col]

		// Eliminate.
		pivot := aug[col][col]
		for row := col + 1; row < n; row++ {
			factor := aug[row][col] / pivot
			for j := col; j <= n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	// Back substitution.
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = aug[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= aug[i][j] * x[j]
		}
		x[i] /= aug[i][i]
	}
	return x
}
