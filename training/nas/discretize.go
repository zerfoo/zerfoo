package nas

import (
	"errors"
	"fmt"

	"github.com/zerfoo/ztensor/tensor"
)

// DiscretizedArch represents a concrete architecture obtained by selecting the
// argmax operation for each edge in the cell DAG.
type DiscretizedArch struct {
	Cell        Cell
	TotalParams int64
}

// DefaultOpParams returns estimated parameter counts for each operation type,
// assuming a spatial dimension of 32x32 with 64 input/output channels.
func DefaultOpParams() map[OpType]int64 {
	channels := int64(64)
	return map[OpType]int64{
		OpConv3x3:     9 * channels * channels,         // K^2 * C_in * C_out
		OpConv5x5:     25 * channels * channels,        // K^2 * C_in * C_out
		OpSepConv3x3:  9*channels + channels*channels,  // depthwise + pointwise
		OpSepConv5x5:  25*channels + channels*channels, // depthwise + pointwise
		OpAvgPool3x3:  0,
		OpMaxPool3x3:  0,
		OpSkipConnect: 0,
		OpZero:        0,
	}
}

// Discretize converts continuous DARTS architecture weights (alpha) into a
// concrete cell architecture by selecting the argmax operation per edge. It
// validates the resulting architecture against a maximum parameter budget.
//
// The alpha slice must have length numEdges * numOps, laid out as
// [edge0_op0, edge0_op1, ..., edge0_opN, edge1_op0, ...]. This matches the
// alpha parameter shape from DARTSLayer when search space has numOps candidates.
func Discretize[T tensor.Numeric](alpha []T, space *SearchSpace, maxParams int64) (*DiscretizedArch, error) {
	if space == nil {
		return nil, errors.New("nas: Discretize requires a non-nil SearchSpace")
	}

	numOps := len(space.Ops)
	if numOps == 0 {
		return nil, errors.New("nas: SearchSpace has no operations")
	}

	numEdges := space.numEdges()
	if len(alpha) != numEdges*numOps {
		return nil, fmt.Errorf("nas: alpha length %d does not match numEdges(%d) * numOps(%d) = %d",
			len(alpha), numEdges, numOps, numEdges*numOps)
	}

	// Build the canonical edge list (from, to) in the same order as SearchSpace.
	type pair struct{ from, to int }
	pairs := make([]pair, 0, numEdges)
	for i := range space.NumNodes {
		for j := i + 1; j < space.NumNodes; j++ {
			pairs = append(pairs, pair{i, j})
		}
	}

	// Select argmax op per edge.
	opParams := DefaultOpParams()
	edges := make([]Edge, numEdges)
	var totalParams int64

	for e := range numEdges {
		offset := e * numOps
		bestIdx := 0
		bestVal := alpha[offset]
		for k := 1; k < numOps; k++ {
			if greaterThan(alpha[offset+k], bestVal) {
				bestIdx = k
				bestVal = alpha[offset+k]
			}
		}

		selectedOp := space.Ops[bestIdx]
		edges[e] = Edge{
			From: pairs[e].from,
			To:   pairs[e].to,
			Op:   selectedOp,
		}

		if p, ok := opParams[selectedOp]; ok {
			totalParams += p
		}
	}

	if maxParams > 0 && totalParams > maxParams {
		return nil, fmt.Errorf("nas: discretized architecture has %d params, exceeds max_params %d",
			totalParams, maxParams)
	}

	return &DiscretizedArch{
		Cell: Cell{
			NumNodes: space.NumNodes,
			Edges:    edges,
		},
		TotalParams: totalParams,
	}, nil
}

// greaterThan returns true if a > b for any ordered numeric type.
func greaterThan[T tensor.Numeric](a, b T) bool {
	return a > b
}
