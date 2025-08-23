// Package training provides tools for training neural networks.
package training

import (
	"github.com/zerfoo/zerfoo/graph"
	"github.com/zerfoo/zerfoo/tensor"
)

// Batch groups the stable inputs for a single training step.
//
// Inputs are provided as a map keyed by the graph's input nodes. Targets
// are provided as a single tensor; strategies may interpret targets
// appropriately for the chosen loss.
type Batch[T tensor.Numeric] struct {
	Inputs  map[graph.Node[T]]*tensor.TensorNumeric[T]
	Targets *tensor.TensorNumeric[T]
}
