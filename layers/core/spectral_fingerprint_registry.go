package core

import (
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// BuildSpectralFingerprint constructs a SpectralFingerprint node from attributes.
// Required attributes:
// - "window" (int): window length to analyze (must be > 1)
// - "top_k" (int): number of non-DC bins to output (must be > 0).
func BuildSpectralFingerprint[T tensor.Numeric](
	engine compute.Engine[T],
	ops numeric.Arithmetic[T],
	_ string,
	_ map[string]*graph.Parameter[T],
	attributes map[string]interface{},
) (graph.Node[T], error) {
	window, _ := attributes["window"].(int)
	topK, _ := attributes["top_k"].(int)

	return NewSpectralFingerprint[T](engine, ops, window, topK)
}

// Statically assert that the type implements the graph.Node interface.
