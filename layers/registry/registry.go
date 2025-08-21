// package registry provides a central registration point for all layer builders.
// This package exists to break the import cycle between the model package and the
// individual layer packages.
package registry

import (
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/embeddings"
	"github.com/zerfoo/zerfoo/layers/normalization"
	_ "github.com/zerfoo/zerfoo/layers/shape" // Blank import to register the Shape layer
	_ "github.com/zerfoo/zerfoo/layers/gather" // Blank import to register the Gather layer
	_ "github.com/zerfoo/zerfoo/layers/transpose" // Blank import to register the Transpose layer
	"github.com/zerfoo/zerfoo/model"
	"github.com/zerfoo/zerfoo/tensor"
)

func init() {
	// Activations
	model.RegisterLayer("FastGelu", activations.BuildFastGelu[tensor.Float32])

	// Normalization
	model.RegisterLayer("RMSNorm", normalization.BuildRMSNorm[tensor.Float32])
	model.RegisterLayer("SimplifiedLayerNormalization", normalization.BuildSimplifiedLayerNormalization[tensor.Float32])
	model.RegisterLayer("SkipSimplifiedLayerNormalization", normalization.BuildSkipSimplifiedLayerNormalization[tensor.Float32])
}
