// Package registry provides a central registration point for all layer builders.
package registry

import (
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/gather"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/reducesum"
	"github.com/zerfoo/zerfoo/layers/transpose"
	"github.com/zerfoo/zerfoo/model"
)

// RegisterAll registers all available layers with the model builder.
func RegisterAll() {
	// Activations
	model.RegisterLayer("FastGelu", activations.BuildFastGelu[float32])

	// Attention
	model.RegisterLayer("GroupQueryAttention", attention.BuildGroupQueryAttention[float32])

	// Core
	model.RegisterLayer("Shape", core.BuildShape[float32])
	model.RegisterLayer("Mul", core.BuildMul[float32])
	model.RegisterLayer("Sub", core.BuildSub[float32])
	model.RegisterLayer("Unsqueeze", core.BuildUnsqueeze[float32])
	model.RegisterLayer("Cast", core.BuildCast[float32])
	model.RegisterLayer("Concat", core.BuildConcat[float32])
	model.RegisterLayer("MatMul", core.BuildMatMul[float32])
	model.RegisterLayer("Reshape", core.BuildReshape[float32])
	model.RegisterLayer("RotaryEmbedding", core.BuildRotaryEmbedding[float32])
	model.RegisterLayer("SpectralFingerprint", core.BuildSpectralFingerprint[float32])

	// Embeddings

	// Gather
	model.RegisterLayer("Gather", gather.BuildGather[float32])

	// Normalization
	model.RegisterLayer("RMSNorm", normalization.BuildRMSNorm[float32])
	model.RegisterLayer("SimplifiedLayerNormalization", normalization.BuildSimplifiedLayerNormalization[float32])
	model.RegisterLayer("SkipSimplifiedLayerNormalization", normalization.BuildSkipSimplifiedLayerNormalization[float32])

	// ReduceSum
	model.RegisterLayer("ReduceSum", reducesum.BuildReduceSum[float32])

	// Transpose
	model.RegisterLayer("Transpose", transpose.BuildTranspose[float32])
}
