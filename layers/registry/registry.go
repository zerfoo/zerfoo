// Package registry provides a central registration point for all layer builders.
package registry

import (
	"github.com/zerfoo/zerfoo/layers/activations"
	"github.com/zerfoo/zerfoo/layers/attention"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/layers/gather"
	"github.com/zerfoo/zerfoo/layers/normalization"
	"github.com/zerfoo/zerfoo/layers/reducesum"
	"github.com/zerfoo/zerfoo/layers/regularization"
	"github.com/zerfoo/zerfoo/layers/transpose"
	"github.com/zerfoo/zerfoo/model"
)

// RegisterAll registers all available layers with the model builder.
func RegisterAll() {
	// Activations
	model.RegisterLayer("FastGelu", activations.BuildFastGelu[float32])
	model.RegisterLayer("Gelu", activations.BuildGelu[float32])
	model.RegisterLayer("Tanh", activations.BuildTanh[float32])
	model.RegisterLayer("Sigmoid", activations.BuildSigmoid[float32])
	model.RegisterLayer("Softmax", activations.BuildSoftmax[float32])
	model.RegisterLayer("Erf", activations.BuildErf[float32])

	// Attention
	model.RegisterLayer("GroupQueryAttention", attention.BuildGroupQueryAttention[float32])
	model.RegisterLayer("GlobalAttention", attention.BuildGlobalAttention[float32])
	model.RegisterLayer("MultiHeadLatentAttention", attention.BuildMultiHeadLatentAttention[float32])

	// Core
	model.RegisterLayer("Add", core.BuildAdd[float32])
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
	model.RegisterLayer("FiLM", core.BuildFiLM[float32])
	model.RegisterLayer("Slice", core.BuildSlice[float32])
	model.RegisterLayer("Pad", core.BuildPad[float32])
	model.RegisterLayer("TopK", core.BuildTopK[float32])
	model.RegisterLayer("Conv", core.BuildConv2d[float32])
	model.RegisterLayer("GlobalAveragePool", core.BuildGlobalAveragePool[float32])
	model.RegisterLayer("Resize", core.BuildResize[float32])
	model.RegisterLayer("MoEGate", core.BuildMoEGate[float32])
	model.RegisterLayer("MixtureOfExperts", core.BuildMixtureOfExperts[float32])
	model.RegisterLayer("FFN", core.BuildFFN[float32])
	model.RegisterLayer("Pow", core.BuildPow[float32])
	model.RegisterLayer("Div", core.BuildDiv[float32])
	model.RegisterLayer("Sqrt", core.BuildSqrt[float32])
	model.RegisterLayer("Neg", core.BuildNeg[float32])
	model.RegisterLayer("Cos", core.BuildCos[float32])
	model.RegisterLayer("Sin", core.BuildSin[float32])
	model.RegisterLayer("ReduceMean", core.BuildReduceMean[float32])
	model.RegisterLayer("Equal", core.BuildEqual[float32])
	model.RegisterLayer("Greater", core.BuildGreater[float32])
	model.RegisterLayer("Where", core.BuildWhere[float32])
	model.RegisterLayer("Expand", core.BuildExpand[float32])
	model.RegisterLayer("Range", core.BuildRange[float32])
	model.RegisterLayer("ConstantOfShape", core.BuildConstantOfShape[float32])
	model.RegisterLayer("ScatterND", core.BuildScatterND[float32])
	model.RegisterLayer("Trilu", core.BuildTrilu[float32])
	model.RegisterLayer("Max", core.BuildMax[float32])
	model.RegisterLayer("LessOrEqual", core.BuildLessOrEqual[float32])
	model.RegisterLayer("Or", core.BuildOr[float32])
	model.RegisterLayer("Squeeze", core.BuildSqueeze[float32])
	model.RegisterLayer("Tile", core.BuildTile[float32])
	model.RegisterLayer("Mod", core.BuildMod[float32])
	model.RegisterLayer("Gemm", core.BuildGemm[float32])

	// Embeddings

	// Gather
	model.RegisterLayer("Gather", gather.BuildGather[float32])

	// Normalization
	model.RegisterLayer("RMSNorm", normalization.BuildRMSNorm[float32])
	model.RegisterLayer("LayerNormalization", normalization.BuildLayerNormalization[float32])
	model.RegisterLayer("SimplifiedLayerNormalization", normalization.BuildSimplifiedLayerNormalization[float32])
	model.RegisterLayer("SkipSimplifiedLayerNormalization", normalization.BuildSkipSimplifiedLayerNormalization[float32])
	model.RegisterLayer("BatchNormalization", normalization.BuildBatchNormalization[float32])

	// Regularization
	model.RegisterLayer("Dropout", regularization.BuildDropout[float32])
	model.RegisterLayer("FeatureDropout", regularization.BuildFeatureDropout[float32])

	// ReduceSum
	model.RegisterLayer("ReduceSum", reducesum.BuildReduceSum[float32])

	// Transpose
	model.RegisterLayer("Transpose", transpose.BuildTranspose[float32])
}
