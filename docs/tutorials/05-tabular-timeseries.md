# Part 5: Tabular and Time-Series ML

This tutorial covers Zerfoo's built-in models for structured data prediction and time-series forecasting. These packages let you train and run inference on tabular datasets and temporal data entirely in Go, using the same tensor and compute infrastructure that powers Zerfoo's language model inference.

## Tabular ML Overview

The `tabular` package provides neural network architectures for structured (tabular) data classification. It currently supports four architectures, each offering different trade-offs between simplicity and expressiveness:

| Architecture | Type | Key Idea |
|-------------|------|----------|
| `Model` (MLP) | Feedforward | Configurable multi-layer perceptron baseline |
| `FTTransformer` | Transformer | Feature Tokenizer + Transformer; each feature becomes a token |
| `TabNet` | Attention-based | Sequential attention for feature selection with sparsemax |
| `SAINT` | Transformer | Self-Attention and Intersample Attention across samples |
| `TabResNet` | Residual MLP | MLP with skip connections between hidden layers |

All models output predictions for 3 classes and are built on `ztensor` tensors and `compute.Engine[float32]`.

## Training a Tabular Model

The `tabular.Train` function handles the full training loop: data splitting, model initialization, AdamW optimization, and cross-entropy loss computation.

```go
package main

import (
	"fmt"
	"log"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/zerfoo/tabular"
)

func main() {
	// Prepare your data: each row is a feature vector, each label is 0, 1, or 2.
	data := [][]float64{
		{0.5, 1.2, -0.3, 0.8},
		{1.1, -0.4, 0.7, 1.5},
		{-0.2, 0.9, 1.1, -0.6},
		// ... more rows
	}
	labels := []int{0, 1, 2} // 0=Long, 1=Short, 2=Flat

	eng := compute.NewCPUEngine[float32]()
	ops := numeric.Float32Arithmetic{}

	model, err := tabular.Train(data, labels,
		tabular.TrainConfig{
			Epochs:          100,
			BatchSize:       32,
			LearningRate:    0.001,
			WeightDecay:     1e-4,
			ValidationSplit: 0.2,
		},
		tabular.ModelConfig{
			HiddenDims:  []int{64, 32},
			DropoutRate:  0.1,
			Activation:   tabular.ActivationGELU,
		},
		eng, ops,
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Training complete")
	_ = model // Use model.Predict for inference
}
```

## Using FTTransformer

The FTTransformer (Feature Tokenizer + Transformer) treats each numeric feature as a learned token embedding, prepends a CLS token, and processes the sequence through transformer encoder layers. It is particularly effective when features have complex interactions.

```go
ft, err := tabular.NewFTTransformer(tabular.FTTransformerConfig{
	NumFeatures: 10,
	DToken:      32,
	NHeads:      4,
	NLayers:     3,
	DFFN:        64,
	DropoutRate: 0.1,
}, eng, ops)
if err != nil {
	log.Fatal(err)
}

// Run inference on a feature vector.
ctx := context.Background()
prediction, err := ft.Predict(ctx, featureVector)
```

## Using TabNet

TabNet uses sequential attention to select which features to focus on at each decision step. It provides built-in feature importance through its attention masks.

```go
tn, err := tabular.NewTabNet(tabular.TabNetConfig{
	InputDim:              10,
	OutputDim:             3,
	NSteps:                5,
	RelaxationFactor:      1.5,
	SparsityCoefficient:   1e-3,
	FeatureTransformerDim: 64,
}, eng, ops)
```

## Using SAINT

SAINT (Self-Attention and Intersample Attention) extends the transformer approach by adding attention across samples in a batch, not just across features within a single sample. This can capture relationships between data points.

```go
saint, err := tabular.NewSAINT(tabular.SAINTConfig{
	NumFeatures:          10,
	DModel:               32,
	NHeads:               4,
	NLayers:              3,
	InterSampleAttention: true,
}, eng, ops)
```

## Using TabResNet

TabResNet is a simple but strong baseline: an MLP with residual (skip) connections between hidden layers. It often performs surprisingly well on tabular data.

```go
rn, err := tabular.NewTabResNet(tabular.TabResNetConfig{
	InputDim:   10,
	OutputDim:  3,
	HiddenDims: []int{64, 64, 32},
	Activation: tabular.ActivationGELU,
	Norm:       tabular.NormLayer,
}, eng, ops)
```

## Time-Series Forecasting

The `timeseries` package provides neural architectures for multi-horizon forecasting. These models accept temporal feature sequences and produce point forecasts with optional quantile estimates.

### Temporal Fusion Transformer (TFT)

The TFT is designed for multi-horizon probabilistic forecasting. It uses variable selection networks to identify relevant features, gated residual networks for nonlinear processing, and temporal self-attention to capture long-range dependencies.

```go
import "github.com/zerfoo/zerfoo/timeseries"

tft, err := timeseries.NewTFT(timeseries.TFTConfig{
	NumStaticFeatures: 3,
	NumTimeFeatures:   8,
	DModel:            32,
	NHeads:            4,
	NHorizons:         12,
	Quantiles:         []float64{0.1, 0.5, 0.9},
}, eng, ops)
if err != nil {
	log.Fatal(err)
}

forecast, err := tft.Forecast(ctx, staticFeatures, temporalFeatures)
// forecast.Horizons contains point forecasts for each horizon.
// forecast.Quantiles[0.1] contains the 10th percentile estimates.
```

### N-BEATS

N-BEATS (Neural Basis Expansion Analysis for Time Series) is a pure deep learning approach that decomposes the forecast into interpretable components: trend and seasonality.

```go
nbeats, err := timeseries.NewNBEATS(timeseries.NBEATSConfig{
	InputLength:     60,
	OutputLength:    12,
	StackTypes:      []timeseries.StackType{timeseries.StackTrend, timeseries.StackSeasonality},
	NBlocksPerStack: 3,
	HiddenDim:       128,
	NHarmonics:      8,
}, eng, ops)
if err != nil {
	log.Fatal(err)
}

forecast, err := nbeats.Forecast(ctx, inputSeries)
```

The `StackTrend` type uses polynomial basis expansion, while `StackSeasonality` uses Fourier basis expansion with configurable harmonics. You can also use `StackGeneric` for learned basis functions.

### PatchTST

PatchTST (Patch Time-Series Transformer) divides the input time series into patches (similar to how vision transformers patch images), embeds them, and processes the patch sequence through a standard transformer encoder.

```go
ptst, err := timeseries.NewPatchTST(timeseries.PatchTSTConfig{
	InputLength:        96,
	PatchLength:        16,
	Stride:             8,
	DModel:             64,
	NHeads:             4,
	NLayers:            3,
	OutputDim:          12,
	ChannelIndependent: true,
}, eng, ops)
if err != nil {
	log.Fatal(err)
}

forecast, err := ptst.Forecast(ctx, inputSeries)
```

Setting `ChannelIndependent` to `true` processes each channel (variable) through the same transformer independently, which often works well for multivariate forecasting.

## AutoML

Zerfoo includes an AutoML command that searches over architectures and hyperparameters:

```bash
zerfoo automl --data train.csv --target label --task classification
```

The AutoML system evaluates different tabular architectures (MLP, FTTransformer, TabNet, SAINT, TabResNet) and tunes hyperparameters, returning the best-performing model.

## Saving and Loading Models

Trained tabular models can be saved to disk and loaded later:

```go
import "github.com/zerfoo/zerfoo/tabular"

// Save a trained model.
err := tabular.Save(model, "model.bin")

// Load a saved model.
loaded, err := tabular.Load("model.bin", eng, ops)
```

## Summary

The `tabular` and `timeseries` packages bring modern neural architectures for structured data and forecasting into the Go ecosystem. They share the same tensor and compute infrastructure as Zerfoo's language model inference, so you can run tabular ML and LLM inference in the same Go process with a single dependency tree.

For more information, refer to the package documentation:
- `github.com/zerfoo/zerfoo/tabular`
- `github.com/zerfoo/zerfoo/timeseries`
