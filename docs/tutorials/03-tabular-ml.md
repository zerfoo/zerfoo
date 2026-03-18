# Tutorial 3: Tabular ML with tabular.Model

The `tabular` package provides a configurable MLP for classification on structured (tabular) data. It uses the `ztensor` compute engine for all arithmetic, meaning the same model code runs on CPU today and GPU tomorrow without changes.

The model outputs one of three classes: `Long`, `Short`, or `Flat` — designed for financial signal generation, but usable for any 3-class classification task.

## Prerequisites

```bash
mkdir tabular-demo && cd tabular-demo
go mod init example.com/tabular-demo
go get github.com/zerfoo/zerfoo@latest
```

## Understanding the API

```go
// ModelConfig defines the network architecture.
type ModelConfig struct {
    InputDim    int        // number of input features
    HiddenDims  []int      // size of each hidden layer
    DropoutRate float64    // dropout probability during training
    Activation  Activation // ActivationReLU or ActivationGELU
}

// TrainConfig defines training hyperparameters.
type TrainConfig struct {
    Epochs          int
    BatchSize       int
    LearningRate    float64
    WeightDecay     float64
    ValidationSplit float64 // fraction of data held out for validation
}

// Train runs full training and returns a ready-to-use model.
// labels[i] must be in [0, 3): 0=Long, 1=Short, 2=Flat.
func Train(data [][]float64, labels []int, config TrainConfig, mc ModelConfig,
    engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Model, error)

// Predict runs inference on a single feature vector.
func (m *Model) Predict(features []float64) (Direction, float64, error)

// Save/Load persist the model in ZTAB binary format.
func Save(model *Model, path string) error
func Load(path string, engine compute.Engine[float32], ops numeric.Arithmetic[float32]) (*Model, error)
```

## Step 1: Prepare Your Data

```go
package main

import (
	"fmt"
	"log"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func main() {
	// Synthesize 1000 samples with 8 features each.
	// In practice, load from a CSV or database.
	const (
		nSamples = 1000
		nFeatures = 8
	)

	data := make([][]float64, nSamples)
	labels := make([]int, nSamples)
	for i := range data {
		row := make([]float64, nFeatures)
		for j := range row {
			row[j] = rand.NormFloat64()
		}
		data[i] = row
		labels[i] = i % 3 // 0=Long, 1=Short, 2=Flat
	}

	// Create a CPU compute engine. On GPU hosts, swap in a CUDA engine
	// without changing any other code.
	engine := compute.NewCPUEngine[float32]()
	ops := numeric.NewArithmetic[float32]()

	trainAndPredict(data, labels, engine, ops)
}
```

## Step 2: Train the Model

```go
func trainAndPredict(data [][]float64, labels []int,
	engine compute.Engine[float32], ops numeric.Arithmetic[float32]) {

	mc := tabular.ModelConfig{
		InputDim:    8,
		HiddenDims:  []int{64, 32}, // two hidden layers
		DropoutRate: 0.1,
		Activation:  tabular.ActivationReLU,
	}

	tc := tabular.TrainConfig{
		Epochs:          20,
		BatchSize:       64,
		LearningRate:    0.001,
		WeightDecay:     1e-4,
		ValidationSplit: 0.2, // hold out 20% for validation
	}

	model, err := tabular.Train(data, labels, tc, mc, engine, ops)
	if err != nil {
		log.Fatal("train:", err)
	}

	// Save the trained model to disk.
	if err := tabular.Save(model, "model.ztab"); err != nil {
		log.Fatal("save:", err)
	}
	fmt.Println("Model saved to model.ztab")
}
```

Each epoch logs training loss and, if `ValidationSplit > 0`, validation loss and accuracy via the standard `log/slog` structured logger.

## Step 3: Run Inference

```go
func predict(engine compute.Engine[float32], ops numeric.Arithmetic[float32]) {
	// Load from disk.
	model, err := tabular.Load("model.ztab", engine, ops)
	if err != nil {
		log.Fatal("load:", err)
	}

	// A single feature vector — must match InputDim.
	features := []float64{0.3, -1.2, 0.8, 0.1, -0.5, 1.4, 0.0, -0.7}

	direction, confidence, err := model.Predict(features)
	if err != nil {
		log.Fatal("predict:", err)
	}

	fmt.Printf("Direction: %s (confidence: %.1f%%)\n", direction, confidence*100)
	// Direction: Long (confidence: 72.4%)
}
```

## Step 4: Batch Prediction

For throughput, loop over a slice of feature vectors:

```go
func batchPredict(model *tabular.Model, batch [][]float64) {
	for i, features := range batch {
		dir, conf, err := model.Predict(features)
		if err != nil {
			log.Printf("sample %d: %v", i, err)
			continue
		}
		fmt.Printf("Sample %d: %s (%.2f)\n", i, dir, conf)
	}
}
```

## Step 5: Switch Activation to GELU

GELU often converges better on smoother data distributions:

```go
mc := tabular.ModelConfig{
	InputDim:   8,
	HiddenDims: []int{128, 64, 32},
	Activation: tabular.ActivationGELU,
}
```

## Complete Example

```go
package main

import (
	"fmt"
	"log"
	"math/rand/v2"

	"github.com/zerfoo/zerfoo/tabular"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
)

func main() {
	engine := compute.NewCPUEngine[float32]()
	ops := numeric.NewArithmetic[float32]()

	// Generate synthetic data.
	data := make([][]float64, 500)
	labels := make([]int, 500)
	for i := range data {
		row := make([]float64, 6)
		for j := range row {
			row[j] = rand.NormFloat64()
		}
		data[i] = row
		labels[i] = i % 3
	}

	// Train.
	model, err := tabular.Train(data, labels,
		tabular.TrainConfig{
			Epochs:          10,
			BatchSize:       32,
			LearningRate:    0.001,
			ValidationSplit: 0.15,
		},
		tabular.ModelConfig{
			InputDim:   6,
			HiddenDims: []int{32, 16},
			Activation: tabular.ActivationReLU,
		},
		engine, ops,
	)
	if err != nil {
		log.Fatal(err)
	}

	// Predict.
	dir, conf, err := model.Predict([]float64{0.1, -0.2, 0.5, 1.1, -0.8, 0.3})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s (%.1f%% confidence)\n", dir, conf*100)
}
```

## Understanding the Output Classes

| Value | Constant | Meaning |
|-------|----------|---------|
| 0 | `tabular.Long` | Positive signal |
| 1 | `tabular.Short` | Negative signal |
| 2 | `tabular.Flat` | No signal / neutral |

`Direction.String()` returns `"Long"`, `"Short"`, or `"Flat"`.

## Next Steps

- [Tutorial 4: Time-Series Forecasting](04-timeseries-forecasting.md) — use TFT, N-BEATS, and PatchTST for sequential predictions
- [Tutorial 5: Edge Deployment](05-edge-deployment.md) — cross-compile for ARM devices
