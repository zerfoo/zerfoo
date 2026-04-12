package timeseries

import (
	"fmt"
	"math"

	"github.com/zerfoo/zerfoo/crossasset"
)

// CrossAssetConfig holds configuration for a cross-asset attention engine
// that wraps the crossasset.Model for the timeseries training interface.
type CrossAssetConfig struct {
	NSources          int     // number of asset sources (maps to channels)
	FeaturesPerSource int     // features per source (maps to input length)
	DModel            int     // transformer hidden dimension
	NHeads            int     // number of attention heads
	NLayers           int     // number of cross-attention layers
	LearningRate      float64 // training learning rate
	Epochs            int     // training epochs
	BatchSize         int     // mini-batch size (0 = full batch)
}

// CrossAsset wraps a crossasset.Model for the timeseries windowed training
// interface. It adapts the multi-source classification model to the
// TrainWindowed / PredictWindowed contract used by other backends (DLinear,
// FreTS, PatchTST, etc.).
//
// Mapping between timeseries and crossasset conventions:
//   - windows[sample][channel][inputLen] <-> data[sample][source][features]
//   - labels: flat float64 with 3 values per source (one-hot encoded direction)
//   - predictions: flat float64 with 3 values per source (softmax probabilities)
type CrossAsset struct {
	model  *crossasset.Model
	config CrossAssetConfig
}

// NewCrossAsset creates a new CrossAsset engine with the given configuration.
func NewCrossAsset(config CrossAssetConfig) (*CrossAsset, error) {
	if config.NSources <= 0 {
		return nil, fmt.Errorf("crossasset_engine: NSources must be positive, got %d", config.NSources)
	}
	if config.FeaturesPerSource <= 0 {
		return nil, fmt.Errorf("crossasset_engine: FeaturesPerSource must be positive, got %d", config.FeaturesPerSource)
	}
	if config.DModel <= 0 {
		return nil, fmt.Errorf("crossasset_engine: DModel must be positive, got %d", config.DModel)
	}
	if config.NHeads <= 0 {
		return nil, fmt.Errorf("crossasset_engine: NHeads must be positive, got %d", config.NHeads)
	}
	if config.DModel%config.NHeads != 0 {
		return nil, fmt.Errorf("crossasset_engine: DModel (%d) must be divisible by NHeads (%d)", config.DModel, config.NHeads)
	}
	if config.NLayers <= 0 {
		return nil, fmt.Errorf("crossasset_engine: NLayers must be positive, got %d", config.NLayers)
	}

	lr := config.LearningRate
	if lr <= 0 {
		lr = 0.001
	}

	m := crossasset.NewModel(crossasset.Config{
		NSources:          config.NSources,
		FeaturesPerSource: config.FeaturesPerSource,
		DModel:            config.DModel,
		NHeads:            config.NHeads,
		NLayers:           config.NLayers,
		LearningRate:      lr,
	})

	return &CrossAsset{model: m, config: config}, nil
}

// TrainWindowed trains the cross-asset model on windowed data.
//
// windows shape: [nSamples][nSources][featuresPerSource].
// labels shape: flat slice of length nSamples * nSources, where each value
// is the integer direction class (0=Long, 1=Short, 2=Flat) encoded as float64.
//
// The TrainConfig.LR and TrainConfig.Epochs fields override the config defaults
// if set. Other TrainConfig fields (Beta1, Beta2, etc.) are not used because
// the underlying crossasset.Model uses its own SGD optimizer.
func (ca *CrossAsset) TrainWindowed(windows [][][]float64, labels []float64, config TrainConfig) (*TrainResult, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("crossasset_engine: empty training set")
	}

	ns := ca.config.NSources
	expectedLabels := nSamples * ns
	if len(labels) != expectedLabels {
		return nil, fmt.Errorf("crossasset_engine: expected %d labels (nSamples*nSources), got %d", expectedLabels, len(labels))
	}

	// Validate window shapes.
	for i, w := range windows {
		if len(w) != ns {
			return nil, fmt.Errorf("crossasset_engine: window %d has %d sources, expected %d", i, len(w), ns)
		}
		for s, feat := range w {
			if len(feat) != ca.config.FeaturesPerSource {
				return nil, fmt.Errorf("crossasset_engine: window %d source %d has %d features, expected %d", i, s, len(feat), ca.config.FeaturesPerSource)
			}
		}
	}

	// Convert float64 windows to float32 for the crossasset model.
	windows32 := make([][][]float32, nSamples)
	for i, w := range windows {
		windows32[i] = make([][]float32, ns)
		for s, feat := range w {
			windows32[i][s] = make([]float32, len(feat))
			for f, v := range feat {
				windows32[i][s][f] = float32(v)
			}
		}
	}

	// Convert flat float64 labels to [nSamples][nSources] int labels.
	intLabels := make([][]int, nSamples)
	for i := 0; i < nSamples; i++ {
		intLabels[i] = make([]int, ns)
		for s := 0; s < ns; s++ {
			intLabels[i][s] = int(math.Round(labels[i*ns+s]))
			if intLabels[i][s] < 0 || intLabels[i][s] > 2 {
				return nil, fmt.Errorf("crossasset_engine: label[%d][%d]=%d out of range [0,2]", i, s, intLabels[i][s])
			}
		}
	}

	epochs := config.Epochs
	if epochs <= 0 {
		epochs = ca.config.Epochs
	}
	if epochs <= 0 {
		epochs = 100
	}

	lr := config.LR
	if lr <= 0 {
		lr = ca.config.LearningRate
	}
	if lr <= 0 {
		lr = 0.001
	}

	batchSize := config.BatchSize
	if batchSize <= 0 {
		batchSize = ca.config.BatchSize
	}
	if batchSize <= 0 {
		batchSize = nSamples
	}

	result := &TrainResult{
		LossHistory: make([]float64, epochs),
	}

	// Train one epoch at a time to record per-epoch loss.
	for epoch := 0; epoch < epochs; epoch++ {
		err := ca.model.Train(windows32, intLabels, crossasset.TrainConfig{
			Epochs:       1,
			BatchSize:    batchSize,
			LearningRate: lr,
		})
		if err != nil {
			return nil, fmt.Errorf("crossasset_engine: train epoch %d: %w", epoch, err)
		}

		// Compute cross-entropy loss over all samples.
		epochLoss := ca.computeLoss(windows32, intLabels)
		result.LossHistory[epoch] = epochLoss
		result.FinalLoss = epochLoss

		if !isFinite(epochLoss) {
			return nil, fmt.Errorf("crossasset_engine: training diverged at epoch %d: loss=%v", epoch, epochLoss)
		}
	}

	result.Metrics = map[string]float64{
		"cross_entropy": result.FinalLoss,
	}
	return result, nil
}

// PredictWindowed runs inference on windowed data.
//
// windows shape: [nSamples][nSources][featuresPerSource].
// Returns flat predictions of length nSamples * nSources * 3 where each
// triplet is the softmax probability for [Long, Short, Flat].
func (ca *CrossAsset) PredictWindowed(modelPath string, windows [][][]float64) ([]float64, error) {
	nSamples := len(windows)
	if nSamples == 0 {
		return nil, fmt.Errorf("crossasset_engine: empty input")
	}

	ns := ca.config.NSources
	out := make([]float64, 0, nSamples*ns*3)

	for i, w := range windows {
		if len(w) != ns {
			return nil, fmt.Errorf("crossasset_engine: window %d has %d sources, expected %d", i, len(w), ns)
		}

		// Convert float64 window to float32.
		w32 := make([][]float32, ns)
		for s, feat := range w {
			w32[s] = make([]float32, len(feat))
			for f, v := range feat {
				w32[s][f] = float32(v)
			}
		}

		dirs, confs, err := ca.model.Predict(w32)
		if err != nil {
			return nil, fmt.Errorf("crossasset_engine: predict sample %d: %w", i, err)
		}

		// Convert direction + confidence to a soft probability vector per source.
		for s := 0; s < ns; s++ {
			probs := directionToProbs(dirs[s], float64(confs[s]))
			out = append(out, probs[:]...)
		}
	}

	return out, nil
}

// computeLoss calculates the average cross-entropy loss over all samples.
func (ca *CrossAsset) computeLoss(data [][][]float32, labels [][]int) float64 {
	ns := ca.config.NSources
	totalLoss := 0.0
	count := 0

	for i, sample := range data {
		dirs, confs, err := ca.model.Predict(sample)
		if err != nil {
			continue
		}
		for s := 0; s < ns; s++ {
			probs := directionToProbs(dirs[s], float64(confs[s]))
			target := labels[i][s]
			if target >= 0 && target < 3 {
				p := probs[target]
				if p < 1e-15 {
					p = 1e-15
				}
				totalLoss -= math.Log(p)
			}
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return totalLoss / float64(count)
}

// directionToProbs converts a direction index and confidence score into a
// 3-element probability vector [Long, Short, Flat].
func directionToProbs(dir int, conf float64) [3]float64 {
	var probs [3]float64
	remaining := (1.0 - conf) / 2.0
	for i := 0; i < 3; i++ {
		if i == dir {
			probs[i] = conf
		} else {
			probs[i] = remaining
		}
	}
	return probs
}
