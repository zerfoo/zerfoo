// Command ts_train trains a PatchTST time-series signal model on offline feature data.
//
// Usage:
//
//	ts_train --features-dir features/ --epochs 50 --output ts_model.gguf
package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/zerfoo/ztensor/compute"
	ztensorgguf "github.com/zerfoo/ztensor/gguf"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/ztensor/types"
	"github.com/zerfoo/zerfoo/inference/timeseries"
	"github.com/zerfoo/zerfoo/training/loss"
	"github.com/zerfoo/zerfoo/training/optimizer"
)

// config holds parsed command-line flags.
type config struct {
	featuresDir string
	patchLen    int
	stride      int
	hiddenDim   int
	numHeads    int
	numLayers   int
	horizon     int
	epochs      int
	batchSize   int
	lr          float64
	valSplit    float64
	patience    int
	output      string
}

func parseFlags() config {
	var cfg config
	flag.StringVar(&cfg.featuresDir, "features-dir", "features/", "directory of CSV feature files")
	flag.IntVar(&cfg.patchLen, "patch-len", 16, "patch length")
	flag.IntVar(&cfg.stride, "stride", 8, "patch stride")
	flag.IntVar(&cfg.hiddenDim, "hidden-dim", 128, "model hidden dimension")
	flag.IntVar(&cfg.numHeads, "num-heads", 8, "attention heads")
	flag.IntVar(&cfg.numLayers, "num-layers", 6, "transformer encoder layers")
	flag.IntVar(&cfg.horizon, "horizon", 1, "forecast horizon steps")
	flag.IntVar(&cfg.epochs, "epochs", 50, "max training epochs")
	flag.IntVar(&cfg.batchSize, "batch-size", 32, "batch size")
	flag.Float64Var(&cfg.lr, "lr", 1e-4, "learning rate")
	flag.Float64Var(&cfg.valSplit, "val-split", 0.2, "fraction for validation")
	flag.IntVar(&cfg.patience, "patience", 5, "early stopping patience")
	flag.StringVar(&cfg.output, "output", "ts_model.gguf", "output GGUF path")
	flag.Parse()
	return cfg
}

// tick represents a single timestamped feature vector loaded from CSV.
type tick struct {
	timestamp time.Time
	features  []float64
}

// loadCSVFeatures reads a CSV file with columns: timestamp, f1, f2, ...
func loadCSVFeatures(path string) ([]tick, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)

	// Skip header.
	if _, err := r.Read(); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var ticks []tick
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read row: %w", err)
		}
		if len(record) < 2 {
			return nil, fmt.Errorf("row has %d columns, need at least 2", len(record))
		}

		ts, err := time.Parse(time.RFC3339, record[0])
		if err != nil {
			return nil, fmt.Errorf("parse timestamp %q: %w", record[0], err)
		}

		features := make([]float64, len(record)-1)
		for i, v := range record[1:] {
			features[i], err = strconv.ParseFloat(v, 64)
			if err != nil {
				return nil, fmt.Errorf("parse feature column %d value %q: %w", i+1, v, err)
			}
		}

		ticks = append(ticks, tick{timestamp: ts, features: features})
	}

	return ticks, nil
}

// loadAllFeatures loads all CSV files from a directory and concatenates ticks.
func loadAllFeatures(dir string) ([]tick, int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, 0, fmt.Errorf("read features dir: %w", err)
	}

	var allTicks []tick
	numVars := 0

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if filepath.Ext(entry.Name()) != ".csv" {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		ticks, err := loadCSVFeatures(path)
		if err != nil {
			return nil, 0, fmt.Errorf("load %s: %w", entry.Name(), err)
		}
		if len(ticks) > 0 {
			if numVars == 0 {
				numVars = len(ticks[0].features)
			} else if len(ticks[0].features) != numVars {
				return nil, 0, fmt.Errorf("file %s has %d features, expected %d", entry.Name(), len(ticks[0].features), numVars)
			}
			allTicks = append(allTicks, ticks...)
		}
	}

	if len(allTicks) == 0 {
		return nil, 0, fmt.Errorf("no feature data found in %s", dir)
	}

	return allTicks, numVars, nil
}

// timeOrderedSplit splits ticks at the given fraction, preserving time order.
// The first (1-valFraction) ticks go to train, the rest to val.
func timeOrderedSplit(ticks []tick, valFraction float64) (train, val []tick) {
	n := len(ticks)
	splitIdx := int(math.Round(float64(n) * (1 - valFraction)))
	if splitIdx < 1 {
		splitIdx = 1
	}
	if splitIdx >= n {
		splitIdx = n - 1
	}
	return ticks[:splitIdx], ticks[splitIdx:]
}

// buildWindows creates input/target windows from sequential ticks.
// Input: seqLen consecutive feature vectors. Target: next horizon vectors.
func buildWindows(ticks []tick, seqLen, horizon, numVars int) (inputs, targets [][]float32) {
	n := len(ticks)
	for i := 0; i+seqLen+horizon <= n; i++ {
		input := make([]float32, seqLen*numVars)
		for j := 0; j < seqLen; j++ {
			for k := 0; k < numVars; k++ {
				input[j*numVars+k] = float32(ticks[i+j].features[k])
			}
		}
		target := make([]float32, horizon*numVars)
		for j := 0; j < horizon; j++ {
			for k := 0; k < numVars; k++ {
				target[j*numVars+k] = float32(ticks[i+seqLen+j].features[k])
			}
		}
		inputs = append(inputs, input)
		targets = append(targets, target)
	}
	return inputs, targets
}

// saveModelGGUF writes model parameters to a GGUF file.
func saveModelGGUF(path string, params []*graph.Parameter[float32]) error {
	if len(params) == 0 {
		return fmt.Errorf("no parameters to save")
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	w := ztensorgguf.NewWriter()
	w.AddMetadataString("general.architecture", "patchtst")

	for _, p := range params {
		data := make([]float32, len(p.Value.Data()))
		copy(data, p.Value.Data())
		w.AddTensorF32(p.Name, p.Value.Shape(), data)
	}

	return w.Write(f)
}

// computeQuantileLossAndGrad computes QuantileLoss and its gradient w.r.t. model output.
func computeQuantileLossAndGrad(engine compute.Engine[float32], output, targetTensor *tensor.TensorNumeric[float32], quantiles []float32) (float32, *tensor.TensorNumeric[float32], error) {
	outputData := output.Data()
	targetData := targetTensor.Data()
	totalElems := len(outputData)
	numQ := len(quantiles)

	// Build preds [totalElems, numQ] with output replicated per quantile.
	predsData := make([]float32, totalElems*numQ)
	for j := 0; j < totalElems; j++ {
		for q := 0; q < numQ; q++ {
			predsData[j*numQ+q] = outputData[j]
		}
	}
	preds, err := tensor.New[float32]([]int{totalElems, numQ}, predsData)
	if err != nil {
		return 0, nil, err
	}
	tgts, err := tensor.New[float32]([]int{totalElems}, targetData)
	if err != nil {
		return 0, nil, err
	}

	lossVal, err := loss.QuantileLoss(engine, preds, tgts, quantiles)
	if err != nil {
		return 0, nil, err
	}

	// Pinball loss gradient: dL/dpred = -q if target >= pred, (1-q) if target < pred.
	// Averaged over quantiles and samples.
	gradData := make([]float32, totalElems)
	for j := 0; j < totalElems; j++ {
		var g float32
		for q := 0; q < numQ; q++ {
			if targetData[j] >= outputData[j] {
				g -= quantiles[q]
			} else {
				g += 1 - quantiles[q]
			}
		}
		gradData[j] = g / float32(numQ*totalElems)
	}

	grad, err := tensor.New[float32](output.Shape(), gradData)
	if err != nil {
		return 0, nil, err
	}
	return lossVal, grad, nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "ts_train: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg := parseFlags()

	log.Printf("loading features from %s", cfg.featuresDir)
	allTicks, numVars, err := loadAllFeatures(cfg.featuresDir)
	if err != nil {
		return err
	}
	log.Printf("loaded %d ticks with %d features", len(allTicks), numVars)

	// Time-ordered train/val split — no shuffling to prevent data leakage.
	trainTicks, valTicks := timeOrderedSplit(allTicks, cfg.valSplit)
	log.Printf("train: %d ticks, val: %d ticks", len(trainTicks), len(valTicks))

	// Sequence length: at least patchLen + stride to get 2+ patches.
	seqLen := cfg.patchLen * 2
	if seqLen < cfg.patchLen+cfg.stride {
		seqLen = cfg.patchLen + cfg.stride
	}

	trainInputs, trainTargets := buildWindows(trainTicks, seqLen, cfg.horizon, numVars)
	valInputs, valTargets := buildWindows(valTicks, seqLen, cfg.horizon, numVars)

	if len(trainInputs) == 0 {
		return fmt.Errorf("not enough training data for seqLen=%d horizon=%d (have %d ticks)", seqLen, cfg.horizon, len(trainTicks))
	}
	log.Printf("train windows: %d, val windows: %d", len(trainInputs), len(valInputs))

	// Build PatchTST model.
	ops := numeric.Float32Ops{}
	var engine compute.Engine[float32]
	gpuEngine, gpuErr := compute.NewGPUEngine[float32](ops)
	if gpuErr == nil {
		engine = gpuEngine
		defer gpuEngine.Close()
		log.Printf("using GPU engine (CUDA)")
	} else {
		engine = compute.NewCPUEngine[float32](ops)
		log.Printf("using CPU engine (CUDA not available: %v)", gpuErr)
	}

	patchCfg := timeseries.PatchTSTConfig{
		PatchLen:  cfg.patchLen,
		Stride:    cfg.stride,
		NumLayers: cfg.numLayers,
		NumHeads:  cfg.numHeads,
		DModel:    cfg.hiddenDim,
		Horizon:   cfg.horizon,
		NumVars:   numVars,
	}

	patchGraph, err := timeseries.BuildPatchTST[float32](patchCfg, engine)
	if err != nil {
		return fmt.Errorf("build PatchTST: %w", err)
	}

	params := patchGraph.Parameters()
	opt := optimizer.NewAdamW[float32](engine, float32(cfg.lr), 0.9, 0.999, 1e-8, 0.01)

	quantiles := []float32{0.1, 0.5, 0.9}

	log.Printf("model: PatchTST (patch=%d stride=%d layers=%d heads=%d dim=%d horizon=%d vars=%d params=%d)",
		cfg.patchLen, cfg.stride, cfg.numLayers, cfg.numHeads, cfg.hiddenDim, cfg.horizon, numVars, len(params))
	log.Printf("training: epochs=%d batch=%d lr=%.1e patience=%d output=%s",
		cfg.epochs, cfg.batchSize, cfg.lr, cfg.patience, cfg.output)

	ctx := context.Background()
	bestValLoss := float32(math.MaxFloat32)
	patienceCounter := 0

	for epoch := 1; epoch <= cfg.epochs; epoch++ {
		// Training.
		var trainLossSum float64
		trainSteps := 0

		for i := 0; i < len(trainInputs); i += cfg.batchSize {
			end := i + cfg.batchSize
			if end > len(trainInputs) {
				end = len(trainInputs)
			}
			batchLen := end - i

			batchInputData := make([]float32, 0, batchLen*seqLen*numVars)
			batchTargetData := make([]float32, 0, batchLen*cfg.horizon*numVars)
			for j := i; j < end; j++ {
				batchInputData = append(batchInputData, trainInputs[j]...)
				batchTargetData = append(batchTargetData, trainTargets[j]...)
			}

			input, err := tensor.New[float32]([]int{batchLen, seqLen, numVars}, batchInputData)
			if err != nil {
				return fmt.Errorf("create input tensor: %w", err)
			}

			output, err := patchGraph.Forward(ctx, input)
			if err != nil {
				return fmt.Errorf("forward: %w", err)
			}

			targetTensor, err := tensor.New[float32](output.Shape(), batchTargetData)
			if err != nil {
				return fmt.Errorf("create target tensor: %w", err)
			}

			lossVal, grad, err := computeQuantileLossAndGrad(engine, output, targetTensor, quantiles)
			if err != nil {
				return fmt.Errorf("loss: %w", err)
			}
			trainLossSum += float64(lossVal)
			trainSteps++

			// Clear gradients.
			for _, p := range params {
				p.ClearGradient()
			}

			// Backward pass through graph.
			if err := patchGraph.Backward(ctx, types.FullBackprop, grad); err != nil {
				return fmt.Errorf("backward: %w", err)
			}

			// Optimizer step.
			if err := opt.Step(ctx, params); err != nil {
				return fmt.Errorf("optimizer step: %w", err)
			}
		}

		trainLoss := float32(0)
		if trainSteps > 0 {
			trainLoss = float32(trainLossSum / float64(trainSteps))
		}

		// Validation.
		var valLossSum float64
		valSteps := 0

		for i := 0; i < len(valInputs); i += cfg.batchSize {
			end := i + cfg.batchSize
			if end > len(valInputs) {
				end = len(valInputs)
			}
			batchLen := end - i

			batchInputData := make([]float32, 0, batchLen*seqLen*numVars)
			batchTargetData := make([]float32, 0, batchLen*cfg.horizon*numVars)
			for j := i; j < end; j++ {
				batchInputData = append(batchInputData, valInputs[j]...)
				batchTargetData = append(batchTargetData, valTargets[j]...)
			}

			input, err := tensor.New[float32]([]int{batchLen, seqLen, numVars}, batchInputData)
			if err != nil {
				return fmt.Errorf("create val input: %w", err)
			}

			output, err := patchGraph.Forward(ctx, input)
			if err != nil {
				return fmt.Errorf("val forward: %w", err)
			}

			targetTensor, err := tensor.New[float32](output.Shape(), batchTargetData)
			if err != nil {
				return fmt.Errorf("create val target: %w", err)
			}

			lossVal, _, err := computeQuantileLossAndGrad(engine, output, targetTensor, quantiles)
			if err != nil {
				return fmt.Errorf("val loss: %w", err)
			}
			valLossSum += float64(lossVal)
			valSteps++
		}

		valLoss := float32(math.MaxFloat32)
		if valSteps > 0 {
			valLoss = float32(valLossSum / float64(valSteps))
		}

		log.Printf("epoch %d/%d  train_loss=%.6f  val_loss=%.6f  best_val_loss=%.6f",
			epoch, cfg.epochs, trainLoss, valLoss, bestValLoss)

		if valLoss < bestValLoss {
			bestValLoss = valLoss
			patienceCounter = 0
			if err := saveModelGGUF(cfg.output, params); err != nil {
				return fmt.Errorf("save model: %w", err)
			}
			log.Printf("  saved best model to %s (val_loss=%.6f)", cfg.output, valLoss)
		} else {
			patienceCounter++
			if patienceCounter >= cfg.patience {
				log.Printf("early stopping at epoch %d (no improvement for %d epochs)", epoch, cfg.patience)
				break
			}
		}
	}

	log.Printf("training complete. best val_loss=%.6f", bestValLoss)
	return nil
}
