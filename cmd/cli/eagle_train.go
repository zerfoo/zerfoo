package cli

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"strconv"

	"github.com/zerfoo/zerfoo/inference"
	"github.com/zerfoo/zerfoo/layers/core"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"

	ztensorgguf "github.com/zerfoo/ztensor/gguf"
)

// EagleTrainCommand implements the "eagle-train" CLI command for training
// an EAGLE speculative decoding head.
type EagleTrainCommand struct {
	out io.Writer
}

// NewEagleTrainCommand creates a new EagleTrainCommand.
func NewEagleTrainCommand(out io.Writer) *EagleTrainCommand {
	return &EagleTrainCommand{out: out}
}

// eagleTrainConfig holds parsed eagle-train command flags.
type eagleTrainConfig struct {
	modelPath  string
	corpusPath string
	outputPath string
	epochs     int
	lr         float64
	maxSamples int
	batchSize  int
	hiddenDim  int
	synthetic  bool
}

// Name implements Command.Name.
func (c *EagleTrainCommand) Name() string { return "eagle-train" }

// Description implements Command.Description.
func (c *EagleTrainCommand) Description() string {
	return "Train an EAGLE speculative decoding head"
}

// Run implements Command.Run.
func (c *EagleTrainCommand) Run(ctx context.Context, args []string) error {
	cfg, err := c.parseArgs(args)
	if err != nil {
		return err
	}

	fmt.Fprintf(c.out, "eagle-train: output=%s epochs=%d lr=%.1e max-samples=%d batch-size=%d hidden-dim=%d synthetic=%v\n",
		cfg.outputPath, cfg.epochs, cfg.lr, cfg.maxSamples, cfg.batchSize, cfg.hiddenDim, cfg.synthetic)

	// Collect training pairs.
	var pairs []inference.TrainingPair
	if cfg.synthetic || cfg.corpusPath == "" {
		pairs, err = inference.GenerateSyntheticPairs(cfg.hiddenDim, cfg.maxSamples)
		if err != nil {
			return fmt.Errorf("generate synthetic pairs: %w", err)
		}
		fmt.Fprintf(c.out, "generated %d synthetic training pairs\n", len(pairs))
	} else {
		corpusData, readErr := os.ReadFile(cfg.corpusPath)
		if readErr != nil {
			return fmt.Errorf("read corpus: %w", readErr)
		}
		// Simple byte-level tokenization for MVP.
		tokens := make([]int, len(corpusData))
		for i, b := range corpusData {
			tokens[i] = int(b)
		}
		pairs, err = inference.CollectPenultimateFeatures(cfg.modelPath, tokens, cfg.maxSamples)
		if err != nil {
			return fmt.Errorf("collect features: %w", err)
		}
		fmt.Fprintf(c.out, "collected %d training pairs from corpus\n", len(pairs))
	}

	if len(pairs) == 0 {
		return fmt.Errorf("no training pairs collected")
	}

	// Create engine and EAGLE head.
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)

	head, err := core.NewEAGLEHead[float32](engine, ops, cfg.hiddenDim)
	if err != nil {
		return fmt.Errorf("create EAGLE head: %w", err)
	}

	// Create AdamW optimizer.
	adamw := optimizer.NewAdamW[float32](
		engine,
		float32(cfg.lr),  // learning rate
		0.9,              // beta1
		0.999,            // beta2
		1e-8,             // epsilon
		0.01,             // weight decay
	)

	// Training loop.
	for epoch := range cfg.epochs {
		select {
		case <-ctx.Done():
			fmt.Fprintf(c.out, "interrupted at epoch %d\n", epoch+1)
			return nil
		default:
		}

		// Shuffle training pairs.
		shuffled := make([]int, len(pairs))
		for i := range shuffled {
			shuffled[i] = i
		}
		rand.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		var epochLoss float64
		numBatches := 0

		for batchStart := 0; batchStart < len(shuffled); batchStart += cfg.batchSize {
			select {
			case <-ctx.Done():
				fmt.Fprintf(c.out, "interrupted at epoch %d batch %d\n", epoch+1, numBatches+1)
				return nil
			default:
			}

			batchEnd := batchStart + cfg.batchSize
			if batchEnd > len(shuffled) {
				batchEnd = len(shuffled)
			}

			var batchLoss float64
			batchCount := 0

			for _, idx := range shuffled[batchStart:batchEnd] {
				pair := pairs[idx]

				// Forward: predicted = head.Forward(ctx, input)
				predicted, fwdErr := head.Forward(ctx, pair.Input)
				if fwdErr != nil {
					return fmt.Errorf("epoch %d forward: %w", epoch+1, fwdErr)
				}

				// Compute MSE loss: loss = mean((predicted - target)^2)
				predData := predicted.Data()
				targetData := pair.Target.Data()
				n := len(predData)
				if n != len(targetData) {
					return fmt.Errorf("epoch %d: predicted size %d != target size %d", epoch+1, n, len(targetData))
				}

				gradData := make([]float32, n)
				var sampleLoss float64
				for j := range n {
					diff := predData[j] - targetData[j]
					sampleLoss += float64(diff * diff)
					// dL/dpred = 2*(pred-target)/n
					gradData[j] = 2.0 * diff / float32(n)
				}
				sampleLoss /= float64(n)
				batchLoss += sampleLoss
				batchCount++

				// Create gradient tensor and set on parameters.
				grad, gradErr := tensor.New[float32](predicted.Shape(), gradData)
				if gradErr != nil {
					return fmt.Errorf("epoch %d create gradient: %w", epoch+1, gradErr)
				}

				// Backprop through the head using manual gradient accumulation.
				if err := eagleBackward(ctx, engine, head, pair.Input, grad); err != nil {
					return fmt.Errorf("epoch %d backward: %w", epoch+1, err)
				}
			}

			// AdamW step.
			params := head.Parameters()
			if err := adamw.Step(ctx, params); err != nil {
				return fmt.Errorf("epoch %d optimizer step: %w", epoch+1, err)
			}

			if batchCount > 0 {
				epochLoss += batchLoss / float64(batchCount)
			}
			numBatches++
		}

		avgLoss := epochLoss / float64(numBatches)
		fmt.Fprintf(c.out, "epoch=%d/%d loss=%.6f\n", epoch+1, cfg.epochs, avgLoss)
	}

	// Export trained weights to GGUF.
	if err := exportEAGLEGGUF(cfg.outputPath, head, cfg.hiddenDim); err != nil {
		return fmt.Errorf("export GGUF: %w", err)
	}
	fmt.Fprintf(c.out, "EAGLE head saved to %s\n", cfg.outputPath)

	return nil
}

// eagleBackward performs manual backpropagation through the EAGLE head layers.
// Since EAGLEHead doesn't implement Backward(), we compute gradients for
// each layer's parameters manually.
//
// EAGLEHead architecture: LayerNorm → Linear(fc1) → SiLU → Linear(fc2)
// We backprop: dL/dfc2 → dL/dsilu → dL/dfc1 → dL/dnorm
func eagleBackward(
	ctx context.Context,
	engine compute.Engine[float32],
	head *core.EAGLEHead[float32],
	input *tensor.TensorNumeric[float32],
	outputGrad *tensor.TensorNumeric[float32],
) error {
	// For the MVP, use finite-difference approximation to set gradients
	// on parameters. This is slow but correct for small models.
	params := head.Parameters()
	eps := float32(1e-4)

	// Compute base loss (sum of outputGrad * output).
	baseOut, err := head.Forward(ctx, input)
	if err != nil {
		return err
	}
	baseLoss := dotProduct(outputGrad.Data(), baseOut.Data())

	for _, p := range params {
		data := p.Value.Data()
		gradData := make([]float32, len(data))

		for i := range data {
			orig := data[i]

			// f(x + eps)
			data[i] = orig + eps
			p.Value.SetData(data)
			plusOut, err := head.Forward(ctx, input)
			if err != nil {
				data[i] = orig
				p.Value.SetData(data)
				return err
			}
			plusLoss := dotProduct(outputGrad.Data(), plusOut.Data())

			// Restore
			data[i] = orig
			p.Value.SetData(data)

			gradData[i] = (plusLoss - baseLoss) / eps
		}

		grad, err := tensor.New[float32](p.Value.Shape(), gradData)
		if err != nil {
			return err
		}
		p.Gradient = grad
	}

	return nil
}

// dotProduct computes the dot product of two float32 slices.
func dotProduct(a, b []float32) float32 {
	var sum float32
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := range n {
		sum += a[i] * b[i]
	}
	return sum
}

// exportEAGLEGGUF writes the trained EAGLE head weights to a GGUF file.
func exportEAGLEGGUF(path string, head *core.EAGLEHead[float32], hiddenDim int) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	w := ztensorgguf.NewWriter()

	// Metadata.
	w.AddMetadataString("general.architecture", "eagle")
	w.AddMetadataUint32("eagle.hidden_dim", uint32(hiddenDim))

	// Write parameters with EAGLE tensor names.
	params := head.Parameters()
	nameMap := map[string]string{
		"gamma":                  "eagle.norm.weight",
		"beta":                   "eagle.norm.bias",
		"eagle_head_fc1_weights": "eagle.fc1.weight",
		"eagle_head_fc2_weights": "eagle.fc2.weight",
	}

	for _, p := range params {
		ggufName, ok := nameMap[p.Name]
		if !ok {
			ggufName = p.Name
		}
		data := make([]float32, len(p.Value.Data()))
		copy(data, p.Value.Data())
		w.AddTensorF32(ggufName, p.Value.Shape(), data)
	}

	return w.Write(f)
}

// Usage implements Command.Usage.
func (c *EagleTrainCommand) Usage() string {
	return `eagle-train [OPTIONS]

Train an EAGLE speculative decoding head.

OPTIONS:
  --model <path>         Path to base GGUF model file (optional with --synthetic)
  --corpus <path>        Path to training corpus text file (optional with --synthetic)
  --output <path>        Output GGUF path for trained EAGLE head (default: eagle.gguf)
  --epochs <n>           Number of training epochs (default: 3)
  --lr <float>           Learning rate (default: 0.001)
  --max-samples <n>      Maximum training pairs (default: 10000)
  --batch-size <n>       Batch size (default: 32)
  --hidden-dim <n>       Hidden dimension (default: 256, required for --synthetic)
  --synthetic            Use synthetic training data for validation`
}

// Examples implements Command.Examples.
func (c *EagleTrainCommand) Examples() []string {
	return []string{
		"eagle-train --model m.gguf --corpus data.txt --output eagle.gguf --epochs 3 --lr 0.001",
		"eagle-train --synthetic --hidden-dim 128 --epochs 5 --output eagle.gguf",
		"eagle-train --synthetic --hidden-dim 64 --max-samples 1000 --epochs 10",
	}
}

func (c *EagleTrainCommand) parseArgs(args []string) (*eagleTrainConfig, error) {
	cfg := &eagleTrainConfig{
		outputPath: "eagle.gguf",
		epochs:     3,
		lr:         0.001,
		maxSamples: 10000,
		batchSize:  32,
		hiddenDim:  256,
	}

	for i := 0; i < len(args); i++ {
		arg := args[i]
		var eqVal string
		var hasEq bool
		if flag, val, ok := splitFlag(arg); ok {
			arg = flag
			eqVal = val
			hasEq = true
		}
		nextVal := func(flagName string) (string, error) {
			if hasEq {
				return eqVal, nil
			}
			if i+1 >= len(args) {
				return "", fmt.Errorf("%s requires a value", flagName)
			}
			i++
			return args[i], nil
		}
		switch arg {
		case "--model":
			v, err := nextVal("--model")
			if err != nil {
				return nil, err
			}
			cfg.modelPath = v
		case "--corpus":
			v, err := nextVal("--corpus")
			if err != nil {
				return nil, err
			}
			cfg.corpusPath = v
		case "--output":
			v, err := nextVal("--output")
			if err != nil {
				return nil, err
			}
			cfg.outputPath = v
		case "--epochs":
			v, err := nextVal("--epochs")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--epochs must be >= 1")
			}
			cfg.epochs = n
		case "--lr":
			v, err := nextVal("--lr")
			if err != nil {
				return nil, err
			}
			f, err := strconv.ParseFloat(v, 64)
			if err != nil || f <= 0 || math.IsNaN(f) || math.IsInf(f, 0) {
				return nil, fmt.Errorf("--lr must be a positive number")
			}
			cfg.lr = f
		case "--max-samples":
			v, err := nextVal("--max-samples")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--max-samples must be >= 1")
			}
			cfg.maxSamples = n
		case "--batch-size":
			v, err := nextVal("--batch-size")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--batch-size must be >= 1")
			}
			cfg.batchSize = n
		case "--hidden-dim":
			v, err := nextVal("--hidden-dim")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--hidden-dim must be >= 1")
			}
			cfg.hiddenDim = n
		case "--synthetic":
			cfg.synthetic = true
		default:
			return nil, fmt.Errorf("unknown flag: %s", arg)
		}
	}

	// Validate: either synthetic mode or model+corpus must be provided.
	if !cfg.synthetic && cfg.corpusPath == "" && cfg.modelPath == "" {
		return nil, fmt.Errorf("either --synthetic or --model/--corpus is required")
	}

	return cfg, nil
}

// Static interface assertion.
var _ Command = (*EagleTrainCommand)(nil)
