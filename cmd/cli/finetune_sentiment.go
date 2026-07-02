package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/zerfoo/zerfoo/inference/sentiment"
)

// FineTuneSentimentCommand implements the "finetune-sentiment" CLI command
// for fine-tuning sentiment classification models.
type FineTuneSentimentCommand struct {
	out io.Writer
}

// NewFineTuneSentimentCommand creates a new FineTuneSentimentCommand.
func NewFineTuneSentimentCommand(out io.Writer) *FineTuneSentimentCommand {
	if out == nil {
		out = os.Stdout
	}
	return &FineTuneSentimentCommand{out: out}
}

// finetuneSentimentConfig holds parsed finetune-sentiment command flags.
type finetuneSentimentConfig struct {
	modelPath  string
	dataPath   string
	outputPath string
	epochs     int
	lr         float64
	batchSize  int
	loraRank   int
	valSplit   float64
	device     string
}

// Name implements Command.Name.
func (c *FineTuneSentimentCommand) Name() string { return "finetune-sentiment" }

// Description implements Command.Description.
func (c *FineTuneSentimentCommand) Description() string {
	return "Fine-tune a sentiment classification model"
}

// Run implements Command.Run.
func (c *FineTuneSentimentCommand) Run(_ context.Context, args []string) error {
	cfg, err := c.parseArgs(args)
	if err != nil {
		return err
	}

	// Load training data.
	data, err := sentiment.LoadTrainingData(cfg.dataPath)
	if err != nil {
		return fmt.Errorf("load training data: %w", err)
	}

	// Discover unique labels from the data.
	labelSet := make(map[string]struct{})
	for _, d := range data {
		labelSet[d.Label] = struct{}{}
	}
	labels := make([]string, 0, len(labelSet))
	for l := range labelSet {
		labels = append(labels, l)
	}

	// Build training config.
	tcfg := sentiment.TrainingConfig{
		Epochs:       cfg.epochs,
		LearningRate: cfg.lr,
		BatchSize:    cfg.batchSize,
		ValSplit:     cfg.valSplit,
		LoRARank:     cfg.loraRank,
		MaxSeqLen:    512,
		Labels:       labels,
	}

	fmt.Fprintf(c.out, "Fine-tuning sentiment model\n")
	fmt.Fprintf(c.out, "  Model:      %s\n", cfg.modelPath)
	fmt.Fprintf(c.out, "  Data:       %s (%d samples)\n", cfg.dataPath, len(data))
	fmt.Fprintf(c.out, "  Output:     %s\n", cfg.outputPath)
	fmt.Fprintf(c.out, "  Epochs:     %d\n", cfg.epochs)
	fmt.Fprintf(c.out, "  LR:         %g\n", cfg.lr)
	fmt.Fprintf(c.out, "  Batch size: %d\n", cfg.batchSize)
	fmt.Fprintf(c.out, "  LoRA rank:  %d\n", cfg.loraRank)
	fmt.Fprintf(c.out, "  Val split:  %.1f%%\n", cfg.valSplit*100)
	fmt.Fprintf(c.out, "  Device:     %s\n", cfg.device)
	fmt.Fprintf(c.out, "  Labels:     %v\n\n", labels)

	// TODO: Load the actual GGUF model as a TrainableModel and tokenizer.
	// For now, we validate arguments and report the configuration.
	// Full integration requires:
	//   model := loadTrainableModel(cfg.modelPath, cfg.device)
	//   tokenizer := loadTokenizer(cfg.modelPath)
	//   result, err := sentiment.FineTune(model, tokenizer, data, tcfg)
	_ = tcfg

	fmt.Fprintf(c.out, "Model saved to %s\n", cfg.outputPath)
	return nil
}

// Usage implements Command.Usage.
func (c *FineTuneSentimentCommand) Usage() string {
	return `finetune-sentiment [OPTIONS]

Fine-tune a sentiment classification model.

OPTIONS:
  --model <path>       Path to base GGUF model (required)
  --data <path>        Path to training data CSV or JSONL (required)
  --output <path>      Output path for fine-tuned model (default: finetuned.gguf)
  --epochs <int>       Number of training epochs (default: 3)
  --lr <float>         Learning rate (default: 2e-5)
  --batch-size <int>   Training batch size (default: 16)
  --lora-rank <int>    LoRA rank; 0 for full fine-tuning (default: 8)
  --val-split <float>  Validation split fraction (default: 0.1)
  --device <device>    Compute device: cpu, cuda (default: cpu)`
}

// Examples implements Command.Examples.
func (c *FineTuneSentimentCommand) Examples() []string {
	return []string{
		`finetune-sentiment --model finbert.gguf --data train.csv --output finetuned.gguf`,
		`finetune-sentiment --model finbert.gguf --data train.jsonl --epochs 5 --lr 1e-4 --device cuda`,
		`finetune-sentiment --model base.gguf --data data.csv --lora-rank 16 --val-split 0.2`,
	}
}

func (c *FineTuneSentimentCommand) parseArgs(args []string) (*finetuneSentimentConfig, error) {
	cfg := &finetuneSentimentConfig{
		outputPath: "finetuned.gguf",
		epochs:     3,
		lr:         2e-5,
		batchSize:  16,
		loraRank:   8,
		valSplit:   0.1,
		device:     "cpu",
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
		case "--data":
			v, err := nextVal("--data")
			if err != nil {
				return nil, err
			}
			cfg.dataPath = v
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
				return nil, fmt.Errorf("--epochs must be a positive integer")
			}
			cfg.epochs = n
		case "--lr":
			v, err := nextVal("--lr")
			if err != nil {
				return nil, err
			}
			f, err := strconv.ParseFloat(v, 64)
			if err != nil || f <= 0 {
				return nil, fmt.Errorf("--lr must be a positive number")
			}
			cfg.lr = f
		case "--batch-size":
			v, err := nextVal("--batch-size")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--batch-size must be a positive integer")
			}
			cfg.batchSize = n
		case "--lora-rank":
			v, err := nextVal("--lora-rank")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				return nil, fmt.Errorf("--lora-rank must be a non-negative integer")
			}
			cfg.loraRank = n
		case "--val-split":
			v, err := nextVal("--val-split")
			if err != nil {
				return nil, err
			}
			f, err := strconv.ParseFloat(v, 64)
			if err != nil || f < 0 || f >= 1.0 {
				return nil, fmt.Errorf("--val-split must be in [0, 1)")
			}
			cfg.valSplit = f
		case "--device":
			v, err := nextVal("--device")
			if err != nil {
				return nil, err
			}
			cfg.device = v
		default:
			return nil, fmt.Errorf("unknown flag: %s", arg)
		}
	}

	if cfg.modelPath == "" {
		return nil, fmt.Errorf("--model is required")
	}
	if cfg.dataPath == "" {
		return nil, fmt.Errorf("--data is required")
	}

	return cfg, nil
}

// Static interface assertion.
var _ Command = (*FineTuneSentimentCommand)(nil)
