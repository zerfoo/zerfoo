package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
)

// TrainCommand implements real tabular training. General GGUF/FSDP training
// paths return an unsupported error rather than synthetic success.
type TrainCommand struct {
	out io.Writer
}

// NewTrainCommand creates a new TrainCommand.
func NewTrainCommand(out io.Writer) *TrainCommand {
	if out == nil {
		out = os.Stdout
	}
	return &TrainCommand{out: out}
}

// trainConfig holds parsed train command flags.
type trainConfig struct {
	modelPath  string
	dataPath   string
	worldSize  int
	rank       int
	masterAddr string
	masterPort int
	outputPath string
	epochs     int
	batchSize  int
	lr         float64
}

// Name implements Command.Name.
func (c *TrainCommand) Name() string { return "train" }

// Description implements Command.Description.
func (c *TrainCommand) Description() string {
	return "Train a numeric classifier from CSV data"
}

// Run implements Command.Run.
func (c *TrainCommand) Run(ctx context.Context, args []string) error {
	if len(args) > 0 && args[0] == "tabular" {
		return c.runTabularTrain(ctx, args[1:])
	}
	if _, err := c.parseArgs(args); err != nil {
		return err
	}
	return fmt.Errorf("GGUF/FSDP training is unsupported; use train tabular --data <CSV> --target <column> --output <bundle>")
}

// Usage implements Command.Usage.
func (c *TrainCommand) Usage() string {
	return `train [OPTIONS]

Train a numeric classifier:
  train tabular --data <CSV> --target <column> --output <bundle>
    [--recipe linear|mlp] [--epochs 200] [--batch-size 15] [--lr 0.01]
    [--seed 42] [--split stratified|group|temporal] [--group <column>]
    [--time <column>] [--dataset-manifest <JSON>] [--device cpu|cuda]
    [--max-steps <n>] [--max-duration 2m]
Output is versioned NDJSON progress and result/error events.

Legacy flags below are parsed for actionable errors; general GGUF/FSDP
training is unsupported. Synthetic training is a test-only demonstration.

OPTIONS:
  --config <path>        Path to GGUF model file (required)
  --data <path>          Path to training data (required)
  --output <path>        Checkpoint output path (default: checkpoint.gguf)
  --world-size <n>       Number of GPUs / processes (default: 1)
  --rank <n>             Process rank, 0 = coordinator (default: 0)
  --master-addr <addr>   Coordinator address (default: localhost)
  --master-port <port>   Coordinator port (default: 29500)
  --epochs <n>           Number of training epochs (default: 1)
  --batch-size <n>       Batch size (default: 4)
  --lr <float>           Learning rate (default: 1e-4)`
}

// Examples implements Command.Examples.
func (c *TrainCommand) Examples() []string {
	return []string{"train tabular --data iris.csv --target species --recipe mlp --output iris.bundle"}
}

func (c *TrainCommand) parseArgs(args []string) (*trainConfig, error) {
	cfg := &trainConfig{
		worldSize:  1,
		rank:       0,
		masterAddr: "localhost",
		masterPort: 29500,
		outputPath: "checkpoint.gguf",
		epochs:     1,
		batchSize:  4,
		lr:         1e-4,
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
		case "--config":
			v, err := nextVal("--config")
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
		case "--world-size":
			v, err := nextVal("--world-size")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("--world-size must be >= 1")
			}
			cfg.worldSize = n
		case "--rank":
			v, err := nextVal("--rank")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				return nil, fmt.Errorf("--rank must be >= 0")
			}
			cfg.rank = n
		case "--master-addr":
			v, err := nextVal("--master-addr")
			if err != nil {
				return nil, err
			}
			cfg.masterAddr = v
		case "--master-port":
			v, err := nextVal("--master-port")
			if err != nil {
				return nil, err
			}
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 || n > 65535 {
				return nil, fmt.Errorf("--master-port must be in [0, 65535]")
			}
			cfg.masterPort = n
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
		default:
			return nil, fmt.Errorf("unknown flag: %s", arg)
		}
	}

	if cfg.modelPath == "" {
		return nil, fmt.Errorf("--config is required")
	}
	if cfg.dataPath == "" {
		return nil, fmt.Errorf("--data is required")
	}
	if cfg.rank >= cfg.worldSize {
		return nil, fmt.Errorf("--rank must be in [0, world-size)")
	}

	return cfg, nil
}

// Static interface assertion.
var _ Command = (*TrainCommand)(nil)
