package cli

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/zerfoo/zerfoo/distributed/coordinator"
	"github.com/zerfoo/zerfoo/distributed/fsdp"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
)

// TrainCommand implements the "train" CLI command for local and distributed
// training. It reuses the distributed/ and training/ packages and supports
// single-GPU (--world-size 1) and multi-GPU modes.
type TrainCommand struct {
	out io.Writer
}

// NewTrainCommand creates a new TrainCommand.
func NewTrainCommand(out io.Writer) *TrainCommand {
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
	return "Train a model locally or distributed across multiple GPUs"
}

// Run implements Command.Run.
func (c *TrainCommand) Run(ctx context.Context, args []string) error {
	cfg, err := c.parseArgs(args)
	if err != nil {
		return err
	}

	fmt.Fprintf(c.out, "train: rank=%d world-size=%d master=%s:%d\n",
		cfg.rank, cfg.worldSize, cfg.masterAddr, cfg.masterPort)
	fmt.Fprintf(c.out, "  model=%s data=%s output=%s\n", cfg.modelPath, cfg.dataPath, cfg.outputPath)
	fmt.Fprintf(c.out, "  epochs=%d batch-size=%d lr=%.1e\n", cfg.epochs, cfg.batchSize, cfg.lr)

	if cfg.worldSize == 1 {
		return c.runLocal(ctx, cfg)
	}

	if cfg.rank == 0 {
		return c.runCoordinator(ctx, cfg)
	}
	return c.runWorker(ctx, cfg)
}

// Usage implements Command.Usage.
func (c *TrainCommand) Usage() string {
	return `train [OPTIONS]

Train a model locally or distributed across multiple GPUs.

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
	return []string{
		"train --config model.gguf --data train.jsonl",
		"train --config model.gguf --data train.jsonl --epochs 3 --batch-size 8 --lr 5e-5",
		"train --config model.gguf --data train.jsonl --world-size 2 --rank 0",
	}
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

// trainModel implements training.Model[float32] for the FSDP trainer.
type trainModel struct {
	params []*graph.Parameter[float32]
}

func newTrainModel(paramSize int) (*trainModel, error) {
	data := make([]float32, paramSize)
	for i := range data {
		data[i] = float32(i%7-3) * 0.01
	}
	t, err := tensor.New[float32]([]int{paramSize}, data)
	if err != nil {
		return nil, err
	}
	p, err := graph.NewParameter[float32]("weights", t, tensor.New[float32])
	if err != nil {
		return nil, err
	}
	return &trainModel{params: []*graph.Parameter[float32]{p}}, nil
}

func (m *trainModel) Parameters() []*graph.Parameter[float32] { return m.params }

func (m *trainModel) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs")
	}
	return inputs[0], nil
}

func (m *trainModel) Backward(_ context.Context, grad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	if len(m.params) > 0 {
		m.params[0].Gradient = grad
	}
	return []*tensor.TensorNumeric[float32]{grad}, nil
}

// Ensure trainModel satisfies training.Model.
var _ training.Model[float32] = (*trainModel)(nil)

// runLocal runs training on a single process without coordination.
func (c *TrainCommand) runLocal(ctx context.Context, cfg *trainConfig) error {
	return c.trainLoop(ctx, cfg)
}

// runCoordinator starts the coordinator and runs the training loop.
func (c *TrainCommand) runCoordinator(ctx context.Context, cfg *trainConfig) error {
	addr := fmt.Sprintf("%s:%d", cfg.masterAddr, cfg.masterPort)
	coord := coordinator.NewCoordinator(c.out, 30*time.Second)
	if err := coord.Start(addr); err != nil {
		return fmt.Errorf("coordinator start: %w", err)
	}
	defer coord.Stop()

	fmt.Fprintf(c.out, "coordinator listening on %s\n", addr)
	return c.trainLoop(ctx, cfg)
}

// runWorker connects to the coordinator and runs the training loop.
func (c *TrainCommand) runWorker(ctx context.Context, cfg *trainConfig) error {
	fmt.Fprintf(c.out, "worker rank=%d connecting to %s:%d\n", cfg.rank, cfg.masterAddr, cfg.masterPort)
	return c.trainLoop(ctx, cfg)
}

// trainLoop runs the FSDP training loop with a synthetic model.
func (c *TrainCommand) trainLoop(ctx context.Context, cfg *trainConfig) error {
	const paramSize = 64

	mdl, err := newTrainModel(paramSize)
	if err != nil {
		return fmt.Errorf("create model: %w", err)
	}

	sharded := fsdp.NewShardedModule[float32](mdl, cfg.rank, cfg.worldSize, nil)

	totalSteps := cfg.epochs * (paramSize / cfg.batchSize)
	if totalSteps == 0 {
		totalSteps = 1
	}

	step := 0
	start := time.Now()
	for epoch := 0; epoch < cfg.epochs; epoch++ {
		for batch := 0; batch < paramSize/cfg.batchSize; batch++ {
			select {
			case <-ctx.Done():
				fmt.Fprintf(c.out, "interrupted at epoch=%d step=%d\n", epoch+1, step+1)
				return nil
			default:
			}

			inputData := make([]float32, cfg.batchSize)
			for i := range inputData {
				inputData[i] = float32(step*cfg.batchSize+i) * 0.001
			}
			input, err := tensor.New[float32]([]int{cfg.batchSize}, inputData)
			if err != nil {
				return fmt.Errorf("create input: %w", err)
			}

			_, err = sharded.Forward(ctx, input)
			if err != nil {
				return fmt.Errorf("forward: %w", err)
			}

			var loss float32
			for _, v := range inputData {
				loss += v * v
			}
			loss /= float32(cfg.batchSize)

			gradData := make([]float32, paramSize)
			for i := range gradData {
				gradData[i] = float32(i) * float32(cfg.lr) * 0.01
			}
			grad, err := tensor.New[float32]([]int{paramSize}, gradData)
			if err != nil {
				return fmt.Errorf("create grad: %w", err)
			}

			_, err = sharded.Backward(ctx, grad, input)
			if err != nil {
				return fmt.Errorf("backward: %w", err)
			}

			step++
			elapsed := time.Since(start).Seconds()
			tokPerSec := float64(step*cfg.batchSize) / elapsed
			fmt.Fprintf(c.out, "epoch=%d step=%d/%d loss=%.6f tok/s=%.1f\n",
				epoch+1, step, totalSteps, loss, tokPerSec)
		}
	}

	if cfg.rank == 0 {
		if err := fsdp.SaveCheckpoint(cfg.outputPath, sharded, cfg.rank); err != nil {
			return fmt.Errorf("save checkpoint: %w", err)
		}
		fmt.Fprintf(c.out, "checkpoint saved to %s\n", cfg.outputPath)
	}

	return nil
}

// Static interface assertion.
var _ Command = (*TrainCommand)(nil)
