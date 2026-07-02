// Command train_distributed launches distributed training using FSDP.
//
// Usage:
//
//	zerfoo train-distributed --config model.gguf --data train.jsonl --world-size 2 --rank 0
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/zerfoo/zerfoo/distributed/coordinator"
	"github.com/zerfoo/zerfoo/distributed/fsdp"
	"github.com/zerfoo/zerfoo/training/optimizer"
	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/numeric"
	"github.com/zerfoo/ztensor/tensor"
)

// config holds parsed CLI flags.
type config struct {
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

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintf(os.Stderr, "train-distributed: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags(args []string) (*config, error) {
	fs := flag.NewFlagSet("train-distributed", flag.ContinueOnError)
	cfg := &config{}

	fs.StringVar(&cfg.modelPath, "config", "", "path to GGUF model file (required)")
	fs.StringVar(&cfg.dataPath, "data", "", "path to training data file (required)")
	fs.IntVar(&cfg.worldSize, "world-size", 1, "number of GPUs / processes")
	fs.IntVar(&cfg.rank, "rank", 0, "this process rank (0 = coordinator)")
	fs.StringVar(&cfg.masterAddr, "master-addr", "localhost", "coordinator address")
	fs.IntVar(&cfg.masterPort, "master-port", 29500, "coordinator port")
	fs.StringVar(&cfg.outputPath, "output", "checkpoint.gguf", "checkpoint output path")
	fs.IntVar(&cfg.epochs, "epochs", 1, "number of training epochs")
	fs.IntVar(&cfg.batchSize, "batch-size", 4, "batch size")
	fs.Float64Var(&cfg.lr, "lr", 1e-4, "learning rate")

	if err := fs.Parse(args); err != nil {
		return nil, err
	}

	if cfg.modelPath == "" {
		return nil, fmt.Errorf("--config is required")
	}
	if cfg.dataPath == "" {
		return nil, fmt.Errorf("--data is required")
	}
	if cfg.worldSize < 1 {
		return nil, fmt.Errorf("--world-size must be >= 1")
	}
	if cfg.rank < 0 || cfg.rank >= cfg.worldSize {
		return nil, fmt.Errorf("--rank must be in [0, world-size)")
	}
	if cfg.masterPort < 0 || cfg.masterPort > 65535 {
		return nil, fmt.Errorf("--master-port must be in [0, 65535]")
	}

	return cfg, nil
}

func run(args []string, stdout, stderr io.Writer) error {
	cfg, err := parseFlags(args)
	if err != nil {
		return err
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	fmt.Fprintf(stdout, "train-distributed: rank=%d world-size=%d master=%s:%d\n",
		cfg.rank, cfg.worldSize, cfg.masterAddr, cfg.masterPort)
	fmt.Fprintf(stdout, "  model=%s data=%s output=%s\n", cfg.modelPath, cfg.dataPath, cfg.outputPath)
	fmt.Fprintf(stdout, "  epochs=%d batch-size=%d lr=%.1e\n", cfg.epochs, cfg.batchSize, cfg.lr)

	if cfg.rank == 0 {
		return runCoordinator(ctx, cfg, stdout)
	}
	return runWorker(ctx, cfg, stdout)
}

// runCoordinator starts the coordinator gRPC server and runs the training loop.
func runCoordinator(ctx context.Context, cfg *config, out io.Writer) error {
	addr := fmt.Sprintf("%s:%d", cfg.masterAddr, cfg.masterPort)
	coord := coordinator.NewCoordinator(out, 30*time.Second)
	if err := coord.Start(addr); err != nil {
		return fmt.Errorf("coordinator start: %w", err)
	}
	defer coord.Stop()

	fmt.Fprintf(out, "coordinator listening on %s\n", addr)

	return trainLoop(ctx, cfg, out)
}

// runWorker connects to the coordinator and runs the training loop.
func runWorker(ctx context.Context, cfg *config, out io.Writer) error {
	fmt.Fprintf(out, "worker rank=%d connecting to %s:%d\n", cfg.rank, cfg.masterAddr, cfg.masterPort)
	return trainLoop(ctx, cfg, out)
}

// stubModel implements training.Model[float32] for the FSDP trainer.
type stubModel struct {
	params []*graph.Parameter[float32]
}

func newStubModel(paramSize int) (*stubModel, error) {
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
	return &stubModel{params: []*graph.Parameter[float32]{p}}, nil
}

func (m *stubModel) Parameters() []*graph.Parameter[float32] { return m.params }

func (m *stubModel) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	if len(inputs) == 0 {
		return nil, fmt.Errorf("no inputs")
	}
	return inputs[0], nil
}

func (m *stubModel) Backward(_ context.Context, grad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	// Set gradient on the parameter.
	if len(m.params) > 0 {
		m.params[0].Gradient = grad
	}
	return []*tensor.TensorNumeric[float32]{grad}, nil
}

// trainLoop runs the FSDP training loop with a synthetic model and AdamW optimizer.
func trainLoop(ctx context.Context, cfg *config, out io.Writer) error {
	const paramSize = 64

	model, err := newStubModel(paramSize)
	if err != nil {
		return fmt.Errorf("create model: %w", err)
	}

	sharded := fsdp.NewShardedModule[float32](model, cfg.rank, cfg.worldSize, nil)

	// Create compute engine and AdamW optimizer.
	ops := numeric.Float32Ops{}
	engine := compute.NewCPUEngine[float32](ops)
	opt := optimizer.NewAdamW[float32](engine, float32(cfg.lr), 0.9, 0.999, 1e-8, 0.01)
	params := model.Parameters()

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
				fmt.Fprintf(out, "interrupted at epoch=%d step=%d\n", epoch+1, step+1)
				return nil
			default:
			}

			// Synthetic input and gradient.
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

			// Compute synthetic loss (MSE-like).
			var loss float32
			for _, v := range inputData {
				loss += v * v
			}
			loss /= float32(cfg.batchSize)

			// Synthetic gradient for backward.
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

			// Optimizer step: update parameters using AdamW.
			if err := opt.Step(ctx, params); err != nil {
				return fmt.Errorf("optimizer step: %w", err)
			}

			step++
			elapsed := time.Since(start).Seconds()
			tokPerSec := float64(step*cfg.batchSize) / elapsed
			fmt.Fprintf(out, "epoch=%d step=%d/%d loss=%.6f tok/s=%.1f\n",
				epoch+1, step, totalSteps, loss, tokPerSec)
		}
	}

	// Save checkpoint from rank 0.
	if cfg.rank == 0 {
		if err := fsdp.SaveCheckpoint(cfg.outputPath, sharded, cfg.rank); err != nil {
			return fmt.Errorf("save checkpoint: %w", err)
		}
		fmt.Fprintf(out, "checkpoint saved to %s\n", cfg.outputPath)
	}

	return nil
}
