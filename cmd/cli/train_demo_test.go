package cli

import (
	"context"
	"fmt"
	"github.com/zerfoo/zerfoo/distributed/fsdp"
	"github.com/zerfoo/zerfoo/training"
	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"time"
)

// Synthetic FSDP demonstration retained only as a test fixture.
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
