// Package parallel provides multi-GPU parallelism strategies for inference,
// including tensor parallelism and pipeline parallelism.
package parallel

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// PipelineParallelConfig configures pipeline parallelism across multiple GPUs.
type PipelineParallelConfig struct {
	// NumStages is the number of pipeline stages (typically one per GPU).
	NumStages int
	// NumLayers is the total number of transformer layers in the model.
	NumLayers int
	// MicroBatchSize is the number of micro-batches to split the input into.
	// More micro-batches reduce the bubble ratio but increase scheduling overhead.
	MicroBatchSize int
}

// Validate checks that the configuration is well-formed.
func (c PipelineParallelConfig) Validate() error {
	if c.NumStages < 1 {
		return errors.New("pipeline parallel: NumStages must be >= 1")
	}
	if c.NumLayers < 1 {
		return errors.New("pipeline parallel: NumLayers must be >= 1")
	}
	if c.NumStages > c.NumLayers {
		return errors.New("pipeline parallel: NumStages must be <= NumLayers")
	}
	if c.MicroBatchSize < 1 {
		return errors.New("pipeline parallel: MicroBatchSize must be >= 1")
	}
	return nil
}

// StageAssignment maps transformer layers to pipeline stages.
type StageAssignment struct {
	// StageForLayer maps layer index to stage index.
	StageForLayer []int
	// LayersPerStage stores the layer indices assigned to each stage.
	LayersPerStage [][]int
}

// AssignLayers distributes transformer layers across pipeline stages as
// evenly as possible. Remainder layers are distributed to the first stages.
func AssignLayers(numLayers, numStages int) StageAssignment {
	sa := StageAssignment{
		StageForLayer:  make([]int, numLayers),
		LayersPerStage: make([][]int, numStages),
	}

	base := numLayers / numStages
	remainder := numLayers % numStages

	layerIdx := 0
	for stage := range numStages {
		count := base
		if stage < remainder {
			count++
		}
		for range count {
			sa.StageForLayer[layerIdx] = stage
			sa.LayersPerStage[stage] = append(sa.LayersPerStage[stage], layerIdx)
			layerIdx++
		}
	}
	return sa
}

// LayerFunc is a function that processes a single transformer layer.
// It receives the layer index, the input activation tensor, and the engine
// for the stage that owns the layer. It returns the output activation.
type LayerFunc[T tensor.Numeric] func(layerIdx int, input *tensor.TensorNumeric[T], engine compute.Engine[T]) (*tensor.TensorNumeric[T], error)

// PipelineScheduler computes the execution schedule for pipeline parallelism.
// It determines the order of (stage, micro-batch) pairs for both forward passes
// (inference) and backward passes (training, not yet implemented).
type PipelineScheduler struct {
	config     PipelineParallelConfig
	assignment StageAssignment
}

// NewPipelineScheduler creates a scheduler for the given configuration.
func NewPipelineScheduler(cfg PipelineParallelConfig) (*PipelineScheduler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &PipelineScheduler{
		config:     cfg,
		assignment: AssignLayers(cfg.NumLayers, cfg.NumStages),
	}, nil
}

// Assignment returns the layer-to-stage assignment.
func (s *PipelineScheduler) Assignment() StageAssignment {
	return s.assignment
}

// ScheduleStep represents one unit of work: process a micro-batch on a stage.
type ScheduleStep struct {
	Stage      int
	MicroBatch int
}

// ForwardSchedule returns the GPipe-style forward schedule. Each step pairs
// a stage with a micro-batch index. In GPipe, all micro-batches flow through
// stage 0 first, then stage 1, etc., but micro-batches can overlap across
// stages. The schedule is ordered by clock cycle.
//
// For S stages and M micro-batches, the schedule has S+M-1 clock cycles.
// At each clock cycle c, all stages s where 0 <= c-s < M execute concurrently.
func (s *PipelineScheduler) ForwardSchedule() [][]ScheduleStep {
	numStages := s.config.NumStages
	numMB := s.config.MicroBatchSize
	numClocks := numStages + numMB - 1

	schedule := make([][]ScheduleStep, numClocks)
	for clock := range numClocks {
		var steps []ScheduleStep
		for stage := range numStages {
			mb := clock - stage
			if mb >= 0 && mb < numMB {
				steps = append(steps, ScheduleStep{Stage: stage, MicroBatch: mb})
			}
		}
		schedule[clock] = steps
	}
	return schedule
}

// BubbleRatio computes the fraction of idle time (bubbles) in the pipeline.
// For GPipe with S stages and M micro-batches:
//
//	bubble_ratio = (S - 1) / (S + M - 1)
//
// A lower ratio means better GPU utilization. With 4 stages and 16 micro-batches,
// the bubble ratio is 3/19 ~ 15.8%.
func (s *PipelineScheduler) BubbleRatio() float64 {
	numStages := float64(s.config.NumStages)
	numMB := float64(s.config.MicroBatchSize)
	return (numStages - 1) / (numStages + numMB - 1)
}

// PipelineExecutor runs micro-batches through the pipeline stages.
// Each stage has its own engine (typically mapped to a different GPU).
type PipelineExecutor[T tensor.Numeric] struct {
	scheduler *PipelineScheduler
	engines   []compute.Engine[T]
	layerFn   LayerFunc[T]
}

// NewPipelineExecutor creates an executor with one engine per stage.
// The engines slice must have exactly NumStages elements.
func NewPipelineExecutor[T tensor.Numeric](
	scheduler *PipelineScheduler,
	engines []compute.Engine[T],
	layerFn LayerFunc[T],
) (*PipelineExecutor[T], error) {
	if len(engines) != scheduler.config.NumStages {
		return nil, fmt.Errorf("pipeline parallel: need %d engines (one per stage), got %d",
			scheduler.config.NumStages, len(engines))
	}
	if layerFn == nil {
		return nil, errors.New("pipeline parallel: layerFn must not be nil")
	}
	return &PipelineExecutor[T]{
		scheduler: scheduler,
		engines:   engines,
		layerFn:   layerFn,
	}, nil
}

// Execute runs pipeline-parallel inference on the given micro-batches.
// microBatches must have exactly MicroBatchSize elements.
// Returns the output activation for each micro-batch after the final stage.
func (e *PipelineExecutor[T]) Execute(ctx context.Context, microBatches []*tensor.TensorNumeric[T]) ([]*tensor.TensorNumeric[T], error) {
	numMB := e.scheduler.config.MicroBatchSize
	if len(microBatches) != numMB {
		return nil, fmt.Errorf("pipeline parallel: expected %d micro-batches, got %d",
			numMB, len(microBatches))
	}

	numStages := e.scheduler.config.NumStages
	assignment := e.scheduler.assignment

	// activations[stage][mb] holds the output of stage for micro-batch mb.
	// activations[-1][mb] is the input micro-batch (before stage 0).
	activations := make([][](*tensor.TensorNumeric[T]), numStages+1)
	for i := range activations {
		activations[i] = make([]*tensor.TensorNumeric[T], numMB)
	}
	// Stage 0 inputs come from the micro-batches directly.
	for mb := range numMB {
		activations[0][mb] = microBatches[mb]
	}

	schedule := e.scheduler.ForwardSchedule()

	for _, clockSteps := range schedule {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// Execute all steps in this clock cycle concurrently.
		var wg sync.WaitGroup
		errs := make([]error, len(clockSteps))

		for i, step := range clockSteps {
			wg.Add(1)
			go func(idx int, s ScheduleStep) {
				defer wg.Done()
				input := activations[s.Stage][s.MicroBatch]
				if input == nil {
					errs[idx] = fmt.Errorf("pipeline parallel: nil activation at stage %d, micro-batch %d", s.Stage, s.MicroBatch)
					return
				}

				// Run all layers assigned to this stage sequentially.
				current := input
				for _, layerIdx := range assignment.LayersPerStage[s.Stage] {
					out, err := e.layerFn(layerIdx, current, e.engines[s.Stage])
					if err != nil {
						errs[idx] = fmt.Errorf("pipeline parallel: layer %d on stage %d: %w", layerIdx, s.Stage, err)
						return
					}
					current = out
				}

				// Store output for the next stage.
				if s.Stage+1 < numStages+1 {
					activations[s.Stage+1][s.MicroBatch] = current
				}
			}(i, step)
		}
		wg.Wait()

		for _, err := range errs {
			if err != nil {
				return nil, err
			}
		}
	}

	// Collect final outputs from the last stage.
	return activations[numStages], nil
}

// SplitMicroBatches splits a batch tensor along dimension 0 into n micro-batches.
// The batch size (dimension 0) must be divisible by n.
func SplitMicroBatches[T tensor.Numeric](
	batch *tensor.TensorNumeric[T],
	n int,
	engine compute.Engine[T],
) ([]*tensor.TensorNumeric[T], error) {
	shape := batch.Shape()
	if len(shape) == 0 {
		return nil, errors.New("pipeline parallel: cannot split scalar tensor")
	}
	batchSize := shape[0]
	if batchSize%n != 0 {
		return nil, fmt.Errorf("pipeline parallel: batch size %d not divisible by %d micro-batches", batchSize, n)
	}
	mbSize := batchSize / n

	data := batch.Data()
	stride := len(data) / batchSize

	microBatches := make([]*tensor.TensorNumeric[T], n)
	for i := range n {
		start := i * mbSize * stride
		end := (i + 1) * mbSize * stride
		mbShape := make([]int, len(shape))
		copy(mbShape, shape)
		mbShape[0] = mbSize
		mb, err := tensor.New[T](mbShape, data[start:end])
		if err != nil {
			return nil, fmt.Errorf("pipeline parallel: create micro-batch %d: %w", i, err)
		}
		microBatches[i] = mb
	}
	return microBatches, nil
}

// ConcatMicroBatches concatenates micro-batch outputs along dimension 0.
func ConcatMicroBatches[T tensor.Numeric](
	microBatches []*tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if len(microBatches) == 0 {
		return nil, errors.New("pipeline parallel: no micro-batches to concatenate")
	}
	if len(microBatches) == 1 {
		return microBatches[0], nil
	}

	baseShape := microBatches[0].Shape()
	totalBatch := 0
	for i, mb := range microBatches {
		s := mb.Shape()
		if len(s) != len(baseShape) {
			return nil, fmt.Errorf("pipeline parallel: micro-batch %d has %d dims, expected %d", i, len(s), len(baseShape))
		}
		for d := 1; d < len(s); d++ {
			if s[d] != baseShape[d] {
				return nil, fmt.Errorf("pipeline parallel: micro-batch %d dim %d is %d, expected %d", i, d, s[d], baseShape[d])
			}
		}
		totalBatch += s[0]
	}

	outShape := make([]int, len(baseShape))
	copy(outShape, baseShape)
	outShape[0] = totalBatch

	var allData []T
	for _, mb := range microBatches {
		allData = append(allData, mb.Data()...)
	}

	return tensor.New[T](outShape, allData)
}
