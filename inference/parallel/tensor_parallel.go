package parallel

import (
	"context"
	"fmt"

	"github.com/zerfoo/ztensor/compute"
	"github.com/zerfoo/ztensor/tensor"
)

// TensorParallelConfig configures tensor parallelism across GPUs.
type TensorParallelConfig struct {
	// NumGPUs is the number of devices to split layers across.
	NumGPUs int
	// DeviceIDs identifies each device. Length must equal NumGPUs.
	DeviceIDs []int
}

// Validate checks that the configuration is consistent.
func (c *TensorParallelConfig) Validate() error {
	if c.NumGPUs < 1 {
		return fmt.Errorf("parallel: NumGPUs must be >= 1, got %d", c.NumGPUs)
	}
	if len(c.DeviceIDs) != c.NumGPUs {
		return fmt.Errorf("parallel: DeviceIDs length %d != NumGPUs %d", len(c.DeviceIDs), c.NumGPUs)
	}
	return nil
}

// AllReducer performs an element-wise sum reduction across devices.
// Implementations may use NCCL for real multi-GPU, or a simple in-process
// sum for CPU-based testing.
type AllReducer[T tensor.Numeric] interface {
	// AllReduceSum sums the tensor across all ranks and returns the result.
	// The returned tensor replaces the input on each rank.
	AllReduceSum(ctx context.Context, t *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error)
}

// SplitMode describes how a weight matrix is partitioned.
type SplitMode int

const (
	// ColumnSplit splits the weight along the output (column) dimension.
	// Each shard computes a slice of the output; results are concatenated.
	ColumnSplit SplitMode = iota
	// RowSplit splits the weight along the input (row) dimension.
	// Each shard computes a partial sum; results are reduced via AllReduce.
	RowSplit
)

// ShardedWeight holds one device's slice of a weight matrix along with
// metadata about the split.
type ShardedWeight[T tensor.Numeric] struct {
	// Shard is the weight slice assigned to this rank.
	Shard *tensor.TensorNumeric[T]
	// Rank is the device index for this shard.
	Rank int
	// Mode indicates column or row split.
	Mode SplitMode
}

// SplitLinearColumnWise splits a 2-D weight tensor [inFeatures, outFeatures]
// along the output dimension into numShards equal parts.
// Each shard has shape [inFeatures, outFeatures/numShards].
func SplitLinearColumnWise[T tensor.Numeric](
	engine compute.Engine[T],
	weight *tensor.TensorNumeric[T],
	numShards int,
) ([]*ShardedWeight[T], error) {
	shape := weight.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("parallel: SplitLinearColumnWise requires 2-D weight, got shape %v", shape)
	}
	outFeatures := shape[1]
	if outFeatures%numShards != 0 {
		return nil, fmt.Errorf("parallel: output dimension %d not divisible by %d shards", outFeatures, numShards)
	}

	parts, err := engine.Split(context.Background(), weight, numShards, 1)
	if err != nil {
		return nil, fmt.Errorf("parallel: column split: %w", err)
	}

	shards := make([]*ShardedWeight[T], numShards)
	for i, part := range parts {
		shards[i] = &ShardedWeight[T]{
			Shard: part,
			Rank:  i,
			Mode:  ColumnSplit,
		}
	}
	return shards, nil
}

// SplitLinearRowWise splits a 2-D weight tensor [inFeatures, outFeatures]
// along the input dimension into numShards equal parts.
// Each shard has shape [inFeatures/numShards, outFeatures].
func SplitLinearRowWise[T tensor.Numeric](
	engine compute.Engine[T],
	weight *tensor.TensorNumeric[T],
	numShards int,
) ([]*ShardedWeight[T], error) {
	shape := weight.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("parallel: SplitLinearRowWise requires 2-D weight, got shape %v", shape)
	}
	inFeatures := shape[0]
	if inFeatures%numShards != 0 {
		return nil, fmt.Errorf("parallel: input dimension %d not divisible by %d shards", inFeatures, numShards)
	}

	parts, err := engine.Split(context.Background(), weight, numShards, 0)
	if err != nil {
		return nil, fmt.Errorf("parallel: row split: %w", err)
	}

	shards := make([]*ShardedWeight[T], numShards)
	for i, part := range parts {
		shards[i] = &ShardedWeight[T]{
			Shard: part,
			Rank:  i,
			Mode:  RowSplit,
		}
	}
	return shards, nil
}

// ColumnParallelLinear computes a column-parallel linear projection.
// The weight has already been split column-wise: shard shape [inFeatures, outFeatures/N].
// Input shape:  [batch, seqLen, inFeatures].
// Output shape: [batch, seqLen, outFeatures/N] (partial; caller concatenates across ranks).
func ColumnParallelLinear[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	input *tensor.TensorNumeric[T],
	shard *ShardedWeight[T],
) (*tensor.TensorNumeric[T], error) {
	return engine.MatMul(ctx, input, shard.Shard)
}

// RowParallelLinear computes a row-parallel linear projection followed by
// AllReduce to sum partial results across ranks.
// The weight has been split row-wise: shard shape [inFeatures/N, outFeatures].
// The input must be the rank-local slice: shape [batch, seqLen, inFeatures/N].
// Output shape after AllReduce: [batch, seqLen, outFeatures].
func RowParallelLinear[T tensor.Numeric](
	ctx context.Context,
	engine compute.Engine[T],
	input *tensor.TensorNumeric[T],
	shard *ShardedWeight[T],
	reducer AllReducer[T],
) (*tensor.TensorNumeric[T], error) {
	partial, err := engine.MatMul(ctx, input, shard.Shard)
	if err != nil {
		return nil, fmt.Errorf("parallel: row-parallel matmul: %w", err)
	}
	return reducer.AllReduceSum(ctx, partial)
}

// TensorParallelLayer represents a single transformer linear layer that has
// been split across multiple ranks for tensor-parallel execution.
type TensorParallelLayer[T tensor.Numeric] struct {
	// Shards holds one weight shard per rank.
	Shards []*ShardedWeight[T]
	// Mode is the split strategy.
	Mode SplitMode
}

// TensorParallelWrapper coordinates tensor-parallel execution of a
// transformer's linear layers across N simulated or real GPU ranks.
type TensorParallelWrapper[T tensor.Numeric] struct {
	config  TensorParallelConfig
	engines []compute.Engine[T]
	reducer AllReducer[T]
	layers  []*TensorParallelLayer[T]
}

// NewTensorParallelWrapper creates a wrapper that distributes linear layers
// across engines (one per rank). The engines slice length must equal
// config.NumGPUs. The reducer is called after row-parallel layers to sum
// partial results.
func NewTensorParallelWrapper[T tensor.Numeric](
	config TensorParallelConfig,
	engines []compute.Engine[T],
	reducer AllReducer[T],
) (*TensorParallelWrapper[T], error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if len(engines) != config.NumGPUs {
		return nil, fmt.Errorf("parallel: engines length %d != NumGPUs %d", len(engines), config.NumGPUs)
	}
	if reducer == nil {
		return nil, fmt.Errorf("parallel: AllReducer must not be nil")
	}
	return &TensorParallelWrapper[T]{
		config:  config,
		engines: engines,
		reducer: reducer,
	}, nil
}

// AddColumnParallelLayer splits a weight column-wise and registers the
// resulting shards as a new layer.
func (w *TensorParallelWrapper[T]) AddColumnParallelLayer(
	weight *tensor.TensorNumeric[T],
) error {
	shards, err := SplitLinearColumnWise(w.engines[0], weight, w.config.NumGPUs)
	if err != nil {
		return err
	}
	w.layers = append(w.layers, &TensorParallelLayer[T]{
		Shards: shards,
		Mode:   ColumnSplit,
	})
	return nil
}

// AddRowParallelLayer splits a weight row-wise and registers the resulting
// shards as a new layer.
func (w *TensorParallelWrapper[T]) AddRowParallelLayer(
	weight *tensor.TensorNumeric[T],
) error {
	shards, err := SplitLinearRowWise(w.engines[0], weight, w.config.NumGPUs)
	if err != nil {
		return err
	}
	w.layers = append(w.layers, &TensorParallelLayer[T]{
		Shards: shards,
		Mode:   RowSplit,
	})
	return nil
}

// ForwardLayer executes one tensor-parallel layer on a single rank.
// For column-parallel: returns the rank's partial output (caller gathers).
// For row-parallel: returns the AllReduced full output.
func (w *TensorParallelWrapper[T]) ForwardLayer(
	ctx context.Context,
	layerIdx int,
	rank int,
	input *tensor.TensorNumeric[T],
) (*tensor.TensorNumeric[T], error) {
	if layerIdx < 0 || layerIdx >= len(w.layers) {
		return nil, fmt.Errorf("parallel: layer index %d out of range [0, %d)", layerIdx, len(w.layers))
	}
	if rank < 0 || rank >= w.config.NumGPUs {
		return nil, fmt.Errorf("parallel: rank %d out of range [0, %d)", rank, w.config.NumGPUs)
	}

	layer := w.layers[layerIdx]
	shard := layer.Shards[rank]
	engine := w.engines[rank]

	switch layer.Mode {
	case ColumnSplit:
		return ColumnParallelLinear(ctx, engine, input, shard)
	case RowSplit:
		return RowParallelLinear(ctx, engine, input, shard, w.reducer)
	default:
		return nil, fmt.Errorf("parallel: unknown split mode %d", layer.Mode)
	}
}

// NumLayers returns the number of registered parallel layers.
func (w *TensorParallelWrapper[T]) NumLayers() int {
	return len(w.layers)
}

// Config returns a copy of the tensor parallel configuration.
func (w *TensorParallelWrapper[T]) Config() TensorParallelConfig {
	return w.config
}

// SumAllReducer is a simple in-process AllReducer that sums partials from
// all ranks. It is intended for single-process CPU testing where multiple
// "ranks" are simulated with separate engine instances operating on
// partitioned tensors.
type SumAllReducer[T tensor.Numeric] struct {
	// partials collects one tensor per rank for the current AllReduce call.
	// In real multi-GPU usage, NCCL handles this; here we simulate it by
	// requiring the caller to register partials before calling AllReduceSum.
	partials []*tensor.TensorNumeric[T]
	engine   compute.Engine[T]
}

// NewSumAllReducer creates an AllReducer that sums tensors in-process.
// numRanks is the number of partials expected per reduction.
func NewSumAllReducer[T tensor.Numeric](engine compute.Engine[T], numRanks int) *SumAllReducer[T] {
	return &SumAllReducer[T]{
		partials: make([]*tensor.TensorNumeric[T], 0, numRanks),
		engine:   engine,
	}
}

// AddPartial registers a partial result from one rank. Once all partials
// are registered, AllReduceSum can be called.
func (r *SumAllReducer[T]) AddPartial(t *tensor.TensorNumeric[T]) {
	r.partials = append(r.partials, t)
}

// Reset clears accumulated partials for the next reduction round.
func (r *SumAllReducer[T]) Reset() {
	r.partials = r.partials[:0]
}

// AllReduceSum returns the element-wise sum of all registered partials.
// After the call, partials are cleared automatically.
func (r *SumAllReducer[T]) AllReduceSum(ctx context.Context, t *tensor.TensorNumeric[T]) (*tensor.TensorNumeric[T], error) {
	if len(r.partials) == 0 {
		// No other partials registered — this is the only rank or a
		// passthrough scenario. Return the input unchanged.
		return t, nil
	}

	result := r.partials[0]
	for i := 1; i < len(r.partials); i++ {
		var err error
		result, err = r.engine.Add(ctx, result, r.partials[i])
		if err != nil {
			return nil, fmt.Errorf("parallel: SumAllReducer: %w", err)
		}
	}
	// Add the current rank's tensor.
	var err error
	result, err = r.engine.Add(ctx, result, t)
	if err != nil {
		return nil, fmt.Errorf("parallel: SumAllReducer: %w", err)
	}

	r.partials = r.partials[:0]
	return result, nil
}
