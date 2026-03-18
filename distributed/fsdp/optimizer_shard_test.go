package fsdp

import (
	"math"
	"testing"
)

func TestShardedOptimizer(t *testing.T) {
	const worldSize = 8
	const fullParamSize = 1024

	t.Run("moment_tensors_sized_1_over_N", func(t *testing.T) {
		shardSize := fullParamSize / worldSize
		opt := NewShardedAdamW[float32](0, worldSize, 0.001, 0.9, 0.999, 1e-8, 0.01)

		// Do one step to initialize moment buffers.
		grads := map[string][]float32{
			"weight": make([]float32, shardSize),
		}
		for i := range grads["weight"] {
			grads["weight"][i] = 0.1
		}
		opt.Step(grads)

		// Verify moment buffer sizes.
		if len(opt.m1["weight"]) != shardSize {
			t.Errorf("m1 size = %d, want %d (fullParamSize/worldSize)", len(opt.m1["weight"]), shardSize)
		}
		if len(opt.m2["weight"]) != shardSize {
			t.Errorf("m2 size = %d, want %d (fullParamSize/worldSize)", len(opt.m2["weight"]), shardSize)
		}
	})

	t.Run("memory_bytes_leq_full_div_worldSize", func(t *testing.T) {
		shardSize := fullParamSize / worldSize
		opt := NewShardedAdamW[float32](0, worldSize, 0.001, 0.9, 0.999, 1e-8, 0.01)

		grads := map[string][]float32{
			"weight_1": make([]float32, shardSize),
			"weight_2": make([]float32, shardSize*2),
		}
		for k := range grads {
			for i := range grads[k] {
				grads[k][i] = 0.1
			}
		}
		opt.Step(grads)

		// Full optimizer would have 2 moment buffers * total params * 4 bytes.
		totalFullParams := fullParamSize + fullParamSize*2 // weight_1 + weight_2
		fullOptimizerBytes := int64(totalFullParams) * 2 * 4 // 2 moments * float32
		maxShardedBytes := fullOptimizerBytes / int64(worldSize)

		mem := opt.MemoryBytes()
		if mem > maxShardedBytes {
			t.Errorf("MemoryBytes() = %d, want <= %d (full_optimizer_bytes / worldSize)", mem, maxShardedBytes)
		}
		if mem != maxShardedBytes {
			t.Errorf("MemoryBytes() = %d, want exactly %d for evenly divisible sizes", mem, maxShardedBytes)
		}
	})

	t.Run("convergence_on_synthetic_quadratic", func(t *testing.T) {
		// Minimize f(x) = sum(x_i^2) using sharded optimizer.
		// Each rank holds a shard of x and its gradient (2*x_i).
		const shardSize = 16
		const steps = 200

		opt := NewShardedAdamW[float32](0, worldSize, 0.1, 0.9, 0.999, 1e-8, 0.0)

		// Initialize parameter shard.
		params := map[string][]float32{
			"x": make([]float32, shardSize),
		}
		for i := range params["x"] {
			params["x"][i] = float32(i+1) * 0.1 // non-zero initial values
		}

		initialLoss := quadraticLoss(params["x"])

		for step := 0; step < steps; step++ {
			// Compute gradient: df/dx_i = 2*x_i
			grads := map[string][]float32{
				"x": make([]float32, shardSize),
			}
			for i, v := range params["x"] {
				grads["x"][i] = 2 * v
			}
			opt.StepOnParams(params, grads)
		}

		finalLoss := quadraticLoss(params["x"])
		if finalLoss >= initialLoss {
			t.Errorf("loss did not decrease: initial=%f, final=%f", initialLoss, finalLoss)
		}
		// After 50 steps of AdamW on a simple quadratic, loss should decrease significantly.
		if finalLoss > initialLoss*0.01 {
			t.Errorf("loss did not converge sufficiently: initial=%f, final=%f (ratio=%f)",
				initialLoss, finalLoss, finalLoss/initialLoss)
		}
	})

	t.Run("all_ranks_independent_shards", func(t *testing.T) {
		const shardSize = 8
		// Each rank should be able to optimize independently.
		for rank := 0; rank < worldSize; rank++ {
			opt := NewShardedAdamW[float32](rank, worldSize, 0.01, 0.9, 0.999, 1e-8, 0.0)
			params := map[string][]float32{
				"w": make([]float32, shardSize),
			}
			for i := range params["w"] {
				params["w"][i] = float32(rank+1) * 0.1
			}

			initialLoss := quadraticLoss(params["w"])
			for step := 0; step < 20; step++ {
				grads := map[string][]float32{
					"w": make([]float32, shardSize),
				}
				for i, v := range params["w"] {
					grads["w"][i] = 2 * v
				}
				opt.StepOnParams(params, grads)
			}

			finalLoss := quadraticLoss(params["w"])
			if finalLoss >= initialLoss {
				t.Errorf("rank %d: loss did not decrease: initial=%f, final=%f", rank, initialLoss, finalLoss)
			}
		}
	})

	t.Run("weight_decay_applied", func(t *testing.T) {
		const shardSize = 4
		// With weight decay and zero gradients, parameters should shrink.
		opt := NewShardedAdamW[float32](0, worldSize, 0.1, 0.9, 0.999, 1e-8, 0.1)

		params := map[string][]float32{
			"w": {1.0, 2.0, 3.0, 4.0},
		}
		zeroGrads := map[string][]float32{
			"w": {0.0, 0.0, 0.0, 0.0},
		}

		initial := make([]float32, shardSize)
		copy(initial, params["w"])

		opt.StepOnParams(params, zeroGrads)

		for i, v := range params["w"] {
			if math.Abs(float64(v)) >= math.Abs(float64(initial[i])) {
				t.Errorf("param[%d]: weight decay did not shrink value: before=%f, after=%f", i, initial[i], v)
			}
		}
	})

	t.Run("step_counter_increments", func(t *testing.T) {
		opt := NewShardedAdamW[float32](0, worldSize, 0.001, 0.9, 0.999, 1e-8, 0.01)
		grads := map[string][]float32{"w": {0.1}}

		for i := 0; i < 5; i++ {
			opt.Step(grads)
			if opt.step != i+1 {
				t.Errorf("after %d steps, step counter = %d", i+1, opt.step)
			}
		}
	})
}

// quadraticLoss computes sum(x_i^2).
func quadraticLoss(x []float32) float32 {
	var loss float32
	for _, v := range x {
		loss += v * v
	}
	return loss
}
