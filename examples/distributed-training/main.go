// Command distributed-training demonstrates setting up FSDP distributed
// training with gradient accumulation using the zerfoo distributed and
// training packages.
//
// This example uses synthetic data and a toy model. In production, replace
// the toy model with a real architecture and the synthetic data with your
// dataset loader.
//
// Usage:
//
//	go build -o distributed-training ./examples/distributed-training/
//	./distributed-training
package main

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/zerfoo/ztensor/graph"
	"github.com/zerfoo/ztensor/tensor"
	"github.com/zerfoo/zerfoo/distributed"
	"github.com/zerfoo/zerfoo/distributed/fsdp"
	"github.com/zerfoo/zerfoo/training"
)

// toyModel is a minimal model that satisfies training.Model[float32].
// It has two trainable parameter tensors and performs a simple linear
// transformation: output = input * weights + bias.
type toyModel struct {
	weights *graph.Parameter[float32]
	bias    *graph.Parameter[float32]
}

func newToyModel(inputDim, outputDim int) *toyModel {
	rng := rand.New(rand.NewPCG(42, 0))
	wData := make([]float32, inputDim*outputDim)
	for i := range wData {
		wData[i] = float32(rng.NormFloat64()) * 0.01
	}
	bData := make([]float32, outputDim)

	wTensor, err := tensor.New[float32]([]int{inputDim, outputDim}, wData)
	if err != nil {
		panic(fmt.Sprintf("creating weight tensor: %v", err))
	}
	bTensor, err := tensor.New[float32]([]int{outputDim}, bData)
	if err != nil {
		panic(fmt.Sprintf("creating bias tensor: %v", err))
	}

	wParam, err := graph.NewParameter("linear.weight", wTensor, tensor.New[float32])
	if err != nil {
		panic(fmt.Sprintf("creating weight parameter: %v", err))
	}
	bParam, err := graph.NewParameter("linear.bias", bTensor, tensor.New[float32])
	if err != nil {
		panic(fmt.Sprintf("creating bias parameter: %v", err))
	}

	return &toyModel{
		weights: wParam,
		bias:    bParam,
	}
}

func (m *toyModel) Forward(_ context.Context, inputs ...*tensor.TensorNumeric[float32]) (*tensor.TensorNumeric[float32], error) {
	// Simplified forward: just return bias (real models do matrix math via engine).
	return m.bias.Value, nil
}

func (m *toyModel) Backward(_ context.Context, grad *tensor.TensorNumeric[float32], _ ...*tensor.TensorNumeric[float32]) ([]*tensor.TensorNumeric[float32], error) {
	return []*tensor.TensorNumeric[float32]{grad}, nil
}

func (m *toyModel) Parameters() []*graph.Parameter[float32] {
	return []*graph.Parameter[float32]{m.weights, m.bias}
}

// Verify toyModel implements training.Model[float32].
var _ training.Model[float32] = (*toyModel)(nil)

func main() {
	const (
		inputDim    = 64
		outputDim   = 16
		worldSize   = 2 // simulated number of GPUs
		microSteps  = 4 // gradient accumulation steps
		totalSteps  = 8
	)

	fmt.Println("=== Distributed Training with FSDP Example ===")
	fmt.Printf("World size: %d, Gradient accumulation steps: %d\n\n", worldSize, microSteps)

	// --- Step 1: Create the model ---
	model := newToyModel(inputDim, outputDim)
	fmt.Printf("Created toy model with %d parameters\n", len(model.Parameters()))
	for _, p := range model.Parameters() {
		fmt.Printf("  %s: shape %v (%d elements)\n", p.Name, p.Value.Shape(), len(p.Value.Data()))
	}

	// --- Step 2: Configure a WorkerNode ---
	// In production, each rank runs in a separate process with its own
	// WorkerNode. The coordinator address and worker addresses come from
	// your cluster orchestrator (e.g., Kubernetes).
	workerCfg := distributed.WorkerNodeConfig{
		WorkerAddress:      "localhost:50051",
		CoordinatorAddress: "localhost:50050",
		WorldSize:          worldSize,
	}
	fmt.Printf("\nWorkerNode config: worker=%s coordinator=%s world_size=%d\n",
		workerCfg.WorkerAddress, workerCfg.CoordinatorAddress, workerCfg.WorldSize)

	// --- Step 3: Wrap the model with FSDP sharding ---
	// In a real setup, you pass a live NCCLComm. Here we pass nil to
	// demonstrate the API structure without requiring multiple GPUs.
	//
	// NOTE: This will panic if you actually call AllGather/ReduceScatter
	// with a nil comm. The purpose is to show the API shape.
	fmt.Println("\n--- FSDP Sharding (simulated, rank 0) ---")
	fmt.Println("In production, each rank creates a ShardedModule with a live NCCLComm:")
	fmt.Println("  sharded := fsdp.NewShardedModule(model, rank, worldSize, comm)")
	fmt.Println("  accum   := fsdp.NewGradAccum(sharded, microSteps)")

	// We demonstrate GradAccum with a nil-comm ShardedModule to show the
	// gradient accumulation logic (which does not touch NCCL).
	sharded := fsdp.NewShardedModule[float32](model, 0, worldSize, nil)
	accum := fsdp.NewGradAccum[float32](sharded, microSteps)

	// --- Step 4: Simulate a training loop with gradient accumulation ---
	fmt.Printf("\n--- Training Loop (%d micro-steps) ---\n", totalSteps)
	rng := rand.New(rand.NewPCG(123, 0))

	for step := 0; step < totalSteps; step++ {
		// Generate synthetic gradients (in production, these come from backward()).
		grads := map[string][]float32{
			"linear.weight": syntheticGradients(rng, inputDim*outputDim/worldSize),
			"linear.bias":   syntheticGradients(rng, outputDim/worldSize),
		}

		ready := accum.Accumulate(grads)
		fmt.Printf("  Step %d: accumulated gradients, ready=%v\n", step+1, ready)

		if ready {
			averaged := accum.Sync()
			fmt.Printf("  -> Synced! Averaged gradients for %d parameters\n", len(averaged))
			for name, g := range averaged {
				fmt.Printf("     %s: %d elements, first=%.6f\n", name, len(g), g[0])
			}
			// In production, apply averaged gradients via optimizer:
			//   optimizer.Step(averaged)
		}
	}

	fmt.Println("\n=== Done ===")
	fmt.Println("In production, call workerNode.Stop() for graceful shutdown.")
	os.Exit(0)
}

// syntheticGradients generates random gradient values for demonstration.
func syntheticGradients(rng *rand.Rand, size int) []float32 {
	g := make([]float32, size)
	for i := range g {
		g[i] = float32(rng.NormFloat64()) * 0.001
	}
	return g
}
